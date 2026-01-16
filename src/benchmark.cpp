#include "board.h"
#include "mcts.h"
#include "mcts_block.h"
#include "mcts_leaf_parallel.h"
#include "mcts_tree_cuda.h"
#include "mcts_tree_openmp.h"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include <cuda_runtime.h>

// Helper to create a bitmask from coordinates (row 0-7, col 0-7)
constexpr uint64_t BIT(int row, int col) { return 1ULL << (row * 8 + col); }

void run_benchmark(std::string name, MCTS &agent, Board const &board)
{
    std::cout << "Running benchmark for: " << name << std::endl;

    auto start = std::chrono::high_resolution_clock::now();
    Move move = agent.get_best_move(board);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    std::cout << "  Best Move: " << move_to_gtp(move) << std::endl;
    std::cout << "  Time: " << elapsed.count() << "s" << std::endl;
    std::cout << "  PPS: " << agent.get_pps() << std::endl;
    std::cout << "  --------------------------------" << std::endl;
}

// === ARENA IMPLEMENTATION ===

enum class BotType { Serial, TreeOMP, TreeCUDA, LeafCUDA, BlockCUDA };

struct BotConfig {
    std::string name;
    BotType type;
    std::chrono::milliseconds time_limit;
};

std::unique_ptr<MCTS> create_bot(BotType type, std::chrono::milliseconds time_limit)
{
    switch (type) {
    case BotType::Serial:
        return std::make_unique<MCTS>(time_limit);
    case BotType::TreeOMP:
        return std::make_unique<MCTSTreeOMP>(time_limit);
    case BotType::TreeCUDA:
        return std::make_unique<MCTSTree>(time_limit);
    case BotType::LeafCUDA:
        return std::make_unique<MCTSLeafParallel>(time_limit);
    case BotType::BlockCUDA:
        return std::make_unique<MCTSBlock>(time_limit);
    default:
        return std::make_unique<MCTS>(time_limit);
    }
}

struct EloEntry {
    double rating = 1200.0;
    int wins = 0;
    int losses = 0;
    int draws = 0;
    int games_played = 0;
};

// Returns 1.0 if p1 wins, 0.0 if p2 wins, 0.5 for draw
double play_match(BotConfig p1_config, BotConfig p2_config, bool verbose = false)
{
    auto p1 = create_bot(p1_config.type, p1_config.time_limit);
    auto p2 = create_bot(p2_config.type, p2_config.time_limit);

    // Setup initial board
    uint64_t white = BIT(3, 3) | BIT(4, 4);
    uint64_t black = BIT(3, 4) | BIT(4, 3);
    Board board(black, white);

    bool p1_turn = true; // Black starts
    int moves = 0;

    if (verbose) {
        std::cout << "Match: " << p1_config.name << " (Black) vs " << p2_config.name
                  << " (White)\n";
    }

    while (!board.is_terminal()) {
        Move move;
        if (p1_turn) {
            move = p1->get_best_move(board);
        } else {
            move = p2->get_best_move(board);
        }

        if (move == 0 && verbose)
            std::cout << "Pass\n";

        board.move(move);
        p1_turn = !p1_turn;
        moves++;
    }

    int black_score = board.is_black_turn()
                          ? Board::count_moves(board.get_curr_player_mask())
                          : Board::count_moves(board.get_opp_player_mask());
    int white_score = board.is_black_turn()
                          ? Board::count_moves(board.get_opp_player_mask())
                          : Board::count_moves(board.get_curr_player_mask());

    if (verbose) {
        std::cout << "Game Over. Black: " << black_score << ", White: " << white_score
                  << "\n";
    }

    double result;
    if (black_score > white_score)
        result = 1.0;
    else if (white_score > black_score)
        result = 0.0;
    else
        result = 0.5;
    
    // Explicitly destroy bots to free GPU memory
    p1.reset();
    p2.reset();
    
    // Force CUDA synchronization to ensure GPU resources are freed
    cudaDeviceSynchronize();
    
    return result;
}

void update_elo(EloEntry &r1, EloEntry &r2, double actual_score_p1)
{
    double k = 32.0;
    double expected_p1 = 1.0 / (1.0 + std::pow(10.0, (r2.rating - r1.rating) / 400.0));
    double expected_p2 = 1.0 - expected_p1;

    double actual_score_p2 = 1.0 - actual_score_p1;

    r1.rating += k * (actual_score_p1 - expected_p1);
    r2.rating += k * (actual_score_p2 - expected_p2);

    r1.games_played++;
    r2.games_played++;

    if (actual_score_p1 == 1.0) {
        r1.wins++;
        r2.losses++;
    } else if (actual_score_p1 == 0.0) {
        r1.losses++;
        r2.wins++;
    } else {
        r1.draws++;
        r2.draws++;
    }
}

void run_arena(int rounds_per_pair = 2, std::chrono::milliseconds time_limit = std::chrono::milliseconds(100))
{
    std::vector<BotConfig> bots = {
        {"Serial MCTS", BotType::Serial, time_limit},
        {"OMP MCTS", BotType::TreeOMP, time_limit},
        {"CUDA Tree", BotType::TreeCUDA, time_limit},
        {"CUDA Leaf", BotType::LeafCUDA, time_limit},
        {"CUDA Block", BotType::BlockCUDA, time_limit},
    };

    std::map<std::string, EloEntry> ratings;
    for (auto const &bot : bots) {
        ratings[bot.name] = EloEntry();
    }

    std::cout << "\n=== STARTING ARENA TOURNAMENT ===\n";
    std::cout << "Bots: " << bots.size() << "\n";
    std::cout << "Rounds per pair: " << rounds_per_pair
              << " (each plays Black and White)\n";

    for (size_t i = 0; i < bots.size(); ++i) {
        for (size_t j = i + 1; j < bots.size(); ++j) {
            for (int r = 0; r < rounds_per_pair; ++r) {
                // Game 1: i is Black, j is White
                double res1 = play_match(bots[i], bots[j]);
                update_elo(ratings[bots[i].name], ratings[bots[j].name], res1);
                std::cout << bots[i].name << " vs " << bots[j].name << ": "
                          << (res1 == 1.0 ? "P1 Win"
                                          : (res1 == 0.0 ? "P2 Win" : "Draw"))
                          << "\n";

                // Game 2: j is Black, i is White
                double res2 = play_match(bots[j], bots[i]);
                update_elo(ratings[bots[j].name], ratings[bots[i].name], res2);
                std::cout << bots[j].name << " vs " << bots[i].name << ": "
                          << (res2 == 1.0 ? "P1 Win"
                                          : (res2 == 0.0 ? "P2 Win" : "Draw"))
                          << "\n";
            }
        }
    }

    std::cout << "\n=== FINAL STANDINGS ===\n";
    std::cout << std::left << std::setw(20) << "Bot Name" << std::setw(10) << "Rating"
              << std::setw(8) << "Wins" << std::setw(8) << "Losses" << std::setw(8)
              << "Draws" << "\n";
    std::cout << "--------------------------------------------------------\n";

    // Sort by rating
    std::vector<std::pair<std::string, EloEntry>> sorted_ratings(ratings.begin(),
                                                                 ratings.end());
    std::sort(
        sorted_ratings.begin(), sorted_ratings.end(),
        [](auto const &a, auto const &b) { return a.second.rating > b.second.rating; });

    for (auto const &entry : sorted_ratings) {
        std::cout << std::left << std::setw(20) << entry.first << std::setw(10)
                  << std::fixed << std::setprecision(1) << entry.second.rating
                  << std::setw(8) << entry.second.wins << std::setw(8)
                  << entry.second.losses << std::setw(8) << entry.second.draws << "\n";
    }
}

int main()
{
    // Setup initial board
    uint64_t white = BIT(3, 3) | BIT(4, 4);
    uint64_t black = BIT(3, 4) | BIT(4, 3);
    Board board(black, white);

    std::cout << "Benchmarking MCTS implementations..." << std::endl;
    // std::cout << "Board state:\n" << board << std::endl;

    /////////////////////////////////////////////////////////////////
    std::cout << "PPS:" << std::endl;

    // 1. Serial MCTS
    {
        std::cout << "--- Serial MCTS ---" << std::endl;
        // Run for 5 seconds
        MCTS mcts_serial(std::chrono::milliseconds(5000));
        run_benchmark("Serial MCTS (5s)", mcts_serial, board);
    }

    // 2. Parallel Tree MCTS (OpenMP)
    {
        std::cout << "--- Parallel Tree MCTS (OpenMP) ---" << std::endl;
        // Run for 5 seconds
        MCTSTreeOMP mcts_parallel(std::chrono::milliseconds(5000));
        run_benchmark("Parallel MCTS (5s)", mcts_parallel, board);
    }
    // 3. Parallel Tree MCTS (CUDA)
    {
        std::cout << "--- Parallel Tree MCTS (CUDA) ---" << std::endl;
        MCTSTree mcts_parallel(std::chrono::milliseconds(5000));
        run_benchmark("Parallel MCTS (5s)", mcts_parallel, board);
    }

    // 4. Leaf Parallel MCTS (CUDA)
    {
        std::cout << "--- Leaf Parallel MCTS (CUDA) ---" << std::endl;
        MCTSLeafParallel mcts_leaf(std::chrono::milliseconds(5000));
        run_benchmark("Leaf Parallel MCTS (5s)", mcts_leaf, board);
    }

    // 5. Block Parallel MCTS (CUDA)
    {
        std::cout << "--- Block Parallel MCTS (CUDA) ---" << std::endl;
        MCTSBlock mcts_block(std::chrono::milliseconds(5000));
        run_benchmark("Block Parallel MCTS (5s)", mcts_block, board);
    }

    // Run Arena with reduced time limit to prevent tree explosion
    run_arena(1, std::chrono::milliseconds(1000)); // 1 second per move

    return 0;
}
