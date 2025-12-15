#include "board.h"
#include "mcts.h"
#include "mcts_tree_openmp.h"
#include <chrono>
#include <iostream>
#include <string>

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

int main()
{
    // Setup initial board
    uint64_t white = BIT(3, 3) | BIT(4, 4);
    uint64_t black = BIT(3, 4) | BIT(4, 3);
    Board board(black, white);

    std::cout << "Benchmarking MCTS implementations..." << std::endl;
    std::cout << "Board state:\n" << board << std::endl;

    // 1. Serial MCTS
    {
        std::cout << "--- Serial MCTS ---" << std::endl;
        // Run for 5 seconds
        MCTS mcts_serial(std::chrono::milliseconds(5000));
        run_benchmark("Serial MCTS (5s)", mcts_serial, board);
    }

    // 2. Parallel MCTS (OpenMP)
    {
        std::cout << "--- Parallel MCTS (OpenMP) ---" << std::endl;
        // Run for 5 seconds
        MCTSTree mcts_parallel(std::chrono::milliseconds(5000));
        run_benchmark("Parallel MCTS (5s)", mcts_parallel, board);
    }

    return 0;
}
