#include "board.h"
#include "mcts.h"
#include "mcts_block.h"
#include "mcts_leaf_parallel.h"
#include "mcts_tree_cuda.h"
#include "mcts_tree_openmp.h"
#include <chrono>
#include <cstdint>
#include <iostream>
#include <memory>
#include <sstream>
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

    /////////////////////////////////////////////////////////////////
    std::unique_ptr<MCTS> player1 =
        std::make_unique<MCTS>(std::chrono::milliseconds(1000));
    std::unique_ptr<MCTS> player2 =
        std::make_unique<MCTSLeafParallel>(std::chrono::milliseconds(1000));

    std::cout << "=== BOTHELLO VERSUS ARENA ===\n";
    std::cout << "Player 1 (Black): CPU MCTS\n";
    std::cout << "Player 2 (White): GPU Leaf Parallel MCTS\n";
    std::cout << "Time Config: " << 1000 << " ms per move\n";
    std::cout << "=============================\n\n";

    int turn = 0;
    static bool is_p1_turn = true; // Player 1 (Black) starts

    while (!board.is_terminal()) {
        turn++;
        std::cout << "\n--- Turn " << turn << " ---\n";
        std::cout << board << "\n";

        std::string p_name =
            is_p1_turn ? "Black (CPU MCTS)" : "White (GPU Leaf Parallel MCTS)";
        std::cout << p_name << " to move...\n";

        Move best_move = 0;
        double pps = 0;

        if (is_p1_turn) {
            best_move = player1->get_best_move(board);
            pps = player1->get_pps();
        } else {
            best_move = player2->get_best_move(board);
            pps = player2->get_pps();
        }

        if (best_move == 0)
            std::cout << "Player passes.\n";
        else
            std::cout << "Selected move: " << move_to_gtp(best_move) << "\n";
        std::cout << "Performance: " << pps << " PPS\n";

        board.move(best_move);
        is_p1_turn = !is_p1_turn;
    }

    std::cout << "\nGame Over!\n";
    std::cout << "Final Board:\n" << board << "\n";

    // Count score
    std::stringstream ss;
    ss << board;

    std::string s = ss.str();
    int b_count = 0;
    int w_count = 0;
    for (char c : s) {
        if (c == '*')
            b_count++;
        if (c == 'O')
            w_count++;
    }

    std::cout << "Final Score - Black (P1): " << b_count << " | White (P2): " << w_count
              << "\n";
    if (b_count > w_count)
        std::cout << "Winner: Player 1 (Black)\n";
    else if (w_count > b_count)
        std::cout << "Winner: Player 2 (White)\n";
    else
        std::cout << "Draw\n";
    return 0;
}
