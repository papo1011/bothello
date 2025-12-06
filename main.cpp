#include "src/board.h"
#include "src/mcts.h"
#include <iostream>
#include <unistd.h>

// Helper to create a bitmask from coordinates (row 0-7, col 0-7)
// Assumes row 0 is top, col 0 is left (A).
constexpr uint64_t BIT(int row, int col) { return 1ULL << (row * 8 + col); }

int main()
{
    // Standard Othello starting position:
    // White: D4 (3,3) and E5 (4,4)
    // Black: E4 (3,4) and D5 (4,3)
    uint64_t white = BIT(3, 3) | BIT(4, 4);
    uint64_t black = BIT(3, 4) | BIT(4, 3);

    Board board(black, white);

    std::cout << "Initial board:\n" << board << std::endl;

    MCTS mcts(std::chrono::milliseconds(10000));

    Move best_move = mcts.get_best_move(board);

    std::cout << "MCTS selected move: " << move_to_gtp(best_move) << std::endl;

    board.move(best_move);

    // Just to print the correct symbols after one move.
    std::cout << "Board after move:\n" << board << std::endl;

    best_move = mcts.get_best_move(board);

    std::cout << "MCTS selected move: " << move_to_gtp(best_move) << std::endl;
    board.move(best_move);
    std::cout << "Board after move:\n" << board << std::endl;

    best_move = mcts.get_best_move(board);

    std::cout << "MCTS selected move: " << move_to_gtp(best_move) << std::endl;
    board.move(best_move);
    std::cout << "Board after move:\n" << board << std::endl;
    return 0;
}
