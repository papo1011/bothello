#include "src/board.h"
#include "src/mcts.h"
#include <iostream>
#include <unistd.h>

// Helper to create a bitmask from coordinates (row 0-7, col 0-7)
// Assumes row 0 is top, col 0 is left (A).
constexpr uint64_t BIT(int row, int col) { return 1ULL << (row * 8 + col); }

int main()
{
    uint64_t white = BIT(3, 3) | BIT(4, 4);
    uint64_t black = BIT(3, 4) | BIT(4, 3);

    Board board(black, white);

    std::cout << "Initial board:\n" << board << std::endl;

    MCTS mcts(std::chrono::milliseconds(10 * 1000));

    int const max_moves = 100;
    double sum_pps = 0.0;
    int moves_played = 0;

    for (int i = 0; i < max_moves; i++) {
        if (board.is_terminal())
            break;

        Move best_move = mcts.get_best_move(board);

        double pps = mcts.get_pps();
        sum_pps += pps;
        moves_played++;

        std::cout << "Move " << (i + 1) << ": ";
        if (i % 2 == 0)
            std::cout << "MCTS is playing Black *" << std::endl;
        else
            std::cout << "MCTS is playing White O" << std::endl;
        std::cout << "MCTS selected move: " << move_to_gtp(best_move) << std::endl;
        if (best_move == 0) {
            std::cout << "Player passes." << std::endl;
        }
        std::cout << "Playouts per second: " << pps << std::endl;

        board.move(best_move);

        std::cout << "Board after move:\n" << board << std::endl;
    }

    if (moves_played > 0) {
        double avg_pps = sum_pps / moves_played;
        std::cout << "Average playouts per second over " << moves_played
                  << " moves: " << avg_pps << std::endl;
    }

    auto [black_score, white_score] = board.score();
    std::cout << "Final score - Black: " << black_score << "  White: " << white_score
              << std::endl;

    if (black_score > white_score) {
        std::cout << "Winner: Black" << std::endl;
    } else if (white_score > black_score) {
        std::cout << "Winner: White" << std::endl;
    } else {
        std::cout << "Result: Draw" << std::endl;
    }

    return 0;
}
