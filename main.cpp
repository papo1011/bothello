#include "src/board.h"

#if defined(BOT_CPU)
#    include "src/mcts.h"
using BotType = MCTS;
#elif defined(BOT_OMP)
#    include "src/mcts_tree_openmp.h"
using BotType = MCTSTreeOMP;
#elif defined(BOT_LEAF)
#    include "src/mcts_leaf_parallel.h"
using BotType = MCTSLeafParallel;
#elif defined(BOT_BLOCK)
#    include "src/mcts_block.h"
using BotType = MCTSBlock;
#else // Default or BOT_CUDA
#    include "src/mcts_tree_cuda.h"
using BotType = MCTSTree;
#endif

#include <ctime>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <unistd.h>

// Helper to create a bitmask from coordinates (row 0-7, col 0-7)
// Assumes row 0 is top, col 0 is left (A).
constexpr uint64_t BIT(int row, int col) { return 1ULL << (row * 8 + col); }

int main()
{
    std::time_t t = std::time(nullptr);
    std::tm tm = *std::localtime(&t);

    std::ostringstream name_builder;
    name_builder << "sim_" << std::put_time(&tm, "%H_%M_%d_%m_%Y") << ".log";
    std::string filename = name_builder.str();

    std::ofstream log(filename);
    if (!log) {
        std::cerr << "Error: impossible to open file " << filename << "\n";
        return 1;
    }

    uint64_t white = BIT(3, 3) | BIT(4, 4);
    uint64_t black = BIT(3, 4) | BIT(4, 3);

    Board board(black, white);

    log << "Initial board:\n" << board << std::endl;

    BotType mcts(std::chrono::milliseconds(1000));

    int const max_moves = 100;
    int moves_played = 0;

    double pps_move1 = -1.0;
    double pps_move30 = -1.0;
    double pps_move55 = -1.0;

    for (int i = 0; i < max_moves; i++) {
        if (board.is_terminal())
            break;

        Move best_move = mcts.get_best_move(board);

        double pps = mcts.get_pps();
        moves_played++;

        if (moves_played == 1)
            pps_move1 = pps;
        else if (moves_played == 30)
            pps_move30 = pps;
        else if (moves_played == 55)
            pps_move55 = pps;

        log << "Move " << moves_played << ": ";
        if (moves_played % 2 == 1)
            log << "MCTS is playing Black *" << std::endl;
        else
            log << "MCTS is playing White O" << std::endl;

        log << "MCTS selected move: " << move_to_gtp(best_move) << std::endl;
        if (best_move == 0) {
            log << "Player passes." << std::endl;
        }
        log << "Playouts per second: " << pps << std::endl;

        board.move(best_move);

        log << "Board after move:\n" << board << std::endl;
    }

    if (pps_move1 >= 0.0)
        log << "PPS at move 1: " << pps_move1 << std::endl;
    if (pps_move30 >= 0.0)
        log << "PPS at move 30: " << pps_move30 << std::endl;
    if (pps_move55 >= 0.0)
        log << "PPS at move 55: " << pps_move55 << std::endl;

    int black_score, white_score;
    board.get_score(black_score, white_score);
    log << "Final score - Black: " << black_score << "  White: " << white_score
        << std::endl;

    if (black_score > white_score) {
        log << "Winner: Black" << std::endl;
    } else if (white_score > black_score) {
        log << "Winner: White" << std::endl;
    } else {
        log << "Result: Draw" << std::endl;
    }

    return 0;
}
