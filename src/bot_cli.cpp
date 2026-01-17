#include "src/board.h"
#include "src/mcts.h"
#include "src/mcts_tree_openmp.h"
#include "src/mcts_leaf_parallel.h"
#include "src/mcts_block.h"
#include "src/mcts_tree_cuda.h"

#include <chrono>
#include <iostream>
#include <memory>
#include <string>
#include <sstream>

static void print_usage()
{
    std::cerr << "Usage: bot_cli --bot <cpu|omp|leaf|block|cuda> --black <uint64> --white <uint64> [--time-ms <ms>] [--is-black-turn <0|1>]\n";
}

int main(int argc, char **argv)
{
    std::string bot = "cpu";
    uint64_t black = 0;
    uint64_t white = 0;
    int time_ms = 5000;
    bool is_black_turn = true;

    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if (a == "--bot" && i + 1 < argc) {
            bot = argv[++i];
        } else if (a == "--black" && i + 1 < argc) {
            black = std::stoull(argv[++i]);
        } else if (a == "--white" && i + 1 < argc) {
            white = std::stoull(argv[++i]);
        } else if (a == "--time-ms" && i + 1 < argc) {
            time_ms = std::stoi(argv[++i]);
        } else if (a == "--server") {
            // server mode flag - handled later
            continue;
        } else if (a == "--is-black-turn" && i + 1 < argc) {
            is_black_turn = (std::stoi(argv[++i]) != 0);
        } else if (a == "--help") {
            print_usage();
            return 0;
        } else {
            print_usage();
            return 1;
        }
    }

    // Determine if we're running in server mode (persistent bot)
    bool server_mode = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--server") {
            server_mode = true;
            break;
        }
    }

    if (!server_mode) {
        Board b;
        if (is_black_turn) {
            b = Board(black, white, true);
        } else {
            b = Board(white, black, false);
        }

        std::unique_ptr<MCTS> bot_ptr;

        try {
            if (bot == "cpu") {
                bot_ptr = std::make_unique<MCTS>(std::chrono::milliseconds(time_ms));
            } else if (bot == "omp") {
                bot_ptr = std::make_unique<MCTSTreeOMP>(std::chrono::milliseconds(time_ms));
            } else if (bot == "leaf") {
                bot_ptr = std::make_unique<MCTSLeafParallel>(std::chrono::milliseconds(time_ms));
            } else if (bot == "block") {
                bot_ptr = std::make_unique<MCTSBlock>(std::chrono::milliseconds(time_ms));
            } else if (bot == "cuda") {
                bot_ptr = std::make_unique<MCTSTree>(std::chrono::milliseconds(time_ms));
            } else {
                std::cerr << "Unknown bot type: " << bot << "\n";
                return 2;
            }
        } catch (std::exception &e) {
            std::cerr << "Failed to construct bot: " << e.what() << "\n";
            return 3;
        }

        auto move = bot_ptr->get_best_move(b);
        double pps = bot_ptr->get_pps();

        int out_index = -1;
        if (move != 0) {
            out_index = __builtin_ctzll(move); // index of least significant set bit
        }

        std::cout << "{\"move\":" << out_index << ",\"pps\":" << pps << "}\n";
        return 0;
    }

    // Server mode: construct bot once and serve multiple requests via stdin
    std::unique_ptr<MCTS> bot_ptr;
    try {
        if (bot == "cpu") {
            bot_ptr = std::make_unique<MCTS>(std::chrono::milliseconds(time_ms));
        } else if (bot == "omp") {
            bot_ptr = std::make_unique<MCTSTreeOMP>(std::chrono::milliseconds(time_ms));
        } else if (bot == "leaf") {
            bot_ptr = std::make_unique<MCTSLeafParallel>(std::chrono::milliseconds(time_ms));
        } else if (bot == "block") {
            bot_ptr = std::make_unique<MCTSBlock>(std::chrono::milliseconds(time_ms));
        } else if (bot == "cuda") {
            bot_ptr = std::make_unique<MCTSTree>(std::chrono::milliseconds(time_ms));
        } else {
            std::cerr << "Unknown bot type: " << bot << "\n";
            return 2;
        }
    } catch (std::exception &e) {
        std::cerr << "Failed to construct bot: " << e.what() << "\n";
        return 3;
    }

    // Expect input lines of the form: <black> <white> <time_ms> <is_black_turn>
    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty())
            continue;
        std::istringstream iss(line);
        uint64_t in_black = 0;
        uint64_t in_white = 0;
        int in_time_ms = time_ms;
        int in_is_black = is_black_turn ? 1 : 0;

        if (!(iss >> in_black >> in_white >> in_time_ms >> in_is_black)) {
            std::cerr << "Invalid server input: " << line << "\n";
            continue;
        }

        bool in_black_turn = (in_is_black != 0);
        Board b2;
        if (in_black_turn) {
            b2 = Board(in_black, in_white, true);
        } else {
            b2 = Board(in_white, in_black, false);
        }

        bot_ptr->set_time_limit(std::chrono::milliseconds(in_time_ms));

        auto mv = bot_ptr->get_best_move(b2);
        double pps = bot_ptr->get_pps();

        int out_index = -1;
        if (mv != 0) {
            out_index = __builtin_ctzll(mv);
        }

        std::cout << "{\"move\":" << out_index << ",\"pps\":" << pps << "}\n";
        std::cout.flush();
    }

    return 0;
}
