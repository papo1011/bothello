#include <iostream>
#include <string>
#include <vector>
#include <memory>
#include <chrono>
#include <cstdlib>
#include <ctime>
#include <sstream>
#include "src/board.h"
#include "src/mcts.h"
#include "src_gpu/parallel_mcts.h"
#include "src_gpu/gpu_config.h"

// Abstract Agent Interface to simplify the loop
class Agent {
public:
    virtual ~Agent() = default;
    virtual Move get_move(const Board& state) = 0;
    virtual std::string name() const = 0;
    virtual double get_pps_stats() const = 0;
};

class CpuAgent : public Agent {
    MCTS mcts;
public:
    CpuAgent(int time_ms) : mcts(std::chrono::milliseconds(time_ms)) {}
    Move get_move(const Board& state) override { return mcts.get_best_move(state); }
    std::string name() const override { return "CPU (MCTS)"; }
    double get_pps_stats() const override { return mcts.get_pps(); }
};

class GpuAgent : public Agent {
    ParallelMCTS mcts;
    std::string type_name;
public:
    GpuAgent(int time_ms, SimulationBackend backend) 
        : mcts(std::chrono::milliseconds(time_ms), backend) {
        type_name = "GPU (CUDA)";
    }
    Move get_move(const Board& state) override { return mcts.get_best_move(state); }
    std::string name() const override { return type_name; }
    double get_pps_stats() const override { return mcts.get_pps(); }
};

// Helper: Factory
std::unique_ptr<Agent> create_agent(std::string type, int time_ms) {
    if (type == "cpu") return std::make_unique<CpuAgent>(time_ms);
    if (type == "cuda" || type == "gpu") return std::make_unique<GpuAgent>(time_ms, SimulationBackend::CUDA_PURE);
    std::cerr << "Unknown agent type: " << type << ". Defaulting to CPU.\n";
    return std::make_unique<CpuAgent>(time_ms);
}

// Helper: Move to String
std::string to_string(Move m) {
    return move_to_gtp(m);
}

int main(int argc, char** argv) {
    // Usage: ./bothello_versus <player1_type> <player2_type> <time_ms>
    // Types: "cpu", "cuda" (or "gpu")
    
    std::string p1_type = "cpu";
    std::string p2_type = "cuda";
    int time_ms = 1000;

    if (argc > 1) p1_type = argv[1];
    if (argc > 2) p2_type = argv[2];
    if (argc > 3) time_ms = std::stoi(argv[3]);
    
    // Handle "random" for backward compatibility if user uses old syntax? 
    // The user asked for "gpu against gpu".
    // Let's assume strict usage for now.
    
    auto player1 = create_agent(p1_type, time_ms);
    auto player2 = create_agent(p2_type, time_ms);

    std::cout << "=== BOTHELLO VERSUS ARENA ===\n";
    std::cout << "Player 1 (Black): " << player1->name() << "\n";
    std::cout << "Player 2 (White): " << player2->name() << "\n";
    std::cout << "Time Config: " << time_ms << " ms per move\n";
    std::cout << "=============================\n\n";

    // Setup standard board
    auto BIT = [](int r, int c) { return 1ULL << (r * 8 + c); };
    uint64_t white = BIT(3, 3) | BIT(4, 4);
    uint64_t black = BIT(3, 4) | BIT(4, 3);
    Board board(black, white);

    int turn = 0;
    static bool is_p1_turn = true; // Player 1 (Black) starts

    while (!board.is_terminal()) {
        turn++;
        std::cout << "\n--- Turn " << turn << " ---\n";
        std::cout << board << "\n";
        
        std::string p_name = is_p1_turn ? "Black (" + player1->name() + ")" : "White (" + player2->name() + ")";
        std::cout << p_name << " to move...\n";
        
        Move best_move = 0;
        double pps = 0;
        
        if (is_p1_turn) {
            best_move = player1->get_move(board);
            pps = player1->get_pps_stats();
        } else {
            best_move = player2->get_move(board);
            pps = player2->get_pps_stats();
        }
        
        if (best_move == 0) std::cout << "Player passes.\n";
        else std::cout << "Selected move: " << to_string(best_move) << "\n";
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
    for(char c : s) {
        if (c == '*') b_count++;
        if (c == 'O') w_count++;
    }
    
    std::cout << "Final Score - Black (P1): " << b_count << " | White (P2): " << w_count << "\n";
    if (b_count > w_count) std::cout << "Winner: Player 1 (Black)\n";
    else if (w_count > b_count) std::cout << "Winner: Player 2 (White)\n";
    else std::cout << "Draw\n";

    return 0;
}
