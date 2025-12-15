#pragma once

#include "mcts.h"
#include "board.h"
#include <memory>
#include <string>
#include <chrono>

// Enum to select the simulation backend
enum class SimulationBackend {
    CPU,
    CUDA_PURE
};

// GPU-accelerated Monte Carlo Tree Search for Othello/Reversi
// 
// KEY CONCEPT: Leaf Parallelization
// ================================
// Unlike traditional MCTS which runs one simulation per tree iteration,
// this implementation runs LEAF_SIMULATIONS parallel GPU simulations
// from each leaf node that is reached during tree traversal.
//
// Each MCTS iteration:
//   1. Selection: Traverse tree using UCB to find a leaf node
//   2. Expansion: Add one new child to the tree
//   3. Simulation: Launch LEAF_SIMULATIONS parallel GPU playouts from the new leaf
//   4. Backpropagation: Update all ancestor nodes with aggregated results
//
// This amortizes GPU kernel launch overhead and exploits massive parallelism
// at the cost of slightly less precise tree statistics per simulation.
class LeafParallelMCTS {
public:
    LeafParallelMCTS(int iterations, SimulationBackend backend);
    LeafParallelMCTS(std::chrono::milliseconds time_limit, SimulationBackend backend);
    ~LeafParallelMCTS();
    
    Move get_best_move(Board const& state);
    double get_pps() const;  // Returns playouts per second (total simulations / time)

private:
    int iterations;
    std::chrono::milliseconds time_limit;
    bool use_time_limit = false;
    SimulationBackend backend;
    
    int last_executed_iterations = 0;
    double last_duration_seconds = 0.0;
    
    void run_parallel_simulations(Node* node, int n_sims);
    
    // Helpers
    // Pure CUDA implementation
    void run_cuda_simulations(gpu::DeviceBoard initial_state, int n_sims, int* results);
    
    void* rng_states = nullptr; // curandState* 
    int* d_results = nullptr;   // Device memory for results
    bool rng_initialized = false;
};
