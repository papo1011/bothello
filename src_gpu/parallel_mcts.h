#pragma once

#include "../src/mcts.h"
#include "gpu_board.cuh"
#include <memory>
#include <string>
#include <chrono>

// Enum to select the simulation backend
enum class SimulationBackend {
    CPU,
    CUDA_PURE
};

// GPU-accelerated Monte Carlo Tree Search for Othello/Reversi
// Each MCTS iteration expands one tree node and runs LEAF_SIMULATIONS parallel game playouts
class ParallelMCTS {
public:
    ParallelMCTS(int iterations, SimulationBackend backend);
    ParallelMCTS(std::chrono::milliseconds time_limit, SimulationBackend backend);
    ~ParallelMCTS();
    
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
