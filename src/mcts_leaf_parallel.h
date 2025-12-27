#pragma once

#include "board.h"
#include "mcts.h"
#include <chrono>
#include <memory>
#include <string>

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
class MCTSLeafParallel : public MCTS {
  public:
    using MCTS::MCTS;
    ~MCTSLeafParallel();

    Move get_best_move(Board const &state) override;
    double get_pps() const override;

  private:
    void run_cuda_simulations(Board initial_state, int n_sims, int *results);

    void *rng_states = nullptr; // curandState*
    int *d_results = nullptr;   // Device memory for results
    bool rng_initialized = false;

    void initialize_gpu();
};
