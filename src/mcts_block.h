#pragma once

#include "board.h"
#include "mcts.h"

// GPU-accelerated Monte Carlo Tree Search for Othello
//
// BLOCK PARALLEL MCTS
// =============================================================
// This implementation uses a Block Parallel scheme where each CUDA Block
// acts as an independent search agent maintaining its own local tree.
//
// Architecture:
//   - Inter-Block: Multiple independent trees (Root Parallelism)
//     Each block explores its own tree stored in a private memory partition.
//   - Intra-Block: Parallel Simulations (Leaf Parallelism)
//     Inside a block, threads cooperate to run multiple simulations in parallel
//     for the same leaf node to maximize SIMD efficiency
//
// Each MCTS iteration:
//   1. Selection: Thread 0 traverses its local tree using UCB to find a leaf
//   2. Expansion: Thread 0 adds one new child to the local tree
//   3. Simulation: All threads in the block launch parallel playouts
//      (batch simulations) from the new leaf state
//   4. Backpropagation: Thread 0 updates the local branch with the aggregated
//      results from all threads
//
// Final results are aggregated by the Host (CPU) summing visits from all trees
class MCTSBlock : public MCTS {
  public:
    using MCTS::MCTS;
    ~MCTSBlock();

    Move get_best_move(Board const &state) override;

  private:
    void *d_states = nullptr; // RNG states per thread
    bool is_gpu_initialized = false;

    GpuNode *d_nodes = nullptr;
    int *d_node_counts = nullptr;

    int *h_stop_flag = nullptr;
    int *d_stop_flag = nullptr;

    // Conservative configurations to avoid OOM on small GPUs
    int num_blocks = 128;
    int threads_per_block = 64;
    int nodes_per_tree = 10000;

    void initialize_gpu();
    void free_gpu();
};
