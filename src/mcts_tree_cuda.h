#pragma once

#include "board.h"
#include "mcts.h"

struct GpuNode;

class MCTSTree : public MCTS {
  public:
    using MCTS::MCTS;
    ~MCTSTree();

    Move get_best_move(Board const &state) override;

  private:
    void *d_states = nullptr;
    bool is_gpu_initialized = false;

    // Memory pool for GPU nodes
    GpuNode *d_nodes = nullptr;
    int *d_node_count = nullptr;
    int *h_stop_flag = nullptr; // Host pinned memory
    int *d_stop_flag = nullptr; // Device pointer to mapped memory
    int max_nodes = 5000000; // 5 million nodes

    void initialize_gpu(int num_threads);
    void free_gpu();
};
