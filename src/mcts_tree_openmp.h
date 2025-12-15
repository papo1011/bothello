#pragma once

#include "board.h"
#include "mcts.h"
#include <omp.h>

struct NodeTree : public Node {
    NodeTree(Board const &state, Node *parent = nullptr, uint64_t move = 0);
    ~NodeTree() override;

    omp_lock_t node_mutex;

    void lock();
    void unlock();

    NodeTree *best_child(double c_param = 1.414) const override;
    Node *expand() override;
};

class MCTSTree : public MCTS {
  public:
    using MCTS::MCTS;

    Move get_best_move(Board const &state) override;

  private:
    double default_policy(Board state);
};
