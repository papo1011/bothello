#pragma once

#include "board.h"
#include "mcts.h"
#include <omp.h>

struct NodeTreeOMP : public Node {
    NodeTreeOMP(Board const &state, Node *parent = nullptr, uint64_t move = 0);
    ~NodeTreeOMP() override;

    omp_lock_t node_mutex;

    void lock();
    void unlock();

    NodeTreeOMP *best_child(double c_param = 1.414) const override;
    Node *expand() override;
};

class MCTSTreeOMP : public MCTS {
  public:
    using MCTS::MCTS;

    Move get_best_move(Board const &state) override;

  private:
    double default_policy(Board state);
};
