#pragma once

#include "board.h"
#include <memory>
#include <random>
#include <vector>

// Monte-Carlo Tree Search Node structure
struct Node {
    Board state;

    // Pointer to the parent node is used for backpropagation
    Node *parent;

    // The move that created this node (0 indicates a PASS)
    uint64_t move_from_parent;

    // Using unique_ptr for automatic memory management (RAII)
    std::vector<std::unique_ptr<Node>> children;

    // Moves that have not been expanded yet
    std::vector<uint64_t> untried_moves;

    double wins;
    int visits;

    // 0 for Root's player, 1 for Opponent
    int player_moved_to_create_node;

    Node(Board const &state, Node *parent = nullptr, uint64_t move = 0);

    bool is_fully_expanded() const;
    bool is_terminal() const;

    // Selects the best child using the UCB formula
    Node *best_child(double c_param = 1.414) const;

    // Expands the tree by adding a new child node
    Node *expand();

    // Updates the stats for this node
    void update(double result);
};

class MCTS {
  public:
    MCTS(int iterations);

    Move get_best_move(Board const &state);

  private:
    int iterations;

    /*  Marsenne Twister random number generator
    Used instead of rand() because avoids modulo bias */
    std::mt19937 shuffler{std::random_device{}()};

    /*  Selection phase: traverse the tree to a leaf node using UCB
    Expansion phase: create new child node and return pointer to it */
    Node *tree_policy(Node *node);

    // Simulation phase: play out a random game from the given state
    double default_policy(Board state);

    // Backpropagation phase: update the nodes with the simulation result
    void backpropagate(Node *node, double result);
};
