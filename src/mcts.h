#pragma once

#include "board.h"
#include <memory>
#include <vector>

// MCTS Node structure
struct Node {
    Board state;
    Node* parent;
    uint64_t move_from_parent; // The move that created this node (0 indicates a PASS)

    // Using unique_ptr for automatic memory management (RAII)
    std::vector<std::unique_ptr<Node>> children;

    // Moves that have not been expanded yet
    std::vector<uint64_t> untried_moves;

    double wins;
    int visits;
    int player_moved_to_create_node; // 0 for Root's player, 1 for Opponent

    Node(const Board& state, Node* parent = nullptr, uint64_t move = 0);

    bool is_fully_expanded() const;
    bool is_terminal() const;

    // Selects the best child using the UCB1 formula
    Node* best_child(double c_param = 1.414) const;

    // Expands the tree by adding a new child node
    Node* expand();

    // Updates the stats for this node
    void update(double result);
};