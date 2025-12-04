#include "mcts.h"
#include <algorithm>
#include <random>

// Converts the bitmask MoveList (uint64_t) into a generic vector of moves
std::vector<Move> get_moves_as_vector(MoveList move_list)
{
    std::vector<Move> moves;
    for (int i = 0; i < 64; ++i) {
        Move m = 1ULL << i;
        if (move_list & m) {
            moves.push_back(m);
        }
    }
    return moves;
}

Node::Node(Board const &state, Node *parent, uint64_t move)
    : state(state)
    , parent(parent)
    , move_from_parent(move)
    , wins(0)
    , visits(0)
{
    MoveList moves = state.list_available_legal_moves();
    untried_moves = get_moves_as_vector(moves);

    if (untried_moves.empty()) {
        // check if eventually the opponent has legal moves after a PASS
        Board copy = state;
        copy.flip();

        if (copy.is_there_a_legal_move_available()) {
            untried_moves.push_back(0); // move 0 represents a PASS
        }
    }

    /* Multi-threading random engine
    used to ensure exploration diversity of the available moves,
    instead of always expanding in the same order */
    static thread_local std::mt19937 shuffler{std::random_device{}()};
    std::shuffle(untried_moves.begin(), untried_moves.end(), shuffler);

    // parent is nullptr for Root node
    if (parent == nullptr) {
        // first player to move is Black, so Root is white
        player_moved_to_create_node = 1;
    } else {
        // swap players
        player_moved_to_create_node = 1 - parent->player_moved_to_create_node;
    }
}

bool Node::is_fully_expanded() const { return untried_moves.empty(); }

bool Node::is_terminal() const { return untried_moves.empty() && children.empty(); }
