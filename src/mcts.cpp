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

Node *Node::best_child(double c_param) const
{
    Node *best = nullptr;
    double best_value = -std::numeric_limits<double>::infinity();

    // pick the child with the highest UCB value
    for (auto const &child : children) {
        double ucb1 = (child->wins / child->visits) +
                      c_param * std::sqrt(2.0 * std::log(this->visits) / child->visits);

        if (ucb1 > best_value) {
            best_value = ucb1;
            best = child.get();
        }
    }
    return best;
}

Node *Node::expand()
{
    uint64_t move = untried_moves.back();
    untried_moves.pop_back();

    Board next_state = state;
    next_state.move(move);

    auto child = std::make_unique<Node>(next_state, this, move);
    Node *child_ptr = child.get();

    children.push_back(std::move(child));

    return child_ptr;
}

void Node::update(double result)
{
    visits++;
    wins += result;
}

MCTS::MCTS(int iterations)
    : iterations(iterations)
{
}

Move MCTS::get_best_move(Board const &state)
{
    Node root(state);

    if (root.is_terminal()) {
        return 0;
    }

    for (int i = 0; i < iterations; ++i) {
        Node *leaf = tree_policy(&root);
        double result = default_policy(leaf->state);
        backpropagate(leaf, result);
    }

    /* The best node is the child with the most visits,
    because UCB directs search traffic toward promising paths,
    the node with the most visits is implicitly the one
    the algorithm has consistently evaluated as the strongest */
    Node *best_node = nullptr;
    int max_visits = -1;

    for (auto const &child : root.children) {
        if (child->visits > max_visits) {
            max_visits = child->visits;
            best_node = child.get();
        }
    }

    return best_node ? best_node->move_from_parent : 0;
}

Node *MCTS::tree_policy(Node *node)
{
    while (!node->is_terminal()) {
        if (!node->is_fully_expanded()) {
            return node->expand();
        } else {
            if (node->children.empty())
                return node;
            node = node->best_child();
        }
    }
    return node;
}
