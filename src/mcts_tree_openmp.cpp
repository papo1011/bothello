#include "mcts_tree_openmp.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>

NodeTree::NodeTree(Board const &state, Node *parent, uint64_t move)
    : Node(state, parent, move)
{
    omp_init_lock(&node_mutex);
}

NodeTree::~NodeTree() { omp_destroy_lock(&node_mutex); }

void NodeTree::lock() { omp_set_lock(&node_mutex); }

void NodeTree::unlock() { omp_unset_lock(&node_mutex); }

NodeTree *NodeTree::best_child(double c_param) const
{
    // With atomics, we don't need to lock to read visits/wins
    int parent_visits = this->visits;

    // Avoid log(0) if parent visits update is lagging behind children updates
    if (parent_visits < 1)
        parent_visits = 1;

    NodeTree *best = nullptr;
    double best_value = -std::numeric_limits<double>::infinity();

    for (auto const &child : children) {
        NodeTree *child_node = static_cast<NodeTree *>(child.get());

        double wins = child_node->wins;
        int visits = child_node->visits;

        double ucb;
        if (visits == 0) {
            ucb = std::numeric_limits<double>::infinity();
        } else {
            ucb = (wins / visits) +
                  c_param * std::sqrt(2.0 * std::log((double)parent_visits) / visits);
        }

        if (ucb > best_value) {
            best_value = ucb;
            best = child_node;
        }
    }
    return best;
}

Node *NodeTree::expand()
{
    // Assumes caller holds the lock on this node
    uint64_t move = untried_moves.back();
    untried_moves.pop_back();

    Board next_state = state;
    next_state.move(move);

    auto child = std::make_unique<NodeTree>(next_state, this, move);
    NodeTree *child_ptr = child.get();

    children.push_back(std::move(child));

    if (untried_moves.empty()) {
        fully_expanded = true;
    }

    return child_ptr;
}

Move MCTSTree::get_best_move(Board const &state)
{
    NodeTree root(state);

    auto start_time = std::chrono::steady_clock::now();

    int total_iterations = 0;

#pragma omp parallel
    {
        int thread_iterations = 0;

        // Each thread runs simulations until time is up
        while (true) {
            auto now = std::chrono::steady_clock::now();
            if (now - start_time > time_limit) {
                break;
            }

            NodeTree *node = &root;

            // Selection
            while (!node->is_terminal()) {
                if (node->is_fully_expanded()) {
                    // No lock needed if fully expanded
                    node = node->best_child();
                } else {
                    node->lock();
                    if (!node->is_fully_expanded()) {
                        // Expand
                        Node *new_child = node->expand();
                        node->unlock();
                        node = static_cast<NodeTree *>(new_child);
                        break; // Reached a new node, start simulation
                    } else {
                        // Select
                        node->unlock();
                        node = node->best_child();
                    }
                }
                if (node == nullptr) {
                    // Should not happen if logic is correct, but prevents segfault
                    break;
                }
            }

            if (node == nullptr) {
                continue;
            }

            // Simulation
            double result = default_policy(node->state);

            // Backpropagation
            while (node) {
                // With atomics, update is thread-safe without locks
                node->update(result);
                node = static_cast<NodeTree *>(node->parent);
            }

            thread_iterations++;
        }

#pragma omp atomic
        total_iterations += thread_iterations;
    }

    auto end_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    last_duration_seconds = elapsed.count();
    last_executed_iterations = total_iterations;

    // Select best move (most visits)
    auto best_child = std::max_element(
        root.children.begin(), root.children.end(),
        [](auto const &a, auto const &b) { return a->visits < b->visits; });

    if (best_child == root.children.end()) {
        // Should not happen unless no moves available?
        return 0;
    }

    return (*best_child)->move_from_parent;
}

double MCTSTree::default_policy(Board state)
{
    // Thread-local random engine
    static thread_local std::mt19937 local_shuffler{std::random_device{}()};

    Board current = state;
    int consecutive_passes = 0;

    while (consecutive_passes < 2) {
        MoveList available = current.list_available_legal_moves();

        if (available == 0) {
            current.move(0);
            consecutive_passes++;
            continue;
        }
        consecutive_passes = 0;

        std::vector<Move> moves = get_moves_as_vector(available);

        std::uniform_int_distribution<> dis(0, moves.size() - 1);
        Move random_move = moves[dis(local_shuffler)];

        current.move(random_move);
    }

    int my_score, opp_score;
    state.get_score(my_score, opp_score);
    if (my_score > opp_score)
        return 1.0;
    if (my_score > opp_score)
        return 0.0;
    return 0.5;
}
