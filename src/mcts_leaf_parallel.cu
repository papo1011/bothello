#include "board.h"
#include "gpu_config.h"
#include "mcts_leaf_parallel.h"
#include <curand_kernel.h>
#include <iostream>
#include <limits>

#define GPU_CHECK(ans)                        \
    {                                         \
        gpuAssert((ans), __FILE__, __LINE__); \
    }

inline void gpuAssert(cudaError_t code, char const *file, int line, bool abort = true)
{
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

// =========================================================================================
// CUDA KERNELS FOR LEAF-PARALLEL MCTS
// =========================================================================================

__global__ void setup_kernel(curandState *state, unsigned long seed, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, id, 0, &state[id]);
    }
}

// Simulates a random playout from the given board state
// Returns 1 if Black wins, 0 if White wins or draw
__device__ int simulate_game_from_leaf(Board state, curandState *localState)
{
    int consecutive_passes = 0;
    while (consecutive_passes < 2) {
        MoveList available = state.list_available_legal_moves();

        if (available == 0) {
            state.move(0); // Pass
            consecutive_passes++;
            continue;
        }
        consecutive_passes = 0;

        // Pick random move
        int n_moves = Board::count_moves(available);
        unsigned int r = curand(localState);
        int idx = r % n_moves;

        Move move = Board::get_nth_move(available, idx);
        state.move(move);
    }

    // Game over - determine winner
    // Return 1 if Black wins, 0 if White wins or draw
    int my_score, opp_score;
    state.get_score(my_score, opp_score);

    int black_score = state.is_black_turn() ? my_score : opp_score;
    int white_score = state.is_black_turn() ? opp_score : my_score;

    if (black_score > white_score)
        return 1;
    return 0; // White win or draw
}

// Kernel that launches n_sims parallel playouts from a leaf node state
// Each thread simulates one complete game and accumulates Black wins
// Uses shared memory reduction to minimize global atomic contention
__global__ void leaf_playout_kernel(Board leaf_state, int n_sims, int *results,
                                    curandState *globalStates)
{
    // Shared memory for block-level reduction
    __shared__ int block_sum;
    
    // Initialize shared memory (only one thread per block)
    if (threadIdx.x == 0) {
        block_sum = 0;
    }
    __syncthreads();
    
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n_sims) {
        // Copy RNG state to local memory
        curandState localState = globalStates[id];

        int result = simulate_game_from_leaf(leaf_state, &localState);

        // Save state back for continuity
        globalStates[id] = localState;

        // First: accumulate within block using shared memory atomic
        // (much faster than global memory atomic)
        atomicAdd(&block_sum, result);
    }
    __syncthreads();
    
    // Only one thread per block writes to global memory
    // Reduces global atomics from n_sims to (n_sims / blockDim.x)
    if (threadIdx.x == 0) {
        atomicAdd(results, block_sum);
    }
}

// =========================================================================================
// MCTSLeafParallel Implementation
// =========================================================================================

void MCTSLeafParallel::initialize_gpu()
{
    if (rng_initialized)
        return;

    // Allocate RNG states
    int max_sims = gpu_config::MAX_SIMULATIONS_BUFFER;
    GPU_CHECK(cudaMalloc((void **)&rng_states, max_sims * sizeof(curandState)));

    // Initialize RNG states
    int threads = gpu_config::BLOCK_SIZE;
    int blocks = (max_sims + threads - 1) / threads;
    setup_kernel<<<blocks, threads>>>((curandState *)rng_states, time(NULL), max_sims);
    GPU_CHECK(cudaGetLastError());
    rng_initialized = true;

    // Allocate results buffer
    GPU_CHECK(cudaMalloc((void **)&d_results, sizeof(int)));
}

MCTSLeafParallel::~MCTSLeafParallel()
{
    if (rng_states)
        cudaFree(rng_states);
    if (d_results)
        cudaFree(d_results);
}

// Helper for MCTS tree traversal to a leaf
Node *tree_policy(Node *node)
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

// Backpropagate batch simulation results up the tree
// Uses iterative approach to prevent stack overflow in deep trees
void backpropagate_leaf_parallel(Node *node, double avg_result, int n_sims, bool root_is_black)
{
    Node *current = node;
    while (current != nullptr) {
        double batch_wins = avg_result * n_sims; // Total Black wins from simulations
        current->visits.fetch_add(n_sims);

        // Map player_moved_to_create_node to actual color based on root's color
        // Root always has value=1, children alternate 0/1
        bool black_moved_to_create_this_node;
        if (root_is_black) {
            // Root=Black's turn with value=1 means White moved to create it
            // So value=0 means Black moved, value=1 means White moved
            black_moved_to_create_this_node = (current->player_moved_to_create_node == 0);
        } else {
            // Root=White's turn with value=1 means Black moved to create it
            // So value=1 means Black moved, value=0 means White moved
            black_moved_to_create_this_node = (current->player_moved_to_create_node == 1);
        }

        // Credit wins to the player who moved to create this node
        double wins_to_add = black_moved_to_create_this_node ? batch_wins : (n_sims - batch_wins);

        // Atomic update
        double old_wins = current->wins.load();
        while (!current->wins.compare_exchange_weak(old_wins, old_wins + wins_to_add)) {
        }

        current = current->parent;
    }
}

Move MCTSLeafParallel::get_best_move(Board const &state)
{
    // Initialize GPU on first use
    initialize_gpu();

    Node root(state); // Root node

    // Early exit: Check if there are any valid moves
    Move valid_moves = state.list_available_legal_moves();
    if (valid_moves == 0) {
        std::cout << "No legal moves available. Passing immediately." << std::endl;
        return 0; // Pass
    }

    using clock = std::chrono::steady_clock;
    auto start_time = clock::now();

    last_executed_iterations = 0;
    int total_leaves_evaluated = 0;

    bool use_time_limit = (time_limit != std::chrono::milliseconds::max());

    // Main MCTS loop with leaf parallelization
    while (last_executed_iterations < iterations) {
        if (use_time_limit) {
            auto current_now = clock::now();
            if ((current_now - start_time) >= time_limit) {
                break;
            }
        }

        // 1. Selection & Expansion: Navigate to a leaf node
        Node *leaf_node = tree_policy(&root);

        // 2. Simulation: Run parallel GPU playouts from this leaf
        // Board auto-flips after every move, so use node's actual state directly
        Board leaf_state(leaf_node->state.get_curr_player_mask(),
                         leaf_node->state.get_opp_player_mask(),
                         leaf_node->state.is_black_turn());

        // Number of parallel simulations per leaf
        int n_sims = gpu_config::LEAF_SIMULATIONS;

        // Reset result buffer and launch GPU simulations
        GPU_CHECK(cudaMemset(d_results, 0, sizeof(int)));
        run_cuda_simulations(leaf_state, n_sims, d_results);

        // Retrieve results (number of Black wins)
        int total_black_wins = 0;
        GPU_CHECK(cudaMemcpy(&total_black_wins, d_results, sizeof(int),
                             cudaMemcpyDeviceToHost));

        double avg_result = (double)total_black_wins / n_sims;

        // 3. Backpropagation: Update tree with batch results
        bool root_is_black = root.state.is_black_turn();
        backpropagate_leaf_parallel(leaf_node, avg_result, n_sims, root_is_black);

        last_executed_iterations++;
        total_leaves_evaluated++;
    }

    auto end_time = clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    last_duration_seconds = elapsed.count();

    // Print performance metrics
    // int total_sims = total_leaves_evaluated * gpu_config::LEAF_SIMULATIONS;
    // double time_per_leaf_ms =
    //     (last_duration_seconds * 1000.0) / std::max(1, total_leaves_evaluated);
    // std::cout << "Leaf-Parallel MCTS Performance: " << total_leaves_evaluated
    //           << " leaves evaluated, " << total_sims << " total simulations, "
    //           << time_per_leaf_ms << " ms/leaf, " << get_pps() << " playouts/sec"
    //           << std::endl;

    // Select best move based on visit count
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

double MCTSLeafParallel::get_pps() const
{
    if (last_duration_seconds > 0.0) {
        return (last_executed_iterations * gpu_config::LEAF_SIMULATIONS) /
               last_duration_seconds;
    }
    return 0.0;
}

void MCTSLeafParallel::run_cuda_simulations(Board leaf_state, int n_sims, int *results)
{
    int threads = gpu_config::BLOCK_SIZE;
    int blocks = (n_sims + threads - 1) / threads;
    leaf_playout_kernel<<<blocks, threads>>>(leaf_state, n_sims, results,
                                             (curandState *)rng_states);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
}
