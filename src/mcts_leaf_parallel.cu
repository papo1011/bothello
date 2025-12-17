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
__global__ void leaf_playout_kernel(Board leaf_state, int n_sims, int *results,
                                    curandState *globalStates)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n_sims) {
        // Copy RNG state to local memory
        curandState localState = globalStates[id];

        int result = simulate_game_from_leaf(leaf_state, &localState);

        // Save state back for continuity
        globalStates[id] = localState;

        // Accumulate Black wins atomically
        atomicAdd(results, result);
    }
}

// =========================================================================================
// LeafParallelMCTS Implementation
// =========================================================================================

LeafParallelMCTS::LeafParallelMCTS(int iterations, SimulationBackend backend)
    : iterations(iterations)
    , time_limit(std::chrono::milliseconds::max())
    , use_time_limit(false)
    , backend(backend)
{
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

LeafParallelMCTS::LeafParallelMCTS(std::chrono::milliseconds time_limit,
                                   SimulationBackend backend)
    : iterations(std::numeric_limits<int>::max())
    , time_limit(time_limit)
    , use_time_limit(true)
    , backend(backend)
{
    // Allocate RNG states
    int max_sims = gpu_config::MAX_SIMULATIONS_BUFFER;
    GPU_CHECK(cudaMalloc((void **)&rng_states, max_sims * sizeof(curandState)));

    // Initialize RNG states
    int threads = gpu_config::BLOCK_SIZE;
    int blocks = (max_sims + threads - 1) / threads;
    setup_kernel<<<blocks, threads>>>((curandState *)rng_states, time(NULL), max_sims);
    GPU_CHECK(cudaGetLastError());
    rng_initialized = true;

    GPU_CHECK(cudaMalloc((void **)&d_results, sizeof(int)));
}

LeafParallelMCTS::~LeafParallelMCTS()
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
// Unlike standard MCTS (1 simulation per iteration), we ran n_sims parallel simulations
void backpropagate_leaf_parallel(Node *node, double avg_result, int n_sims)
{
    while (node != nullptr) {
        // avg_result is the fraction of simulations won by Black (0.0 to 1.0)
        double batch_wins = avg_result * n_sims; // Total Black wins in the batch

        // Update visits (atomic int)
        node->visits.fetch_add(n_sims);

        // Update wins (atomic double - requires compare-exchange loop)
        double wins_to_add;
        if (node->player_moved_to_create_node == 0) {
            wins_to_add = batch_wins;
        } else {
            wins_to_add = n_sims - batch_wins;
        }

        // Compare-and-swap loop for atomic double
        double old_wins = node->wins.load();
        while (!node->wins.compare_exchange_weak(old_wins, old_wins + wins_to_add)) {
            // Loop continues until successful update
        }

        node = node->parent;
    }
}

Move LeafParallelMCTS::get_best_move(Board const &state)
{
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

        // 2. Simulation: Run LEAF_SIMULATIONS parallel GPU playouts from this leaf
        // Convert node state to Board for GPU execution
        // Node->player_moved_to_create_node indicates who moved to reach this state:
        //   0 = Black moved to create node -> current turn is White
        //   1 = White moved to create node -> current turn is Black
        bool is_black_turn = (leaf_node->player_moved_to_create_node == 1);

        Board leaf_state(leaf_node->state.get_curr_player_mask(),
                         leaf_node->state.get_opp_player_mask(), is_black_turn);

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
        backpropagate_leaf_parallel(leaf_node, avg_result, n_sims);

        last_executed_iterations++;
        total_leaves_evaluated++;
    }

    auto end_time = clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    last_duration_seconds = elapsed.count();

    // Print performance metrics
    int total_sims = total_leaves_evaluated * gpu_config::LEAF_SIMULATIONS;
    double time_per_leaf_ms =
        (last_duration_seconds * 1000.0) / std::max(1, total_leaves_evaluated);
    std::cout << "Leaf-Parallel MCTS Performance: " << total_leaves_evaluated
              << " leaves evaluated, " << total_sims << " total simulations, "
              << time_per_leaf_ms << " ms/leaf, " << get_pps() << " playouts/sec"
              << std::endl;

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

double LeafParallelMCTS::get_pps() const
{
    if (last_duration_seconds > 0.0) {
        return (last_executed_iterations * gpu_config::LEAF_SIMULATIONS) /
               last_duration_seconds;
    }
    return 0.0;
}

void LeafParallelMCTS::run_cuda_simulations(Board leaf_state, int n_sims, int *results)
{
    int threads = gpu_config::BLOCK_SIZE;
    int blocks = (n_sims + threads - 1) / threads;
    leaf_playout_kernel<<<blocks, threads>>>(leaf_state, n_sims, results,
                                             (curandState *)rng_states);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
}
