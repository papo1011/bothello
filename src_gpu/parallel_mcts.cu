#include "parallel_mcts.h"
#include "gpu_config.h"
#include <curand_kernel.h>
#include <iostream>
#include <limits>
#include <cmath>

#define GPU_CHECK(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

// =========================================================================================
// CUDA KERNELS
// =========================================================================================

__global__ void setup_kernel(curandState *state, unsigned long seed, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, id, 0, &state[id]);
    }
}

__device__ int simulate_game(gpu::DeviceBoard state, curandState* localState) {
    int consecutive_passes = 0;
    while (consecutive_passes < 2) {
        gpu::MoveList available = state.list_available_legal_moves();
        
        if (available == 0) {
            state.move(0); // Pass
            consecutive_passes++;
            continue;
        }
        consecutive_passes = 0;

        // Pick random move
        int n_moves = gpu::DeviceBoard::count_moves(available);
        // Generate random index in [0, n_moves-1]
        // curand() returns a random unsigned int, modulo gives uniform distribution
        unsigned int r = curand(localState);
        int idx = r % n_moves;
        
        gpu::Move move = gpu::DeviceBoard::get_nth_move(available, idx);
        state.move(move);
    }
    
    // Game over - determine winner
    // Return 1 if Black wins, 0 if White wins or draw
    // This is consistent with MCTS expectation: 1.0 = Black win, 0.0 = White win
    // Note: Draws are treated as White wins (0) for simplicity with integer atomics
    // Using floats would allow 0.5 for draws, but atomicAdd on floats is slower
    
    int my_score, opp_score;
    state.get_score(my_score, opp_score);
    
    int black_score = state.is_black_turn ? my_score : opp_score;
    int white_score = state.is_black_turn ? opp_score : my_score;
    
    if (black_score > white_score) return 1;
    return 0; // White win or draw
}

__global__ void playout_kernel(gpu::DeviceBoard initial_state, int n_sims, int* results, curandState *globalStates)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n_sims) {
        // Copy state to local memory (registers)
        curandState localState = globalStates[id];
        
        int result = simulate_game(initial_state, &localState);
        
        // Save state back if we want continuity (not strictly needed for independent sims but good practice)
        globalStates[id] = localState;
        
        // Accumulate result
        atomicAdd(results, result);
    }
}



// =========================================================================================
// Implementation
// =========================================================================================

ParallelMCTS::ParallelMCTS(int iterations, SimulationBackend backend)
    : iterations(iterations), time_limit(std::chrono::milliseconds::max()), use_time_limit(false), backend(backend)
{
    // Allocate RNG states
    int max_sims = gpu_config::MAX_SIMULATIONS_BUFFER;
    GPU_CHECK(cudaMalloc((void**)&rng_states, max_sims * sizeof(curandState)));
    
    // Initialize
    int threads = gpu_config::BLOCK_SIZE;
    int blocks = (max_sims + threads - 1) / threads;
    setup_kernel<<<blocks, threads>>>((curandState*)rng_states, time(NULL), max_sims);
    GPU_CHECK(cudaGetLastError());
    rng_initialized = true;
    
    // Allocate results buffer
    GPU_CHECK(cudaMalloc((void**)&d_results, sizeof(int)));
}

ParallelMCTS::ParallelMCTS(std::chrono::milliseconds time_limit, SimulationBackend backend)
    : iterations(std::numeric_limits<int>::max()), time_limit(time_limit), use_time_limit(true), backend(backend)
{
    // Duplicate initialization logic (should refactor but copy-paste for safety)
    int max_sims = gpu_config::MAX_SIMULATIONS_BUFFER;
    GPU_CHECK(cudaMalloc((void**)&rng_states, max_sims * sizeof(curandState)));
    
    int threads = gpu_config::BLOCK_SIZE;
    int blocks = (max_sims + threads - 1) / threads;
    setup_kernel<<<blocks, threads>>>((curandState*)rng_states, time(NULL), max_sims);
    GPU_CHECK(cudaGetLastError());
    rng_initialized = true;
    
    GPU_CHECK(cudaMalloc((void**)&d_results, sizeof(int)));
}

ParallelMCTS::~ParallelMCTS() {
    if (rng_states) cudaFree(rng_states);
    if (d_results) cudaFree(d_results);
}

// Duplicate helper for MCTS traversal
Node* tree_policy(Node* node) {
    while (!node->is_terminal()) {
        if (!node->is_fully_expanded()) {
            return node->expand();
        } else {
            if (node->children.empty()) return node;
            node = node->best_child();
        }
    }
    return node;
}

// Backpropagate batch simulation results up the tree
// Unlike standard MCTS (1 simulation per iteration), we ran n_sims parallel simulations
void backpropagate_parallel(Node* node, double avg_result, int n_sims) {
    while (node != nullptr) {
        // Update node statistics with batch results
        // avg_result is the fraction of simulations won by Black (0.0 to 1.0)
        // We accumulate n_sims visits and proportional wins
        
        double batch_wins = avg_result * n_sims; // Total Black wins in the batch
        
        // Direct member access for efficiency (Node members are public)
        // Perspective flip: if Black moved to create this node, add Black wins
        // If White moved to create this node, add White wins (n_sims - batch_wins)
        
        node->visits += n_sims;
        
        if (node->player_moved_to_create_node == 0) {
            node->wins += batch_wins;
        } else {
            node->wins += (n_sims - batch_wins);
        }
        
        node = node->parent;
    }
}

Move ParallelMCTS::get_best_move(Board const& state)
{
    Node root(state); // Root node
    
    // Early exit: Check if there are any valid moves
    Move valid_moves = state.list_available_legal_moves();
    if (valid_moves == 0) {
        std::cout << "No legal moves available. Passing immediately." << std::endl;
        return 0; // Pass
    }
    
    // ... setup timers ...
    using clock = std::chrono::steady_clock;
    auto start_time = clock::now();
    
    last_executed_iterations = 0;
    int total_leaves_evaluated = 0;
    
    while (last_executed_iterations < iterations) {
        if (use_time_limit) {
            auto current_now = clock::now();
            if ((current_now - start_time) >= time_limit) {
                break;
            }
        }
    
        Node* node = tree_policy(&root);
        
        // MCTS Simulation Phase - run parallel GPU simulations from this node
        // Convert node state to DeviceBoard for GPU execution
        // Node->player_moved_to_create_node indicates who moved to reach this state:
        //   0 = Black moved to create node -> current turn is White
        //   1 = White moved to create node -> current turn is Black
        // Root node has player_moved_to_create_node=1, so Black moves first
        bool is_black_turn = (node->player_moved_to_create_node == 1);
        
        gpu::DeviceBoard device_state(
            node->state.get_curr_player_mask(), 
            node->state.get_opp_player_mask(), 
            is_black_turn
        );
        
        // Number of parallel simulations per leaf
        int n_sims = gpu_config::LEAF_SIMULATIONS;
        
        double avg_result = 0.0;
        
        GPU_CHECK(cudaMemset(d_results, 0, sizeof(int)));
        
        run_cuda_simulations(device_state, n_sims, d_results);
        
        int total_wins = 0;
        GPU_CHECK(cudaMemcpy(&total_wins, d_results, sizeof(int), cudaMemcpyDeviceToHost));
        
        avg_result = (double)total_wins / n_sims;
        
        backpropagate_parallel(node, avg_result, n_sims);
        
        last_executed_iterations++;
        total_leaves_evaluated++;
    }
    
    auto end_time = clock::now();
    std::chrono::duration<double> elapsed = end_time - start_time;
    last_duration_seconds = elapsed.count();
    
    // Print performance metrics
    int total_sims = total_leaves_evaluated * gpu_config::LEAF_SIMULATIONS;
    double time_per_leaf_ms = (last_duration_seconds * 1000.0) / std::max(1, total_leaves_evaluated);
    std::cout << "Performance: " << total_leaves_evaluated << " leaves evaluated, "
              << total_sims << " total simulations, "
              << time_per_leaf_ms << " ms/leaf" << std::endl;
    
    // Select best move
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

double ParallelMCTS::get_pps() const {
    if (last_duration_seconds > 0.0) {
        return (last_executed_iterations * gpu_config::LEAF_SIMULATIONS) / last_duration_seconds;
    }
    return 0.0;
}

void ParallelMCTS::run_cuda_simulations(gpu::DeviceBoard initial_state, int n_sims, int* results) {
    int threads = gpu_config::BLOCK_SIZE;
    int blocks = (n_sims + threads - 1) / threads;
    playout_kernel<<<blocks, threads>>>(initial_state, n_sims, results, (curandState*)rng_states);
    GPU_CHECK(cudaGetLastError());
    GPU_CHECK(cudaDeviceSynchronize());
}
