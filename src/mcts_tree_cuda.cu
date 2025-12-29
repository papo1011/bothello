#include "mcts_tree_cuda.h"
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <thread>

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

struct GpuNode {
    // No Board state. State is maintained during traversal.

    int first_child;  // Index in pool. -1 for null.
    int next_sibling; // Index in pool. -1 for null.
    int parent;       // Index in pool. -1 for null.

    uint64_t untried_moves; // 8 bytes.

    int visits; // 4 bytes.
    float wins; // 4 bytes.

    uint8_t move_idx; // 0-63. 1 byte.
    bool is_terminal; // 1 byte.

    __host__ __device__ void init(Board s, int p, Move m)
    {
        parent = p;
#ifdef __CUDA_ARCH__
        move_idx = (m == 0) ? 64 : (__ffsll(m) - 1);
#else
        move_idx = (m == 0) ? 64 : (__builtin_ffsll(m) - 1);
#endif

        first_child = -1;
        next_sibling = -1;
        visits = 0;
        wins = 0.0f;
        untried_moves = s.list_available_legal_moves();
        is_terminal = s.is_terminal();
    }
};

static __device__ float calculate_ucb(int parent_visits, int child_visits,
                                      float child_wins, double c_param)
{
    if (child_visits == 0)
        return 1e20f; // Infinity
    return (child_wins / child_visits) +
           c_param * sqrtf(2.0f * logf((float)parent_visits) / child_visits);
}

__device__ int get_random_int(curandState *state, int max)
{
    return curand(state) % max;
}

__device__ int simulate_game(Board state, curandState *localState)
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

    int my_score, opp_score;
    state.get_score(my_score, opp_score);

    // Openmp use 1, 0 and 0.5 but here we use int
    if (my_score > opp_score)
        return 2;
    if (my_score < opp_score)
        return 0;
    return 1;
}

__device__ float simulate_game_float(Board state, curandState *localState)
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

    int my_score, opp_score;
    state.get_score(my_score, opp_score);

    if (my_score > opp_score)
        return 1.0f;
    if (my_score < opp_score)
        return 0.0f;
    return 0.5f;
}

__global__ void mcts_kernel(GpuNode *nodes, int *node_count, int max_nodes,
                            curandState *globalStates, int volatile *stop_flag,
                            double c_param, Board root_state)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    curandState localState = globalStates[tid];

    while (*stop_flag == 0) {
        int node_idx = 0; // Root is always at index 0
        Board cur_board = root_state;

        while (true) {
            if (nodes[node_idx].is_terminal) {
                break;
            }

            // Check if fully expanded
            MoveList untried = nodes[node_idx].untried_moves;

            if (untried != 0) {
                bool expanded = false;

                // Copy untried to local to iterate
                MoveList current_untried = nodes[node_idx].untried_moves;
                while (current_untried != 0) {

                    // Pick random bit to reduce contention
                    int n_moves = Board::count_moves(current_untried);
                    int idx = curand(&localState) % n_moves;
                    Move move = Board::get_nth_move(current_untried, idx);

                    unsigned long long old_val =
                        atomicCAS((unsigned long long *)&nodes[node_idx].untried_moves,
                                  (unsigned long long)current_untried,
                                  (unsigned long long)(current_untried & ~move));

                    if (old_val == current_untried) {
                        // Try to allocate first
                        int new_idx = atomicAdd(node_count, 1);
                        if (new_idx >= max_nodes) {
                            // Memory full, we can't use this node.
                            // We lost the move from untried_moves, but we can't add the
                            // child. Just simulate.
                            goto simulation;
                        }

                        Board next_state = cur_board;
                        next_state.move(move);

                        nodes[new_idx].init(next_state, node_idx, move);

                        int old_head;
                        do {
                            old_head = nodes[node_idx].first_child;
                            nodes[new_idx].next_sibling = old_head;
                            __threadfence(); // Ensure next_sibling write is visible
                        } while (atomicCAS((int *)&nodes[node_idx].first_child,
                                           old_head, new_idx) != old_head);

                        node_idx = new_idx;
                        cur_board = next_state;
                        expanded = true;
                        break;
                    } else {
                        current_untried = nodes[node_idx].untried_moves;
                    }
                }

                if (expanded) {
                    goto simulation;
                }
            }

            int best_child_idx = -1;
            float best_val = -1e20f;
            int parent_visits = nodes[node_idx].visits;

            int curr_child_idx = nodes[node_idx].first_child;
            while (curr_child_idx != -1) {
                float val = calculate_ucb(parent_visits, nodes[curr_child_idx].visits,
                                          nodes[curr_child_idx].wins, c_param);
                if (val > best_val) {
                    best_val = val;
                    best_child_idx = curr_child_idx;
                }
                curr_child_idx = nodes[curr_child_idx].next_sibling;
            }

            if (best_child_idx == -1) {
                // Should not happen if fully expanded and not terminal
                break;
            }

            // Update board and descend
            int move_i = nodes[best_child_idx].move_idx;
            Move move = (move_i == 64) ? 0 : (1ULL << move_i);
            cur_board.move(move);
            node_idx = best_child_idx;
        }

    simulation:
        // Also possible to have the int version
        float result = simulate_game_float(cur_board, &localState);

        while (node_idx != -1) {
            atomicAdd(&nodes[node_idx].visits, 1);
            atomicAdd(&nodes[node_idx].wins, result);
            node_idx = nodes[node_idx].parent;
        }
    }

    globalStates[tid] = localState;
}

static __global__ void setup_rng_kernel(curandState *state, unsigned long seed, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n) {
        curand_init(seed, id, 0, &state[id]);
    }
}

MCTSTree::~MCTSTree() { free_gpu(); }

void MCTSTree::free_gpu()
{
    if (d_nodes) {
        cudaFree(d_nodes);
        d_nodes = nullptr;
    }
    if (d_node_count) {
        cudaFree(d_node_count);
        d_node_count = nullptr;
    }
    if (d_states) {
        cudaFree(d_states);
        d_states = nullptr;
    }
    if (h_stop_flag) {
        cudaFreeHost(h_stop_flag);
        h_stop_flag = nullptr;
        d_stop_flag = nullptr;
    }
    is_gpu_initialized = false;
}

void MCTSTree::initialize_gpu(int num_threads)
{
    if (is_gpu_initialized)
        return;

    GPU_CHECK(cudaMalloc(&d_nodes, max_nodes * sizeof(GpuNode)));
    GPU_CHECK(cudaMalloc(&d_node_count, sizeof(int)));
    GPU_CHECK(cudaMalloc(&d_states, num_threads * sizeof(curandState)));

    // Use mapped pinned memory for stop flag
    GPU_CHECK(cudaHostAlloc(&h_stop_flag, sizeof(int), cudaHostAllocMapped));
    GPU_CHECK(cudaHostGetDevicePointer(&d_stop_flag, h_stop_flag, 0));

    // Init RNG
    int blocks = (num_threads + 127) / 128;
    setup_rng_kernel<<<blocks, 128>>>((curandState *)d_states, time(NULL), num_threads);
    GPU_CHECK(cudaDeviceSynchronize());

    is_gpu_initialized = true;
}

Move MCTSTree::get_best_move(Board const &state)
{
    int num_threads = 1024 * 4; // 4096 threads
    // Total iterations = 4096 * 10 = 40k per launch.

    initialize_gpu(num_threads);

    // Reset node count
    int zero = 1; // Start at 1 (0 is root)
    GPU_CHECK(cudaMemcpy(d_node_count, &zero, sizeof(int), cudaMemcpyHostToDevice));

    // Init root node
    GpuNode root_host;
    root_host.init(state, -1, 0);
    GPU_CHECK(
        cudaMemcpy(&d_nodes[0], &root_host, sizeof(GpuNode), cudaMemcpyHostToDevice));

    // Reset stop flag
    *h_stop_flag = 0;

    // Run MCTS
    auto start_time = std::chrono::steady_clock::now();

    int blocks = (num_threads + 127) / 128;
    mcts_kernel<<<blocks, 128>>>(d_nodes, d_node_count, max_nodes,
                                 (curandState *)d_states, d_stop_flag, 1.414, state);

    // Wait for time limit
    while (true) {
        auto now = std::chrono::steady_clock::now();
        if (now - start_time > time_limit) {
            break;
        }

        // Check if memory is full?
        // Check if everything allrght

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Stop kernel
    *h_stop_flag = 1;
    GPU_CHECK(cudaDeviceSynchronize());

    last_duration_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
            .count();

    // We don't track total iterations exactly anymore, but we can estimate or read root
    // visits
    GpuNode root;
    GPU_CHECK(cudaMemcpy(&root, &d_nodes[0], sizeof(GpuNode), cudaMemcpyDeviceToHost));
    last_executed_iterations = root.visits;

    Move best_move = 0;
    int max_visits = -1;

    int curr_child_idx = root.first_child;
    while (curr_child_idx != -1) {
        GpuNode child;
        GPU_CHECK(cudaMemcpy(&child, &d_nodes[curr_child_idx], sizeof(GpuNode),
                             cudaMemcpyDeviceToHost));

        if (child.visits > max_visits) {
            max_visits = child.visits;
            best_move = (child.move_idx == 64) ? 0 : (1ULL << child.move_idx);
        }

        curr_child_idx = child.next_sibling;
    }

    return best_move;
}
