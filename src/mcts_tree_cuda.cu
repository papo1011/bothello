#include "mcts_tree_cuda.h"
#include "gpu_config.h"
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
    int first_child; // Index in pool. -1 for null/unexpanded. -2 for locking.
    int parent;      // Index in pool.

    int visits;
    float wins;

    uint8_t move_idx;     // 0-63. 64 for pass/root.
    uint8_t num_children; // Number of children in the contiguous block
    bool is_terminal;

    __host__ __device__ void init(Board s, int p, Move m)
    {
        parent = p;
#ifdef __CUDA_ARCH__
        move_idx = (m == 0) ? 64 : (__ffsll(m) - 1);
#else
        move_idx = (m == 0) ? 64 : (__builtin_ffsll(m) - 1);
#endif

        first_child = -1;
        num_children = 0;
        visits = 0;
        wins = 0.0f;
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

    // Shared variables for the root node to reduce contention
    __shared__ int shared_root_visits;
    __shared__ float shared_root_wins;

    if (threadIdx.x == 0) {
        shared_root_visits = 0;
        shared_root_wins = 0.0f;
    }
    __syncthreads();

    while (*stop_flag == 0) {
        int node_idx = 0; // Root is always at index 0
        Board cur_board = root_state;

        // Virtual Loss (Root)
        atomicAdd(&shared_root_visits, 1);

        while (true) {
            if (nodes[node_idx].is_terminal) {
                break;
            }

            int first_child = nodes[node_idx].first_child;

            // EXPANSION PHASE: "Expand All" strategy
            // If not expanded (-1), try to expand.
            if (first_child == -1) {
                // Attempt to lock the node for expansion
                int old = atomicCAS(&nodes[node_idx].first_child, -1, -2);

                if (old == -1) {
                    // WE ARE THE EXPANDER
                    MoveList untried = cur_board.list_available_legal_moves();
                    int n_moves = Board::count_moves(untried);
                    bool is_pass = (n_moves == 0); // Implicit pass if not terminal
                    int needed_nodes = is_pass ? 1 : n_moves;

                    int start_idx = atomicAdd(node_count, needed_nodes);
                    if (start_idx + needed_nodes > max_nodes) {
                        nodes[node_idx].first_child = -1;
                        goto simulation;
                    }

                    // Initialize all children in contiguous memory
                    if (is_pass) {
                        // Create one child representing PASS
                        Board next_state = cur_board;
                        next_state.move(0);
                        nodes[start_idx].init(next_state, node_idx, 0);
                        // Virtual Loss: First visitor visits this new node immediately
                        nodes[start_idx].visits = 1;
                    } else {
                        // Create children for all moves
                        int child_offset = 0;
                        MoveList temp_moves = untried;
                        while (temp_moves != 0) {
#ifdef __CUDA_ARCH__
                            int idx = __ffsll(temp_moves) - 1;
#else
                            int idx = __builtin_ffsll(temp_moves) - 1;
#endif
                            Move move = (1ULL << idx);
                            temp_moves &= ~move;

                            Board next_state = cur_board;
                            next_state.move(move);

                            nodes[start_idx + child_offset].init(next_state, node_idx,
                                                                 move);

                            child_offset++;
                        }
                    }

                    nodes[node_idx].num_children = (uint8_t)needed_nodes;
                    __threadfence(); // Ensure children are visible
                    nodes[node_idx].first_child = start_idx; // Unlock and publish

                    // Fall through to selection.
                    first_child = start_idx;
                } else if (old == -2) {
                    // Someone else is expanding.
                    // We treat this node as a leaf for this simulation to avoid
                    // spinning.
                    goto simulation;
                } else {
                    // Already expanded by someone else between our check and CAS
                    first_child = old;
                }
            }

            // Check again for -2 if we fell through
            if (first_child == -2)
                goto simulation;

            // SELECTION PHASE
            int num_children = nodes[node_idx].num_children;
            int best_child_offset = -1;
            float best_val = -1e20f;
            int parent_visits = nodes[node_idx].visits;

            // Use shared memory values for root if we are at root
            if (node_idx == 0) {
                parent_visits += atomicAdd(&shared_root_visits, 0);
            }

            // Iterate contiguous children
            for (int i = 0; i < num_children; i++) {
                int child_idx = first_child + i;
                int c_visits = nodes[child_idx].visits;
                float c_wins = nodes[child_idx].wins;

                float val = calculate_ucb(parent_visits, c_visits, c_wins, c_param);
                if (val > best_val) {
                    best_val = val;
                    best_child_offset = i;
                }
            }

            if (best_child_offset == -1) {
                // Should not happen
                break;
            }

            int best_child_idx = first_child + best_child_offset;

            // Virtual Loss: Update child visits immediately
            atomicAdd(&nodes[best_child_idx].visits, 1);

            // Update board and descend
            int move_i = nodes[best_child_idx].move_idx;
            Move move = (move_i == 64) ? 0 : (1ULL << move_i);
            cur_board.move(move);
            node_idx = best_child_idx;
        }

    simulation:
        float result = simulate_game_float(cur_board, &localState);

        while (node_idx != -1) {
            bool processed = false;

            // Shared Memory Accumulation for Root
            if (node_idx == 0) {
                // Visits were handled with Virtual Loss (start of loop)
                atomicAdd(&shared_root_wins, result);
                processed = true;
            }

            if (!processed) {
                // Warp thingy
                unsigned int mask =
                    __match_any_sync(__activemask(), (unsigned long long)node_idx);
                // int leader = __ffs(mask) - 1;

                atomicAdd(&nodes[node_idx].wins, result);
            }

            node_idx = nodes[node_idx].parent;
        }

        // Flush logic: Periodically flush shared memory to global.
        // We do this check at the end of every simulation loop.
        // Thread 0 of the block handles the flush.
        if (threadIdx.x == 0) {
            int v = atomicExch(&shared_root_visits, 0);
            if (v > 0) {
                atomicAdd(&nodes[0].visits, v);
            }

            float w = atomicExch(&shared_root_wins, 0.0f);
            if (w != 0.0f) {
                atomicAdd(&nodes[0].wins, w);
            }
        }
    }

    // Final flush on kernel exit
    __syncthreads(); // Wait for all threads to finish their last loop iteration
    if (threadIdx.x == 0) {
        int v = atomicExch(&shared_root_visits, 0);
        if (v > 0)
            atomicAdd(&nodes[0].visits, v);

        float w = atomicExch(&shared_root_wins, 0.0f);
        if (w != 0.0f)
            atomicAdd(&nodes[0].wins, w);
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
    int blocks = (num_threads + gpu_config::BLOCK_SIZE - 1) / gpu_config::BLOCK_SIZE;
    setup_rng_kernel<<<blocks, gpu_config::BLOCK_SIZE>>>((curandState *)d_states, time(NULL), num_threads);
    GPU_CHECK(cudaDeviceSynchronize());

    is_gpu_initialized = true;
}

Move MCTSTree::get_best_move(Board const &state)
{
    int num_threads = gpu_config::NUM_BLOCKS * gpu_config::BLOCK_SIZE;

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

    mcts_kernel<<<gpu_config::NUM_BLOCKS, gpu_config::BLOCK_SIZE>>>(d_nodes, d_node_count, max_nodes,
                                 (curandState *)d_states, d_stop_flag, 1.414, state);

    // Wait for time limit
    while (true) {
        auto now = std::chrono::steady_clock::now();
        if (now - start_time > time_limit) {
            break;
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Stop kernel
    *h_stop_flag = 1;
    GPU_CHECK(cudaDeviceSynchronize());

    last_duration_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
            .count();

    GpuNode root;
    GPU_CHECK(cudaMemcpy(&root, &d_nodes[0], sizeof(GpuNode), cudaMemcpyDeviceToHost));
    last_executed_iterations = root.visits;

    Move best_move = 0;
    int max_visits = -1;

    int first_child = root.first_child;
    int num_children = root.num_children;

    for (int i = 0; i < num_children; ++i) {
        GpuNode child;
        GPU_CHECK(cudaMemcpy(&child, &d_nodes[first_child + i], sizeof(GpuNode),
                             cudaMemcpyDeviceToHost));

        if (child.visits > max_visits) {
            max_visits = child.visits;
            best_move = (child.move_idx == 64) ? 0 : (1ULL << child.move_idx);
        }
    }

    return best_move;
}
