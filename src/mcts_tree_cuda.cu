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

    // Return from BLACK's absolute perspective (like mcts_block.cu)
    int my_score, opp_score;
    state.get_score(my_score, opp_score);

    int black_score = state.is_black_turn() ? my_score : opp_score;
    int white_score = state.is_black_turn() ? opp_score : my_score;

    if (black_score > white_score)
        return 1.0f;
    if (white_score > black_score)
        return 0.0f;
    return 0.5f;
}

__global__ void mcts_kernel(GpuNode *nodes, int *node_count, int max_nodes,
                            curandState *globalStates, int volatile *stop_flag,
                            double c_param, Board root_state, bool root_is_black)
{
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    int lane = threadIdx.x % 32;
    curandState localState = globalStates[tid];

    // Shared variables for the root node to reduce contention
    __shared__ int shared_root_visits[8]; // One per warp (max 8 warps in 256 threads)
    __shared__ float shared_root_wins[8];
    int warp_idx = threadIdx.x / 32;

    if (lane == 0) {
        shared_root_visits[warp_idx] = 0;
        shared_root_wins[warp_idx] = 0.0f;
    }
    __syncthreads();

    while (*stop_flag == 0) {
        int node_idx = 0; // Root is always at index 0
        Board cur_board = root_state;
        int depth = 0;

        // Virtual Loss (Root)
        if (lane == 0) {
            atomicAdd(&shared_root_visits[warp_idx], 1);
        }

        while (true) {
            if (nodes[node_idx].is_terminal) {
                break;
            }

            int first_child = nodes[node_idx].first_child;

            // EXPANSION PHASE: "Expand All" strategy
            if (first_child == -1) {
                // Only lane 0 attempts to lock
                int old = -3;
                if (lane == 0) {
                    old = atomicCAS(&nodes[node_idx].first_child, -1, -2);
                }
                old = __shfl_sync(0xffffffff, old, 0);

                if (old == -1) {
                    // WE ARE THE EXPANDER (Lane 0 handles memory)
                    int start_idx = -1;
                    int needed_nodes = 0;
                    bool is_pass = false;

                    if (lane == 0) {
                        MoveList untried = cur_board.list_available_legal_moves();
                        int n_moves = Board::count_moves(untried);
                        is_pass = (n_moves == 0);
                        needed_nodes = is_pass ? 1 : n_moves;

                        start_idx = atomicAdd(node_count, needed_nodes);
                        if (start_idx + needed_nodes > max_nodes) {
                            nodes[node_idx].first_child = -1;
                            start_idx = -1;
                        }
                    }
                    start_idx = __shfl_sync(0xffffffff, start_idx, 0);
                    needed_nodes = __shfl_sync(0xffffffff, needed_nodes, 0);
                    is_pass = __shfl_sync(0xffffffff, is_pass, 0);

                    if (start_idx == -1)
                        goto simulation;

                    // Root's turn info for init
                    if (lane == 0) {
                        if (is_pass) {
                            Board next_state = cur_board;
                            next_state.move(0);
                            nodes[start_idx].init(next_state, node_idx, 0);
                            nodes[start_idx].visits = 1;
                        } else {
                            MoveList untried = cur_board.list_available_legal_moves();
                            int child_offset = 0;
                            while (untried != 0) {
                                int idx = __ffsll(untried) - 1;
                                Move move = (1ULL << idx);
                                untried &= ~move;

                                Board next_state = cur_board;
                                next_state.move(move);
                                nodes[start_idx + child_offset].init(next_state, node_idx,
                                                                     move);
                                child_offset++;
                            }
                        }
                        nodes[node_idx].num_children = (uint8_t)needed_nodes;
                        __threadfence();
                        nodes[node_idx].first_child = start_idx;
                    }
                    first_child = start_idx;
                } else if (old == -2) {
                    goto simulation;
                } else {
                    first_child = old;
                }
            }

            if (first_child == -2)
                goto simulation;

            // SELECTION PHASE (Lane 0 computes best, then broadcasts)
            int best_child_idx = -1;
            if (lane == 0) {
                int num_children = nodes[node_idx].num_children;
                int best_offset = -1;
                float best_val = -1e20f;
                int parent_visits = nodes[node_idx].visits;
                parent_visits += atomicAdd(&shared_root_visits[warp_idx], 0);

                for (int i = 0; i < num_children; i++) {
                    int child_idx = first_child + i;
                    float val = calculate_ucb(parent_visits, nodes[child_idx].visits,
                                              nodes[child_idx].wins, c_param);
                    if (val > best_val) {
                        best_val = val;
                        best_offset = i;
                    }
                }
                if (best_offset != -1) {
                    best_child_idx = first_child + best_offset;
                    atomicAdd(&nodes[best_child_idx].visits, 1);
                }
            }
            best_child_idx = __shfl_sync(0xffffffff, best_child_idx, 0);

            if (best_child_idx == -1)
                break;

            // Update board logic (All threads keep board in sync)
            int move_i = nodes[best_child_idx].move_idx;
            Move move = (move_i == 64) ? 0 : (1ULL << move_i);
            cur_board.move(move);
            node_idx = best_child_idx;
            depth++;
        }

    simulation:
        // PARALLEL SIMULATION: All 32 threads in warp simulate
        float black_wins = simulate_game_float(cur_board, &localState);

        // BACKPROPAGATION: Warp reduction
        for (int offset = 16; offset > 0; offset /= 2) {
            black_wins += __shfl_down_sync(0xffffffff, black_wins, offset);
        }
        // Lane 0 now has the sum of all 32 simulation results
        float warp_total_black_wins = __shfl_sync(0xffffffff, black_wins, 0);
        int warp_visits = 32;

        int d = depth;
        int curr = node_idx;
        while (curr != -1) {
            bool mover_is_black;
            if (d % 2 != 0) {
                mover_is_black = root_is_black;
            } else {
                mover_is_black = !root_is_black;
            }

            float wins_to_add = mover_is_black ? warp_total_black_wins
                                               : (warp_visits - warp_total_black_wins);

            if (lane == 0) {
                if (curr == 0) {
                    atomicAdd(&shared_root_wins[warp_idx], wins_to_add);
                } else {
                    atomicAdd(&nodes[curr].wins, wins_to_add);
                    // visits were already handled with Virtual Loss (but only by 1,
                    // we need to add the other 31)
                    atomicAdd(&nodes[curr].visits, warp_visits - 1);
                }
            }

            curr = nodes[curr].parent;
            d--;
        }

        // Periodically flush shared memory to global
        if (lane == 0) {
            int v = atomicExch(&shared_root_visits[warp_idx], 0);
            if (v > 0)
                atomicAdd(&nodes[0].visits, v);

            float w = atomicExch(&shared_root_wins[warp_idx], 0.0f);
            if (w != 0.0f)
                atomicAdd(&nodes[0].wins, w);
        }
    }

    __syncthreads();
    if (lane == 0) {
        int v = atomicExch(&shared_root_visits[warp_idx], 0);
        if (v > 0)
            atomicAdd(&nodes[0].visits, v);

        float w = atomicExch(&shared_root_wins[warp_idx], 0.0f);
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

    bool root_is_black = state.is_black_turn();

    mcts_kernel<<<gpu_config::NUM_BLOCKS, gpu_config::BLOCK_SIZE>>>(d_nodes, d_node_count, max_nodes,
                                 (curandState *)d_states, d_stop_flag, 1.414, state, root_is_black);

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
