#include "mcts_block.h"
#include <chrono>
#include <cuda_runtime.h>
#include <curand_kernel.h>
#include <iostream>
#include <map>
#include <thread>
#include <vector>

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
    int first_child;
    int next_sibling;
    int parent;

    uint64_t untried_moves;

    int visits;
    float wins;

    uint8_t move_idx;
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
        return 1e20f;
    return (child_wins / child_visits) +
           c_param * sqrtf(2.0f * logf((float)parent_visits) / child_visits);
}

// Parallel simulation executed by all threads in the block
__device__ float default_policy_block(Board state, curandState *localState)
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

        int n_moves = Board::count_moves(available);
        unsigned int r = curand(localState);
        int idx = r % n_moves;

        Move move = Board::get_nth_move(available, idx);
        state.move(move);
    }

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

__global__ void init_roots_kernel(GpuNode *all_nodes, int nodes_per_tree,
                                  Board root_state, int num_blocks)
{
    int bid = blockIdx.x * blockDim.x + threadIdx.x;
    if (bid < num_blocks) {
        all_nodes[bid * nodes_per_tree].init(root_state, -1, 0);
    }
}

__global__ void mcts_block_kernel(GpuNode *all_nodes, int *all_node_counts,
                                  int nodes_per_tree, curandState *globalStates,
                                  int volatile *stop_flag, double c_param,
                                  Board root_state, bool root_is_black)
{
    int tid = threadIdx.x;
    int bid = blockIdx.x;

    GpuNode *my_tree = &all_nodes[bid * nodes_per_tree];
    int *my_node_count = &all_node_counts[bid];

    int global_tid = bid * blockDim.x + tid;
    curandState localState = globalStates[global_tid];

    __shared__ int shared_node_idx;
    __shared__ Board shared_board;
    __shared__ float block_total_black_wins;
    __shared__ int shared_depth;

    while (*stop_flag == 0) {

        // SELECTION PHASE (Thread 0 only)
        if (tid == 0) {
            int node_idx = 0;
            Board cur_board = root_state;
            int depth = 0;

            while (true) {
                if (my_tree[node_idx].is_terminal)
                    break;
                if (my_tree[node_idx].untried_moves != 0)
                    break;

                int best_child = -1;
                float best_val = -1e20f;
                int p_visits = my_tree[node_idx].visits;

                int curr = my_tree[node_idx].first_child;
                while (curr != -1) {
                    float ucb = calculate_ucb(p_visits, my_tree[curr].visits,
                                              my_tree[curr].wins, c_param);
                    if (ucb > best_val) {
                        best_val = ucb;
                        best_child = curr;
                    }
                    curr = my_tree[curr].next_sibling;
                }

                if (best_child == -1)
                    break;

                int move_i = my_tree[best_child].move_idx;
                Move move = (move_i == 64) ? 0 : (1ULL << move_i);
                cur_board.move(move);
                node_idx = best_child;
                depth++;
            }

            shared_node_idx = node_idx;
            shared_board = cur_board;
            shared_depth = depth;
        }
        __syncthreads();

        // EXPANSION PHASE (Thread 0 only)
        if (tid == 0) {
            int node_idx = shared_node_idx;
            MoveList untried = my_tree[node_idx].untried_moves;

            if (!my_tree[node_idx].is_terminal && untried != 0) {
                int n_moves = Board::count_moves(untried);
                int idx = curand(&localState) % n_moves;
                Move move = Board::get_nth_move(untried, idx);

                my_tree[node_idx].untried_moves &= ~move;

                int new_idx = *my_node_count;
                if (new_idx < nodes_per_tree) {
                    (*my_node_count)++;

                    Board next_state = shared_board;
                    next_state.move(move);
                    my_tree[new_idx].init(next_state, node_idx, move);

                    my_tree[new_idx].next_sibling = my_tree[node_idx].first_child;
                    my_tree[node_idx].first_child = new_idx;

                    shared_node_idx = new_idx;
                    shared_board = next_state;
                    shared_depth++;
                }
            }
        }
        __syncthreads();

        // PARALLEL SIMULATION (All Threads)
        Board my_board = shared_board;
        float my_result = default_policy_block(my_board, &localState);

        // reduction (All Threads)
        if (tid == 0)
            block_total_black_wins = 0;
        __syncthreads();
        atomicAdd(&block_total_black_wins, my_result);
        __syncthreads();

        // BACKPROPAGATION PHASE (Thread 0 only)
        if (tid == 0) {
            int curr = shared_node_idx;
            float black_wins_batch = block_total_black_wins;
            int visits = blockDim.x;
            int d = shared_depth; // Profondità corrente (Leaf)

            while (curr != -1) {
                my_tree[curr].visits += visits;

                bool mover_is_black;
                if (d % 2 != 0) {
                    mover_is_black = root_is_black;
                } else {
                    mover_is_black = !root_is_black;
                }

                if (mover_is_black) {
                    my_tree[curr].wins += black_wins_batch;
                } else {
                    my_tree[curr].wins += (visits - black_wins_batch);
                }

                curr = my_tree[curr].parent;
                d--;
            }
        }
        __syncthreads();

        globalStates[global_tid] = localState;
    }
}

__global__ void setup_rng_kernel(curandState *state, unsigned long seed, int n)
{
    int id = blockIdx.x * blockDim.x + threadIdx.x;
    if (id < n)
        curand_init(seed, id, 0, &state[id]);
}

MCTSBlock::~MCTSBlock() { free_gpu(); }

void MCTSBlock::free_gpu()
{
    if (d_nodes) {
        cudaFree(d_nodes);
        d_nodes = nullptr;
    }
    if (d_node_counts) {
        cudaFree(d_node_counts);
        d_node_counts = nullptr;
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

void MCTSBlock::initialize_gpu()
{
    if (is_gpu_initialized)
        return;

    int total_threads = num_blocks * threads_per_block;
    size_t total_nodes_mem = (size_t)num_blocks * nodes_per_tree * sizeof(GpuNode);

    GPU_CHECK(cudaMalloc(&d_nodes, total_nodes_mem));
    GPU_CHECK(cudaMalloc(&d_node_counts, num_blocks * sizeof(int)));
    GPU_CHECK(cudaMalloc(&d_states, total_threads * sizeof(curandState)));
    GPU_CHECK(cudaHostAlloc(&h_stop_flag, sizeof(int), cudaHostAllocMapped));
    GPU_CHECK(cudaHostGetDevicePointer(&d_stop_flag, h_stop_flag, 0));

    int grid_size = (total_threads + 127) / 128;
    setup_rng_kernel<<<grid_size, 128>>>((curandState *)d_states, time(NULL),
                                         total_threads);
    GPU_CHECK(cudaDeviceSynchronize());

    is_gpu_initialized = true;
}

Move MCTSBlock::get_best_move(Board const &state)
{
    initialize_gpu();

    // Reset counters: All blocks start at index 1 (0 is root)
    std::vector<int> initial_counts(num_blocks, 1);
    GPU_CHECK(cudaMemcpy(d_node_counts, initial_counts.data(), num_blocks * sizeof(int),
                         cudaMemcpyHostToDevice));

    // Initialize Roots for each partition
    int init_grid = (num_blocks + 127) / 128;
    init_roots_kernel<<<init_grid, 128>>>(d_nodes, nodes_per_tree, state, num_blocks);
    GPU_CHECK(cudaGetLastError());

    // Reset flag and launch
    *h_stop_flag = 0;
    auto start_time = std::chrono::steady_clock::now();

    bool root_is_black = state.is_black_turn();

    mcts_block_kernel<<<num_blocks, threads_per_block>>>(
        d_nodes, d_node_counts, nodes_per_tree, (curandState *)d_states, d_stop_flag,
        1.414, state, root_is_black);

    while (true) {
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - start_time;
        if (elapsed > time_limit) {
            break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    *h_stop_flag = 1;
    GPU_CHECK(cudaDeviceSynchronize());

    // Record duration for PPS
    last_duration_seconds =
        std::chrono::duration<double>(std::chrono::steady_clock::now() - start_time)
            .count();

    // Aggregate Results on CPU and compute iterations
    std::map<Move, long long> aggregate_visits;
    long long total_root_visits = 0;

    for (int b = 0; b < num_blocks; b++) {
        size_t root_offset = (size_t)b * nodes_per_tree;

        GpuNode root;
        GPU_CHECK(cudaMemcpy(&root, &d_nodes[root_offset], sizeof(GpuNode),
                             cudaMemcpyDeviceToHost));

        // Sum root visits from all block trees
        total_root_visits += root.visits;

        int curr = root.first_child;
        int safety = 0;
        while (curr != -1 && safety++ < 200) {
            GpuNode child;
            GPU_CHECK(cudaMemcpy(&child, &d_nodes[root_offset + curr], sizeof(GpuNode),
                                 cudaMemcpyDeviceToHost));

            Move m = (child.move_idx == 64) ? 0 : (1ULL << child.move_idx);

            if (aggregate_visits.find(m) == aggregate_visits.end()) {
                aggregate_visits[m] = child.visits;
            } else {
                aggregate_visits[m] += child.visits;
            }

            curr = child.next_sibling;
        }
    }

    // Store iterations for PPS
    last_executed_iterations = (double)total_root_visits;

    // Pick Best Move
    Move best_move = 0;
    long long max_visits = -1;

    for (auto const &pair : aggregate_visits) {
        Move move = pair.first;
        long long visits = pair.second;

        if (visits > max_visits) {
            max_visits = visits;
            best_move = move;
        }
    }

    if (best_move == 0 && !aggregate_visits.empty()) {
        best_move = aggregate_visits.begin()->first;
    }

    return best_move;
}
