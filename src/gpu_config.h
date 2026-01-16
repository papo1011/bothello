#pragma once

namespace gpu_config {

// GPU Configuration for Leaf-Parallel MCTS
// Tuned for RTX 5060 Laptop (Blackwell architecture, 8GB VRAM, ~3,584 CUDA cores)
// Using compute_89 for CUDA 12.0 compiler compatibility

constexpr int BLOCK_SIZE = 256;         // Threads per block (standard warp multiple)
constexpr int NUM_BLOCKS = 64;          // Number of blocks to launch
constexpr int LEAF_SIMULATIONS = 65536; // Optimal for RTX 5060
constexpr int MAX_SIMULATIONS_BUFFER = LEAF_SIMULATIONS;
constexpr int BLOCK_SIM_NODE_PER_TREE =
    10000; // Number of nodes for each block's local tree

} // namespace gpu_config
