#pragma once

namespace gpu_config {

// GPU Configuration for Leaf-Parallel MCTS
// Tuned for RTX 5060 Laptop (Blackwell architecture, 8GB VRAM, ~3,584 CUDA cores)
// Using compute_89 for CUDA 12.0 compiler compatibility

constexpr int BLOCK_SIZE = 256;         // Threads per block (standard warp multiple)
constexpr int NUM_BLOCKS = 64;          // Number of blocks to launch
constexpr int LEAF_SIMULATIONS = 16384; // Parallel game simulations per MCTS leaf node
constexpr int MAX_SIMULATIONS_BUFFER = 100000; // Max curandState buffer size for memory safety
} // namespace gpu_config
