#!/bin/bash

echo "Profiling bothello_tree_cuda with Nsight Systems..."

nsys profile --stats=true --trace=cuda,nvtx,osrt --cuda-memory-usage=true -o profiling/nsys/bothello_tree_cuda_profile ./build/bothello_tree_cuda > profiling/nsys/bothello_tree_cuda_profile.txt

echo "Profiling bothello_leaf with Nsight Systems..."

nsys profile --stats=true --trace=cuda,nvtx,osrt --cuda-memory-usage=true -o profiling/nsys/bothello_leaf_profile ./build/bothello_leaf > profiling/nsys/bothello_leaf_profile.txt  

echo "Profiling bothello_block with Nsight Systems..."

nsys profile --stats=true --trace=cuda,nvtx,osrt --cuda-memory-usage=true -o profiling/nsys/bothello_block_profile ./build/bothello_block > profiling/nsys/bothello_block_profile.txt

echo "NSYS profiles saved in profiling/nsys directory."

echo "Profiling bothello_tree_cuda with Nsight Compute..."

sudo ~/NVIDIA-Nsight-Compute-2025.4/ncu --set full --target-processes all --export profiling/ncu/bothello_tree_cuda_ncu_report.ncu-rep --launch-skip 0 --launch-count 1 ./build/bothello_tree_cuda > profiling/ncu/bothello_tree_cuda_ncu_report.txt

ncu --import profiling/ncu/bothello_tree_cuda_ncu_report.ncu-rep > profiling/ncu/bothello_tree_cuda_ncu_report_converted.txt

echo "Profiling bothello_leaf with Nsight Compute..."

sudo ~/NVIDIA-Nsight-Compute-2025.4/ncu --set full --target-processes all --export profiling/ncu/bothello_leaf_ncu_report.ncu-rep --launch-skip 0 --launch-count 1 ./build/bothello_leaf > profiling/ncu/bothello_leaf_ncu_report.txt

ncu --import profiling/ncu/bothello_leaf_ncu_report.ncu-rep > profiling/ncu/bothello_leaf_ncu_report_converted.txt

echo "Profiling bothello_block with Nsight Compute..."

sudo ~/NVIDIA-Nsight-Compute-2025.4/ncu --set full --target-processes all --export profiling/ncu/bothello_block_ncu_report.ncu-rep --launch-skip 0 --launch-count 1 ./build/bothello_block > profiling/ncu/bothello_block_ncu_report.txt

ncu --import profiling/ncu/bothello_block_ncu_report.ncu-rep > profiling/ncu/bothello_block_ncu_report_converted.txt

echo "NCU reports saved in profiling/ncu directory."