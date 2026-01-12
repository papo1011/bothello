cmake_minimum_required(VERSION 3.14)

project(BotHello CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Set CUDA architecture for RTX 5060 Laptop
# Using compute_89 (Ada Lovelace) for backward compatibility with CUDA 12.0 compiler
# RTX 5060 (Blackwell) is backward compatible and will work fine
set(CMAKE_CUDA_ARCHITECTURES 89)

# Set CMake policy for CUDA architectures
cmake_policy(SET CMP0104 NEW)

# Default to Release if not specified
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
	message(STATUS "Setting build type to 'Release' as none was specified.")
	set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo")
endif()

# Explicitly set compiler options based on build type
if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    message(STATUS "Configuring for Debug build: Adding debug flags")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-g>)
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O0>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-G>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-g>)
elseif(CMAKE_BUILD_TYPE STREQUAL "Release")
    message(STATUS "Configuring for Release build: Adding optimization flags")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
endif()

# Create a library for the core logic
add_library(bothello_core
    src/board.cpp
    src/board.h
    src/mcts.h
    src/mcts.cpp
	src/mcts_leaf_parallel.cu
    src/mcts_tree_cuda.h
    src/mcts_tree_cuda.cu
	src/mcts_tree_openmp.h
	src/mcts_tree_openmp.cpp
	src/mcts_block.h
    src/mcts_block.cu
)

# Include the root directory so we can include "src/board.h"
target_include_directories(bothello_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

find_package(OpenMP REQUIRED)
target_link_libraries(bothello_core PUBLIC OpenMP::OpenMP_CXX)

# Main executable (Default - maps to CUDA Tree)
add_executable(bothello main.cpp)
target_link_libraries(bothello PRIVATE bothello_core)

# CPU Bot
add_executable(bothello_cpu main.cpp)
target_compile_definitions(bothello_cpu PRIVATE BOT_CPU)
target_link_libraries(bothello_cpu PRIVATE bothello_core)

# OpenMP Bot
add_executable(bothello_omp main.cpp)
target_compile_definitions(bothello_omp PRIVATE BOT_OMP)
target_link_libraries(bothello_omp PRIVATE bothello_core)

# CUDA Bot (Tree)
add_executable(bothello_cuda main.cpp)
target_compile_definitions(bothello_cuda PRIVATE BOT_CUDA)
target_link_libraries(bothello_cuda PRIVATE bothello_core)

# Leaf Parallel Bot
add_executable(bothello_leaf main.cpp)
target_compile_definitions(bothello_leaf PRIVATE BOT_LEAF)
target_link_libraries(bothello_leaf PRIVATE bothello_core)

# Block Parallel Bot
add_executable(bothello_block main.cpp)
target_compile_definitions(bothello_block PRIVATE BOT_BLOCK)
target_link_libraries(bothello_block PRIVATE bothello_core)

# Benchmark executable
add_executable(benchmark src/benchmark.cpp)
target_link_libraries(benchmark PRIVATE bothello_core)

option(BUILD_TESTS "Build tests" OFF)

if(BUILD_TESTS)
	# Testing setup
	include(FetchContent)
	FetchContent_Declare(
		googletest
		URL https://github.com/google/googletest/archive/03597a01ee50ed33e9dfd640b249b4be3799d395.zip)
	# Prevent overriding the parent project's compiler/linker settings
	set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
	FetchContent_MakeAvailable(googletest)

	enable_testing()

	# Test executable
	add_executable(unit_tests tests/test_board.cpp)
	target_link_libraries(unit_tests PRIVATE bothello_core gtest_main)

	include(GoogleTest)
	gtest_discover_tests(unit_tests)
endif()
