cmake_minimum_required(VERSION 3.14)

project(BotHello CXX CUDA)

option(BUILD_TESTS "Build tests" OFF)

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
	set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release" "MinSizeRel" "RelWithDebInfo" "Profile")
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
elseif(CMAKE_BUILD_TYPE STREQUAL "Profile")
    message(STATUS "Configuring for Profile build: Adding optimization and lineinfo flags")
    add_compile_options($<$<COMPILE_LANGUAGE:CXX>:-O3>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-O3>)
    add_compile_options($<$<COMPILE_LANGUAGE:CUDA>:-lineinfo>)
endif()

find_package(OpenMP REQUIRED)

# Add NVTX library
add_subdirectory(c)

# --- Modular Libraries ---

# 1. Board Library
add_library(bothello_board
    src/board.cpp
    src/board.h
)
# Include the root directory so we can include "src/board.h"
target_include_directories(bothello_board PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# 2. MCTS CPU / Base Library
add_library(bothello_mcts_cpu
    src/mcts.cpp
    src/mcts.h
)
target_link_libraries(bothello_mcts_cpu PUBLIC bothello_board)

# 3. MCTS OpenMP Library
add_library(bothello_mcts_tree_omp
    src/mcts_tree_openmp.cpp
    src/mcts_tree_openmp.h
)
target_link_libraries(bothello_mcts_tree_omp PUBLIC bothello_board bothello_mcts_cpu OpenMP::OpenMP_CXX)

# 4. MCTS CUDA Tree Library
add_library(bothello_mcts_tree_cuda
    src/mcts_tree_cuda.cu
    src/mcts_tree_cuda.h
)
target_link_libraries(bothello_mcts_tree_cuda PUBLIC bothello_board bothello_mcts_cpu)

# 5. MCTS Leaf Parallel Library
add_library(bothello_mcts_leaf
    src/mcts_leaf_parallel.cu
    src/mcts_leaf_parallel.h
)
target_link_libraries(bothello_mcts_leaf PUBLIC bothello_board bothello_mcts_cpu)

# 6. MCTS Block Parallel Library
add_library(bothello_mcts_block
    src/mcts_block.cu
    src/mcts_block.h
)
target_link_libraries(bothello_mcts_block PUBLIC bothello_board bothello_mcts_cpu)

# 7. BotHello Core - Aggregates all libraries
add_library(bothello_core INTERFACE)
target_link_libraries(bothello_core INTERFACE
    bothello_board
    bothello_mcts_cpu
    bothello_mcts_tree_omp
    bothello_mcts_tree_cuda
    bothello_mcts_leaf
    bothello_mcts_block
)

# Determine if we should link NVTX
set(NVTX_LIB "")
if(CMAKE_BUILD_TYPE STREQUAL "Profile" OR CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(NVTX_LIB nvtx3-cpp)
    add_compile_definitions(USE_NVTX)
    message(STATUS "NVTX Enabled for ${CMAKE_BUILD_TYPE} build")
endif()

if(NOT BUILD_TESTS)
    # --- Executables ---

    # CPU Bot
    add_executable(bothello_cpu main.cpp)
    target_compile_definitions(bothello_cpu PRIVATE BOT_CPU)
    target_link_libraries(bothello_cpu PRIVATE bothello_mcts_cpu ${NVTX_LIB})

    # OpenMP Bot
    add_executable(bothello_tree_omp main.cpp)
    target_compile_definitions(bothello_tree_omp PRIVATE BOT_OMP)
    target_link_libraries(bothello_tree_omp PRIVATE bothello_mcts_tree_omp ${NVTX_LIB})

    # CUDA Bot (Tree)
    add_executable(bothello_tree_cuda main.cpp)
    target_compile_definitions(bothello_tree_cuda PRIVATE BOT_CUDA)
    target_link_libraries(bothello_tree_cuda PRIVATE bothello_mcts_tree_cuda ${NVTX_LIB})

    # Leaf Parallel Bot
    add_executable(bothello_leaf main.cpp)
    target_compile_definitions(bothello_leaf PRIVATE BOT_LEAF)
    target_link_libraries(bothello_leaf PRIVATE bothello_mcts_leaf ${NVTX_LIB})

    # Block Parallel Bot
    add_executable(bothello_block main.cpp)
    target_compile_definitions(bothello_block PRIVATE BOT_BLOCK)
    target_link_libraries(bothello_block PRIVATE bothello_mcts_block ${NVTX_LIB})

    # Benchmark executable
    add_executable(benchmark src/benchmark.cpp)
    target_link_libraries(benchmark PRIVATE bothello_core)
endif()

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
