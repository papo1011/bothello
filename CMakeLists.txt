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

# Create a library for the core logic
add_library(bothello_core
    src/board.cpp
    src/board.h
    src/mcts.h
    src/mcts.cpp
	src/mcts_leaf_parallel.cu
)

# Include the root directory so we can include "src/board.h"
target_include_directories(bothello_core PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

# Main executable
add_executable(bothello main.cpp)
target_link_libraries(bothello PRIVATE bothello_core)

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
