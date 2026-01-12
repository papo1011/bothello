#!/bin/bash
set -e

BUILD_DIR="build"
BUILD_TYPE="Debug"
BUILD_TESTS="OFF"
DO_TEST="false"

function show_help {
    echo "Usage: ./compile.sh [options] [clean]"
    echo ""
    echo "Options:"
    echo "  -d          Build in Debug mode (default)"
    echo "  -r          Build in Release mode"
    echo "  -p          Build in Profile mode (-O3 -lineinfo)"
    echo "  -t          Build and run tests"
    echo "  clean       Remove build directory"
    echo "  -h, --help  Show this help message"
}

# Parse arguments
for arg in "$@"
do
    case $arg in
        -d)
            BUILD_TYPE="Debug"
            ;;
        -r)
            BUILD_TYPE="Release"
            ;;
        -p)
            BUILD_TYPE="Profile"
            ;;
        -t)
            BUILD_TESTS="ON"
            DO_TEST="true"
            ;;
        clean)
            echo "Removing build directory..."
            rm -rf "$BUILD_DIR"
            exit 0
            ;;
        -h|--help)
            show_help
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            show_help
            exit 1
            ;;
    esac
done

echo "Configuring CMake (Type: $BUILD_TYPE, Tests: $BUILD_TESTS)..."
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE" -DBUILD_TESTS="$BUILD_TESTS"

echo "Building project..."
cmake --build "$BUILD_DIR" -- -j$(nproc)

if [ "$DO_TEST" = "true" ]; then
    echo "Running tests..."
    cd "$BUILD_DIR" && ctest --output-on-failure
fi
