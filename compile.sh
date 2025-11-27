#!/bin/bash
set -e

BUILD_TYPE="Debug"
BUILD_DIR="build"
MODE="build"
EXPLICIT_TYPE=false

# Parse arguments
for arg in "$@"
do
    case $arg in
        -d)
            if [ "$MODE" == "test" ]; then
                echo "Error: -d cannot be used with -t"
                exit 1
            fi
            if [ "$EXPLICIT_TYPE" = true ] && [ "$BUILD_TYPE" == "Release" ]; then
                echo "Error: Cannot specify both -d and -r"
                exit 1
            fi
            BUILD_TYPE="Debug"
            EXPLICIT_TYPE=true
            ;;
        -r)
            if [ "$MODE" == "test" ]; then
                echo "Error: -r cannot be used with -t"
                exit 1
            fi
            if [ "$EXPLICIT_TYPE" = true ] && [ "$BUILD_TYPE" == "Debug" ]; then
                echo "Error: Cannot specify both -d and -r"
                exit 1
            fi
            BUILD_TYPE="Release"
            EXPLICIT_TYPE=true
            ;;
        -t)
            if [ "$EXPLICIT_TYPE" = true ]; then
                echo "Error: -t cannot be used with -d or -r"
                exit 1
            fi
            MODE="test"
            ;;
        run)
            MODE="run"
            ;;
        clean)
            echo "Removing build directory..."
            rm -rf $BUILD_DIR
            exit 0
            ;;
        -h|--help)
            echo "Usage: ./build.sh [options] [command]"
            echo ""
            echo "Options:"
            echo "  -d          Build in Debug mode (default)"
            echo "  -r          Build in Release mode"
            echo "  -t          Build and run tests (cannot be used with -d or -r)"
            echo "  -h, --help  Show this help message"
            echo ""
            echo "Commands:"
            echo "  run         Run the main application after building"
            echo "  clean       Remove the build directory"
            exit 0
            ;;
        *)
            echo "Unknown option: $arg"
            exit 1
            ;;
    esac
done

echo "Configuring CMake for $BUILD_TYPE..."
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
if [ "$MODE" == "test" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DBUILD_TESTS=ON"
fi
cmake -S . -B "$BUILD_DIR" $CMAKE_ARGS

echo "Building project..."
cmake --build "$BUILD_DIR" -- -j$(nproc)

if [ "$MODE" == "test" ]; then
    echo "Running tests..."
    # (cd "$BUILD_DIR" && ctest --output-on-failure)
	(cd "$BUILD_DIR" && ctest --verbose)
elif [ "$MODE" == "run" ]; then
    echo "Running application..."
    ./"$BUILD_DIR"/bothello
fi
