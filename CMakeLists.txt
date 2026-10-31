cmake_minimum_required(VERSION 3.18)

# Set project name and languages (CXX + CUDA)
project(LucidCamera LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES 87) # NVIDIA Jetson AGX Orin (sm_87)

# 1. Find OpenCV
find_package(OpenCV REQUIRED)

# 2. Define base paths
set(ARENA_SDK_ROOT "/home/swdep/Downloads/arena_ptp/ArenaSDK_v0.1.105_Linux_ARM64/ArenaSDK_Linux_ARM64")
set(TENSORRT_ROOT "/media/swdep/500GB_SSD_PERCEP/black_magic/Armory/TensorRT-11.3.0.99")

# 3. Add Include Directories
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/src
    ${ARENA_SDK_ROOT}/include
    ${ARENA_SDK_ROOT}/GenICam/library/CPP/include
    ${TENSORRT_ROOT}/include
    /usr/local/cuda/include
    ${OpenCV_INCLUDE_DIRS}
)

# 4. Add Library Search Paths
link_directories(
    ${ARENA_SDK_ROOT}/lib
    ${ARENA_SDK_ROOT}/GenICam/library/lib/Linux64_ARM
    ${TENSORRT_ROOT}/lib
    /usr/local/cuda/lib64
)

# 5. Automatically discover all source and header files in include and src folders (with CONFIGURE_DEPENDS)
file(GLOB_RECURSE SRC_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu"
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
)

file(GLOB_RECURSE INC_FILES CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/*.cuh"
)

# 6. Build test_app (from main.cu and all src/include files) using NVCC
add_executable(test_app main.cu ${SRC_FILES} ${INC_FILES})
target_link_libraries(test_app
    arena
    GCBase_gcc54_v3_3_LUCID
    GenApi_gcc54_v3_3_LUCID
    pthread
    nvinfer
    cudart
    ${OpenCV_LIBS}
)

# Ensure RPATH points to TensorRT and Arena libraries at runtime
set_target_properties(test_app PROPERTIES
    BUILD_WITH_INSTALL_RPATH TRUE
    INSTALL_RPATH "${TENSORRT_ROOT}/lib;${ARENA_SDK_ROOT}/lib;/usr/local/cuda/lib64"
)