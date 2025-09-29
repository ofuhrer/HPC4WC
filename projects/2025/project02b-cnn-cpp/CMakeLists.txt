cmake_minimum_required(VERSION 3.18)
project(cnn_cpp)

# ------------------------------
# Options
# ------------------------------
option(BUILD_GPU "Build CUDA targets (GPU layers + test)" ON)

# ------------------------------
# C++ standard
# ------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ------------------------------
# LibTorch
# ------------------------------
set(CMAKE_PREFIX_PATH "${CMAKE_SOURCE_DIR}/libtorch")
find_package(Torch REQUIRED)

# ------------------------------
# OpenMP
# ------------------------------
find_package(OpenMP)
if (OpenMP_CXX_FOUND)
  message(STATUS "OpenMP found: enabling")
else()
  message(WARNING "OpenMP NOT found; optimized targets will run single-threaded")
endif()

# ------------------------------
# cnpy + zlib
# ------------------------------
add_library(cnpy STATIC external/cnpy/cnpy.cpp)
target_include_directories(cnpy PUBLIC external/cnpy)
find_package(ZLIB REQUIRED)
target_link_libraries(cnpy PUBLIC ZLIB::ZLIB)

# ------------------------------
# Include paths
# ------------------------------
include_directories(
  ${CMAKE_SOURCE_DIR}/include
  ${CMAKE_SOURCE_DIR}/include/naive
  ${CMAKE_SOURCE_DIR}/include/stencil
  ${CMAKE_SOURCE_DIR}/include/naive_opt
  ${CMAKE_SOURCE_DIR}/include/stencil_opt
  ${CMAKE_SOURCE_DIR}/include/stencil_opt_alt
  ${CMAKE_SOURCE_DIR}/include/naive_opt_alt
)

# ------------------------------
# Naive layers
# ------------------------------
add_library(cnn_layers STATIC
  src/naive/conv2d_naive.cpp
  src/naive/relu_naive.cpp
  src/naive/maxpool_naive.cpp
  src/naive/linear_naive.cpp
  src/naive/cnn_naive.cpp
)
target_include_directories(cnn_layers PUBLIC include/naive)

# ------------------------------
# Stencil layers
# ------------------------------
add_library(cnn_stencil_layers STATIC
  src/stencil/conv2d_stencil.cpp
  src/stencil/relu_stencil.cpp
  src/stencil/maxpool_stencil.cpp
  src/stencil/linear_stencil.cpp
  src/stencil/cnn_stencil.cpp
)
target_include_directories(cnn_stencil_layers PUBLIC include/stencil)

# ------------------------------
# GPU layers (optional)
# ------------------------------
if (BUILD_GPU)
  enable_language(CUDA)
  set(CMAKE_CUDA_STANDARD 17)
  set(CMAKE_CUDA_STANDARD_REQUIRED ON)

  # Default safe arch list unless user provided one
  if (NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 80;86;89;90)
    message(STATUS "CMAKE_CUDA_ARCHITECTURES not set; defaulting to: ${CMAKE_CUDA_ARCHITECTURES}")
  endif()

  add_library(cnn_gpu_layers STATIC
      src/gpu/conv2d_cuda.cu
      src/gpu/relu_cuda.cu
      src/gpu/maxpool_cuda.cu
      src/gpu/linear_cuda.cu
      src/gpu/cnn_cuda.cpp
  )
  target_include_directories(cnn_gpu_layers PUBLIC include/gpu)
  set_target_properties(cnn_gpu_layers PROPERTIES
      CUDA_SEPARABLE_COMPILATION ON
      POSITION_INDEPENDENT_CODE ON
  )
endif()

# ------------------------------
# Naive optimized layers
# ------------------------------
add_library(cnn_opt_layers STATIC
  src/naive_opt/conv2d_opt.cpp
  src/naive_opt/relu_opt.cpp
  src/naive_opt/maxpool_opt.cpp
  src/naive_opt/linear_opt.cpp
  src/naive_opt/cnn_opt.cpp
)
target_include_directories(cnn_opt_layers PUBLIC include/naive_opt)
if (OpenMP_CXX_FOUND)
  target_link_libraries(cnn_opt_layers PUBLIC OpenMP::OpenMP_CXX)
endif()

# ------------------------------
# Stencil optimized layers
# ------------------------------
add_library(cnn_stencil_opt_layers STATIC
  src/stencil_opt/conv2d_stencil_opt.cpp
  src/stencil_opt/relu_stencil_opt.cpp
  src/stencil_opt/maxpool_stencil_opt.cpp
  src/stencil_opt/linear_stencil_opt.cpp
  src/stencil_opt/cnn_stencil_opt.cpp
)
target_include_directories(cnn_stencil_opt_layers PUBLIC include/stencil_opt)
if (OpenMP_CXX_FOUND)
  target_link_libraries(cnn_stencil_opt_layers PUBLIC OpenMP::OpenMP_CXX)
endif()

# ------------------------------
# Stencil optimized ALT
# ------------------------------
add_library(cnn_stencil_opt_alt_layers STATIC
  src/stencil_opt_alt/conv2d_stencil_alt.cpp
  src/stencil_opt_alt/relu_stencil_alt.cpp
  src/stencil_opt_alt/maxpool_stencil_alt.cpp
  src/stencil_opt_alt/linear_stencil_alt.cpp
  src/stencil_opt_alt/cnn_stencil_alt.cpp
)
target_include_directories(cnn_stencil_opt_alt_layers PUBLIC include/stencil_opt_alt)
if (OpenMP_CXX_FOUND)
  target_link_libraries(cnn_stencil_opt_alt_layers PUBLIC OpenMP::OpenMP_CXX)
endif()

# ------------------------------
# Naive optimized ALT
# ------------------------------
add_library(cnn_opt_alt_layers STATIC
  src/naive_opt_alt/conv2d_opt_alt.cpp
  src/naive_opt_alt/relu_opt_alt.cpp
  src/naive_opt_alt/maxpool_opt_alt.cpp
  src/naive_opt_alt/linear_opt_alt.cpp
  src/naive_opt_alt/cnn_opt_alt.cpp
)
target_include_directories(cnn_opt_alt_layers PUBLIC include/naive_opt_alt)
if (OpenMP_CXX_FOUND)
  target_link_libraries(cnn_opt_alt_layers PUBLIC OpenMP::OpenMP_CXX)
endif()

# ------------------------------
# Test binaries
# ------------------------------
add_executable(test_mnist src/naive/test_mnist.cpp)
target_link_libraries(test_mnist PRIVATE cnn_layers cnpy)

add_executable(test_mnist_stencil src/stencil/test_mnist_stencil.cpp)
target_link_libraries(test_mnist_stencil PRIVATE cnn_stencil_layers cnpy)

if (BUILD_GPU)
  add_executable(test_mnist_gpu src/gpu/test_mnist_gpu.cpp)
  target_link_libraries(test_mnist_gpu PRIVATE cnn_gpu_layers cnpy "${TORCH_LIBRARIES}")
endif()

add_executable(test_mnist_opt src/naive_opt/test_mnist_opt.cpp)
target_link_libraries(test_mnist_opt PRIVATE cnn_opt_layers cnpy)
if (OpenMP_CXX_FOUND)
  target_link_libraries(test_mnist_opt PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(test_mnist_stencil_opt src/stencil_opt/test_mnist_stencil_opt.cpp)
target_link_libraries(test_mnist_stencil_opt PRIVATE cnn_stencil_opt_layers cnpy)
if (OpenMP_CXX_FOUND)
  target_link_libraries(test_mnist_stencil_opt PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(test_mnist_stencil_alt src/stencil_opt_alt/test_mnist_stencil_alt.cpp)
target_link_libraries(test_mnist_stencil_alt PRIVATE cnn_stencil_opt_alt_layers cnpy)
if (OpenMP_CXX_FOUND)
  target_link_libraries(test_mnist_stencil_alt PRIVATE OpenMP::OpenMP_CXX)
endif()

add_executable(test_mnist_opt_alt src/naive_opt_alt/test_mnist_opt_alt.cpp)
target_link_libraries(test_mnist_opt_alt PRIVATE cnn_opt_alt_layers cnpy)
if (OpenMP_CXX_FOUND)
  target_link_libraries(test_mnist_opt_alt PRIVATE OpenMP::OpenMP_CXX)
endif()


# ------------------------------
# Default build type
# ------------------------------
if (NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "Build type" FORCE)
endif()