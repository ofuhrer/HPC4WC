cmake_minimum_required(VERSION 3.18)
project(CUDAArccosProject LANGUAGES CUDA CXX)

# Enable CUDA
enable_language(CUDA)

# Set CUDA standard
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# Add global compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")

# Optional flag for laptop builds
option(LAPTOP_BUILD "Disable sm_90 architecture for local/laptop builds" OFF)

if(NOT LAPTOP_BUILD)
    message(STATUS "Cluster build: setting CUDA architecture to sm_90")
    set(CMAKE_CUDA_ARCHITECTURES 90)
else()
    message(STATUS "Laptop build: using default CUDA architecture")
endif()

# Define source files
set(CUDA_SOURCES
    src/cuda/main.cu
    src/cuda/arccos_cuda.cu
)

# Add the executable
add_executable(cuda_arccos ${CUDA_SOURCES})

# Include the headers
target_include_directories(cuda_arccos PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

# Add stencil GPU executable
set(STENCIL_SOURCES
    src/cuda/stencil2d-gpu.cu
    src/cuda/stencil2d_helper/stencil_kernels.cu
)

add_executable(stencil2d_gpu ${STENCIL_SOURCES})
target_include_directories(stencil2d_gpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

# Add stencil GPU executable with multiple streams
set(STENCIL_STREAM_SOURCES
    src/cuda/stencil2d-gpu_streams.cu
    src/cuda/stencil2d_helper/stencil_kernels.cu
)

add_executable(stencil2d_gpu_streams ${STENCIL_STREAM_SOURCES})
target_include_directories(stencil2d_gpu_streams PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

# Add stencil CPU baseline executable
set(STENCIL_BASE_SOURCES
    src/cuda/stencil2d_helper/stencil2d-base.cpp
)

add_executable(stencil2d_cpu ${STENCIL_BASE_SOURCES})
target_include_directories(stencil2d_cpu PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src/cuda)

# Directories with Python/runtime scripts
set(CONTAINS_PYTHON
    ${PROJECT_SOURCE_DIR}/src/gt4py
    ${PROJECT_SOURCE_DIR}/src/reference
    ${PROJECT_SOURCE_DIR}/tests
)

# Create symlinks for Python/runtime folders in build directory
foreach(INPUT_DIR ${CONTAINS_PYTHON})
    get_filename_component(DIR_NAME ${INPUT_DIR} NAME)
    add_custom_command(TARGET cuda_arccos POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${INPUT_DIR} $<TARGET_FILE_DIR:cuda_arccos>/${DIR_NAME}
        COMMENT "Creating symlink to ${DIR_NAME} in build directory"
    )
endforeach()
