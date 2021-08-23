cmake_minimum_required(VERSION 3.10)

project(executables)

add_executable(diffusion2d diffusion2d.cpp)
target_link_libraries(diffusion2d eigen HPC4WC)
target_include_directories(diffusion2d PUBLIC HPC4WC)

add_executable(autotune_search autotune_search.cpp)
target_link_libraries(autotune_search eigen HPC4WC)
target_include_directories(autotune_search PUBLIC HPC4WC)

if(${MPI_FOUND})
    add_executable(mpi_diffusion mpi_diffusion.cpp)
    target_link_libraries(mpi_diffusion eigen HPC4WC)
    target_include_directories(mpi_diffusion PUBLIC HPC4WC)
endif(${MPI_FOUND})

add_executable(experiment_openmp_k experiment_openmp_k.cpp)
target_link_libraries(experiment_openmp_k eigen HPC4WC)
target_include_directories(experiment_openmp_k PUBLIC HPC4WC)
add_dependencies(experiment_openmp_k diffusion2d)

add_executable(block_i block_i.cpp)
target_link_libraries(block_i eigen HPC4WC)
target_include_directories(block_i PUBLIC HPC4WC)
add_dependencies(block_i diffusion2d)


add_executable(block_ij block_ij.cpp)
target_link_libraries(block_ij eigen HPC4WC)
target_include_directories(block_ij PUBLIC HPC4WC)
add_dependencies(block_ij diffusion2d)

add_executable(blocking_diff_simple blocking_diff_simple.cpp)
target_link_libraries(blocking_diff_simple eigen HPC4WC)
target_include_directories(blocking_diff_simple PUBLIC HPC4WC)
add_dependencies(blocking_diff_simple diffusion2d)