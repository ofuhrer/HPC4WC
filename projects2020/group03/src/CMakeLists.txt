set (CMAKE_Fortran_MODULE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR})

include (utils.cmake)
include (partitioner.cmake)

add_subdirectory (halo)

add_subdirectory (mpi)
add_subdirectory (orig)

add_subdirectory (openmp)
add_subdirectory (openmp_target)
# add_subdirectory (openacc)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
