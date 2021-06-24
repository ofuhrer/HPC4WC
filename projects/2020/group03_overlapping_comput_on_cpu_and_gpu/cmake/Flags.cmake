# list (REMOVE_ITEM CMAKE_CONFIGURATION_TYPES MinSizeRel)
# list (APPEND CMAKE_CONFIGURATION_TYPES Fast)

add_library (OpenMP  INTERFACE IMPORTED)
add_library (OpenACC INTERFACE IMPORTED)

include (Fortran_FLAGS)
include (CXX_FLAGS)
include (CUDA_FLAGS)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 :
