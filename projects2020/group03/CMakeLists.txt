cmake_minimum_required (VERSION 3.12)
project(stencil2d LANGUAGES Fortran CXX C)

if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_BINARY_DIR})
	message (FATAL_ERROR "In-source builds not allowed. Please make a new directory (called a build directory) and run CMake from there. You may need to remove CMakeCache.txt. ")
endif ()

if (NOT CMAKE_BUILD_TYPE)
	set (CMAKE_BUILD_TYPE "Release")
endif ()

set (CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include (Flags)

find_package (OpenMP REQUIRED COMPONENTS Fortran CXX)
find_package (MPI    )# REQUIRED COMPONENTS Fortran CXX)
find_package (OpenACC)

add_subdirectory (src)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 :
