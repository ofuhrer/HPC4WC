cmake_minimum_required(VERSION 3.11)

project(HPC4WC-Project)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release RelWithDebInfo MinSizeRel." FORCE)
endif(NOT CMAKE_BUILD_TYPE)

set(CMAKE_CXX_STANDARD 17)

find_package(OpenMP)
find_package(MPI)

add_subdirectory(ext)
add_subdirectory(src)
add_subdirectory(tests)