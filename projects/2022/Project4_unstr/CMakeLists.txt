#install cmake
#in folder of source code mkdir build
#cd build
#cmake ../
#make
#./main 

cmake_minimum_required (VERSION 3.16.3)
cmake_policy(SET CMP0076 NEW)
set(CMAKE_CXX_STANDARD 17)
project(unstructured_grids)

include(FetchContent)
 
find_package(Eigen3 3.3 REQUIRED NO_MODULE)
include_directories(${EIGEN3_INCLUDE_DIR})
message(STATUS "EIGEN: " ${EIGEN_VERSION} " (internal)")


set(SOURCES "mesh.cpp" "walltime.cpp")
set(HEADERS "mesh.hpp" "interpolation.hpp" "walltime.hpp" "test_laplacian.hpp")
             
#add_executable(main "${SOURCES1}" "${HEADERS1}" "main.cpp")
add_executable(main "${SOURCES}" "${HEADERS}" "main.cpp")

target_link_libraries(main Eigen3::Eigen)
#target_link_libraries()

#option 1 downloaded gmsh with brew 
find_library("gmsh", [".", "../", "/usr/local/opt/gmsh/lib/"])
target_link_libraries(main gmsh)

#option 2 downloaded gmsh with brew 
#" " absolute path to location were gmsh was installed e.g. brew info gmsh 
# include_directories(foo "/usr/local/opt/gmsh/include")
# target_link_libraries(main "/usr/local/opt/gmsh/lib/libgmsh.4.10.dylib")

 

