cmake_minimum_required(VERSION 3.12)
project(dslgen)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR
   CMAKE_CXX_COMPILER_ID MATCHES "GNU" OR
   CMAKE_CXX_COMPILER_ID MATCHES "Intel")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Wpedantic -Wno-deprecated -march=native")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -g -ggdb -fno-omit-frame-pointer")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3 -fopenmp -march=native -DNDEBUG -s -flto")
endif()

find_package(Python3 REQUIRED COMPONENTS Development)
find_package(Boost 1.68 REQUIRED COMPONENTS python numpy)

include(FetchContent)
FetchContent_Declare(eigen3
    GIT_REPOSITORY https://gitlab.com/libeigen/eigen.git
    GIT_SHALLOW TRUE
)
FetchContent_GetProperties(eigen3)
if(NOT eigen3_POPULATED)
    FetchContent_Populate(eigen3)
endif()
add_library(eigen3 INTERFACE)
target_include_directories(eigen3 INTERFACE "${eigen3_SOURCE_DIR}")

add_library(tsc INTERFACE)
target_include_directories(tsc INTERFACE "${CMAKE_CURRENT_LIST_DIR}/include")

set(SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}" CACHE FILEPATH "directory where 'dslgen.cpp' is located")

set(CMAKE_SHARED_MODULE_PREFIX "")
add_library(dslgen MODULE "${SOURCE_DIR}/dslgen.cpp")

# For easier debugging: change `dslgen.cpp` to include `common_cpp.hpp`
# instead of `common_python.hpp`, uncomment the following line to make
# the target `dslgen` an executable instead of a library, and add your
# own main function using Eigen vectors as the array types.
# add_executable(dslgen "${SOURCE_DIR}/dslgen.cpp")

target_link_libraries(dslgen
    Python3::Python
    Boost::python
    Boost::numpy
    tsc
    eigen3
)
