cmake_minimum_required(VERSION 3.10)
project(BenchmarkBuilder CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PRINT_CSV "Enable CSV printing" OFF)
set(ALIGN_BYTES 32 CACHE STRING "Alignment bytes for memory allocation")

find_package(OpenMP REQUIRED)

if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CCX_OPT -Ofast -march=native)
    message(STATUS "Using GCC optimization flags: ${CCX_OPT}")
    add_compile_options(${CCX_OPT})
elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CCX_OPT -Ofast -march=native)
    message(STATUS "Using Clang optimization flags: ${CCX_OPT}")
    add_compile_options(${CCX_OPT})
endif()

include_directories(${CMAKE_SOURCE_DIR}/include)
file(GLOB SRC_FILES "${CMAKE_SOURCE_DIR}/src/*.cpp")

foreach(SRC_FILE ${SRC_FILES})
    get_filename_component(FILENAME ${SRC_FILE} NAME_WE)
    add_executable(${FILENAME} ${SRC_FILE})

    # Link only OpenMP
    target_link_libraries(${FILENAME} PRIVATE OpenMP::OpenMP_CXX)

    # Only define ALIGN_BYTES now
    target_compile_definitions(${FILENAME} PRIVATE ALIGN_BYTES=${ALIGN_BYTES})

    if(PRINT_CSV)
        target_compile_definitions(${FILENAME} PRIVATE PRINT_CSV)
    endif()
endforeach()
