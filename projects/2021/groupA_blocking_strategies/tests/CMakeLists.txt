cmake_minimum_required(VERSION 3.10)

project(tests)

file(
    GLOB
    SOURCES #
    "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp" #
)

enable_testing()

add_executable(unittests ${SOURCES})
target_link_libraries(unittests 
    PUBLIC HPC4WC
    PUBLIC gtest_main
)

include(GoogleTest)
gtest_discover_tests(unittests)
