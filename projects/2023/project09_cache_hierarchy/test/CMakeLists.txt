# Avoid warning about DOWNLOAD_EXTRACT_TIMESTAMP in CMake 3.24:
if(CMAKE_VERSION VERSION_GREATER_EQUAL "3.24.0")
  cmake_policy(SET CMP0135 NEW)
endif()

include(FetchContent)
FetchContent_Declare(
  googletest
  URL https://github.com/google/googletest/archive/f8d7d77c06936315286eb55f8de22cd23c188571.zip # v1.14
)

# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
FetchContent_MakeAvailable(googletest)

find_package(Eigen3 REQUIRED NO_MODULE)

add_executable(cache_simulator_test src/cache_simulator_test.cpp ../src/cache_simulator.cpp)
target_link_libraries(cache_simulator_test Eigen3::Eigen GTest::gtest_main)
target_compile_definitions(cache_simulator_test PUBLIC CACHE_TRACKING)

add_executable(stencil_test src/stencil_test.cpp ../src/cache_simulator.cpp)
target_link_libraries(stencil_test Eigen3::Eigen GTest::gtest_main)

include(GoogleTest)
gtest_discover_tests(cache_simulator_test)
