add_executable(cache_hierarchy cache_hierarchy.cpp cache_simulator.cpp)
add_executable(cache_hierarchy_hr cache_hierarchy.cpp cache_simulator.cpp)

find_package(Eigen3 REQUIRED NO_MODULE)
target_link_libraries(cache_hierarchy PUBLIC Eigen3::Eigen)
target_link_libraries(cache_hierarchy_hr PUBLIC Eigen3::Eigen)

target_compile_definitions(cache_hierarchy_hr PUBLIC CACHE_TRACKING)
