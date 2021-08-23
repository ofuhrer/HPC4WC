cmake_minimum_required(VERSION 3.10)

project(HPC4WC)

file(
    GLOB
    SOURCE #
    "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp" #
)

file(
    GLOB
    HEADER #
    "${CMAKE_CURRENT_SOURCE_DIR}/include/HPC4WC/*.h" #
)

add_library(${PROJECT_NAME} STATIC ${SOURCE} ${HEADER})
target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_include_directories(${PROJECT_NAME} PUBLIC ${FLAGS_SOURCE_DIR}/include)

if(${OpenMP_FOUND})
    target_link_libraries(${PROJECT_NAME} PUBLIC OpenMP::OpenMP_CXX)
    message("Found OpenMP.")
else(${OpenMP_FOUND})
    message("OpenMP not found.")
endif(${OpenMP_FOUND})

if(${MPI_FOUND})
    target_link_libraries(${PROJECT_NAME} PUBLIC MPI::MPI_CXX)
    message("Found MPI.")
else(${MPI_FOUND})
    message("MPI not found, disabling partitioner.")
    target_compile_definitions(${PROJECT_NAME} PUBLIC -DDISABLE_PARTITIONER)
endif(${MPI_FOUND})

target_link_libraries(${PROJECT_NAME}
    PUBLIC eigen
)
