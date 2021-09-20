cmake_minimum_required(VERSION 3.14)
project(HPWCProject)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

include(FetchContent)

find_package(Eigen3 3.4 QUIET)
if(NOT EIGEN_FOUND)
	FetchContent_Declare(
		Eigen
		GIT_REPOSITORY https://gitlab.com/libeigen/eigen
		GIT_TAG        3.4
	)
	FetchContent_MakeAvailable(Eigen)
endif()

FetchContent_Declare(
	ProgramOptionsHxx
	GIT_REPOSITORY https://github.com/Fytch/ProgramOptions.hxx
)
FetchContent_MakeAvailable(ProgramOptionsHxx)

find_package(OpenMP REQUIRED)

add_executable(HPWCProject main.cpp)
target_link_libraries(HPWCProject Eigen3::Eigen ProgramOptionsHxx OpenMP::OpenMP_CXX)
