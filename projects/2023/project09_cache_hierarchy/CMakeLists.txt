cmake_minimum_required(VERSION 3.14)

project(cache_hierarchy LANGUAGES CXX)

if("${PROJECT_SOURCE_DIR}" STREQUAL "${PROJECT_BINARY_DIR}")
  message(SEND_ERROR "In-source builds are not allowed.")
endif()

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE "Release" CACHE STRING "" FORCE)
endif()

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(UNIX)
  if("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
    # GCC compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fdiagnostics-color=always -Wall -Wextra -Wpedantic -Wcast-qual -Wshadow -Wundef -Wunused-macros -Wextra-semi")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3 -ffast-math -fno-tree-vectorize")
    set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -DDEBUG -Og -ggdb3") # -O0 for inlined functions
  elseif("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang")
    # Clang or AppleClang compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fcolor-diagnostics -Wall -Wextra -Wpedantic -Wno-c++98-compat -Wno-c++98-compat-pedantic")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -DNDEBUG -O3 -DNDEBUG -O3 -ffast-math -fno-vectorize -fno-slp-vectorize")
  elseif("${CMAKE_CXX_COMPILER_ID}" STREQUAL "IntelLLVM")
    # Intel compiler
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fsycl -Wall")
    set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -fast")
  else()
    message(SEND_ERROR "Unsupported compiler: ${CMAKE_CXX_COMPILER_ID}")
  endif()
else()
  message(SEND_ERROR "Unsupported OS.")
endif()

add_subdirectory(src)
include_directories(src)

option(ENABLE_TESTING "Enable the tests" ON)

if(ENABLE_TESTING)
  enable_testing()
  add_subdirectory(test)
endif()
