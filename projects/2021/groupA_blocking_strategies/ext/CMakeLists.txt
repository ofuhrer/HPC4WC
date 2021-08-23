include(FetchContent)

macro(fetch what)
    FetchContent_GetProperties("${what}")
    if(NOT ${${what}_POPULATED})
        message(STATUS "fetching ${what} ...")
        FetchContent_Populate(${what})
    endif()
    mark_as_advanced(${${what}_SOURCE_DIR})
endmacro()

FetchContent_Declare(
    eigen
    URL                 https://gitlab.com/libeigen/eigen/-/archive/3.3.7/eigen-3.3.7.zip
    URL_HASH            MD5=888aab45512cc0c734b3e8f60280daba
)
fetch(eigen)
add_library(eigen INTERFACE)
target_include_directories(eigen INTERFACE ${eigen_SOURCE_DIR})

FetchContent_Declare(
    googletest
    URL                 https://github.com/google/googletest/archive/refs/tags/release-1.11.0.zip
    URL_HASH            MD5=52943a59cefce0ae0491d4d2412c120b
)
fetch(googletest)
# For Windows: Prevent overriding the parent project's compiler/linker settings
set(gtest_force_shared_crt ON CACHE BOOL "" FORCE)
add_subdirectory(${googletest_SOURCE_DIR} ${googletest_BINARY_DIR})

# Simple args-parser
FetchContent_Declare(
    flags
    GIT_REPOSITORY	    https://github.com/sailormoon/flags.git
    GIT_TAG		        master
)
fetch(flags)
set(FLAGS_SOURCE_DIR ${flags_SOURCE_DIR} CACHE INTERNAL "flags source directory")

# Set folder for visual studio
if(WIN32)
    set_property(GLOBAL PROPERTY USE_FOLDERS ON)
    set_target_properties(gtest PROPERTIES FOLDER Ext)
    set_target_properties(gtest_main PROPERTIES FOLDER Ext)
    set_target_properties(gmock PROPERTIES FOLDER Ext)
    set_target_properties(gmock_main PROPERTIES FOLDER Ext)
    set_target_properties(eigen PROPERTIES FOLDER Ext)
endif(WIN32)
