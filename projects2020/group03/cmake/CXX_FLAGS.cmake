set (
        CMAKE_CXX_FLAGS ""
        CACHE STRING "Flags used by the CXX compiler during all build types."
        FORCE
)
set (
        CMAKE_CXX_FLAGS_DEBUG
        CACHE STRING "Flags used by the CXX compiler during Debug builds."
        FORCE
)
set (
        CMAKE_CXX_FLAGS_RELEASE
        CACHE STRING "Flags used by the CXX compiler during Release builds."
        FORCE
)
set (
        CMAKE_CXX_FLAGS_RELWITHDEBINFO
        CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
        FORCE
)

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
	set (
		CMAKE_CXX_FLAGS
		"-std=c++1z -D_USE_MATH_DEFINES=1 -Wall -Wextra -Wnon-virtual-dtor -Wconversion -Wcast-align -Wformat=2 -Wformat-security -Wmissing-declarations -Wstrict-overflow -Wtrampolines -Wreorder -Wsign-promo -pedantic -Wno-sign-conversion -save-temps=obj"
		CACHE STRING "Flags used by the CXX compiler during all build types."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g3 -fno-omit-frame-pointer -ftrapv"
		CACHE STRING "Flags used by the CXX compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -ffast-math -funroll-loops -flto -fno-fat-lto-objects -fomit-frame-pointer -fopt-info"
		CACHE STRING "Flags used by the CXX compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -g2 -flto -fno-lto-objects -fno-omit-frame-pointer"
		CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
		FORCE
	)
	target_compile_options (OpenMP  INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CXX>:-fopenmp>")
	target_link_options    (OpenMP  INTERFACE "SHELL: $<$<LINK_LANGUAGE:CXX>:-fopenmp>")
	target_compile_options (OpenACC INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CXX>:-fopenacc>")
	target_link_options    (OpenACC INTERFACE "SHELL: $<$<LINK_LANGUAGE:CXX>:-fopenacc>")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	set (
		CMAKE_CXX_FLAGS
		"-std=c++17 -D_USE_MATH_DEFINES=1 -Wall -Wextra -Wnon-virtual-dtor -Wconversion -Wformat=2 -Wformat-security -Wmissing-declarations -Woverloaded-virtual -Wreorder -Wsign-promo -pedantic"
		CACHE STRING "Flags used by the CXX compiler during all build types."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g3 -fno-omit-frame-pointer"
		CACHE STRING "Flags used by the CXX compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -unroll-aggressive -ipo -fno-fat-lto-objects -qopt-prefetch -qopt-report -fomit-frame-pointer"
		CACHE STRING "Flags used by the CXX compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -g -ipo -fno-fat-lto-objects -fno-omit-frame-pointer"
		CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
		FORCE
	)
	target_compile_options (OpenMP INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CXX>:-qopenmp>")
	target_link_options    (OpenMP INTERFACE "SHELL: $<$<LINK_LANGUAGE:CXX>:-qopenmp>")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang") 
	set (
		CMAKE_CXX_FLAGS
		"-std=c++17 -march=native -D_USE_MATH_DEFINES=1 -Wall -Wextra -Wnon-virtual-dtor -Wconversion -Wcast-align -Wformat=2 -Wformat-security -Wmissing-declarations -Wstrict-overflow -Woverloaded-virtual -Wreorder -Wsign-promo -pedantic"
		CACHE STRING "Flags used by the CXX compiler during all build types."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g3 -fno-omit-frame-pointer"
		CACHE STRING "Flags used by the CXX compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -ffast-math -fomit-frame-pointer" # -flto
		CACHE STRING "Flags used by the CXX compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		"-UDEBUG -UNDEBUG=1 -O3 -g2 -fno-omit-frame-pointer" # -flto
		CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
		FORCE
	)
	target_compile_options (OpenMP INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CXX>:-fopenmp -fopenmp-targets=nvptx64>")
	target_link_options    (OpenMP INTERFACE "SHELL: $<$<LINK_LANGUAGE:CXX>:-fopenmp -fopenmp-targets=nvptx64>")
elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
	set (
		CMAKE_CXX_FLAGS
		"--c++17 -Wall -pedantic"
		CACHE STRING "Flags used by the CXX compiler during all build types."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g"
		CACHE STRING "Flags used by the CXX compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -fast -Munroll -Minline -Mmovnt -Mconcur -Mipa=fast,inline -Mlist -Minfo=all"
		CACHE STRING "Flags used by the CXX compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CXX_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -gopt"
		CACHE STRING "Flags used by the CXX compiler during RelWithDebInfo builds."
		FORCE
	)
	target_compile_options (OpenMP  INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CXX>:-mp=nonuma>")
	target_link_options    (OpenMP  INTERFACE "SHELL: $<$<LINK_LANGUAGE:CXX>:-mp=nonuma>")
	target_compile_options (OpenACC INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CXX>:-acc -ta=tesla,cc60>")
	target_link_options    (OpenACC INTERFACE "SHELL: $<$<LINK_LANGUAGE:CXX>:-acc -ta=tesla,cc60>")
endif ()

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 :
