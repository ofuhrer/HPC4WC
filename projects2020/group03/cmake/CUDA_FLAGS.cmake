set (
        CMAKE_CUDA_FLAGS
		""
        CACHE STRING "Flags used by the CUDA compiler during all build types."
        FORCE
)
set (
        CMAKE_CUDA_FLAGS_DEBUG
		""
        CACHE STRING "Flags used by the CUDA compiler during Debug builds."
        FORCE
)
set (
        CMAKE_CUDA_FLAGS_RELEASE
		""
        CACHE STRING "Flags used by the CUDA compiler during Release builds."
        FORCE
)
set (
        CMAKE_CUDA_FLAGS_RELWITHDEBINFO
		""
        CACHE STRING "Flags used by the CUDA compiler during RelWithDebInfo builds."
        FORCE
)

if ("${CMAKE_CUDA_COMPILER_ID}" STREQUAL "NVIDIA")
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		set (
			CMAKE_CUDA_FLAGS
			"-std c++14 -D_USE_MATH_DEFINES=1 -Wreorder"
			CACHE STRING "Flags used by the CUDA compiler during all build types."
			FORCE
		)
	else ()
		set (
			CMAKE_CUDA_FLAGS
			"-std c++14 -D_USE_MATH_DEFINES=1 -Wreorder --compiler-bindir ${CMAKE_CXX_COMPILER}"
			CACHE STRING "Flags used by the CUDA compiler during all build types."
			FORCE
		)
	endif ()
	set (
		CMAKE_CUDA_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g -G -src-in-ptx"
		CACHE STRING "Flags used by the CUDA compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_CUDA_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -use_fast_math"
		CACHE STRING "Flags used by the CUDA compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_CUDA_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -lineinfo -g"
		CACHE STRING "Flags used by the CUDA compiler during RelWithDebInfo builds."
		FORCE
	)
	if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "GNU")
		target_compile_options (OpenMP  INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>")
		target_link_options    (OpenMP  INTERFACE "SHELL: $<$<LINK_LANGUAGE:CUDA>:-Xlinker -fopenmp>")
		target_compile_options (OpenACC INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenacc>")
		target_link_options    (OpenACC INTERFACE "SHELL: $<$<LINK_LANGUAGE:CUDA>:-Xlinker -fopenmp>")
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
		target_compile_options (OpenMP  INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -qopenmp>")
		target_link_options    (OpenMP  INTERFACE "SHELL: $<$<LINK_LANGUAGE:CUDA>:-Xlinker -qopenmp>")
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Cray")
		target_compile_options (OpenMP  INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -homp>")
		target_link_options    (OpenMP  INTERFACE "SHELL: $<$<LINK_LANGUAGE:CUDA>:-Xlinker -homp>")
		target_compile_options (OpenACC INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -hacc>")
		target_link_options    (OpenACC INTERFACE "SHELL: $<$<LINK_LANGUAGE:CUDA>:-Xlinker -hacc>")
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Clang")
		# actually calls system gcc
		target_compile_options (OpenMP  INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -fopenmp>")
		target_link_options    (OpenMP  INTERFACE "SHELL: $<$<LINK_LANGUAGE:CUDA>:-Xlinker -fopenmp>")
	elseif ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "PGI")
		target_compile_options (OpenMP  INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -mp=nonuma>")
		target_link_options    (OpenMP  INTERFACE "SHELL: $<$<LINK_LANGUAGE:CUDA>:-Xlinker -mp=nonuma>")
		target_compile_options (OpenACC INTERFACE "SHELL: $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler -acc -Xcompiler -ta=tesla,cc60>")
		target_link_options    (OpenACC INTERFACE "SHELL: $<$<LINK_LANGUAGE:CUDA>:-Xlinker -acc -Xlinker -ta=tesla,cc60>")
	endif ()
endif ()

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 :
