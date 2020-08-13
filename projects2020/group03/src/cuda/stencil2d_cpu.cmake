add_executable (
	stencil2d_cpu.x
		src/update_halo.cpp
		src/apply_stencil_cpu.cpp
		src/stencil2d_cpu.cpp
)
target_include_directories (
	stencil2d_cpu.x
		include
)
target_link_libraries (
	stencil2d_cpu.x
		OpenMP
)
