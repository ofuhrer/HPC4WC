add_executable (
	stencil2d_cpu
		update_halo.cpp
		apply_stencil_cpu.cpp
		stencil2d_cpu.cpp
)
target_link_libraries (
	stencil2d_cpu
		OpenMP
)
