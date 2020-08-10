add_executable (
	stencil2d_hybrid
		update_halo.cpp
		apply_stencil_cpu.cpp
		apply_stencil.cu
		stencil2d_hybrid.cu
)
target_link_libraries (
	stencil2d_hybrid
		OpenMP
)
