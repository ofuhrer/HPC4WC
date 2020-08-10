add_executable (
	stencil2d_gpu
		update_halo.cpp
		apply_stencil.cu
		stencil2d_gpu.cu
)
target_link_libraries (
	stencil2d_gpu
		OpenMP
)
