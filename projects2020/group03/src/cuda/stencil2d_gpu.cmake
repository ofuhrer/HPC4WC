add_executable (
	stencil2d_gpu.x
		src/update_halo.cpp
		src/apply_stencil.cu
		src/stencil2d_gpu.cu
)
target_include_directories (
	stencil2d_gpu.x
		include
)
target_link_libraries (
	stencil2d_gpu.x
		OpenMP
)
