add_executable (
	stencil2d_hybrid.x
		src/update_halo.cpp
		src/apply_stencil_cpu.cpp
		src/apply_stencil.cu
		src/stencil2d_hybrid.cu
)
target_include_directories (
	stencil2d_hybrid.x PRIVATE
		include
)
target_link_libraries (
	stencil2d_hybrid.x
		OpenMP
)
