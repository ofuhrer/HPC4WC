add_executable (
	stencil2d_seq_base_array_cpp
	stencil2d-base-array.cpp
)
target_link_libraries (
	stencil2d_seq_base_array_cpp
		OpenMP
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
