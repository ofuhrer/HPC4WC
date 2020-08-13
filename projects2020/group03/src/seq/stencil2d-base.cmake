add_executable (
	stencil2d_seq_base_cpp
	stencil2d-base.cpp
)
target_link_libraries (
	stencil2d_seq_base_cpp
		OpenMP
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
