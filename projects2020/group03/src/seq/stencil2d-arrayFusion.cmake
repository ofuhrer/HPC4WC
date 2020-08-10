add_executable (
	stencil2d_seq_arrayFusion_cpp
	stencil2d-arrayFusion.cpp
)
target_link_libraries (
	stencil2d_seq_arrayFusion_cpp
		OpenMP::CXX
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
