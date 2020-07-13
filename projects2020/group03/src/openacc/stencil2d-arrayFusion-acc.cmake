add_executable (
	stencil2d-arrayFusion-acc
		stencil2d-arrayFusion-acc.cpp
)
target_link_libraries (
	stencil2d-arrayFusion-acc
		OpenACC::CXX
		OpenMP::CXX
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
