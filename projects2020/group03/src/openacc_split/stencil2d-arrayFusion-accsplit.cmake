add_executable (
	stencil2d-arrayFusion-accsplit
	stencil2d-arrayFusion-accsplit.cpp
)
target_link_libraries (
	stencil2d-arrayFusion-accsplit
		OpenACC
		OpenMP
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
