add_executable (
	stencil2d-base-array-acc
		stencil2d-base-array-acc.cpp
)
target_link_libraries (
	stencil2d-base-array-acc
		OpenACC::CXX
		OpenMP::CXX
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
