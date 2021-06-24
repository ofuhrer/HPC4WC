add_executable (
	stencil2d-orig
		stencil2d-orig.f
)
target_link_libraries (
	stencil2d-orig
		utils
		MPI::MPI_Fortran
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
