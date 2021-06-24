add_executable (
	stencil2d-mpi
		stencil2d-mpi.f
)
target_link_libraries (
	stencil2d-mpi
		utils
		partitioner
		MPI::MPI_Fortran
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
