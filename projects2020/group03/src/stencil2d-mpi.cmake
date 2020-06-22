add_executable (
	stencil2d-mpi
		stencil2d-mpi.f
		m_utils.f
		m_partitioner.f
)
target_link_libraries (
	stencil2d-mpi
		MPI::MPI_Fortran
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
