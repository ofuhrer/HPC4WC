add_executable (
	stencil2d
		stencil2d.f
		m_utils.f
		m_partitioner.f

		diffusion_seq.f
		diffusion_openacc.f
		diffusion_openmp.f
		diffusion_openmp_target.f

		halo_seq.f
		halo_openmp.f
		halo_mpi.f
)

target_link_libraries (
	stencil2d
		OpenMP::OpenMP_Fortran
		MPI::MPI_Fortran
)

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

add_executable (
	stencil2d-orig
		stencil2d-orig.f
		m_utils.f
)
target_link_libraries (
	stencil2d-orig
		MPI::MPI_Fortran
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
