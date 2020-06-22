add_executable (
	stencil2d_seq
		stencil2d.f
		m_utils.f
		m_partitioner.f
		diffusion_seq.f
		halo_mpi.f
)
target_compile_definitions (
	stencil2d_seq PRIVATE
		-D m_halo=m_halo_mpi
		-D m_diffusion=m_diffusion_seq
)
target_link_libraries (
	stencil2d_seq
		MPI::MPI_Fortran
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
