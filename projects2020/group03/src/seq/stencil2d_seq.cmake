add_executable (
	stencil2d_seq
		stencil2d.f
		diffusion_seq.f
)
target_compile_definitions (
	stencil2d_seq PRIVATE
		-D m_diffusion=m_diffusion_seq
)
target_link_libraries (
	stencil2d_seq
		utils
		partitioner
		halo_mpi
		MPI::MPI_Fortran
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
