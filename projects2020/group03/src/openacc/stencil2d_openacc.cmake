add_executable (
	stencil2d_openacc
		stencil2d.f
		diffusion_openacc.f
)
target_compile_definitions (
	stencil2d_openacc PRIVATE
		-D m_diffusion=m_diffusion_openacc
)
target_link_libraries (
	stencil2d_openacc
		utils
		partitioner
		halo_mpi
		OpenACC
		MPI::MPI_Fortran
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
