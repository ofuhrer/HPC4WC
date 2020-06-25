add_executable (
	stencil2d_openmp
		stencil2d.f
		diffusion_openmp.f
)
target_compile_definitions (
	stencil2d_openmp PRIVATE
		-D m_diffusion=m_diffusion_openmp
)
target_link_libraries (
	stencil2d_openmp
		utils
		partitioner
		halo_mpi
		OpenMP::OpenMP_Fortran
		MPI::MPI_Fortran
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
