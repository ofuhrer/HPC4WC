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
		halo_openmp
		OpenMP
		MPI::MPI_Fortran
)

add_executable (
	stencil2d_openmp2
		stencil2d.f
		diffusion_openmp2.f
)
target_compile_definitions (
	stencil2d_openmp2 PRIVATE
		-D m_diffusion=m_diffusion_openmp2
)
target_link_libraries (
	stencil2d_openmp2
		utils
		partitioner
		OpenMP
		MPI::MPI_Fortran
)

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
