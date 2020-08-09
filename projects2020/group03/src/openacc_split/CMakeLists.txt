if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
	message (WARNING "OpenACC not supported by Intel compiler")
else ()
	add_executable (
		stencil2d_openacc_split
			stencil2d.f
			diffusion_openacc_split.f
	)
	target_compile_definitions (
		stencil2d_openacc_split PRIVATE
			-D m_diffusion=m_diffusion_openacc_split
	)
	target_link_libraries (
		stencil2d_openacc_split
			utils
			partitioner
			halo_mpi
			OpenACC::Fortran
			OpenMP::OpenMP_Fortran
			MPI::MPI_Fortran
	)
endif ()

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
