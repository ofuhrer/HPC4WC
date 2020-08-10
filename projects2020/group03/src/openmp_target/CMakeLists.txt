if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
	message (WARNING "OpenMP offloading not supported by Intel compiler")
elseif ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Cray" AND "${CMAKE_BUILD_TYPE}" STREQUAL "Debug")
	message (WARNING "OpenMP offloading not supported on Cray debug builds")
else ()
	add_executable (
		stencil2d_openmp_target
			stencil2d.f
			diffusion_openmp_target.f
	)
	target_compile_definitions (
		stencil2d_openmp_target PRIVATE
			-D m_diffusion=m_diffusion_openmp_target
	)
	target_link_libraries (
		stencil2d_openmp_target
			utils
			partitioner
			halo_openmp_mpi
			OpenMP
			MPI::MPI_Fortran
	)
endif ()

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
