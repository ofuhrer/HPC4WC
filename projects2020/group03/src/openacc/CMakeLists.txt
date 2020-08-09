if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
	message (WARNING "OpenACC not supported by Intel compiler")
else ()
	include (stencil2d_openacc.cmake)
endif ()

if ("${CMAKE_CXX_COMPILER_ID}" STREQUAL "Intel")
	message (WARNING "OpenACC not supported by Intel compiler")
else ()
	include (stencil2d-base-array-acc.cmake)
	include (stencil2d-arrayFusion-acc.cmake)
endif ()

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 shiftwidth=2 :
