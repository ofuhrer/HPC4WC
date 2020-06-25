# list (REMOVE_ITEM CMAKE_CONFIGURATION_TYPES MinSizeRel)
# list (APPEND CMAKE_CONFIGURATION_TYPES Fast)

set (
	CMAKE_Fortran_FLAGS ""
	CACHE STRING "Flags used by the Fortran compiler during all build types."
	FORCE
)
set (
	CMAKE_Fortran_FLAGS_DEBUG ""
	CACHE STRING "Flags used by the Fortran compiler during Debug builds."
	FORCE
)
set (
	CMAKE_Fortran_FLAGS_MINSIZEREL ""
	CACHE STRING "Flags used by the Fortran compiler during MinSizeRel builds."
	FORCE
)
set (
	CMAKE_Fortran_FLAGS_RELEASE ""
	CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
	FORCE
)
# set (
# 	CMAKE_EXE_LINKER_FLAGS ""
# 	CACHE STRING "Flags used by the linker during all build types."
# 	FORCE
# )

add_library (OpenMP::Fortran  INTERFACE IMPORTED)
add_library (OpenACC::Fortran INTERFACE IMPORTED)

if ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "GNU")
	set (
		CMAKE_Fortran_FLAGS
		"-std=f2018 -ffree-form -cpp -Wall -Wextra -Wpedantic -Wsurprising -Wno-maybe-uninitialized"
		CACHE STRING "Flags used by the Fortran compiler during all build types."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g3 -fno-omit-frame-pointer -fcheck=all -ffpe-trap=invalid,zero,overflow,underflow,denormal"
		CACHE STRING "Flags used by the Fortran compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -flto -fno-fat-lto-objects -fomit-frame-pointer -fopt-info"
		CACHE STRING "Flags used by the Fortran compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -g2 -flto -fno-fat-lto-objects -fno-omit-frame-pointer"
		CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
		FORCE
	)
	set_property (TARGET OpenMP::Fortran  PROPERTY INTERFACE_COMPILE_OPTIONS "-fopenmp")
	set_property (TARGET OpenMP::Fortran  PROPERTY INTERFACE_LINK_LIBRARIES  "-fopenmp")
	set_property (TARGET OpenACC::Fortran PROPERTY INTERFACE_COMPILE_OPTIONS "-fopenacc")
	set_property (TARGET OpenACC::Fortran PROPERTY INTERFACE_LINK_LIBRARIES  "-fopenacc")
elseif ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Intel")
	set (
		CMAKE_Fortran_FLAGS
		"-stand f18 -free -warn all -fpp"
		CACHE STRING "Flags used by the Fortran compiler during all build types."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g -check all -ftrapuv -fp-speculation safe"
		CACHE STRING "Flags used by the Fortran compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3" # TODO: vec-report?
		CACHE STRING "Flags used by the Fortran compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -g"
		CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
		FORCE
	)
	set_property (TARGET OpenMP::Fortran PROPERTY INTERFACE_COMPILE_OPTIONS "-qopenmp")
	set_property (TARGET OpenMP::Fortran PROPERTY INTERFACE_LINK_LIBRARIES  "-qopenmp")
elseif ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "Cray") 
	set (
		CMAKE_Fortran_FLAGS
		"-f free -J. -ec -eC -em -ef -en -eT -m1 -M7405,7418"
		CACHE STRING "Flags used by the Fortran compiler during all build types."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -eD -h develop -R bcdps -K trap=denorm,divz,fp,inexact,inv,ovf,unf"
		CACHE STRING "Flags used by the Fortran compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -ra"
		CACHE STRING "Flags used by the Fortran compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1" # TODO
		CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
		FORCE
	)
	set_property (TARGET OpenMP::Fortran  PROPERTY INTERFACE_COMPILE_OPTIONS "-h omp")
	set_property (TARGET OpenMP::Fortran  PROPERTY INTERFACE_LINK_LIBRARIES  "-h omp")
	set_property (TARGET OpenACC::Fortran PROPERTY INTERFACE_COMPILE_OPTIONS "-h acc")
	set_property (TARGET OpenACC::Fortran PROPERTY INTERFACE_LINK_LIBRARIES  "-h acc")
elseif ("${CMAKE_Fortran_COMPILER_ID}" STREQUAL "PGI")
	set (
		CMAKE_Fortran_FLAGS
		"-Mfree -Mpreprocess -Mstandard"
		CACHE STRING "Flags used by the Fortran compiler during all build types."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_DEBUG
		"-DDEBUG=1 -UNDEBUG -O0 -g -Mbounds -Mchkptr -Ktrap=divz,fp,inexact,inv,ovf,unf"
		CACHE STRING "Flags used by the Fortran compiler during Debug builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELEASE
		"-UDEBUG -DNDEBUG=1 -O3 -fast -Mlist"
		CACHE STRING "Flags used by the Fortran compiler during Release builds."
		FORCE
	)
	set (
		CMAKE_Fortran_FLAGS_RELWITHDEBINFO
		"-UDEBUG -DNDEBUG=1 -O2 -gopt" # TODO
		CACHE STRING "Flags used by the Fortran compiler during RelWithDebInfo builds."
		FORCE
	)
	set_property (TARGET OpenMP::Fortran  PROPERTY INTERFACE_COMPILE_OPTIONS "-mp")
	set_property (TARGET OpenMP::Fortran  PROPERTY INTERFACE_LINK_LIBRARIES  "-mp")
	set_property (TARGET OpenACC::Fortran PROPERTY INTERFACE_COMPILE_OPTIONS -acc -ta=tesla)
	set_property (TARGET OpenACC::Fortran PROPERTY INTERFACE_LINK_LIBRARIES  -acc -ta=tesla)
endif ()

# vim : filetype=cmake noexpandtab tabstop=2 softtabs=2 :
