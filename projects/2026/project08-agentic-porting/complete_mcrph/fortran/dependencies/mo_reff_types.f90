! ICON
!
! ---------------------------------------------------------------
! Copyright (C) 2004-2024, DWD, MPI-M, DKRZ, KIT, ETH, MeteoSwiss
! Contact information: icon-model.org
!
! See AUTHORS.TXT for a list of authors
! See LICENSES/ for license information
! SPDX-License-Identifier: BSD-3-Clause
! ---------------------------------------------------------------

! Module to with the types to compute effective radius
!
! Description:
! The module also contains adapted versions of the  routines developed
! by Simon Grueber and Uli Blahack for the optical properties in RRTM
! (only the effective radius, not the optical porperties),

MODULE mo_reff_types

  USE, INTRINSIC :: iso_fortran_env, ONLY: wp => real64

  IMPLICIT NONE
  PRIVATE

  PUBLIC::  t_reff_calc, t_reff_calc_dom, nreff_max_calc

  INTEGER,PARAMETER   ::  nreff_max_calc = 10   ! Maximum number of reff parameterizations
  
  ! Description for the effective radius calculation of different hydrometeors
  ! In case of subgrid and subgrid, 2 parameterizations are are neccesary
  ! We initially allow a maximum of nreff_max_calc       
  TYPE t_reff_calc     


    INTEGER           :: hydrometeor       ! Hydrometeor type
                                           ! 0          Cloud water
                                           ! 1          Ice Crystals
                                           ! 2          Rain
                                           ! 3          Snow
                                           ! 4          Graupel
                                           ! 5          Hail

    INTEGER           :: microph_param     ! Microphysics Parameterization
                                           ! This parameter sets the particle geometry and DSD.
                                           ! 1,3        Cloudice one-moment
                                           ! 2          Graupel  one-moment
                                           ! 4,5,6,7,9  Two-moment
                                           ! 100        Spherical particles (monodisperse if dsd_type not set)
                                           ! 101        RRTM

    INTEGER           :: reff_param        ! Parameterization type of reff
                                           ! 0          Spheroid  : reff = 0.5*c1*x**c2 (x= mean mass)
                                           ! 1          Fu Needles: reff= c5/(c1*x**c2 + c3*x**c4) 

    INTEGER           :: dsd_type          ! Assumed Droplet Size Distribution. It overrules microph_param when /= 0
                                           ! 0          Consistent with microphysics
                                           ! 1          Monodisperse
                                           ! 2          Generalized gamma with given gamma nu

    INTEGER           :: ncn_param         ! Parameterization for the number density
                                           ! 0          Constant cloud number (only for cloud water, it uses cloud_num)
                                           ! 1,2,3      One-mom microphysics  n = c1*(rho q)**[c2] or function.
                                           ! 4,5,6,7,9  From two-mom microphysics   n = qn
                                           ! 101        Uses acdnc (only for water, currently used by RRTM)
                                           ! 102        Uses surface values from cloud_num (only for water)
                                           ! -1         No calculation neccesary (like RRTM for ice) 

    INTEGER           :: grid_scope        ! Calculations over
                                           ! 0          Grid+subgrid
                                           ! 1          Grid only
                                           ! 2          Subgrid only 

    INTEGER           :: ncn_param_incloud ! ncn corresponds to grid values or incloud 
                                           ! 0          Ncn in grid box
                                           ! 1          Ncn ncdn in cloud (larger than grid box)


    REAL(wp), ALLOCATABLE :: reff_coeff(:)     ! Coeeficients of the reff parameterization
    REAL(wp), ALLOCATABLE :: ncn_coeff(:)      ! Coeeficients of the n parameterization 

    REAL(wp)              :: x_min             ! Min hydro. mass admited by the parameterization
    REAL(wp)              :: x_max             ! Max hydro. mass admited by the parameterization
    REAL(wp)              :: r_min             ! Min hydro. radius admited by the parameterization
    REAL(wp)              :: r_max             ! Max hydro. radius admited by the parameterization
    
    REAL(wp)              :: mu                ! Given Gamma parameter in DSD (only for dsd_type=2)
    REAL(wp)              :: nu                ! Given Nu parameter in DSD    (only for dsd_type=2)



    REAL(wp), POINTER, DIMENSION(:,:,:) :: p_q           ! Hydrometeor mixing ratio 
    REAL(wp), POINTER, DIMENSION(:,:,:) :: p_reff        ! Effective radius output
    REAL(wp), POINTER, DIMENSION(:,:,:) :: p_qtot        ! Total hydro. mix. ratio (differentiate total and sub)
    REAL(wp), POINTER, DIMENSION(:,:,:) :: p_ncn3D       ! Hydrometeor number concentration (3D)
    REAL(wp), POINTER, DIMENSION(:,:)   :: p_ncn2D       ! Hdrometoer number concentration at surface (2D)

  CONTAINS
    PROCEDURE :: construct => t_reff_calc_construct
    PROCEDURE :: destruct  => t_reff_calc_destruct
        
  END TYPE t_reff_calc

  ! Structure for calculation in different domains
  TYPE     t_reff_calc_dom
    TYPE(t_reff_calc)          ::  reff_calc_arr(nreff_max_calc) ! Parameterizations for reff calculation in a domain   
    INTEGER                    ::  nreff_calc                    ! Number of effective radius calculations in a domain
  CONTAINS
    PROCEDURE :: destruct  => t_reff_calc_dom_destruct
  END TYPE t_reff_calc_dom

  
CONTAINS

  ! Constructor of t_reff_calc. It only allocates memory and nullifies the pointers of the derived type.
  SUBROUTINE t_reff_calc_construct (me)
    CLASS(t_reff_calc), INTENT(INOUT)     :: me

    !$ACC ENTER DATA CREATE(me)
    ! Allocate memory for parameterizations
    IF( .NOT. ALLOCATED(me%reff_coeff) ) ALLOCATE( me%reff_coeff (4))   ! Coeeficients of the reff parameterization
    IF( .NOT. ALLOCATED(me%ncn_coeff)  ) ALLOCATE( me%ncn_coeff  (3))   ! Coeeficients of the ncn parameterization

    !$ACC ENTER DATA CREATE(me%reff_coeff)

    ! Nullify pointers
    NULLIFY(me%p_q)
    NULLIFY(me%p_reff)
    NULLIFY(me%p_qtot)
    NULLIFY(me%p_ncn3D)


    NULLIFY(me%p_ncn2D)    

  END SUBROUTINE  t_reff_calc_construct
  
  ! The destructor deallocates the memory and nullifies the pointers
  SUBROUTINE t_reff_calc_destruct (me)
    CLASS(t_reff_calc), INTENT(INOUT)     :: me

    !$ACC WAIT(1)
    !$ACC EXIT DATA DELETE(me%reff_coeff)
    IF (ALLOCATED(  me%reff_coeff ))  DEALLOCATE ( me%reff_coeff )         
    IF (ALLOCATED(  me%ncn_coeff  ))  DEALLOCATE ( me%ncn_coeff )
     
 ! Nullify pointers
    NULLIFY(me%p_q)
    NULLIFY(me%p_reff)
    NULLIFY(me%p_qtot)
    NULLIFY(me%p_ncn3D)
    NULLIFY(me%p_ncn2D)    

    !$ACC EXIT DATA DELETE(me)


  END SUBROUTINE  t_reff_calc_destruct
  
  SUBROUTINE t_reff_calc_dom_destruct (me)
    CLASS(t_reff_calc_dom), INTENT(INOUT) :: me
    
    INTEGER  i

    ! Destruct individual calculations
    DO i = 1,nreff_max_calc
      CALL me%reff_calc_arr(i)%destruct()
    END DO

    !$ACC WAIT(1)
    !$ACC EXIT DATA DELETE(me)

  END SUBROUTINE t_reff_calc_dom_destruct
   
END MODULE mo_reff_types
