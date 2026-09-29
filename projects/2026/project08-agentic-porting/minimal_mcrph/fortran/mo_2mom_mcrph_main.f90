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

! Two-moment bulk microphysics after Seifert, Beheng and Blahak
!
! Description:
! This module contains the main subroutine for the two-moment microphysics, and
! the initialization subroutines that calculated the run-time coefficients

!NEC$ options "-finline-max-depth=3 -finline-max-function-size=10000"

MODULE mo_2mom_mcrph_main

!===============================================================================!
! Re-write for ICON 04/2014 by AS:
! Some general notes:
! - This version may need an up-to-date compiler due to some Fortran2003 features
! - Adapted physical constants to ICON
! - Atlas-type fall speed of rain has been changed to SBB2014, GMD
! Some notes on optimization (tests on thunder Thunder):
! - Small penalty for the particle%meanmass etc. functions, but compensated
!   by the exp(b*log(x)) instead of the original power law.
! - Increased q_crit from 1e-9 to 1e-7 for efficiency. Looks ok for WK-test,
!   but has to be tested in a real-case setup with stratiform and cirrus clouds.
! - Replaced some more power laws by exp(a*log(x)), e.g.,
!   in graupel_hail_conv_wet_gamlook()
! - Replaced almost all ()**0.5 by sqrt(), including the **m_f in ventilation
!   coefficients
! - Clipping in rain_freeze is necessary, but removed everywhere else
!===============================================================================!
! Version of May 2015 by AS:
! - New IN and CCN routines implemented based on Hande et al. (HDCP2-M3)
! - gscp=4 has now prognostic QNC and IN depletion (n_inact)
! - gscp=5 has additional budget equations for IN and CCN
!===============================================================================!
! Version of Nov 2015 - Jan 2016 by AS:
! - Including new melting of graupel and hail with explicit melt water
!   based on the work of Vivek Sant
! - gscp=7 has new prognostic QHL and QGL, i.e., prognostic melt water
! - Extended particle types to include meltwater in particle structures
! - Restructuring for cleaner argument lists including a new module named
!   mo_2mom_mcrph_processes and setup subroutines for most coefficients
!===============================================================================!
! To Do:
! - Check conservation of water mass
! - Further optimization might be possible in rain_freeze.
!===============================================================================!
! Further plans (physics) including HDCP2 project:
! - Implement new collision rate parameterizations of SBB2014
! - Implement improved height dependency of terminal fall velocity similar as
!   used in COSMO two-moment code (but may be quite expensive).
! - Write a version with three or four different ice particle species
!   (hom, het, frz, and splinters from ice multiplication)
!===============================================================================!
! Small stuff:
! - Increase alpha_spacefilling?
! - Are the minor differences in the sticking efficiencies important?
!===============================================================================!
! Further plans (restructuring and numerics):
! - Better understand performance issues of semi-implicit solver
! - Introduce logicals llqi_crit=(qi>q_crit), and llqi_zero = (qi>0.0), etc.
!   which are calculated once in the driver
!===============================================================================!

  USE mo_kind,               ONLY: sp, wp
  USE mo_exception,          ONLY: finish, message, txt => message_text
  USE mo_2mom_mcrph_types, ONLY: &
       & particle, particle_frozen, particle_lwf, atmosphere, &
       & particle_sphere, particle_rain_coeffs, particle_cloud_coeffs, &
       & particle_ice_coeffs, particle_snow_coeffs, particle_graupel_coeffs, &
       & aerosol_ccn, aerosol_in, &
       & particle_coeffs, collection_coeffs, rain_riming_coeffs, dep_imm_coeffs, &
       & coll_coeffs_ir_pm, &
       & ltabdminwgg, ltabdminwgh, ltab_estick_ice, ltab_estick_snow, ltab_estick_parti
  ! MINIMAL SCHEME: the incomplete-gamma lookup tables and wet-growth dmin table
  ! (mo_2mom_mcrph_util) are not used, so nothing is imported from that module.
  ! MINIMAL SCHEME: only the retained call tree is imported from processes.
  USE mo_2mom_mcrph_processes, ONLY:                                         &
       &  particle_assign, particle_frozen_assign, particle_lwf_assign,      &
       &  init_2mom_sedi_vel, setup_particle_coeffs,                         &
       &  cloud_freeze, ice_nucleation_homhet, vapor_dep_relaxation,         &
       &  ice_melting, ccn_activation_hdcp2, ccn_activation_sk_4d,           &
       &  set_default_n, cfg_params

  ! Some switches...
  USE mo_2mom_mcrph_processes, ONLY:                                         &
       &  ice_typ, nuc_i_typ, nuc_c_typ, auto_typ, isdebug, isprint,         &
       &  particle_meanmass

  USE mo_timer, ONLY: timers_level, timer_start, timer_stop, timer_phys_2mom_wetgrowth
  USE mo_satad, ONLY: satad_v_3D, satad_v_3D_gpu
  USE mo_stage_dump, ONLY: dump_stage, dump_coeffs, stage_dump_enabled

  IMPLICIT NONE

  PRIVATE

  CHARACTER(len=*), PARAMETER :: routine = 'mo_2mom_mcrph_main'

  ! In place of ICON's mo_math_constants: pi and 4*pi are the only two
  ! constants this scheme needs from it.
  REAL(wp), PARAMETER :: pi  = 3.14159265358979323846_wp
  REAL(wp), PARAMETER :: pi4 = 4.0_wp * pi

  ! for constant droplet number runs
  ! (will be set during init, but currently not used by any implementation)
  real(wp) :: qnc_const = 200.0e6_wp

  ! Derived types that contain run-time coefficients for each particle type
  TYPE(particle_ice_coeffs)      :: ice_coeffs
  TYPE(particle_snow_coeffs)     :: snow_coeffs
  TYPE(particle_graupel_coeffs)  :: graupel_coeffs
  TYPE(particle_sphere)          :: hail_coeffs
  TYPE(particle_cloud_coeffs)    :: cloud_coeffs
  TYPE(particle_rain_coeffs)     :: rain_coeffs
  TYPE(aerosol_ccn)              :: ccn_coeffs
  TYPE(aerosol_in)               :: in_coeffs

  ! MINIMAL SCHEME: the incomplete-gamma lookup tables for the dropped
  ! rain_freeze_gamlook / graupel_hail_conv_wet_gamlook / shedding routines have
  ! been removed.

  ! choice of mu-D relation for rain, default is mu_Dm_rain_typ = 1
  INTEGER, PARAMETER     :: mu_Dm_rain_typ = 1     ! see init_twomoment() for possible choices

  ! Parameter for evaporation of rain, determines change of n_rain during evaporation
  REAL(wp) :: rain_gfak   ! this is set in init_twomoment depending on mu_Dm_rain_typ

  ! debug switches
  LOGICAL, PARAMETER     :: ischeck = .false.    ! frequently check for positive definite q's

  ! some cloud microphysical switches
  LOGICAL, PARAMETER     :: ice_multiplication = .TRUE.  ! default is .true.
  LOGICAL, PARAMETER     :: enhanced_melting   = .TRUE.  ! default is .true.
  LOGICAL, PARAMETER     :: classic_melting_in_lwf_scheme = .False.

  !..Pre-defined particle types (used in init_2mom_scheme)
  TYPE(particle_frozen), PARAMETER :: & ! WE KEEP THIS FOR REFERENCE TO ORIGINAL COSMO SETTING
       &        graupelhail_cosmo5_orig = particle_frozen( & ! graupelhail2test4
       &        'graupelhail_cosmo5_o' ,& !.name
       &        1.000000, & !..nu.........1st shape parameter of the distribution
       &        0.333333, & !..mu.........2nd shape parameter of the distribution
       &        5.30d-04, & !..x_max......maximum particle mean mass
       &        4.19d-09, & !..x_min......minimum particle mean mass
       &        1.42d-01, & !..a_geo......particle geometry prefactor
       &        0.314000, & !..b_geo......particle geometry exponent = 1/3.10
       &        86.89371, & !..a_vel......terminal fall velocity prefactor
       &        0.268325, & !..b_vel......terminal fall velocity exponent
       &        0.780000, & !..a_ven......1st ventilation coefficient (PK, S.541)
       &        0.308000, & !..b_ven......2nd ventilation coefficient (PK, S.541)
       &        2.00,     & !..cap........capacity coefficient
       &        30.0,     & !..vsedi_max..maximum bulk sedimentation velocity
       &        0.10,     & !..vsedi_min..minimum bulk sedimentation velocity
       &        null(),   & !..n pointer..pointer to number density array
       &        null(),   & !..q pointer..pointer to mass density array
       &        null(),   & !..rho_v......pointer to density correction array
       &        1.0,      & !..ecoll_c....maximum collision efficiency with cloud droplets
       &        100.0d-6, & !..D_crit_c...D-threshold for cloud riming
       &        1.000d-6, & !..q_crit_c...q-threshold for cloud riming
       &        0.0       & !..sigma_vel..dispersion of fall velocity for collection kernel
       &        )

  TYPE(particle_frozen), PARAMETER :: &
       &        graupelhail_cosmo5 = particle_frozen( & ! graupelhail2test5
       &        'graupelhail_cosmo5' ,& !.name
       &        1.000000, & !..nu.........1st shape parameter of the distribution
       &        0.333333, & !..mu.........2nd shape parameter of the distribution
       &        5.30d-04, & !..x_max......maximum particle mean mass
       &        4.19d-09, & !..x_min......minimum particle mean mass
       &        1.42d-01, & !..a_geo......particle geometry prefactor
       &        0.314000, & !..b_geo......particle geometry exponent = 1/3.10
       &        100.0,    & !..a_vel......terminal fall velocity prefactor
       &        0.34,     & !..b_vel......terminal fall velocity exponent
       &        0.780000, & !..a_ven......1st ventilation coefficient (PK, S.541)
       &        0.308000, & !..b_ven......2nd ventilation coefficient (PK, S.541)
       &        2.00,     & !..cap........capacity coefficient
       &        80.0,     & !..vsedi_max..maximum bulk sedimentation velocity
       &        0.10,     & !..vsedi_min..minimum bulk sedimentation velocity
       &        null(),   & !..n pointer..pointer to number density array
       &        null(),   & !..q pointer..pointer to mass density array
       &        null(),   & !..rho_v......pointer to density correction array
       &        1.0,      & !..ecoll_c....maximum collision efficiency with cloud droplets
       &        100.0d-6, & !..D_crit_c...D-threshold for cloud riming
       &        1.000d-6, & !..q_crit_c...q-threshold for cloud riming
       &        0.0       & !..sigma_vel..dispersion of fall velocity for collection kernel
       &        )

  TYPE(particle_lwf), PARAMETER :: graupel_vivek = particle_lwf( & ! graupelhail2test5
       &        'graupel_vivek' ,& !.name...Bezeichnung
       &        1.000000, & !..nu.....Breiteparameter der Verteil.
       &        0.333333, & !..mu.....Exp.-parameter der Verteil.
       &        5.30d-04, & !..x_max..maximale Teilchenmasse
       &        4.19d-09, & !..x_min..minimale Teilchenmasse
       &        1.42d-01, & !..a_geo..Koeff. Geometrie
       &        0.314000, & !..b_geo..Koeff. Geometrie = 1/3.10
       &        100.0,    & !..a_vel..Koeff. Fallgesetz
       &        0.34,     & !..b_vel..Koeff. Fallgesetz
       &        0.780000, & !..a_ven..Koeff. Ventilation (PK, S.541)
       &        0.308000, & !..b_ven..Koeff. Ventilation (PK, S.541)
       &        2.00,     & !..cap....Koeff. Kapazitaet
       &        80.0,     & !..vsedi_max
       &        0.10,     & !..vsedi_min
       &        null(),   & !..n pointer
       &        null(),   & !..q pointer
       &        null(),   & !..rho_v pointer
       &        1.0,      & !..ecoll_c
       &        100.0d-6, & !..D_crit_c
       &        1.000d-6, & !..q_crit_c
       &        0.0,      & !..sigma_vel
       &        0.5,      & !..cnorm1 (normalized diameter)
       &        4.0,      & !..cnorm2 (normalized diameter)
       &        0.5,      & !..cnorm3 (normalized diameter)
       &        7.246261, & !..cmelt1 (melting intgral)   !!NEEDS TO BE ADJUSTED
       &        0.666666, & !..cmelt2 (melting intgral)   !!NEEDS TO BE ADJUSTED
       &        null()    ) !..ql pointer


  TYPE(particle_frozen), PARAMETER :: &
       &        hail_cosmo5 = particle_frozen( & ! hailULItest
       &        'hail_cosmo5' ,& !.name...Bezeichnung
       &        1.000000, & !..nu.....Breiteparameter der Verteil.
       &        0.333333, & !..mu.....Exp.-parameter der Verteil.
       &        5.00d-03, & !..x_max..maximale Teilchenmasse
       &        2.60d-9,  & !..x_min..minimale Teilchenmasse
       &        0.1366 ,  & !..a_geo..Koeff. Geometrie
       &        0.333333, & !..b_geo..Koeff. Geometrie = 1/3
       &        39.3    , & !..a_vel..Koeff. Fallgesetz
       &        0.166667, & !..b_vel..Koeff. Fallgesetz
       &        0.780000, & !..a_ven..Koeff. Ventilation (PK, S.541)
       &        0.308000, & !..b_ven..Koeff. Ventilation (PK, S.541)
       &        2.00,     & !..cap....Koeff. Kapazitaet
       &        30.0,     & !..vsedi_max
       &        0.1,      & !..vsedi_min
       &        null(),   & !..n pointer
       &        null(),   & !..q pointer
       &        null(),   & !..rho_v pointer
       &        1.0,      & !..ecoll_c
       &        100.0d-6, & !..D_crit_c
       &        1.000d-6, & !..q_crit_c
       &        0.0       & !..sigma_vel
       &        )

  TYPE(particle_lwf), PARAMETER :: hail_vivek = particle_lwf( & ! hailULItest
       &        'hail_vivek' ,& !.name...Bezeichnung
       &        1.000000, & !..nu.....Breiteparameter der Verteil.
       &        0.333333, & !..mu.....Exp.-parameter der Verteil.
       &        5.40d-04, & !..x_max..maximale Teilchenmasse
       &        2.60d-9,  & !..x_min..minimale Teilchenmasse (Vivek has 4.19d-09)
       &        1.28d-01 ,& !..a_geo..Koeff. Geometrie
       &        0.333333, & !..b_geo..Koeff. Geometrie = 1/3
       &        39.3,     & !..a_vel..Koeff. Fallgesetz
       &        0.166667, & !..b_vel..Koeff. Fallgesetz
       &        0.780000, & !..a_ven..Koeff. Ventilation (PK, S.541)
       &        0.308000, & !..b_ven..Koeff. Ventilation (PK, S.541)
       &        2.00,     & !..cap....Koeff. Kapazitaet
       &        30.0,     & !..vsedi_max
       &        0.1,      & !..vsedi_min
       &        null(),   & !..n pointer
       &        null(),   & !..q pointer
       &        null(),   & !..rho_v pointer
       &        1.0,      & !..ecoll_c
       &        100.0d-6, & !..D_crit_c
       &        1.000d-6, & !..q_crit_c
       &        0.0,      & !..sigma_vel
       &        0.5,      & !..cnorm1 (normalized diameter)
       &        4.0,      & !..cnorm2 (normalized diameter)
       &        0.5,      & !..cnorm3 (normalized diameter)
       &        7.246261, & !..cmelt1 (melting intgral)   !!NEEDS TO BE ADJUSTED
       &        0.666666, & !..cmelt2 (melting intgral)   !!NEEDS TO BE ADJUSTED
       &        null()    ) !..ql pointer

  TYPE(particle), PARAMETER :: cloud_cosmo5 = PARTICLE( &
       &      'cloud_cosmo5',  & !.name...Bezeichnung der Partikelklasse
       &        0.0,      & !..nu.....Breiteparameter der Verteil.
       &        0.333333, & !..mu.....Exp.-parameter der Verteil.
       &        2.60d-10, & !..x_max..maximale Teilchenmasse D=80e-6m
       &        4.20d-15, & !..x_min..minimale Teilchenmasse D=2.e-6m
       &        1.24d-01, & !..a_geo..Koeff. Geometrie
       &        0.333333, & !..b_geo..Koeff. Geometrie = 1/3
       &        3.75d+05, & !..a_vel..Koeff. Fallgesetz
       &        0.666667, & !..b_vel..Koeff. Fallgesetz
       &        0.780000, & !..a_ven..Koeff. Ventilation (PK, S.541)
       &        0.308000, & !..b_ven..Koeff. Ventilation (PK, S.541)
       &        2.00,     & !..cap....Koeff. Kapazitaet
       &        1.0,      & !..vsedi_max
       &        0.0,      & !..vsedi_min
       &        null(),   & !..n pointer
       &        null(),   & !..q pointer
       &        null() )    !..rho_v pointer

  TYPE(particle), PARAMETER :: cloud_nue1mue1 = PARTICLE( &
       &        'cloud_nue1mue1',  & !.name...Bezeichnung der Partikelklasse
       &        1.000000, & !..nu.....Breiteparameter der Verteil.
       &        1.000000, & !..mu.....Exp.-parameter der Verteil.
       &        2.60d-10, & !..x_max..maximale Teilchenmasse D=80e-6m
       &        4.20d-15, & !..x_min..minimale Teilchenmasse D=2.e-6m
       &        1.24d-01, & !..a_geo..Koeff. Geometrie
       &        0.333333, & !..b_geo..Koeff. Geometrie = 1/3
       &        3.75d+05, & !..a_vel..Koeff. Fallgesetz
       &        0.666667, & !..b_vel..Koeff. Fallgesetz
       &        0.780000, & !..a_ven..Koeff. Ventilation (PK, S.541)
       &        0.308000, & !..b_ven..Koeff. Ventilation (PK, S.541)
       &        2.00,     & !..cap....Koeff. Kapazitaet
       &        1.0,      & !..vsedi_max
       &        0.0,      & !..vsedi_min
       &        null(),   & !..n pointer
       &        null(),   & !..q pointer
       &        null() )    !..rho_v pointer

  TYPE(particle_frozen), PARAMETER :: &
       &        ice_cosmo5 =  particle_frozen( & ! iceCRY2test
       &        'ice_cosmo5', & !.name...Bezeichnung der Partikelklasse
       &        0.000000, & !..nu...e..Breiteparameter der Verteil.
       &        0.333333, & !..mu.....Exp.-parameter der Verteil.
       &        1.00d-05, & !..x_max..maximale Teilchenmasse D=???e-2m
       &        1.00d-12, & !..x_min..minimale Teilchenmasse D=200e-6m
       &        0.835000, & !..a_geo..Koeff. Geometrie
       &        0.390000, & !..b_geo..Koeff. Geometrie
       &        2.77d+01, & !..a_vel..Koeff. Fallgesetz
       &        0.215790, & !..b_vel..Koeff. Fallgesetz = 0.41/1.9
       &        0.780000, & !..a_ven..Koeff. Ventilation (PK, S.541)
       &        0.308000, & !..b_ven..Koeff. Ventilation (PK, S.541)
       &        3.0,      & !..cap....Koeff. Kapazitaet
       &        3.0,      & !..vsedi_max
       &        0.0,      & !..vsedi_min
       &        null(),   & !..n pointer
       &        null(),   & !..q pointer
       &        null(),   & !..rho_v pointer
       &        0.80,     & !..ecoll_c
       &        150.0d-6, & !..D_crit_c
       &        1.000d-5, & !..q_crit_c
       &        0.25      & !..sigma_vel
       &        )

  TYPE(particle_frozen), PARAMETER :: &
       &        snow_cosmo5 = particle_frozen( & ! nach Andy Heymsfield (CRYSTAL-FACE)
       &        'snow_cosmo5', & !.name...Bezeichnung der Partikelklasse
       &        0.000000, & !..nu.....Breiteparameter der Verteil.
       &        0.500000, & !..mu.....Exp.-parameter der Verteil.
       &        2.00d-05, & !..x_max..maximale Teilchenmasse D=???e-2m
       &        1.00d-10, & !..x_min..minimale Teilchenmasse D=200e-6m
       &        2.400000, & !..a_geo..Koeff. Geometrie
       &        0.455000, & !..b_geo..Koeff. Geometrie
       &        8.800000, & !..a_vel..Koeff. Fallgesetz
       &        0.150000, & !..b_vel..Koeff. Fallgesetz
       &        0.780000, & !..a_ven..Koeff. Ventilation (PK, S.541)
       &        0.308000, & !..b_ven..Koeff. Ventilation (PK, S.541)
       &        3.00,     & !..cap....Koeff. Kapazitaet
       &        3.0,      & !..vsedi_max
       &        0.1,      & !..vsedi_min
       &        null(),   & !..n pointer
       &        null(),   & !..q pointer
       &        null(),   & !..rho_v pointer
       &        0.80,     & !..ecoll_c
       &        150.0d-6, & !..D_crit_c
       &        1.000d-5, & !..q_crit_c
       &        0.25      & !..sigma_vel
       &        )

  TYPE(particle_frozen), PARAMETER :: &
       &        snowSBB =  particle_frozen(   & !
       &        'snowSBB',& !..name...Bezeichnung der Partikelklasse
       &        0.000000, & !..nu.....Breiteparameter der Verteil.
       &        0.500000, & !..mu.....Exp.-parameter der Verteil.
       &        2.00d-05, & !..x_max..maximale Teilchenmasse
       &        1.00d-10, & !..x_min..minimale Teilchenmasse
       &        5.130000, & !..a_geo..Koeff. Geometrie, x = 0.038*D**2
       &        0.500000, & !..b_geo..Koeff. Geometrie = 1/2
       &        400.0000, & !..a_vel..Koeff. Fallgesetz  8.294000 standard
       &        0.350000, & !..b_vel..Koeff. Fallgesetz  0.125 standard
       &        0.780000, & !..a_ven..Koeff. Ventilation (PK, S.541)
       &        0.308000, & !..b_ven..Koeff. Ventilation (PK, S.541)
       &        3.00,     & !..cap....Koeff. Kapazitaet
       &        3.0,      & !..vsedi_max
       &        0.1,      & !..vsedi_min
       &        null(),   & !..n pointer
       &        null(),   & !..q pointer
       &        null(),   & !..rho_v pointer
       &        0.80,     & !..ecoll_c
       &        150.0d-6, & !..D_crit_c
       &        1.000d-5, & !..q_crit_c
       &        0.25      & !..sigma_vel
       &        )

  TYPE(particle_frozen), PARAMETER :: &
       &        snowSBBcorr =  particle_frozen(   & !
       &        'snowSBBcorr',& !..name...Bezeichnung der Partikelklasse
       &        0.000000, & !..nu.....Breiteparameter der Verteil.
       &        0.500000, & !..mu.....Exp.-parameter der Verteil.
       &        2.00d-05, & !..x_max..maximale Teilchenmasse
       &        3.00d-11, & !..x_min..minimale Teilchenmasse
       &        6.500000, & !..a_geo..Koeff. Geometrie, x = 0.038*D**2
       &        0.495000, & !..b_geo..Koeff. Geometrie = 1/2
       &        7.500000, & !..a_vel..Koeff. Fallgesetz
       &        0.125000, & !..b_vel..Koeff. Fallgesetz
       &        0.780000, & !..a_ven..Koeff. Ventilation (PK, S.541)
       &        0.308000, & !..b_ven..Koeff. Ventilation (PK, S.541)
       &        3.00,     & !..cap....Koeff. Kapazitaet
       &        1.2,      & !..vsedi_max
       &        0.01,     & !..vsedi_min
       &        null(),   & !..n pointer
       &        null(),   & !..q pointer
       &        null(),   & !..rho_v pointer
       &        0.80,     & !..ecoll_c
       &        150.0d-6, & !..D_crit_c
       &        1.000d-5, & !..q_crit_c
       &        0.25      & !..sigma_vel
       &        )

  TYPE(particle), PARAMETER :: rainULI = particle( & ! Blahak, v=v(x) gefittet 6.9.2005
       &        'rainULI', & !..name
       &        0.000000,  & !..nu
       &        0.333333,  & !..mu
       &        3.00d-06,  & !..x_max
       &        2.60d-10,  & !..x_min
       &        1.24d-01,  & !..a_geo
       &        0.333333,  & !..b_geo
       &        114.0137,  & !..a_vel
       &        0.234370,  & !..b_vel
       &        0.780000,  & !..a_ven
       &        0.308000,  & !..b_ven
       &        2.00,      & !..cap
       &        20.0,      & !..vsedi_max
       &        0.1,       & !..vsedi_min
       &        null(),    & !..n pointer
       &        null(),    & !..q pointer
       &        null() )     !..rho_v pointer

  TYPE(particle), PARAMETER :: rainSBB = particle( &
       &        'rainSBB', & !..name
       &        1.000000,  & !..nu
       &        0.333333,  & !..mu
       &        6.50d-05,  & !..x_max  ! 5 mm 
       &        2.60d-10,  & !..x_min  ! 80 mum
       &        1.24d-01,  & !..a_geo
       &        0.333333,  & !..b_geo
       &        114.0137,  & !..a_vel
       &        0.234370,  & !..b_vel
       &        0.780000,  & !..a_ven
       &        0.308000,  & !..b_ven
       &        2.000000,  & !..cap
       &        2.000d+1,  & !..vsedi_max
       &        0.1,       & !..vsedi_min
       &        null(),    & !..n pointer
       &        null(),    & !..q pointer
       &        null() )     !..rho_v pointer

  TYPE(particle_rain_coeffs), PARAMETER :: rainSBBcoeffs = particle_rain_coeffs( &
       &        0.0,0.0,0.0,0.0, & !
       &        9.292000,  & !..alfa
       &        9.623000,  & !..beta
       &        6.222d+2,  & !..gama
       &        6.0000d0,  & !..cmu0
       &        3.000d+1,  & !..cmu1
       &        1.000d+3,  & !..cmu2
       &        1.100d-3,  & !..cmu3 = D_br
       &        1.0000d0,  & !..cmu4
       &        2 )          !..cmu5

  REAL(wp), PARAMETER :: pi6 = pi/6.0_wp, pi8 = pi/8.0_wp ! more pieces of pi

  !..run-time- and location-invariant collection process parameters
  TYPE(collection_coeffs), SAVE :: scr_coeffs  ! snow cloud riming
  TYPE(rain_riming_coeffs),SAVE :: srr_coeffs  ! snow rain riming
  TYPE(rain_riming_coeffs),SAVE :: irr_coeffs  ! ice rain riming
  TYPE(collection_coeffs), SAVE :: icr_coeffs  ! ice cloud riming
  TYPE(collection_coeffs), SAVE :: hrr_coeffs  ! hail rain riming
  TYPE(collection_coeffs), SAVE :: grr_coeffs  ! graupel rain riming
  TYPE(collection_coeffs), SAVE :: hcr_coeffs  ! hail cloud riming
  TYPE(collection_coeffs), SAVE :: gcr_coeffs  ! graupel cloud  riming
  TYPE(collection_coeffs), SAVE :: sic_coeffs  ! snow ice collection
  TYPE(collection_coeffs), SAVE :: hic_coeffs  ! hail ice collection
  TYPE(collection_coeffs), SAVE :: gic_coeffs  ! graupel ice collection
  TYPE(collection_coeffs), SAVE :: hsc_coeffs  ! hail snow collection
  TYPE(collection_coeffs), SAVE :: gsc_coeffs  ! graupel snow collection
  TYPE(coll_coeffs_ir_pm), SAVE :: gshedc_coeffs ! graupel shedding during cloud riming
  TYPE(coll_coeffs_ir_pm), SAVE :: hshedc_coeffs ! hail shedding during cloud riming
  TYPE(coll_coeffs_ir_pm), SAVE :: gshedr_coeffs ! graupel shedding during rain riming
  TYPE(coll_coeffs_ir_pm), SAVE :: hshedr_coeffs ! hail shedding during rain riming

  PUBLIC :: atmosphere, particle, particle_lwf, particle_frozen
  PUBLIC :: init_2mom_scheme, init_2mom_scheme_once, clouds_twomoment
  PUBLIC :: rain_coeffs, ice_coeffs, snow_coeffs, graupel_coeffs, hail_coeffs, &
       &    ccn_coeffs, in_coeffs, cloud_coeffs
  PUBLIC :: qnc_const

  ! DR: The following block is necessarily public as these parameters/coefficients
  !     are required by the ART code
  PUBLIC :: scr_coeffs, srr_coeffs, irr_coeffs, icr_coeffs
  PUBLIC :: hrr_coeffs, grr_coeffs, hcr_coeffs, gcr_coeffs
  PUBLIC :: sic_coeffs, hic_coeffs, gic_coeffs, hsc_coeffs, gsc_coeffs
  PUBLIC :: rain_gfak
  ! These do no longer exist and have been replace by ice_coeffs, etc.
  !  PUBLIC :: vid_params, vgd_params, vhd_params, vsd_params
  !  PUBLIC :: ge_params, he_params, se_params, gm_params, hm_params, sm_params
  ! These have been deleted:
  !  PUBLIC :: graupel_shedding, hail_shedding
  ! END DR

CONTAINS
  !*******************************************************************************
  ! Main subroutine of the two-moment microphysics
  !
  ! All individual processes are called in sequence and do their own time
  ! integration for the mass and number densities, i.e., Marchuk-type operator
  ! splitting. Temperature is only updated in the driver.
  !*******************************************************************************

  SUBROUTINE clouds_twomoment(kstart, kend, dt, use_prog_in, atmo, &
       cloud, rain, ice, snow, graupel, hail, ssat, lexpl_supersat, n_inact, n_cn, &
       n_inpot, n_inagi, luse_agi, iagi_param)

    ! start and end level for the column
    INTEGER, INTENT(in) :: kstart, kend

    ! time step within two-moment scheme
    REAL(wp), INTENT(in) :: dt

    LOGICAL, INTENT(in) :: use_prog_in
    TYPE(atmosphere), INTENT(inout)       :: atmo
    CLASS(particle),  INTENT(inout)       :: cloud, rain
    CLASS(particle_frozen), INTENT(inout) :: ice, snow, graupel, hail

    REAL(wp), DIMENSION(:) :: n_inact

    ! optional arguments for version with prognostic CCN and IN
    ! (for nuc_c_typ > 0 and  use_prog_in=true)
    REAL(wp), DIMENSION(:), OPTIONAL :: n_inpot, n_inagi, n_cn, ssat
    INTEGER, OPTIONAL, INTENT(IN) :: iagi_param
    LOGICAL, OPTIONAL, INTENT(IN) :: luse_agi
    LOGICAL, INTENT(IN) :: lexpl_supersat

    REAL(wp), DIMENSION(SIZE(cloud%n)) :: &
         & dep_rate_ice,    &  ! deposition rate of vapor on ice particles
         & dep_rate_snow,   &  ! deposition rate of vapor on snow particles
         & gmelting,        &  ! ambient atmospheric conditions for melting in lwf scheme
         & cloud_q_old

    REAL(wp) :: delta_q, x_c, n_c

    INTEGER :: k

    IF (isdebug) CALL message(TRIM(routine),"clouds_twomoment start")

    DO k = kstart,kend
      dep_rate_ice(k)  = 0.0_wp
      dep_rate_snow(k) = 0.0_wp
    ENDDO

    IF (isdebug) CALL message(TRIM(routine),'cloud_nucleation')

    IF (nuc_c_typ .EQ. 0) THEN
      IF (isdebug) CALL message(TRIM(routine),'  ... force constant cloud droplet number')

      cloud%n(:) = qnc_const
    ELSEIF (nuc_c_typ < 6) THEN
      IF (isdebug) CALL message(TRIM(routine),'  ... Hande et al CCN activation')
      IF (PRESENT(n_cn)) THEN
        CALL finish(TRIM(routine),&
               & 'Error in two_moment_mcrph: Hande et al activation not supported for progn. aerosol')
      ELSE
        CALL ccn_activation_hdcp2(kstart,kend,atmo,cloud)
      END IF
    ELSE
      IF (isdebug) CALL message(TRIM(routine), &
            & '  ... CCN activation using look-up tables according to Segal & Khain')
      IF (PRESENT(n_cn)) THEN
        CALL ccn_activation_sk_4d(kstart,kend,ccn_coeffs,atmo,cloud,n_cn)
      ELSE
        CALL ccn_activation_sk_4d(kstart,kend,ccn_coeffs,atmo,cloud)
      END IF
    END IF

    IF (ischeck) CALL check(kstart,kend,'start',cloud,rain,ice,snow,graupel,hail)
    CALL dump('ccn')

    ! Set to default values where qnx =0 and qx>0
    CALL set_default_n(kstart, kend, cloud, ice, rain, snow, graupel, hail)

    IF (nuc_c_typ.ne.0) THEN
      DO k=kstart,kend
        cloud%n(k) = MAX(cloud%n(k), cloud%q(k) / cloud%x_max)
        cloud%n(k) = MIN(cloud%n(k), cloud%q(k) / cloud%x_min)
      END DO
    END IF
    ! after set_default_n *and* the cloud-size clip that follows it, matching the
    ! port's step ordering in driver.py
    CALL dump('default_n')

    IF (cfg_params%iicephase .EQ. 1) THEN

      ! homogeneous and heterogeneous ice nucleation
      CALL ice_nucleation_homhet(kstart,kend, use_prog_in, atmo, cloud, ice, n_inact, n_inpot, &
            &                    n_inagi, luse_agi, iagi_param)
      CALL dump('ice_nuc')

      ! homogeneous freezing of cloud droplets
      CALL cloud_freeze(kstart,kend, dt, cloud_coeffs, qnc_const, atmo, cloud, ice)
      IF (ischeck) CALL check(kstart,kend,'cloud_freeze', cloud, rain, ice, snow, graupel,hail)

      DO k=kstart,kend
        ice%n(k) = MIN(ice%n(k), ice%q(k)/ice%x_min)
        ice%n(k) = MAX(ice%n(k), ice%q(k)/ice%x_max)
      END DO

      IF (ischeck) CALL check(kstart,kend,'ice nucleation',cloud,rain,ice,snow,graupel,hail)
      ! after cloud_freeze *and* the ice-size clip -- this is exactly the state
      ! vapor_dep_relaxation is handed, which is what makes it the useful boundary
      ! for isolating the depositional-growth step
      CALL dump('cloud_freeze')

      ! depositional growth of all ice particles
      ! ( store deposition rate of ice and snow for conversion calculation in
      !   ice_riming and snow_riming )
      CALL vapor_dep_relaxation(kstart,kend,dt,ice_coeffs,snow_coeffs,graupel_coeffs,hail_coeffs,&
           &                    atmo,ice,snow,graupel,hail,dep_rate_ice,dep_rate_snow)
      IF (ischeck) CALL check(kstart,kend,'vapor_dep_relaxation',cloud,rain,ice,snow,graupel,hail)
      CALL dump('vapor_dep')

      ! MINIMAL SCHEME: all ice/snow/graupel/hail collision, riming, wet-growth
      ! conversion, rain freezing, and the precip-size melting/evaporation steps
      ! are intentionally omitted here. The only retained sink is the melting of
      ! cloud-ice crystals back to cloud/rain water.
      CALL ice_melting(kstart,kend, atmo, ice, cloud, rain)
      IF (ischeck) CALL check(kstart,kend, 'ice_melting',cloud,rain,ice,snow,graupel,hail)
      CALL dump('ice_melt')

    END IF ! cfg_params%iicephase


    ! MINIMAL SCHEME: warm-rain autoconversion/accretion/self-collection and
    ! rain evaporation are omitted (no precipitation-size processes).

    DO k=kstart,kend

      ! size limits for all hydrometeors
      IF (nuc_c_typ > 0) THEN
        cloud%n(k) = MIN(cloud%n(k), cloud%q(k)/cloud%x_min)
        cloud%n(k) = MAX(cloud%n(k), cloud%q(k)/cloud%x_max)
        ! Hard upper limit for cloud number conc.
        cloud%n(k) = MIN(cloud%n(k), 5000d6)
      END IF

      rain%n(k) = MIN(rain%n(k), rain%q(k)/rain%x_min)
      rain%n(k) = MAX(rain%n(k), rain%q(k)/rain%x_max)
      ice%n(k) = MIN(ice%n(k), ice%q(k)/ice%x_min)
      ice%n(k) = MAX(ice%n(k), ice%q(k)/ice%x_max)
      snow%n(k) = MIN(snow%n(k), snow%q(k)/snow%x_min)
      snow%n(k) = MAX(snow%n(k), snow%q(k)/snow%x_max)
      graupel%n(k) = MIN(graupel%n(k), graupel%q(k)/graupel%x_min)
      graupel%n(k) = MAX(graupel%n(k), graupel%q(k)/graupel%x_max)
      hail%n(k) = MIN(hail%n(k), hail%q(k)/hail%x_min)
      hail%n(k) = MAX(hail%n(k), hail%q(k)/hail%x_max)

    END DO

    IF (ischeck) CALL check(kstart,kend, 'clouds_twomoment end',cloud,rain,ice,snow,graupel,hail)
    IF (isdebug) CALL message(TRIM(routine),"clouds_twomoment end")

  CONTAINS

    ! Output only; no-op unless the driver was given a dump directory. Exists so each
    ! process boundary above reads as a single line instead of a repeated 20-argument
    ! call -- host association supplies atmo/cloud/.../n_cn here.
    !
    ! n_cn and n_inpot are OPTIONAL dummies of clouds_twomoment (absent when the scheme
    ! runs without prognostic aerosol), so they are copied through PRESENT guards rather
    ! than sliced directly, which would be invalid for an absent argument.
    SUBROUTINE dump(stage)
      CHARACTER(len=*), INTENT(in) :: stage

      REAL(wp), DIMENSION(kend-kstart+1) :: ccn_l, inpot_l

      IF (.NOT. stage_dump_enabled()) RETURN

      ccn_l   = 0.0_wp
      inpot_l = 0.0_wp
      IF (PRESENT(n_cn))    ccn_l   = n_cn(kstart:kend)
      IF (PRESENT(n_inpot)) inpot_l = n_inpot(kstart:kend)

      CALL dump_stage(stage, kend-kstart+1, &
           atmo%rho(kstart:kend), atmo%p(kstart:kend), atmo%w(kstart:kend), &
           atmo%T(kstart:kend), atmo%qv(kstart:kend), &
           cloud%q(kstart:kend),   cloud%n(kstart:kend),   &
           rain%q(kstart:kend),    rain%n(kstart:kend),    &
           ice%q(kstart:kend),     ice%n(kstart:kend),     &
           snow%q(kstart:kend),    snow%n(kstart:kend),    &
           graupel%q(kstart:kend), graupel%n(kstart:kend), &
           hail%q(kstart:kend),    hail%n(kstart:kend),    &
           ccn_l, inpot_l, n_inact(kstart:kend))
    END SUBROUTINE dump

  END SUBROUTINE clouds_twomoment

  !*******************************************************************************
  ! This subroutine has to be called once per time step to properly sets
  ! the different hydrometeor classes according to predefined parameter sets
  !*******************************************************************************

  SUBROUTINE init_2mom_scheme(cloud,rain,ice,snow,graupel,hail)
    CLASS(particle),INTENT(inout)        :: cloud, rain
    CLASS(particle_frozen),INTENT(inout) :: ice, snow, graupel, hail

    CALL particle_assign(cloud,cloud_nue1mue1)
    CALL particle_assign(rain,rainSBB)

    IF (cfg_params % nu_r > -900.0_wp) rain%nu = cfg_params%nu_r

    CALL particle_frozen_assign(ice,ice_cosmo5)
    CALL particle_frozen_assign(snow,snowSBB)
!!$    CALL particle_frozen_assign(snow,snowSBBcorr)

    IF (cfg_params % nu_i > -900.0_wp) ice%nu = cfg_params%nu_i
    IF (cfg_params % mu_i > -900.0_wp) ice%mu = cfg_params%mu_i
    IF (cfg_params % ageo_i > -900.0_wp) ice%a_geo = cfg_params%ageo_i
    IF (cfg_params % bgeo_i > -900.0_wp) ice%b_geo = cfg_params%bgeo_i
    IF (cfg_params % avel_i > -900.0_wp) ice%a_vel = cfg_params%avel_i
    IF (cfg_params % bvel_i > -900.0_wp) ice%b_vel = cfg_params%bvel_i
    IF (cfg_params % cap_ice > -900.0_wp) ice%cap = cfg_params%cap_ice

    IF (cfg_params % nu_s > -900.0_wp) snow%nu = cfg_params%nu_s
    IF (cfg_params % mu_s > -900.0_wp) snow%mu = cfg_params%mu_s
    IF (cfg_params % ageo_s > -900.0_wp) snow%a_geo = cfg_params%ageo_s
    IF (cfg_params % bgeo_s > -900.0_wp) snow%b_geo = cfg_params%bgeo_s
    IF (cfg_params % avel_s > -900.0_wp) snow%a_vel = cfg_params%avel_s
    IF (cfg_params % bvel_s > -900.0_wp) snow%b_vel = cfg_params%bvel_s
    IF (cfg_params % cap_snow > -900.0_wp) snow%cap = cfg_params%cap_snow
    IF (cfg_params % vsedi_max_s > -900.0_wp) snow%vsedi_max = cfg_params%vsedi_max_s

    SELECT TYPE (graupel)
    TYPE IS (particle_frozen)
      CALL particle_frozen_assign(graupel,graupelhail_cosmo5)
    TYPE IS (particle_lwf)
      CALL particle_lwf_assign(graupel,graupel_vivek)
    END SELECT

    IF (cfg_params % nu_g > -900.0_wp) graupel%nu = cfg_params%nu_g
    IF (cfg_params % mu_g > -900.0_wp) graupel%mu = cfg_params%mu_g
    IF (cfg_params % ageo_g > -900.0_wp) graupel%a_geo = cfg_params%ageo_g
    IF (cfg_params % bgeo_g > -900.0_wp) graupel%b_geo = cfg_params%bgeo_g
    IF (cfg_params % avel_g > -900.0_wp) graupel%a_vel = cfg_params%avel_g
    IF (cfg_params % bvel_g > -900.0_wp) graupel%b_vel = cfg_params%bvel_g

    SELECT TYPE (hail)
    TYPE IS (particle_frozen)
      CALL particle_frozen_assign(hail,hail_cosmo5)
    TYPE IS (particle_lwf)
      CALL particle_lwf_assign(hail,hail_vivek)
    END SELECT

    IF (cfg_params % nu_h > -900.0_wp) hail%nu = cfg_params%nu_h
    IF (cfg_params % mu_h > -900.0_wp) hail%mu = cfg_params%mu_h
    IF (cfg_params % ageo_h > -900.0_wp) hail%a_geo = cfg_params%ageo_h
    IF (cfg_params % bgeo_h > -900.0_wp) hail%b_geo = cfg_params%bgeo_h
    IF (cfg_params % avel_h > -900.0_wp) hail%a_vel = cfg_params%avel_h
    IF (cfg_params % bvel_h > -900.0_wp) hail%b_vel = cfg_params%bvel_h

  END SUBROUTINE init_2mom_scheme

  !*******************************************************************************
  ! This subroutine has to be called once at the start of the model run by
  ! the main program. It properly sets the parameters for the different hydrometeor
  ! classes according to predefined parameter sets (see above).
  !*******************************************************************************

  SUBROUTINE init_2mom_scheme_once(cloud,rain,ice,snow,graupel,hail,cloud_type)
    INTEGER, INTENT(in)  :: cloud_type
    CLASS(particle), INTENT(inout) :: cloud, rain
    CLASS(particle_frozen), INTENT(inout) :: ice, snow, graupel, hail

    CHARACTER(len=*), PARAMETER :: routine = 'init_2mom_scheme_once'

    ! MINIMAL SCHEME one-time setup.
    !
    ! Only the run-time coefficients the retained call tree needs are computed:
    !   * setup_particle_coeffs -> ventilation/diffusion coeffs (c_i, a_f, b_f)
    !     and c_z, used by vapor_dep_relaxation and cloud_freeze
    !   * init_2mom_sedi_vel    -> bulk sedimentation-velocity coeffs
    ! All collision/riming coefficient setups, the incomplete-gamma lookup tables
    ! (rain_freeze / wet-growth conversion), the sticking-efficiency tables and the
    ! warm-rain autoconversion setup of the full scheme are omitted, since the
    ! processes that use them are not called. Note: for the default cloud particle
    ! (cloud_nue1mue1, mu=1) setup_cloud_autoconversion would not modify c_z, so
    ! dropping it leaves cloud_freeze unchanged.

    CALL init_2mom_scheme(cloud,rain,ice,snow,graupel,hail)

    ice_typ   = cloud_type/1000           ! (0) no ice, (1) no hail (2) with hail
    nuc_i_typ = MOD(cloud_type/100,10)    ! choice of ice nucleation, see ice_nucleation_homhet()
    nuc_c_typ = MOD(cloud_type/10,10)     ! choice of CCN assumptions, see cloud_nucleation()
    auto_typ  = MOD(cloud_type,10)        ! warm-rain scheme choice (unused in minimal scheme)

    ! rain run-time coefficients (kept for a consistent rain particle definition)
    CALL setup_particle_coeffs(rain,rain_coeffs)
    rain_coeffs%alfa = rainSBBcoeffs%alfa
    rain_coeffs%beta = rainSBBcoeffs%beta
    rain_coeffs%gama = rainSBBcoeffs%gama
    rain_coeffs%cmu2 = rainSBBcoeffs%cmu2
    rain_coeffs%cmu4 = rainSBBcoeffs%cmu4
    rain_coeffs%cmu5 = rainSBBcoeffs%cmu5
    rain_coeffs%cmu0 = cfg_params%rain_cmu0
    rain_coeffs%cmu1 = cfg_params%rain_cmu1
    rain_coeffs%cmu3 = cfg_params%rain_cmu3
    rain_coeffs%cmu4 = cfg_params%rain_cmu4

    ! bulk sedimentation-velocity coefficients for the frozen categories
    CALL init_2mom_sedi_vel(ice,ice_coeffs)
    CALL init_2mom_sedi_vel(snow,snow_coeffs)
    CALL init_2mom_sedi_vel(graupel,graupel_coeffs)
    CALL init_2mom_sedi_vel(hail,hail_coeffs)

    ! ventilation / diffusion coefficients used by vapor_dep_relaxation and cloud_freeze
    CALL setup_particle_coeffs(ice,ice_coeffs)
    CALL setup_particle_coeffs(graupel,graupel_coeffs)
    CALL setup_particle_coeffs(hail,hail_coeffs)
    CALL setup_particle_coeffs(snow,snow_coeffs)
    CALL setup_particle_coeffs(cloud,cloud_coeffs)

    ! Segal & Khain CCN activation: build the equidistant lookup table if selected
    IF (nuc_c_typ > 5) THEN
      CALL ccn_activation_sk_4d()
      IF (isprint) CALL message(routine,"Equidistant lookup table for Segal-Khain created")
    END IF

    ! Output only, no-op unless the driver was given a dump directory. Without this
    ! the a_f/b_f/c_i/c_z coefficients are printed nowhere at all (only
    ! init_2mom_sedi_vel's three reach stdout, via `isprint`, and only at 7 digits),
    ! which is why the port's coefficient test could not assert tighter than 1e-6.
    ! cloud_coeffs is a particle_cloud_coeffs and carries no sedimentation
    ! coefficients, so those three are written as zeros for cloud.
    CALL dump_coeffs('ice', ice_coeffs%a_f, ice_coeffs%b_f, ice_coeffs%c_i, ice_coeffs%c_z, &
         ice_coeffs%coeff_alfa_n, ice_coeffs%coeff_alfa_q, ice_coeffs%coeff_lambda)
    CALL dump_coeffs('snow', snow_coeffs%a_f, snow_coeffs%b_f, snow_coeffs%c_i, snow_coeffs%c_z, &
         snow_coeffs%coeff_alfa_n, snow_coeffs%coeff_alfa_q, snow_coeffs%coeff_lambda)
    CALL dump_coeffs('graupel', graupel_coeffs%a_f, graupel_coeffs%b_f, graupel_coeffs%c_i, &
         graupel_coeffs%c_z, graupel_coeffs%coeff_alfa_n, graupel_coeffs%coeff_alfa_q, &
         graupel_coeffs%coeff_lambda)
    CALL dump_coeffs('hail', hail_coeffs%a_f, hail_coeffs%b_f, hail_coeffs%c_i, hail_coeffs%c_z, &
         hail_coeffs%coeff_alfa_n, hail_coeffs%coeff_alfa_q, hail_coeffs%coeff_lambda)
    CALL dump_coeffs('cloud', cloud_coeffs%a_f, cloud_coeffs%b_f, cloud_coeffs%c_i, &
         cloud_coeffs%c_z, 0.0_wp, 0.0_wp, 0.0_wp)

  END SUBROUTINE init_2mom_scheme_once

  SUBROUTINE check(kstart, kend, mtxt, cloud, rain, ice, snow, graupel, hail)
    INTEGER, INTENT(in) :: kstart, kend
    CHARACTER(len=*), INTENT(in) :: mtxt
    CLASS(particle), INTENT(in) :: cloud, rain, ice, snow, graupel, hail

    INTEGER :: k
    REAL(wp), PARAMETER  :: meps = -1e-12_wp

    DO k = kstart,kend
      IF (cloud%q(k) < meps) THEN
        WRITE (txt,'(1X,A,I4,A)') '  qc < 0 at k = ',k,' after '//TRIM(mtxt)
        CALL message(TRIM(routine),TRIM(txt))
        CALL finish (TRIM(routine),TRIM(txt))
      ENDIF
      IF (rain%q(k) < meps) THEN
        WRITE (txt,'(1X,A,I4,A)') '  qr < 0 at k = ',k,' after '//TRIM(mtxt)
        CALL message(TRIM(routine),TRIM(txt))
        CALL finish (TRIM(routine),TRIM(txt))
      ENDIF
      IF (ice%q(k) < meps) THEN
        WRITE (txt,'(1X,A,I4,A)') '  qi < 0 at k = ',k,' after '//TRIM(mtxt)
        CALL message(TRIM(routine),TRIM(txt))
        CALL finish (TRIM(routine),TRIM(txt))
      ENDIF
      IF (snow%q(k) < meps) THEN
        WRITE (txt,'(1X,A,I4,A)') '  qs < 0 at k = ',k,' after '//TRIM(mtxt)
        CALL message(TRIM(routine),TRIM(txt))
        CALL finish (TRIM(routine),TRIM(txt))
      ENDIF
      IF (graupel%q(k) < meps) THEN
        WRITE (txt,'(1X,A,I4,A)') '  qg < 0 at k = ',k,' after '//TRIM(mtxt)
        CALL message(TRIM(routine),TRIM(txt))
        CALL finish (TRIM(routine),TRIM(txt))
      ENDIF
      IF (hail%q(k) < meps) THEN
        WRITE (txt,'(1X,A,I4,A)') '  qh < 0 at k = ',k,' after '//TRIM(mtxt)
        CALL message(TRIM(routine),TRIM(txt))
        CALL finish (TRIM(routine),TRIM(txt))
      ENDIF
    END DO

  END SUBROUTINE check

END MODULE mo_2mom_mcrph_main
