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
  USE mo_2mom_mcrph_util, ONLY: &
       & gamlookuptable,             &  ! For look-up table of incomplete Gamma function
       & nlookup, nlookuphr_dummy,   &  !   array size of table
       & incgfct_lower_lookupcreate, &  !   create table
       & incgfct_lower_lookup,       &  !   interpolation in table, lower incomplete Gamma function
       & incgfct_upper_lookup,       &  !   interpolation in talbe, upper incomplete Gamma function
       & dmin_wg_gr_ltab_equi,       &  ! For look-up table of wet growth diameter
       & init_estick_ltab_equi
  USE mo_2mom_mcrph_processes, ONLY:                                         &
       &  particle_assign, particle_frozen_assign, particle_lwf_assign,      &
       &  sedi_vel_rain, init_2mom_sedi_vel, autoconversionSB,               &
       &  accretionSB, rain_selfcollectionSB, autoconversionKB, accretionKB, &
       &  autoconversionKK, accretionKK, rain_evaporation, evaporation,      &
       &  cloud_freeze, ice_nucleation_homhet,                               &
       &  vapor_dep_relaxation, cloud_mass_growth, rain_freeze_gamlook,      &
       &  cloud_mass_growth_implicit,                                                 &
       &  setup_particle_coeffs, setup_cloud_autoconversion,                 &
       &  setup_ice_selfcollection, ice_selfcollection,                      &
       &  setup_snow_selfcollection, snow_selfcollection,                    &
       &  setup_particle_collection_type1,                                   &
       &  setup_particle_collection_type2,                                   &
       &  setup_particle_coll_pm_type1_bfull,                                &
       &  snow_melting, particle_particle_collection,                        &
       &  particle_melting_lwf,prepare_melting_lwf,                          &
       &  setup_graupel_selfcollection, graupel_selfcollection, ice_melting, &
       &  particle_cloud_riming, particle_rain_riming, graupel_melting,      &
       &  hail_melting_simple, graupel_hail_conv_wet_gamlook, ice_riming,    &
       &  snow_riming, ccn_activation_sk, ccn_activation_hdcp2,              &
       &  ccn_activation_sk_4d, set_default_n, cfg_params

  ! Some switches...
  USE mo_2mom_mcrph_processes, ONLY:                                         &
       &  ice_typ, nuc_i_typ, nuc_c_typ, auto_typ, isdebug, isprint,         &
       &  particle_meanmass

  USE mo_timer, ONLY: timers_level, timer_start, timer_stop, timer_phys_2mom_wetgrowth
  USE mo_satad, ONLY: satad_v_3D, satad_v_3D_gpu

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

  ! .. Look up tables for graupel_hail_conv_wet_gamlook and rain_freeze_gamlook
  TYPE(gamlookuptable) :: graupel_ltable1, graupel_ltable2
  TYPE(gamlookuptable) :: rain_ltable1, rain_ltable2, rain_ltable3
  REAL(wp)             :: rain_nm1, rain_nm2, rain_nm3, rain_g1, rain_g2
  REAL(wp)             :: graupel_nm1, graupel_nm2, graupel_g1, graupel_g2

  ! .. Look up tables for graupel and hail shedding
  TYPE(gamlookuptable) :: gshedc_ltab_dgg_03, gshedc_ltab_drr_01, gshedc_ltab_dgr_02, &
                          gshedc_ltab_tgg_05, gshedc_ltab_tgg_03, gshedc_ltab_tgr_04
  TYPE(gamlookuptable) :: hshedc_ltab_dhh_03, hshedc_ltab_drr_01, hshedc_ltab_dhr_02, &
                          hshedc_ltab_thh_05, hshedc_ltab_thh_03, hshedc_ltab_thr_04
  TYPE(gamlookuptable) :: gshedr_ltab_dgg_03, gshedr_ltab_drr_01, gshedr_ltab_dgr_02, &
                          gshedr_ltab_tgg_05, gshedr_ltab_tgg_03, gshedr_ltab_tgr_04
  TYPE(gamlookuptable) :: hshedr_ltab_dhh_03, hshedr_ltab_drr_01, hshedr_ltab_dhr_02, &
                          hshedr_ltab_thh_05, hshedr_ltab_thh_03, hshedr_ltab_thr_04
  
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
  PUBLIC :: graupel_ltable1, graupel_ltable2
  PUBLIC :: rain_ltable1, rain_ltable2, rain_ltable3
  PUBLIC :: rain_nm1, rain_nm2, rain_nm3, rain_g1, rain_g2
  PUBLIC :: graupel_nm1, graupel_nm2, graupel_g1, graupel_g2
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

    ! Set to default values where qnx =0 and qx>0
    CALL set_default_n(kstart, kend, cloud, ice, rain, snow, graupel, hail)

    IF (nuc_c_typ.ne.0) THEN
      DO k=kstart,kend
        cloud%n(k) = MAX(cloud%n(k), cloud%q(k) / cloud%x_max)
        cloud%n(k) = MIN(cloud%n(k), cloud%q(k) / cloud%x_min)
      END DO
    END IF

    !>>NO
    ! Implementation of cloud droplet mass growth / evaporation equation
    !>>BB
    !Try to move this to the beginning of the cloud scheme to emulate the saturation adjustment
    if (lexpl_supersat) then
      CALL cloud_mass_growth(kstart,kend,dt,atmo,cloud)
    end if

    IF (cfg_params%iicephase .EQ. 1) THEN

      ! homogeneous and heterogeneous ice nucleation
      CALL ice_nucleation_homhet(kstart,kend, use_prog_in, atmo, cloud, ice, n_inact, n_inpot, &
            &                    n_inagi, luse_agi, iagi_param)

      ! homogeneous freezing of cloud droplets
      CALL cloud_freeze(kstart,kend, dt, cloud_coeffs, qnc_const, atmo, cloud, ice)
      IF (ischeck) CALL check(kstart,kend,'cloud_freeze', cloud, rain, ice, snow, graupel,hail)

      DO k=kstart,kend
        ice%n(k) = MIN(ice%n(k), ice%q(k)/ice%x_min)
        ice%n(k) = MAX(ice%n(k), ice%q(k)/ice%x_max)
      END DO

      IF (ischeck) CALL check(kstart,kend,'ice nucleation',cloud,rain,ice,snow,graupel,hail)

      ! depositional growth of all ice particles
      ! ( store deposition rate of ice and snow for conversion calculation in
      !   ice_riming and snow_riming )
      CALL vapor_dep_relaxation(kstart,kend,dt,ice_coeffs,snow_coeffs,graupel_coeffs,hail_coeffs,&
           &                    atmo,ice,snow,graupel,hail,dep_rate_ice,dep_rate_snow)
      IF (ischeck) CALL check(kstart,kend,'vapor_dep_relaxation',cloud,rain,ice,snow,graupel,hail)
    
      !>>NO
      ! Implementation of cloud droplet mass growth / evaporation equation
      if (.not.lexpl_supersat) then
        CALL cloud_mass_growth(kstart,kend,dt,atmo,cloud)
      end if

      ! ice-ice collisions
      CALL ice_selfcollection(kstart,kend,dt,atmo,ice,snow,ice_coeffs,ltab_estick_ice)
      CALL snow_selfcollection(kstart,kend,dt,atmo,snow,snow_coeffs,ltab_estick_snow)
      CALL particle_particle_collection(kstart,kend, dt, atmo, ice, snow, sic_coeffs, ltab_estick_parti)
      IF (ischeck) CALL check(kstart,kend, 'ice and snow collection',cloud,rain,ice,snow,graupel,hail)

      CALL graupel_selfcollection(kstart,kend, dt, atmo, graupel, graupel_coeffs)
      CALL particle_particle_collection(kstart,kend, dt, atmo, ice, graupel, gic_coeffs, ltab_estick_parti)
      CALL particle_particle_collection(kstart,kend, dt, atmo, snow, graupel, gsc_coeffs, ltab_estick_parti)
      IF (ischeck) CALL check(kstart,kend, 'graupel collection',cloud,rain,ice,snow,graupel,hail)

      ! conversion of graupel to hail in wet growth regime
      IF (timers_level > 10) CALL timer_start(timer_phys_2mom_wetgrowth)
      CALL graupel_hail_conv_wet_gamlook(kstart,kend, graupel_ltable1, graupel_ltable2,       &
           &                             graupel_nm1, graupel_nm2, graupel_g1, graupel_g2, &
           &                             ltabdminwgg, atmo, graupel, cloud, rain, ice, snow, hail)
      IF (timers_level > 10) CALL  timer_stop(timer_phys_2mom_wetgrowth)
      IF (ischeck) CALL check(kstart,kend, 'graupel_hail_conv_wet_gamlook',cloud,rain,ice,snow,graupel,hail)

      ! hail collisions
      CALL particle_particle_collection(kstart,kend, dt, atmo, ice, hail, hic_coeffs, ltab_estick_parti)    ! Important?
      CALL particle_particle_collection(kstart,kend, dt, atmo, snow, hail, hsc_coeffs, ltab_estick_parti)
      IF (ischeck) CALL check(kstart,kend, 'hail collection',cloud,rain,ice,snow,graupel,hail)

      ! riming of ice with cloud droplets and rain drops, and conversion to graupel
      CALL ice_riming(kstart,kend, dt, icr_coeffs, irr_coeffs, atmo, ice, cloud, rain, graupel, dep_rate_ice)
      IF (ischeck) CALL check(kstart,kend, 'ice_riming',cloud,rain,ice,snow,graupel,hail)

      ! riming of snow with cloud droplets and rain drops, and conversion to graupel
      CALL snow_riming(kstart,kend, dt, scr_coeffs, srr_coeffs, atmo, snow, cloud, rain, ice, graupel, dep_rate_snow)
      IF (ischeck) CALL check(kstart,kend, 'snow_riming',cloud,rain,ice,snow,graupel,hail)

      ! hail-cloud and hail-rain riming including shedding
      CALL particle_cloud_riming(kstart,kend, dt, atmo, hail, hcr_coeffs, cloud, rain, ice, &
           hshedc_coeffs, snow, ltabdminwgh, &
           hshedc_ltab_dhh_03, hshedc_ltab_dhr_02, hshedc_ltab_drr_01, &
           hshedc_ltab_thh_05, hshedc_ltab_thh_03, hshedc_ltab_thr_04 )
      CALL particle_rain_riming(kstart,kend, dt, atmo, hail, hrr_coeffs, rain, ice, &
           hshedr_coeffs, cloud, snow, ltabdminwgh, &
           hshedr_ltab_dhh_03, hshedr_ltab_dhr_02, hshedr_ltab_drr_01, &
           hshedr_ltab_thh_05, hshedr_ltab_thh_03, hshedr_ltab_thr_04 )
      IF (ischeck) CALL check(kstart,kend, 'hail riming',cloud,rain,ice,snow,graupel,hail)

      ! graupel-cloud and graupel-rain riming including shedding
      CALL particle_cloud_riming(kstart,kend, dt, atmo, graupel, gcr_coeffs, cloud, rain, ice, &
           gshedc_coeffs, snow, ltabdminwgg, &
           gshedc_ltab_dgg_03, gshedc_ltab_dgr_02, gshedc_ltab_drr_01, &
           gshedc_ltab_tgg_05, gshedc_ltab_tgg_03, gshedc_ltab_tgr_04)
      CALL particle_rain_riming(kstart,kend, dt, atmo, graupel, grr_coeffs, rain, ice, &
           gshedr_coeffs, cloud, snow, ltabdminwgg, &
           gshedr_ltab_dgg_03, gshedr_ltab_dgr_02, gshedr_ltab_drr_01, &
           gshedr_ltab_tgg_05, gshedr_ltab_tgg_03, gshedr_ltab_tgr_04)
      IF (ischeck) CALL check(kstart,kend, 'graupel riming',cloud,rain,ice,snow,graupel,hail)

      ! freezing of rain and conversion to ice/graupel/hail
      CALL rain_freeze_gamlook(kstart,kend, dt, rain_ltable1, rain_ltable2, rain_ltable3, &
           &                   rain_nm1, rain_nm2, rain_nm3, rain_g1, rain_g2,         &
           &                   rain_coeffs,atmo,rain,ice,snow,graupel,hail)
      IF (ischeck) CALL check(kstart,kend, 'rain_freeze_gamlook',cloud,rain,ice,snow,graupel,hail)

      ! melting of ice and snow
      CALL ice_melting(kstart,kend, atmo, ice, cloud, rain)
      CALL snow_melting(kstart,kend,dt,snow_coeffs,atmo,snow,rain)

      ! melting of graupel and hail can be simple or LWF-based
      SELECT TYPE (graupel)
      TYPE IS (particle_frozen)
        CALL graupel_melting(kstart,kend,dt,graupel_coeffs,atmo,graupel,rain)
      TYPE IS (particle_lwf)
        CALL prepare_melting_lwf(kstart,kend, atmo, gmelting)
        CALL particle_melting_lwf(kstart,kend, dt, graupel, rain, gmelting)
      END SELECT
      SELECT TYPE (hail)
      TYPE IS (particle_frozen)
        CALL hail_melting_simple(kstart,kend,dt,hail_coeffs,atmo,hail,rain)
      TYPE IS (particle_lwf)
        CALL particle_melting_lwf(kstart,kend, dt, hail, rain, gmelting)
      END SELECT
      IF (ischeck) CALL check(kstart,kend, 'melting',cloud,rain,ice,snow,graupel,hail)

      ! evaporation from melting ice particles
      CALL evaporation(kstart,kend, dt, atmo, snow, snow_coeffs)
      CALL evaporation(kstart,kend, dt, atmo, graupel, graupel_coeffs)
      CALL evaporation(kstart,kend, dt, atmo, hail, hail_coeffs)
      IF (ischeck) CALL check(kstart,kend, 'evaporation of ice',cloud,rain,ice,snow,graupel,hail)

    END IF ! cfg_params%iicephase


    ! warm rain processes
    ! (using something other than SB is somewhat inconsistent and not recommended)
    IF (auto_typ == 1) THEN
       CALL autoconversionKB(kstart,kend, dt, cloud, rain)   ! Beheng (1994)
       CALL accretionKB(kstart,kend, dt, cloud, rain)
       CALL rain_selfcollectionSB(kstart,kend, dt, atmo, rain)
    ELSE IF (auto_typ == 2) THEN
       ! Khairoutdinov and Kogan (2000)
       ! (KK2000 originally assume a 25 micron size threshold)
       CALL autoconversionKK(kstart,kend, dt, cloud, rain)
       CALL accretionKK(kstart,kend, dt, cloud, rain)
       CALL rain_selfcollectionSB(kstart,kend, dt, atmo, rain)
    ELSE IF (auto_typ == 3) THEN
       CALL autoconversionSB(kstart,kend, dt, atmo, cloud_coeffs, cloud, rain) 
       CALL accretionSB(kstart,kend, dt, atmo, cloud, rain)
       CALL rain_selfcollectionSB(kstart,kend, dt, atmo, rain)
    ENDIF
    IF (ischeck) CALL check(kstart,kend,'warm rain',cloud,rain,ice,snow,graupel,hail)

    ! evaporation of rain following Seifert (2008)
    CALL rain_evaporation(kstart,kend, dt, rain_coeffs, rain_gfak, atmo, cloud, rain)

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
    REAL(wp) :: q_r,x_r,q_c,vn_rain_min, vq_rain_min, vn_rain_max, vq_rain_max, rhocorr
    REAL(wp) :: nu, mu, x_s_i

    rhocorr = 1.0_wp
    
    CALL init_2mom_scheme(cloud,rain,ice,snow,graupel,hail)

    ice_typ   = cloud_type/1000           ! (0) no ice, (1) no hail (2) with hail
    nuc_i_typ = MOD(cloud_type/100,10)    ! choice of ice nucleation, see ice_nucleation_homhet()
    nuc_c_typ = MOD(cloud_type/10,10)     ! choice of CCN assumptions, see cloud_nucleation()
    auto_typ  = MOD(cloud_type,10)        ! choice of warm rain scheme, see clouds_twomoment()

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

    CALL message(TRIM(routine), "calculate run-time coefficients")
    WRITE (txt,'(A,I10)') "  cloud_type = ",cloud_type ; CALL message(routine,TRIM(txt))
    WRITE (txt,'(2A)') "     cloud   = ",cloud%name    ; CALL message(routine,TRIM(txt))
    WRITE (txt,'(2A)') "     rain    = ",rain%name     ; CALL message(routine,TRIM(txt))
    WRITE (txt,'(2A)') "     ice     = ",ice%name      ; CALL message(routine,TRIM(txt))
    WRITE (txt,'(2A)') "     snow    = ",snow%name     ; CALL message(routine,TRIM(txt))
    WRITE (txt,'(2A)') "     graupel = ",graupel%name  ; CALL message(routine,TRIM(txt))
    WRITE (txt,'(2A)') "     hail    = ",hail%name     ; CALL message(routine,TRIM(txt))

    ! initialize bulk sedimentation velocities
    ! calculates coeff_alfa_n, coeff_alfa_q, and coeff_lambda
    call init_2mom_sedi_vel(ice,ice_coeffs)
    call init_2mom_sedi_vel(snow,snow_coeffs)
    call init_2mom_sedi_vel(graupel,graupel_coeffs)
    call init_2mom_sedi_vel(hail,hail_coeffs)

    ! look-up table and parameters for rain_freeze_gamlook
    rain_nm1 = (rain%nu+1.0)/rain%mu
    rain_nm2 = (rain%nu+2.0)/rain%mu
    rain_nm3 = (rain%nu+3.0)/rain%mu
    CALL incgfct_lower_lookupcreate(rain_nm1, rain_ltable1, nlookup, nlookuphr_dummy)
    CALL incgfct_lower_lookupcreate(rain_nm2, rain_ltable2, nlookup, nlookuphr_dummy)
    CALL incgfct_lower_lookupcreate(rain_nm3, rain_ltable3, nlookup, nlookuphr_dummy)
    rain_g1 = rain_ltable1%igf(rain_ltable1%n) ! ordinary gamma function of nm1 is the last value in table 1
    rain_g2 = rain_ltable2%igf(rain_ltable2%n) ! ordinary gamma function of nm2 is the last value in table 2

    ! tables and parameters for graupel_hail_conv_wet_gamlook
    graupel_nm1 = (graupel%nu+1.0)/graupel%mu
    graupel_nm2 = (graupel%nu+2.0)/graupel%mu
    CALL incgfct_lower_lookupcreate(graupel_nm1, graupel_ltable1, nlookup, nlookuphr_dummy)
    CALL incgfct_lower_lookupcreate(graupel_nm2, graupel_ltable2, nlookup, nlookuphr_dummy)
    graupel_g1 = graupel_ltable1%igf(graupel_ltable1%n) ! ordinary gamma function of nm1 is the last value in table 1
    graupel_g2 = graupel_ltable2%igf(graupel_ltable2%n) ! ordinary gamma function of nm2 is the last value in table 2

    
    ! .. tables and parameters for graupel shedding during cloud riming:
    !    graupel: partial moment; cloud: full moment
    CALL setup_particle_coll_pm_type1_bfull(graupel,cloud,gshedc_coeffs)
    !    for delta_gg-part:
    CALL incgfct_lower_lookupcreate(gshedc_coeffs%moma(0,3), gshedc_ltab_dgg_03, nlookup, nlookuphr_dummy)
    !    for delta_gr-part:
    CALL incgfct_lower_lookupcreate(gshedc_coeffs%moma(0,2), gshedc_ltab_dgr_02, nlookup, nlookuphr_dummy)
    !    for delta_rr-part:
    CALL incgfct_lower_lookupcreate(gshedc_coeffs%moma(0,1), gshedc_ltab_drr_01, nlookup, nlookuphr_dummy)
    !    for theta_gg-part:
    CALL incgfct_lower_lookupcreate(gshedc_coeffs%moma(0,5), gshedc_ltab_tgg_05, nlookup, nlookuphr_dummy)
    CALL incgfct_lower_lookupcreate(gshedc_coeffs%moma(0,3), gshedc_ltab_tgg_03, nlookup, nlookuphr_dummy)
    !    for theta_gr-part:
    CALL incgfct_lower_lookupcreate(gshedc_coeffs%moma(0,4), gshedc_ltab_tgr_04, nlookup, nlookuphr_dummy)
   
    ! .. tables and parameters for hail shedding during cloud riming:
    !    hail: partial moment; cloud: full moment
    CALL setup_particle_coll_pm_type1_bfull(hail,cloud,hshedc_coeffs)
    !    for delta_hh-part:
    CALL incgfct_lower_lookupcreate(hshedc_coeffs%moma(0,3), hshedc_ltab_dhh_03, nlookup, nlookuphr_dummy)
    !    for delta_hr-part:
    CALL incgfct_lower_lookupcreate(hshedc_coeffs%moma(0,2), hshedc_ltab_dhr_02, nlookup, nlookuphr_dummy)
    !    for delta_rr-part:
    CALL incgfct_lower_lookupcreate(hshedc_coeffs%moma(0,1), hshedc_ltab_drr_01, nlookup, nlookuphr_dummy)
    !    for theta_hh-part:
    CALL incgfct_lower_lookupcreate(hshedc_coeffs%moma(0,5), hshedc_ltab_thh_05, nlookup, nlookuphr_dummy)
    CALL incgfct_lower_lookupcreate(hshedc_coeffs%moma(0,3), hshedc_ltab_thh_03, nlookup, nlookuphr_dummy)
    !    for theta_hr-part:
    CALL incgfct_lower_lookupcreate(hshedc_coeffs%moma(0,4), hshedc_ltab_thr_04, nlookup, nlookuphr_dummy)


    ! .. tables and parameters for graupel shedding during rain riming:
    !    graupel: partial moment; rain: full moment
    CALL setup_particle_coll_pm_type1_bfull(graupel,rain,gshedr_coeffs)
    !    for delta_gg-part:
    CALL incgfct_lower_lookupcreate(gshedr_coeffs%moma(0,3), gshedr_ltab_dgg_03, nlookup, nlookuphr_dummy)
    !    for delta_gr-part:
    CALL incgfct_lower_lookupcreate(gshedr_coeffs%moma(0,2), gshedr_ltab_dgr_02, nlookup, nlookuphr_dummy)
    !    for delta_rr-part:
    CALL incgfct_lower_lookupcreate(gshedr_coeffs%moma(0,1), gshedr_ltab_drr_01, nlookup, nlookuphr_dummy)
    !    for theta_gg-part:
    CALL incgfct_lower_lookupcreate(gshedr_coeffs%moma(0,5), gshedr_ltab_tgg_05, nlookup, nlookuphr_dummy)
    CALL incgfct_lower_lookupcreate(gshedr_coeffs%moma(0,3), gshedr_ltab_tgg_03, nlookup, nlookuphr_dummy)
    !    for theta_gr-part:
    CALL incgfct_lower_lookupcreate(gshedr_coeffs%moma(0,4), gshedr_ltab_tgr_04, nlookup, nlookuphr_dummy)
   
    ! .. tables and parameters for hail shedding during rain riming:
    !    hail: partial moment; rain: full moment
    CALL setup_particle_coll_pm_type1_bfull(hail,rain,hshedr_coeffs)
    !    for delta_hh-part:
    CALL incgfct_lower_lookupcreate(hshedr_coeffs%moma(0,3), hshedr_ltab_dhh_03, nlookup, nlookuphr_dummy)
    !    for delta_hr-part:
    CALL incgfct_lower_lookupcreate(hshedr_coeffs%moma(0,2), hshedr_ltab_dhr_02, nlookup, nlookuphr_dummy)
    !    for delta_rr-part:
    CALL incgfct_lower_lookupcreate(hshedr_coeffs%moma(0,1), hshedr_ltab_drr_01, nlookup, nlookuphr_dummy)
    !    for theta_hh-part:
    CALL incgfct_lower_lookupcreate(hshedr_coeffs%moma(0,5), hshedr_ltab_thh_05, nlookup, nlookuphr_dummy)
    CALL incgfct_lower_lookupcreate(hshedr_coeffs%moma(0,3), hshedr_ltab_thh_03, nlookup, nlookuphr_dummy)
    !    for theta_hr-part:
    CALL incgfct_lower_lookupcreate(hshedr_coeffs%moma(0,4), hshedr_ltab_thr_04, nlookup, nlookuphr_dummy)


    ! .. Lookup tables for sticking efficiencies:
    CALL init_estick_ltab_equi (ltab_estick_ice,   cfg_params%iice_stick,   'estick_cloudice')
    CALL init_estick_ltab_equi (ltab_estick_snow,  cfg_params%isnow_stick,  'estick_snow')
    CALL init_estick_ltab_equi (ltab_estick_parti, cfg_params%iparti_stick, 'estick_inter-categorical')
    
    ! other options for mue-D-relation of raindrops (for sensitivity studies)
    IF (mu_Dm_rain_typ.EQ.0) THEN
      !..constant mue value
      rain_coeffs%cmu0 = 0.0
      rain_coeffs%cmu1 = 0.0
      rain_coeffs%cmu2 = 1.0
      rain_coeffs%cmu3 = 1.0
      rain_coeffs%cmu4 = (rain%nu+1.0_wp)/rain%b_geo - 1.0_wp ! <-- this is the (constant) mue value
      rain_coeffs%cmu5 = 1
      rain_gfak = -1.0  ! In this case gamma = 1 in rain_evaporation
    ELSEIF (mu_Dm_rain_typ.EQ.1) THEN
      !..Axel's mu-Dm-relation for raindrops based on 1d-bin model
      !  (this is the default and the cmus are set in the particle constructor)
      rain_gfak = 1.0
    ELSEIF (mu_Dm_rain_typ.EQ.2) THEN
      !..Modification of mu-Dm-relation for experiments with increased evaporation
      rain_coeffs%cmu0 = 11.0            ! instead of 6.0
      rain_coeffs%cmu1 = 30.0            !
      rain_coeffs%cmu2 = 1.00d+3         !
      rain_coeffs%cmu3 = 1.10d-3         !
      rain_coeffs%cmu4 = 4.0             ! instead of 1.0
      rain_coeffs%cmu5 = 2               !
      rain_gfak = 0.5             ! instead of 1.0
    ELSEIF (mu_Dm_rain_typ.EQ.3) THEN
      !..Jason Milbrandts mu-Dm-relation'
      rain_coeffs%cmu0 = 19.0           !
      rain_coeffs%cmu1 = 19.0           ! Jason Milbrandt's mu-Dm-relation for rain_coeffs
      rain_coeffs%cmu2 = 0.60d+3        ! (Milbrandt&Yau 2005, JAS, Table 1)
      rain_coeffs%cmu3 = 1.80d-3        !
      rain_coeffs%cmu4 = 17.0           !
      rain_coeffs%cmu5 = 1              !
      rain_gfak = -1.0  ! In this case gamma = 1 in rain_evaporation
    ENDIF

    IF (isprint) THEN
      CALL message(TRIM(routine), "init_2mom_scheme: rain coeffs and sedi vel")
      q_r = 1.0e-3_wp
      WRITE (txt,'(2A)') "    name  = ",rain%name ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     alfa  = ",rain_coeffs%alfa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     beta  = ",rain_coeffs%beta ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     gama  = ",rain_coeffs%gama ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     cmu0  = ",rain_coeffs%cmu0 ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     cmu1  = ",rain_coeffs%cmu1 ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     cmu2  = ",rain_coeffs%cmu2 ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     cmu3  = ",rain_coeffs%cmu3 ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     cmu4  = ",rain_coeffs%cmu4 ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,I10)')   "     cmu5  = ",rain_coeffs%cmu5 ; CALL message(routine,TRIM(txt))
      x_r = rain%x_min ; CALL sedi_vel_rain(rain,rain_coeffs,q_r,x_r,rhocorr,vn_rain_min,vq_rain_min)
      x_r = rain%x_max ; CALL sedi_vel_rain(rain,rain_coeffs,q_r,x_r,rhocorr,vn_rain_max,vq_rain_max)
      WRITE(txt,'(A)')       "    out-of-cloud: " ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     vn_rain_min  = ",vn_rain_min ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     vn_rain_max  = ",vn_rain_max ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     vq_rain_min  = ",vq_rain_min ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     vq_rain_max  = ",vq_rain_max ; CALL message(routine,TRIM(txt))
      q_c = 1e-3_wp
      x_r = rain%x_min ; CALL sedi_vel_rain(rain,rain_coeffs,q_r,x_r,rhocorr,vn_rain_min,vq_rain_min,q_c)
      x_r = rain%x_max ; CALL sedi_vel_rain(rain,rain_coeffs,q_r,x_r,rhocorr,vn_rain_max,vq_rain_max,q_c)
      WRITE(txt,'(A)')       "    in-cloud: " ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     vn_rain_min  = ",vn_rain_min ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     vn_rain_max  = ",vn_rain_max ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     vq_rain_min  = ",vq_rain_min ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     vq_rain_max  = ",vq_rain_max ; CALL message(routine,TRIM(txt))
    END IF

    ! initialization for snow_cloud_riming
    CALL setup_particle_collection_type1(snow,cloud,scr_coeffs)

    IF (isprint) THEN
      WRITE(txt,'(A,ES14.7)') "   a_snow      = ",snow%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   b_snow      = ",snow%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   alf_snow    = ",snow%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   bet_snow    = ",snow%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   a_cloud    = ",cloud%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   b_cloud    = ",cloud%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   alf_cloud  = ",cloud%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   bet_cloud  = ",cloud%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_n_ss = ",scr_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_n_sc = ",scr_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_n_cc = ",scr_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_n_ss = ",scr_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_n_sc = ",scr_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_n_cc = ",scr_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_q_ss = ",scr_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_q_sc = ",scr_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_q_cc = ",scr_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_q_ss = ",scr_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_q_sc = ",scr_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_q_cc = ",scr_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! coefficients for snow_rain_riming
    CALL setup_particle_collection_type2(snow,rain,srr_coeffs)

    IF (isprint) THEN
      WRITE(txt,'(A,ES14.7)') "    a_snow     = ",snow%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_snow     = ",snow%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    alf_snow   = ",snow%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    bet_snow   = ",snow%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    a_rain     = ",rain%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_rain     = ",rain%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    alf_rain   = ",rain%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    bet_rain   = ",rain%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_ss = ",srr_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_sr = ",srr_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_rr = ",srr_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_ss = ",srr_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_sr = ",srr_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_rr = ",srr_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_ss = ",srr_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_sr = ",srr_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_rs = ",srr_coeffs%delta_q_ba ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_rr = ",srr_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_ss = ",srr_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_sr = ",srr_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_rs = ",srr_coeffs%theta_q_ba ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_rr = ",srr_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! ice rain riming parameters
    CALL setup_particle_collection_type2(ice,rain,irr_coeffs)

    IF (isprint) THEN
      WRITE(txt,'(A,ES14.7)') "     a_ice      = ",ice%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     b_ice      = ",ice%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     alf_ice    = ",ice%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     bet_ice    = ",ice%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     a_rain    = ",rain%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     b_rain    = ",rain%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     alf_rain  = ",rain%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     bet_rain  = ",rain%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_n_ii = ", irr_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_n_ir = ", irr_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_n_rr = ", irr_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_n_ii = ", irr_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_n_ir = ", irr_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_n_rr = ", irr_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_ii = ", irr_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_ir = ", irr_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_ri = ", irr_coeffs%delta_q_ba ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_rr = ", irr_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_ii = ", irr_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_ir = ", irr_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_ri = ", irr_coeffs%theta_q_ba ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_rr = ", irr_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! ice cloud riming parameter setup
    CALL setup_particle_collection_type1(ice,cloud,icr_coeffs)

    IF (isprint) THEN
      WRITE(txt,'(A,ES14.7)') "    a_ice      = ",ice%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_ice      = ",ice%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    alf_ice    = ",ice%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    bet_ice    = ",ice%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    a_cloud    = ",cloud%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_cloud    = ",cloud%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    alf_cloud  = ",cloud%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    bet_cloud  = ",cloud%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_ii = ", icr_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_ic = ", icr_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_cc = ", icr_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_ii = ", icr_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_ic = ", icr_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_cc = ", icr_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_ii = ", icr_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_ic = ", icr_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_cc = ", icr_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_ii = ", icr_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_ic = ", icr_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_cc = ", icr_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! hail rain riming
    CALL setup_particle_collection_type1(hail,rain,hrr_coeffs)

    IF (isprint) THEN
      WRITE(txt,*) " hail_rain_riming:" ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     a_hail     = ",hail%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     b_hail     = ",hail%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     alf_hail   = ",hail%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     bet_hail   = ",hail%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     a_rain     = ",rain%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     b_rain     = ",rain%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     alf_rain   = ",rain%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     bet_rain   = ",rain%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_n_hh = ",hrr_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_n_hr = ",hrr_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_n_rr = ",hrr_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_n_hh = ",hrr_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_n_hr = ",hrr_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_n_rr = ",hrr_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_hh = ",hrr_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_hr = ",hrr_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_rr = ",hrr_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_hh = ",hrr_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_hr = ",hrr_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_rr = ",hrr_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! graupel rain riming parameter setup
    CALL setup_particle_collection_type1(graupel,rain,grr_coeffs)

    IF (isprint) THEN
      WRITE(txt,'(A,ES14.7)') "     delta_n_gg = ",grr_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_n_gr = ",grr_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_n_rr = ",grr_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_n_gg = ",grr_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_n_gr = ",grr_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_n_rr = ",grr_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_gg = ",grr_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_gr = ",grr_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_rr = ",grr_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_gg = ",grr_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_gr = ",grr_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_rr = ",grr_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! hail cloud riming parameter setup
    CALL setup_particle_collection_type1(hail,cloud,hcr_coeffs)

    IF (isprint) THEN
      WRITE(txt,'(A,ES14.7)') "     delta_n_hh  = ", hcr_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_n_hc  = ", hcr_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_n_cc  = ", hcr_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_n_hh  = ", hcr_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_n_hc  = ", hcr_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_n_cc  = ", hcr_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_hh  = ", hcr_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_hc  = ", hcr_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     delta_q_cc  = ", hcr_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_hh  = ", hcr_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_hc  = ", hcr_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "     theta_q_cc  = ", hcr_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! graupel cloud riming parameters
    CALL setup_particle_collection_type1(graupel,cloud,gcr_coeffs)

    IF (isprint) THEN
      WRITE(txt,'(A,ES14.7)') "    delta_n_gg  = ", gcr_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_gc  = ", gcr_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_cc  = ", gcr_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_gg  = ", gcr_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_gc  = ", gcr_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_cc  = ", gcr_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_gg  = ", gcr_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_gc  = ", gcr_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_cc  = ", gcr_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_gg  = ", gcr_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_gc  = ", gcr_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_cc  = ", gcr_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! snow ice collection parameters setup
    CALL setup_particle_collection_type1(snow,ice,sic_coeffs)

    IF (isprint) THEN
      WRITE(txt,'(A,ES14.7)') "   delta_n_ss = ", sic_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_n_si = ", sic_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_n_ii = ", sic_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_n_ss = ", sic_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_n_si = ", sic_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_n_ii = ", sic_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_q_ss = ", sic_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_q_si = ", sic_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_q_ii = ", sic_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_q_ss = ", sic_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_q_si = ", sic_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_q_ii = ", sic_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! hail ice collection parameter setup
    CALL setup_particle_collection_type1(hail,ice,hic_coeffs)

    IF (isprint) THEN
      WRITE(txt,'(A,ES14.7)') "    delta_n_hh = ", hic_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_hi = ", hic_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_ii = ", hic_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_hh = ", hic_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_hi = ", hic_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_ii = ", hic_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_hh = ", hic_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_hi = ", hic_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_ii = ", hic_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_hh = ", hic_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_hi = ", hic_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_ii = ", hic_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! graupel ice collection parameter setup
    CALL setup_particle_collection_type1(graupel,ice,gic_coeffs)

    IF (isprint) THEN
      WRITE(txt,'(A,ES14.7)') "   a_graupel   = ",graupel%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   b_graupel   = ",graupel%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   alf_graupel = ",graupel%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   bet_graupel = ",graupel%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   a_ice       = ",ice%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   b_ice       = ",ice%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   alf_ice     = ",ice%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   bet_ice     = ",ice%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_n_gg  = ", gic_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_n_gi  = ", gic_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_n_ii  = ", gic_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_n_gg  = ", gic_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_n_gi  = ", gic_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_n_ii  = ", gic_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_q_gg  = ", gic_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_q_gi  = ", gic_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   delta_q_ii  = ", gic_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_q_gg  = ", gic_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_q_gi  = ", gic_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "   theta_q_ii  = ", gic_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! hail snow collection parameter setup
    CALL setup_particle_collection_type1(hail,snow,hsc_coeffs)

    IF (isprint) THEN
      WRITE(txt,'(A,ES14.7)') "    delta_n_hh = ", hsc_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_hs = ", hsc_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_ss = ", hsc_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_hh = ", hsc_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_hs = ", hsc_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_ss = ", hsc_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_hh = ", hsc_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_hs = ", hsc_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_ss = ", hsc_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_hh = ", hsc_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_hs = ", hsc_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_ss = ", hsc_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! graupel snow collection parameter setup
    CALL setup_particle_collection_type1(graupel,snow,gsc_coeffs)

    IF (isprint) THEN
      WRITE(txt,'(A,ES14.7)') "    delta_n_gg = ", gsc_coeffs%delta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_gs = ", gsc_coeffs%delta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_n_ss = ", gsc_coeffs%delta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_gg = ", gsc_coeffs%theta_n_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_gs = ", gsc_coeffs%theta_n_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_n_ss = ", gsc_coeffs%theta_n_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_gg = ", gsc_coeffs%delta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_gs = ", gsc_coeffs%delta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    delta_q_ss = ", gsc_coeffs%delta_q_bb ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_gg = ", gsc_coeffs%theta_q_aa ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_gs = ", gsc_coeffs%theta_q_ab ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    theta_q_ss = ", gsc_coeffs%theta_q_bb ; CALL message(routine,TRIM(txt))
    END IF

    ! ice coeffs
    CALL setup_particle_coeffs(ice,ice_coeffs)
    IF (isprint) THEN
      WRITE(txt,*) "  ice: " ; CALL message(routine,TRIM(txt))
      WRITE (txt,'(A,ES14.7)') "    a_geo   = ",ice%a_geo ; CALL message(routine,TRIM(txt))
      WRITE (txt,'(A,ES14.7)') "    b_geo   = ",ice%b_geo ; CALL message(routine,TRIM(txt))
      WRITE (txt,'(A,ES14.7)') "    a_vel   = ",ice%a_vel ; CALL message(routine,TRIM(txt))
      WRITE (txt,'(A,ES14.7)') "    b_vel   = ",ice%b_vel ; CALL message(routine,TRIM(txt))
      WRITE (txt,'(A,ES14.7)') "    c_i     = ",ice_coeffs%c_i ; CALL message(routine,TRIM(txt))
      WRITE (txt,'(A,ES14.7)') "    a_f     = ",ice_coeffs%a_f ; CALL message(routine,TRIM(txt))
      WRITE (txt,'(A,ES14.7)') "    b_f     = ",ice_coeffs%b_f ; CALL message(routine,TRIM(txt))
    END IF

    ! graupel parameter setup
    CALL setup_particle_coeffs(graupel,graupel_coeffs)
    IF (isprint) THEN
      WRITE(txt,*) "  graupel: " ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    a_geo = ",graupel%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_geo = ",graupel%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    a_vel = ",graupel%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_vel = ",graupel%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    c_g   = ",graupel_coeffs%c_i ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    a_f   = ",graupel_coeffs%a_f ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_f   = ",graupel_coeffs%b_f ; CALL message(routine,TRIM(txt))
    END IF

    ! hail parameter setup
    CALL setup_particle_coeffs(hail,hail_coeffs)
    IF (isprint) THEN
      WRITE(txt,*) "  hail: " ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    a_geo = ",hail%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_geo = ",hail%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    a_vel = ",hail%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_vel = ",hail%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    c_h   = ",hail_coeffs%c_i ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    a_f   = ",hail_coeffs%a_f ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_f   = ",hail_coeffs%b_f ; CALL message(routine,TRIM(txt))
    END IF

    ! snow parameter setup
    CALL setup_particle_coeffs(snow,snow_coeffs)
    IF (isprint) THEN
      WRITE(txt,*) "  snow: " ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    a_geo = ",snow%a_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_geo = ",snow%b_geo ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    a_vel = ",snow%a_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_vel = ",snow%b_vel ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    c_s   = ",snow_coeffs%c_i ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    a_f   = ",snow_coeffs%a_f ; CALL message(routine,TRIM(txt))
      WRITE(txt,'(A,ES14.7)') "    b_f   = ",snow_coeffs%b_f ; CALL message(routine,TRIM(txt))
    END IF

    ! setup selfcollection of ice particles, coeffs are stored in their derived types
    CALL setup_graupel_selfcollection(graupel,graupel_coeffs)
    CALL setup_snow_selfcollection(snow,snow_coeffs)
    CALL setup_ice_selfcollection(ice,ice_coeffs)

    ! setup run-time coeffs for cloud, e.g., used in cloud_freeze and autoconversionSB
    CALL setup_particle_coeffs(cloud,cloud_coeffs)
    CALL setup_cloud_autoconversion(cloud,cloud_coeffs)

    IF (isprint) THEN
      CALL message(routine, "rain_coeffs:")
      WRITE(txt,'(A,ES14.7)') "    c_z= ",rain_coeffs%c_z
      CALL message(routine,TRIM(txt))
    ENDIF
    IF (isprint) THEN
      CALL message(routine, "cloud_coeffs:")
      WRITE(txt,'(A,ES14.7)') "    c_z= ",cloud_coeffs%c_z
      CALL message(routine,TRIM(txt))
    ENDIF

    ! Init SK Activation table
    IF (nuc_c_typ > 5) THEN
      call ccn_activation_sk_4d()
      IF (isprint) THEN
        CALL message(routine,"Equidistant lookup table for Segal-Khain created")
      ENDIF
    END IF

    !$ACC ENTER DATA COPYIN(rain_coeffs, ice_coeffs, snow_coeffs, graupel_coeffs, hail_coeffs, cloud_coeffs) &
    !$ACC   COPYIN(sic_coeffs, gic_coeffs, gsc_coeffs, hic_coeffs, hsc_coeffs, scr_coeffs, srr_coeffs) &
    !$ACC   COPYIN(irr_coeffs, icr_coeffs, hrr_coeffs, grr_coeffs, hcr_coeffs, gcr_coeffs) &
    !$ACC   COPYIN(gshedc_coeffs, gshedr_coeffs, hshedc_coeffs, hshedr_coeffs)

    !$ACC ENTER DATA COPYIN(ltab_estick_ice)
    !$ACC ENTER DATA COPYIN(ltab_estick_ice%x1, ltab_estick_ice%ltable)
    !$ACC ENTER DATA COPYIN(ltab_estick_snow)
    !$ACC ENTER DATA COPYIN(ltab_estick_snow%x1, ltab_estick_snow%ltable)
    !$ACC ENTER DATA COPYIN(ltab_estick_parti)
    !$ACC ENTER DATA COPYIN(ltab_estick_parti%x1, ltab_estick_parti%ltable)

    !$ACC ENTER DATA COPYIN(graupel_ltable1)
    !$ACC ENTER DATA COPYIN(graupel_ltable1%x, graupel_ltable1%xhr, graupel_ltable1%igf, graupel_ltable1%igfhr)
    !$ACC ENTER DATA COPYIN(graupel_ltable2)
    !$ACC ENTER DATA COPYIN(graupel_ltable2%x, graupel_ltable2%xhr, graupel_ltable2%igf, graupel_ltable2%igfhr)

    !$ACC ENTER DATA COPYIN(rain_ltable1)
    !$ACC ENTER DATA COPYIN(rain_ltable1%x, rain_ltable1%xhr, rain_ltable1%igf, rain_ltable1%igfhr)
    !$ACC ENTER DATA COPYIN(rain_ltable2)
    !$ACC ENTER DATA COPYIN(rain_ltable2%x, rain_ltable2%xhr, rain_ltable2%igf, rain_ltable2%igfhr)
    !$ACC ENTER DATA COPYIN(rain_ltable3)
    !$ACC ENTER DATA COPYIN(rain_ltable3%x, rain_ltable3%xhr, rain_ltable3%igf, rain_ltable3%igfhr)
    
    !$ACC ENTER DATA COPYIN(gshedc_ltab_dgg_03)
    !$ACC ENTER DATA COPYIN(gshedc_ltab_dgg_03%x, gshedc_ltab_dgg_03%xhr, gshedc_ltab_dgg_03%igf, gshedc_ltab_dgg_03%igfhr)
    !$ACC ENTER DATA COPYIN(gshedc_ltab_drr_01)
    !$ACC ENTER DATA COPYIN(gshedc_ltab_drr_01%x, gshedc_ltab_drr_01%xhr, gshedc_ltab_drr_01%igf, gshedc_ltab_drr_01%igfhr)
    !$ACC ENTER DATA COPYIN(gshedc_ltab_dgr_02)
    !$ACC ENTER DATA COPYIN(gshedc_ltab_dgr_02%x, gshedc_ltab_dgr_02%xhr, gshedc_ltab_dgr_02%igf, gshedc_ltab_dgr_02%igfhr)
    !$ACC ENTER DATA COPYIN(gshedc_ltab_tgg_05)
    !$ACC ENTER DATA COPYIN(gshedc_ltab_tgg_05%x, gshedc_ltab_tgg_05%xhr, gshedc_ltab_tgg_05%igf, gshedc_ltab_tgg_05%igfhr)
    !$ACC ENTER DATA COPYIN(gshedc_ltab_tgg_03)
    !$ACC ENTER DATA COPYIN(gshedc_ltab_tgg_03%x, gshedc_ltab_tgg_03%xhr, gshedc_ltab_tgg_03%igf, gshedc_ltab_tgg_03%igfhr)
    !$ACC ENTER DATA COPYIN(gshedc_ltab_tgr_04)
    !$ACC ENTER DATA COPYIN(gshedc_ltab_tgr_04%x, gshedc_ltab_tgr_04%xhr, gshedc_ltab_tgr_04%igf, gshedc_ltab_tgr_04%igfhr)

    !$ACC ENTER DATA COPYIN(hshedc_ltab_dhh_03)
    !$ACC ENTER DATA COPYIN(hshedc_ltab_dhh_03%x, hshedc_ltab_dhh_03%xhr, hshedc_ltab_dhh_03%igf, hshedc_ltab_dhh_03%igfhr)
    !$ACC ENTER DATA COPYIN(hshedc_ltab_drr_01)
    !$ACC ENTER DATA COPYIN(hshedc_ltab_drr_01%x, hshedc_ltab_drr_01%xhr, hshedc_ltab_drr_01%igf, hshedc_ltab_drr_01%igfhr)
    !$ACC ENTER DATA COPYIN(hshedc_ltab_dhr_02)
    !$ACC ENTER DATA COPYIN(hshedc_ltab_dhr_02%x, hshedc_ltab_dhr_02%xhr, hshedc_ltab_dhr_02%igf, hshedc_ltab_dhr_02%igfhr)
    !$ACC ENTER DATA COPYIN(hshedc_ltab_thh_05)
    !$ACC ENTER DATA COPYIN(hshedc_ltab_thh_05%x, hshedc_ltab_thh_05%xhr, hshedc_ltab_thh_05%igf, hshedc_ltab_thh_05%igfhr)
    !$ACC ENTER DATA COPYIN(hshedc_ltab_thh_03)
    !$ACC ENTER DATA COPYIN(hshedc_ltab_thh_03%x, hshedc_ltab_thh_03%xhr, hshedc_ltab_thh_03%igf, hshedc_ltab_thh_03%igfhr)
    !$ACC ENTER DATA COPYIN(hshedc_ltab_thr_04)
    !$ACC ENTER DATA COPYIN(hshedc_ltab_thr_04%x, hshedc_ltab_thr_04%xhr, hshedc_ltab_thr_04%igf, hshedc_ltab_thr_04%igfhr)

    !$ACC ENTER DATA COPYIN(gshedr_ltab_dgg_03)
    !$ACC ENTER DATA COPYIN(gshedr_ltab_dgg_03%x, gshedr_ltab_dgg_03%xhr, gshedr_ltab_dgg_03%igf, gshedr_ltab_dgg_03%igfhr)
    !$ACC ENTER DATA COPYIN(gshedr_ltab_drr_01)
    !$ACC ENTER DATA COPYIN(gshedr_ltab_drr_01%x, gshedr_ltab_drr_01%xhr, gshedr_ltab_drr_01%igf, gshedr_ltab_drr_01%igfhr)
    !$ACC ENTER DATA COPYIN(gshedr_ltab_dgr_02)
    !$ACC ENTER DATA COPYIN(gshedr_ltab_dgr_02%x, gshedr_ltab_dgr_02%xhr, gshedr_ltab_dgr_02%igf, gshedr_ltab_dgr_02%igfhr)
    !$ACC ENTER DATA COPYIN(gshedr_ltab_tgg_05)
    !$ACC ENTER DATA COPYIN(gshedr_ltab_tgg_05%x, gshedr_ltab_tgg_05%xhr, gshedr_ltab_tgg_05%igf, gshedr_ltab_tgg_05%igfhr)
    !$ACC ENTER DATA COPYIN(gshedr_ltab_tgg_03)
    !$ACC ENTER DATA COPYIN(gshedr_ltab_tgg_03%x, gshedr_ltab_tgg_03%xhr, gshedr_ltab_tgg_03%igf, gshedr_ltab_tgg_03%igfhr)
    !$ACC ENTER DATA COPYIN(gshedr_ltab_tgr_04)
    !$ACC ENTER DATA COPYIN(gshedr_ltab_tgr_04%x, gshedr_ltab_tgr_04%xhr, gshedr_ltab_tgr_04%igf, gshedr_ltab_tgr_04%igfhr)

    !$ACC ENTER DATA COPYIN(hshedr_ltab_dhh_03)
    !$ACC ENTER DATA COPYIN(hshedr_ltab_dhh_03%x, hshedr_ltab_dhh_03%xhr, hshedr_ltab_dhh_03%igf, hshedr_ltab_dhh_03%igfhr)
    !$ACC ENTER DATA COPYIN(hshedr_ltab_drr_01)
    !$ACC ENTER DATA COPYIN(hshedr_ltab_drr_01%x, hshedr_ltab_drr_01%xhr, hshedr_ltab_drr_01%igf, hshedr_ltab_drr_01%igfhr)
    !$ACC ENTER DATA COPYIN(hshedr_ltab_dhr_02)
    !$ACC ENTER DATA COPYIN(hshedr_ltab_dhr_02%x, hshedr_ltab_dhr_02%xhr, hshedr_ltab_dhr_02%igf, hshedr_ltab_dhr_02%igfhr)
    !$ACC ENTER DATA COPYIN(hshedr_ltab_thh_05)
    !$ACC ENTER DATA COPYIN(hshedr_ltab_thh_05%x, hshedr_ltab_thh_05%xhr, hshedr_ltab_thh_05%igf, hshedr_ltab_thh_05%igfhr)
    !$ACC ENTER DATA COPYIN(hshedr_ltab_thh_03)
    !$ACC ENTER DATA COPYIN(hshedr_ltab_thh_03%x, hshedr_ltab_thh_03%xhr, hshedr_ltab_thh_03%igf, hshedr_ltab_thh_03%igfhr)
    !$ACC ENTER DATA COPYIN(hshedr_ltab_thr_04)
    !$ACC ENTER DATA COPYIN(hshedr_ltab_thr_04%x, hshedr_ltab_thr_04%xhr, hshedr_ltab_thr_04%igf, hshedr_ltab_thr_04%igfhr)

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
