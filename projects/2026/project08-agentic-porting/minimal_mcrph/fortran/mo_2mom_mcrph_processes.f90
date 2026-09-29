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
! Provides various modules and subroutines for two-moment bulk microphysics

!NEC$ options "-finline-max-depth=3 -finline-max-function-size=2000"

MODULE mo_2mom_mcrph_processes

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
! Re-write of sedimentation schemes 03/2019 by UB:
! - Technical re-write of sedi_icon_core() overtaken from COSMO src_twomom_sb.f90:
!   - sedi_icon_core() vectorized version, in principle reproducible on the CRAY
!     but only used #if defined (__SX__) || defined (__NEC_VH__) || defined (__NECSX__)
!   - scalar version sedi_icon_core() for scalar architectures
!   - sedi_icon_core_lwf() including liquid water fraction
! - New internal switches "lboxtracking=.true.", activating the new explicit
!   boxtracking sedimentation method from COSMO src_twomom_f90
!   (http://www.cosmo-model.org/content/model/documentation/core/docu_sedi_twomom.pdf)
!   instead of sedi_icon_core() et al.:
!   - sedi_icon_box_core() (vectorized version #if defined (__SX__) || defined (__NEC_VH__) || defined (__NECSX__))
!   - sedi_icon_box_core_lwf() (vectorized version #if defined (__SX__) || defined (__NEC_VH__) || defined (__NECSX__))
!===============================================================================!
! OpenACC compiler error workarounds (04/2023 by MJ):
! IPSF: Several intermediate pointers have been introduced to circumvent
!       segmentation faults with nvhpc 22.7. The affected lines are marked by
!       the following abbreviation:
!       ! ACCWA (nvhpc 22.7, IPSF, see above)
!       Without these pointer, the compiler or Nvidia runtime is otherwise
!       unable to find the derived type on the accelerator device.
!       This workaround also requires additional WAIT clauses.
!       04/2024: This bug also affects nvhpc 23.3
!===============================================================================!

  USE mo_kind,               ONLY: sp, wp
  USE mo_exception,          ONLY: finish, message, txt => message_text
  USE mo_physical_constants, ONLY: &
       & R_l   => rd,     & ! gas constant of dry air (luft)
       & R_d   => rv,     & ! gas constant of water vapor (dampf)
       & cp    => cpd,    & ! specific heat capacity of air at constant pressure
       & c_w   => clw,    & ! specific heat capacity of water
       & L_wd  => alv,    & ! specific heat of vaporization (wd: wasser->dampf)
       & L_ed  => als,    & ! specific heat of sublimation (ed: eis->dampf)
       & L_ew  => alf,    & ! specific heat of fusion (ew: eis->wasser)
       & T_3   => tmelt,  & ! melting temperature of ice
       & rho_w => rhoh2o, & ! density of liquid water
       & rho_ice => rhoice,&! density of pure ice
       & nu_l  => con_m,  & ! kinematic viscosity of air
       & D_v   => dv0,    & ! diffusivity of water vapor in air at 0 C
       & K_t   => con0_h, & ! heat conductivity of air
       & N_avo => avo,    & ! Avogadro number [1/mol]
       & k_b   => ak,     & ! Boltzmann constant [J/K]
       & grav               ! acceleration due to Earth's gravity
  USE mo_satad, ONLY:     &
       & e_ws  => sat_pres_water,  & ! saturation pressure over liquid water
       & e_es  => sat_pres_ice,    & ! saturation pressure over ice
       & latent_heat_vaporization
  USE mo_2mom_mcrph_types, ONLY: &
       & particle, particle_frozen, particle_lwf, atmosphere, &
       & particle_sphere, particle_rain_coeffs, particle_cloud_coeffs, aerosol_ccn, &
       & particle_ice_coeffs, particle_snow_coeffs, particle_graupel_coeffs, &
       & particle_coeffs, collection_coeffs, rain_riming_coeffs, dep_imm_coeffs, &
       & coll_coeffs_ir_pm, lookupt_1D, lookupt_4D

  USE mo_2mom_mcrph_config,         ONLY: t_cfg_2mom
  USE mo_2mom_mcrph_config_default, ONLY: cfg_2mom_default
  USE mo_2mom_mcrph_util, ONLY: &
       & rat2do3,                    &  ! rational function for lwf-melting scheme
       & dyn_visc_sutherland,        &  ! used in lwf melting scheme
       & Dv_Rasmussen,               &  ! used in lwf melting scheme
       & ka_Rasmussen,               &  ! used in lwf melting scheme
       & lh_evap_RH87,               &  ! used in lwf melting scheme
       & lh_melt_RH87,               &  ! used in lwf melting scheme
       & set_qnc,                    &
       & set_qni,                    &
       & set_qnr,                    &
       & set_qns,                    &
       & set_qng,                    &
       & set_qnh_expPSD_N0const,     &
       & estick_ltab_equi

  IMPLICIT NONE

  PRIVATE

  CHARACTER(len=*), PARAMETER :: routine = 'mo_2mom_mcrph_processes'

  ! In place of ICON's mo_math_constants: pi and 4*pi are the only two
  ! constants this scheme needs from it.
  REAL(wp), PARAMETER :: pi  = 3.14159265358979323846_wp
  REAL(wp), PARAMETER :: pi4 = 4.0_wp * pi

  ! switches for ice scheme, ice nucleation, drop activation and autoconversion
  INTEGER  :: ice_typ, nuc_i_typ, nuc_c_typ, auto_typ

  ! Physical parameters and coefficients which occur only in the two-moment scheme

  ! .. lower limit of tke used in turbulent collision enhancement parameterization:
  REAL(wp), PARAMETER :: tke_min = 0.01_wp ** 2 ! [m^2 s^-2]
  
  ! .. some physical parameters not found in ICON
  REAL(wp), PARAMETER :: T_f     = 233.0_wp     !..below this temperature there is no liquid water

  ! .. some cloud physics parameters
  REAL(wp), PARAMETER :: N_sc = 0.710_wp        !..Schmidt-Zahl (PK, S.541)
  REAL(wp), PARAMETER :: n_f  = 0.333_wp        !..Exponent von N_sc im Vent-koeff. (PK, S.541)

  ! .. for old saturation pressure relations (keep this for some time for testing)
  REAL(wp), PARAMETER :: A_e  = 2.18745584e1_wp !..Konst. Saettigungsdamppfdruck - Eis
  REAL(wp), PARAMETER :: A_w  = 1.72693882e1_wp !..Konst. Saettigungsdamppfdruck - Wasser
  REAL(wp), PARAMETER :: B_e  = 7.66000000e0_wp !..Konst. Saettigungsdamppfdruck - Eis
  REAL(wp), PARAMETER :: B_w  = 3.58600000e1_wp !..Konst. Saettigungsdamppfdruck - Wasser
  REAL(wp), PARAMETER :: e_3  = 6.10780000e2_wp !..Saettigungsdamppfdruck bei T = T_3

  ! .. Autoconversion
  REAL(wp), PARAMETER :: kc_autocon  = 9.44e+9_wp   !..Long-Kernel
    
  ! .. Hallet-Mossop ice multiplication
  REAL(wp), PARAMETER ::           &
       &    C_mult     = 3.5e8_wp, &    !..Koeff. fuer Splintering
       &    T_mult_min = 265.0_wp, &    !..Minimale Temp. Splintering
       &    T_mult_max = 270.0_wp, &    !..Maximale Temp. Splintering
       &    T_mult_opt = 268.0_wp       !..Optimale Temp. Splintering

  ! .. Phillips et al. ice nucleation scheme, see ice_nucleation_homhet() for more details
  REAL(wp) ::                         &
       &    na_dust    = 160.e4_wp,   & ! initial number density of dust [1/m], Phillips08 value 162e3 (never used, reset later)
       &    na_soot    =  25.e6_wp,   & ! initial number density of soot [1/m], Phillips08 value 15e6 (never used, reset later)
       &    na_orga    =  30.e6_wp,   & ! initial number density of organics [1/m3], Phillips08 value 177e6 (never used, reset later)
       &    ni_het_max = 100.0e3_wp,  & ! max number of IN between 1-10 per liter, i.e. 1d3-10d3
       &    ni_hom_max = 5000.0e3_wp    ! number of liquid aerosols between 100-5000 per liter

  INTEGER, PARAMETER ::               & ! Look-up table for Phillips et al. nucleation
       &    ttmax  = 30,              & ! sets limit for temperature in look-up table
       &    ssmax  = 60,              & ! sets limit for ice supersaturation in look-up table
       &    ttstep = 2,               & ! increment for temperature in look-up table
       &    ssstep = 1                  ! increment for ice supersaturation in look-up table

  REAL(sp), DIMENSION(0:100,0:100)  :: &
       &    afrac_dust, &  ! look-up table of activated fraction of dust particles acting as ice nuclei
       &    afrac_soot, &  ! ... of soot particles
       &    afrac_orga     ! ... of organic material

  INCLUDE 'phillips_nucleation_2010.incf'

  ! MINIMAL SCHEME: the LWF (liquid-water-fraction) melting-scheme
  ! coefficient arrays and their include files (hailcoeffs.incf,
  ! grplcoeffs.incf) have been removed -- they were used only by the
  ! dropped particle_melting_lwf / prepare_melting_lwf routines.

  !..Tables for 4D Segal-Khain activation
  TYPE(lookupt_4D) :: otab, tab

  ! Size thresholds for partioning of freezing rain in the hail scheme:
  ! Raindrops smaller than D_rainfrz_ig freeze into cloud ice,
  ! drops between D_rainfrz_ig and D_rainfrz_gh freeze to graupel, and the
  ! largest raindrop freeze directly to hail.
!!$ now this comes from cfg_params:
!!$  REAL(wp), PARAMETER ::               &
!!$       &    D_rainfrz_ig = 0.50e-3_wp, & ! rain --> ice oder graupel
!!$       &    D_rainfrz_gh = 1.25e-3_wp    ! rain --> graupel oder hail

  ! Various parameters for collision and conversion rates
  REAL(wp), PARAMETER ::                  &
       &    ecoll_min    = 0.01_wp          ! ..min. eff. for graupel_cloud, ice_cloud and snow_cloud
!!       &    Tcoll_gg_wet = 270.16_wp
!!       &    ecoll_gg     = 0.10_wp,       &  !..collision efficiency for graupel selfcollection
!!       &    ecoll_gg_wet = 0.40_wp           !    in case of wet graupel
!!$ now this comes from cfg_params:
!!$    &    alpha_spacefilling = 0.01_wp     !..Raumerfuellungskoeff (max. 0.68)

  ! Even more parameters for collision and conversion rates
  REAL(wp), PARAMETER :: &
       &    q_crit_ii = 1.000e-6_wp, & ! q-threshold for ice_selfcollection
       &    D_crit_ii = 5.0e-6_wp,   & ! D-threshold for ice_selfcollection  
!!$ now this comes from cfg_params:
!!$       &    D_conv_ii = 75.00e-6_wp, & ! D-threshold for conversion in ice_selfcollection
       &    q_crit_r  = 1.000e-5_wp, & ! q-threshold for ice_rain_riming and snow_rain_riming
       &    D_crit_r  = 100.0e-6_wp, & ! D-threshold for ice_rain_riming and snow_rain_riming
       &    q_crit_fr = 1.000e-6_wp, & ! q-threshold for rain_freeze
       &    q_crit_c  = 1.000e-6_wp, & ! q-threshold for cloud water
!!$       &    q_crit    = 1.000e-7_wp, & ! q-threshold elsewhere 1e-7 kg/m3 = 1e-4 g/m3 = 0.1 mg/m3
       &    q_crit    = 1.000e-9_wp, & ! q-threshold elsewhere 1e-7 kg/m3 = 1e-4 g/m3 = 0.1 mg/m3
       &    D_conv_sg = 200.0e-6_wp, & ! D-threshold for conversion of snow to graupel
       &    D_conv_ig = 200.0e-6_wp, & ! D-threshold for conversion of ice to graupel 
       &    x_conv    = 0.100e-9_wp, & ! minimum mass of conversion due to riming
       &    D_crit_c  = 10.00e-6_wp, & ! D-threshold for cloud drop collection efficiency
       &    D_coll_c  = 40.00e-6_wp    ! upper bound for diameter in collision efficiency

  REAL(wp), PARAMETER ::           &
       &    T_nuc        = 268.15_wp, & ! lower temperature threshold for ice nucleation, -5 C
       &    T_freeze  = 273.15_wp    ! lower temperature threshold for raindrop freezing

  ! Parameter for evaporation of rain, determines change of n_rain during evaporation
  REAL(wp) :: rain_gfak   ! this is set in init_twomoment

  ! debug switches
  LOGICAL, PARAMETER     :: isdebug = .false.   ! use only when really desperate
  LOGICAL, PARAMETER     :: isprint = .true.    ! print-out initialization values
  
  ! some cloud microphysical switches
  LOGICAL, PARAMETER     :: ice_multiplication = .TRUE.  ! default is .true.
  LOGICAL, PARAMETER     :: enhanced_melting   = .TRUE.  ! default is .true.
  LOGICAL, PARAMETER     :: classic_melting_in_lwf_scheme = .False.

  REAL(wp), PARAMETER    :: pi6 = pi/6.0_wp, pi8 = pi/8.0_wp ! more pieces of pi

  TYPE(t_cfg_2mom) :: cfg_params !.. Container to hold some config params for the actual 2-mom call
  
  ! Parameters
  PUBLIC :: q_crit
  PUBLIC :: cfg_2mom_default, cfg_params
  ! Switches
  PUBLIC :: ice_typ, nuc_i_typ, nuc_c_typ, auto_typ
  PUBLIC :: isdebug, isprint
  ! Functions
  PUBLIC :: particle_meanmass
  PUBLIC :: particle_assign, particle_frozen_assign, particle_lwf_assign
  ! Process Routines (minimal scheme: only the retained call tree is exported)
  PUBLIC :: init_2mom_sedi_vel
  PUBLIC :: cloud_freeze
  PUBLIC :: ice_nucleation_homhet
  PUBLIC :: vapor_dep_relaxation
  PUBLIC :: setup_particle_coeffs
  PUBLIC :: ice_melting
  PUBLIC :: ccn_activation_hdcp2, ccn_activation_sk_4d
  PUBLIC :: moment_gamma
  PUBLIC :: set_default_n

CONTAINS

  !*******************************************************************************
  ! Particle-type constructors (assign a predefined parameter set to a category).
  ! Called by init_2mom_scheme.
  !*******************************************************************************

  subroutine particle_assign(that,this)
    CLASS(particle), INTENT(in)   :: this
    TYPE(particle), INTENT(inout) :: that

    that%name = this%name
    that%nu = this%nu
    that%mu = this%mu
    that%x_max = this%x_max
    that%x_min = this%x_min
    that%a_geo = this%a_geo
    that%b_geo = this%b_geo
    that%a_vel = this%a_vel
    that%b_vel = this%b_vel
    that%a_ven = this%a_ven
    that%b_ven = this%b_ven
    that%cap   = this%cap
    that%vsedi_max = this%vsedi_max
    that%vsedi_min = this%vsedi_min
  END subroutine particle_assign

  subroutine particle_frozen_assign(that,this)
    TYPE(particle_frozen), INTENT(in)    :: this
    TYPE(particle_frozen), INTENT(inout) :: that

    that%name = this%name
    that%nu = this%nu
    that%mu = this%mu
    that%x_max = this%x_max
    that%x_min = this%x_min
    that%a_geo = this%a_geo
    that%b_geo = this%b_geo
    that%a_vel = this%a_vel
    that%b_vel = this%b_vel
    that%a_ven = this%a_ven
    that%b_ven = this%b_ven
    that%cap   = this%cap
    that%vsedi_max = this%vsedi_max
    that%vsedi_min = this%vsedi_min
    that%ecoll_c   = this%ecoll_c
    that%D_crit_c  = this%D_crit_c
    that%q_crit_c  = this%q_crit_c
    that%s_vel     = this%s_vel
  END subroutine particle_frozen_assign

  subroutine particle_lwf_assign(that,this)
    TYPE(particle_lwf), INTENT(in)    :: this
    TYPE(particle_lwf), INTENT(inout) :: that

    that%name = this%name
    that%nu = this%nu
    that%mu = this%mu
    that%x_max = this%x_max
    that%x_min = this%x_min
    that%a_geo = this%a_geo
    that%b_geo = this%b_geo
    that%a_vel = this%a_vel
    that%b_vel = this%b_vel
    that%a_ven = this%a_ven
    that%b_ven = this%b_ven
    that%cap   = this%cap
    that%vsedi_max = this%vsedi_max
    that%vsedi_min = this%vsedi_min

    that%ecoll_c   = this%ecoll_c
    that%D_crit_c  = this%D_crit_c
    that%q_crit_c  = this%q_crit_c
    that%s_vel     = this%s_vel

    that%lwf_cnorm1 = this%lwf_cnorm1
    that%lwf_cnorm2 = this%lwf_cnorm2
    that%lwf_cnorm3 = this%lwf_cnorm3
    that%lwf_cmelt1 = this%lwf_cmelt1
    that%lwf_cmelt2 = this%lwf_cmelt2
  END subroutine particle_lwf_assign

  ! mean mass with limiters, Eq. (94) of SB2006
  ELEMENTAL FUNCTION particle_meanmass(this,q,n) RESULT(xmean)


    CLASS(particle), INTENT(in) :: this
    REAL(wp),        INTENT(in) :: q, n
    REAL(wp)                    :: xmean
    REAL(wp), PARAMETER         :: eps = 1e-20_wp

    xmean = MIN(MAX(q/(n+eps),this%x_min),this%x_max)
  END FUNCTION particle_meanmass


  ! mass-diameter relation, power law, Eq. (32) of SB2006
  ELEMENTAL FUNCTION particle_diameter(this,x) RESULT(D)


    CLASS(particle), INTENT(in) :: this
    REAL(wp),        INTENT(in) :: x
    REAL(wp)                    :: D

    D = this%a_geo * EXP(this%b_geo*LOG(x))    ! D = a_geo * x**b_geo
  END FUNCTION particle_diameter


  ! terminal fall velocity of particles, cf. Eq. (33) of SB2006
! Cray compiler does not support OpenACC in elemental or pure
#ifdef _CRAYFTN
  FUNCTION particle_velocity(this,x) RESULT(v)
#else
  ELEMENTAL FUNCTION particle_velocity(this,x) RESULT(v)
#endif

    CLASS(particle), INTENT(in) :: this
    REAL(wp),        INTENT(in) :: x
    REAL(wp)                    :: v

    v = this%a_vel * EXP(this%b_vel * LOG(x))  ! v = a_vel * x**b_vel
  END FUNCTION particle_velocity


  !*******************************************************************************
  ! (2) More functions working on particle class, these are not CLASS procedures
  !*******************************************************************************

  ! bulk ventilation coefficient, Eq. (88) of SB2006
  REAL(wp) FUNCTION vent_coeff_a(parti,n)
    IMPLICIT NONE
    INTEGER, INTENT(IN)        :: n
    CLASS(particle), INTENT(IN) :: parti

    vent_coeff_a = parti%a_ven * GAMMA((parti%nu+n+parti%b_geo)/parti%mu)                 &
         &                     / GAMMA((parti%nu+1.0_wp)/parti%mu)                        &
         &                   * ( GAMMA((parti%nu+1.0_wp)/parti%mu)                        &
         &                     / GAMMA((parti%nu+2.0_wp)/parti%mu) )**(parti%b_geo+n-1.0_wp)
  END FUNCTION vent_coeff_a


  ! bulk ventilation coefficient, Eq. (89) of SB2006
  REAL(wp) FUNCTION vent_coeff_b(parti,n)
    IMPLICIT NONE
    INTEGER, INTENT(in)         :: n
    CLASS(particle), INTENT(in) :: parti

    REAL(wp), PARAMETER :: m_f = 0.500 ! see PK, S.541. Do not change.

    vent_coeff_b = parti%b_ven                                                  &
         & * GAMMA((parti%nu+n+(m_f+1.0_wp)*parti%b_geo+m_f*parti%b_vel)/parti%mu)  &
         &             / GAMMA((parti%nu+1.0_wp)/parti%mu)                          &
         &           * ( GAMMA((parti%nu+1.0_wp)/parti%mu)                          &
         &             / GAMMA((parti%nu+2.0_wp)/parti%mu)                          &
         &             )**((m_f+1.0_wp)*parti%b_geo+m_f*parti%b_vel+n-1.0_wp)
  END FUNCTION vent_coeff_b


  ! complete mass moment of particle size distribution, Eq (82) of SB2006
  REAL(wp) FUNCTION moment_gamma(p,n)
    IMPLICIT NONE
    INTEGER, INTENT(in)           :: n
    CLASS(particle), INTENT(in)   :: p

    moment_gamma  = GAMMA((n+p%nu+1.0_wp)/p%mu) / GAMMA((p%nu+1.0_wp)/p%mu)        &
         &      * ( GAMMA((  p%nu+1.0_wp)/p%mu) / GAMMA((p%nu+2.0_wp)/p%mu) )**n
  END FUNCTION moment_gamma

  
  ! initialize coefficients for bulk sedimentation velocity
  SUBROUTINE init_2mom_sedi_vel(this,thisCoeffs)
    CLASS(particle), INTENT(in) :: this
    CLASS(particle_sphere), INTENT(out) :: thisCoeffs
    
    CHARACTER(len=*), PARAMETER :: sroutine = 'init_2mom_sedi_vel'
    
    thisCoeffs%coeff_alfa_n = this%a_vel * GAMMA((this%nu+this%b_vel+1.0)/this%mu) / GAMMA((this%nu+1.0)/this%mu)
    thisCoeffs%coeff_alfa_q = this%a_vel * GAMMA((this%nu+this%b_vel+2.0)/this%mu) / GAMMA((this%nu+2.0)/this%mu)
    thisCoeffs%coeff_lambda = GAMMA((this%nu+1.0)/this%mu)/GAMMA((this%nu+2.0)/this%mu)
    
    IF (isprint) THEN
      WRITE (txt,'(2A)') "    name  = ",this%name ; CALL message(sroutine,TRIM(txt))
      WRITE (txt,'(A,D14.7)') "    c_lam = ",thisCoeffs%coeff_lambda ; CALL message(sroutine,TRIM(txt))
      WRITE (txt,'(A,D14.7)') "    alf_n = ",thisCoeffs%coeff_alfa_n ; CALL message(sroutine,TRIM(txt))
      WRITE (txt,'(A,D14.7)') "    alf_q = ",thisCoeffs%coeff_alfa_q ; CALL message(sroutine,TRIM(txt))
    END IF
  END SUBROUTINE init_2mom_sedi_vel


  !*******************************************************************************
  ! Fundamental physical relations
  !*******************************************************************************

  ! Molecular diffusivity of water vapor
  ELEMENTAL FUNCTION diffusivity(T,p) result(D_v)


    REAL(wp), INTENT(IN) :: T,p
    REAL(wp) :: D_v
    ! This is D_v = 8.7602e-5_wp * T_a**(1.81_wp) / p_a
    D_v = 8.7602e-5_wp * EXP(1.81_wp*LOG(T)) / p
    RETURN
  END FUNCTION diffusivity


  SUBROUTINE cloud_freeze(kstart, kend, dt, cloud_coeffs, qnc_const, atmo, cloud_in, ice)
    !*******************************************************************************
    ! This is only the homogeneous freezing of liquid water droplets.              *
    ! Immersion freezing and homogeneous freezing of liquid aerosols are           *
    ! treated in the subroutine ice_nucleation_homhet()                            *
    !*******************************************************************************
    INTEGER,  INTENT(in) :: kstart, kend
    REAL(wp), INTENT(in) :: dt
    REAL(wp), INTENT(in) :: qnc_const
    TYPE(particle_cloud_coeffs), INTENT(in) :: cloud_coeffs
    TYPE(atmosphere), INTENT(inout) :: atmo
    CLASS(particle), INTENT(inout), TARGET :: cloud_in
    CLASS(particle), POINTER :: cloud ! ACCWA (nvhpc 22.7, IPSF, see above)
    CLASS(particle), INTENT(inout) :: ice
    INTEGER :: k
    REAL(wp)           :: fr_q, fr_n, T_a, q_c, x_c, n_c, j_hom, T_c

    REAL(wp), PARAMETER :: log_10 = LOG(10.0_wp)

    cloud => cloud_in ! ACCWA (nvhpc 22.7, IPSF, see above)


    DO k = kstart,kend

      T_a = atmo%T(k)
      IF (T_a < T_3) THEN

        T_c = T_a - T_3
        q_c = cloud%q(k)
        n_c = cloud%n(k)
        IF (q_c > 0.0_wp .and. T_c < -30.0_wp) THEN
          IF (T_c < -50.0_wp) THEN
            fr_q = q_c             !..instantaneous freezing
            fr_n = n_c             !..below -50 C
          ELSE
            x_c = particle_meanmass(cloud, q_c, n_c)

            !..Hom. freezing based on Jeffrey und Austin (1997), see also Cotton und Field (2001)
            !  (note that log in Cotton and Field is log10, not ln)
            IF (T_c > -30.0_wp) THEN
!                 j_hom = 1.0e6_wp/rho_w * 10**(-7.63-2.996*(T_c+30.0))           !..J in 1/(kg s)
               j_hom = 1.0e6_wp/rho_w * EXP((-7.63_wp-2.996_wp*(T_c+30.0_wp))*log_10)
            ELSE
!                 j_hom = 1.0e6_wp/rho_w &
!                      &  * 10**(-243.4-14.75*T_c-0.307*T_c**2-0.00287*T_c**3-0.0000102*T_c**4)
               j_hom = 1.0e6_wp/rho_w &
                    &  * EXP((-243.4_wp-14.75_wp*T_c-0.307_wp*T_c**2-0.00287_wp*T_c**3-0.0000102_wp*T_c**4)*log_10)
            ENDIF

            fr_n  = j_hom * q_c *  dt
            fr_q  = j_hom * q_c * x_c * dt *  cloud_coeffs%c_z
            fr_q  = MIN(fr_q,q_c)
            fr_n  = MIN(fr_n,n_c)
          END IF

          cloud%q(k) = cloud%q(k) - fr_q
          cloud%n(k) = cloud%n(k) - fr_n

          fr_n  = MAX(fr_n, fr_q/cloud%x_max)

          !..special treatment for constant drop number
          IF (nuc_c_typ .EQ. 0) THEN
            ! ... force upper bound in cloud_freeze'
            fr_n = MAX(MIN(fr_n, qnc_const-ice%n(k)), 0.0_wp)
          ENDIF

          ice%q(k)   = ice%q(k) + fr_q
          ice%n(k)   = ice%n(k) + fr_n
        ENDIF
      END IF
    END DO

  END SUBROUTINE cloud_freeze


  SUBROUTINE ice_nucleation_homhet(kstart, kend, use_prog_in, &
       atmo, cloud, ice_in, n_inact, n_inpot, n_inagi, luse_agi, iagi_param)
    !*******************************************************************************
    !                                                                              *
    ! Homogeneous and heterogeneous ice nucleation                                 *
    !                                                                              *
    ! Nucleation scheme is based on the papers:                                    *
    !                                                                              *
    ! "A parametrization of cirrus cloud formation: Homogenous                     *
    ! freezing of supercooled aerosols" by B. Kaercher and                         *
    ! U. Lohmann 2002 (KL02 hereafter)                                             *
    !                                                                              *
    ! "Physically based parameterization of cirrus cloud formation                 *
    ! for use in global atmospheric models" by B. Kaercher, J. Hendricks           *
    ! and U. Lohmann 2006 (KHL06 hereafter)                                        *
    !                                                                              *
    ! and Phillips et al. (2008) with extensions                                   *
    !                                                                              *
    ! implementation by Carmen Koehler and AS                                      *
    !*******************************************************************************
    INTEGER, INTENT(in) :: kstart, kend
    LOGICAL, INTENT(in) :: use_prog_in

    TYPE(atmosphere), INTENT(inout) :: atmo
    CLASS(particle), INTENT(inout), TARGET :: ice_in
    CLASS(particle), POINTER :: ice ! ACCWA (nvhpc 22.7, IPSF, see above)
    CLASS(particle), INTENT(inout) :: cloud
    REAL(wp), DIMENSION(:) :: n_inact
    REAL(wp), DIMENSION(:), OPTIONAL :: n_inpot, n_inagi
    INTEGER, OPTIONAL, INTENT(IN) :: iagi_param
    LOGICAL, OPTIONAL, INTENT(IN) :: luse_agi

    INTEGER              :: k,nuc_typ
    REAL(wp)             :: nuc_n, nuc_q
    REAL(wp)             :: T_a, p_a, ssi
    REAL(wp)             :: q_i,n_i,x_i,r_i

    ! switch for version of Phillips et al. scheme
    ! (but make sure you have the correct INCLUDE file)
    INTEGER, PARAMETER :: iphillips = 2010

    ! switch for Hande et al. ice nucleation, if .true. this turns off Phillips scheme
    LOGICAL  :: use_hdcp2_het

    ! some more constants needed for homogeneous nucleation scheme
    REAL(wp), PARAMETER ::            &
         r_0     = 0.25e-6_wp          , &    ! aerosol particle radius prior to freezing
         alpha_d = 0.5_wp              , &    ! deposition coefficient (KL02; Spichtinger & Gierens 2009)
         M_w     = 18.01528e-3_wp      , &    ! molecular mass of water [kg/mol]
         M_a     = 28.96e-3_wp         , &    ! molecular mass of air [kg/mol]
         ma_w    = M_w / N_avo         , &    ! mass of water molecule [kg]
         svol    = ma_w / rho_ice             ! specific volume of a water molecule in ice

    REAL(wp)  :: e_si
    REAL(wp)  :: ni_hom,ri_hom,mi_hom
    REAL(wp)  :: v_th,n_sat,flux,phi,cool,tau,delta,w_pre,scr
    REAL(wp)  :: ctau, acoeff(3),bcoeff(2), ri_dot
    REAL(wp)  :: kappa,sqrtkap,ren,R_imfc,R_im,R_ik,ri_0

    ! parameters for Hande et al. nucleation parameterization for HDCP2 simulations
    TYPE(dep_imm_coeffs), PARAMETER :: hdcp2_nuc_coeffs(5) &
         = (/ dep_imm_coeffs(   &  ! Spring of Table 1
         nin_imm = 1.5684e5_wp, &
         alf_imm = 0.2466_wp,   &
         bet_imm = 1.2293_wp,   &
         nin_dep = 1.7836e5_wp, &
         alf_dep = 0.0075_wp,   &
         bet_dep = 2.0341_wp),  &
         dep_imm_coeffs(        &  ! Summer
         nin_imm = 2.9694e4_wp, &
         alf_imm = 0.2813_wp,   &
         bet_imm = 1.1778_wp,   &
         nin_dep = 2.6543e4_wp, &
         alf_dep = 0.0020_wp,   &
         bet_dep = 2.5128_wp),  &
         dep_imm_coeffs(        &  ! Autumn
         nin_imm = 4.9920e4_wp, &
         alf_imm = 0.2622_wp,   &
         bet_imm = 1.2044_wp,   &
         nin_dep = 7.7167e4_wp, &
         alf_dep = 0.0406_wp,   &
         bet_dep = 1.4705_wp),  &
         dep_imm_coeffs(        &  ! Winter
         nin_imm = 1.0259e5_wp, &
         alf_imm = 0.2073_wp,   &
         bet_imm = 1.2873_wp,   &
         nin_dep = 1.1663e4_wp, &
         alf_dep = 0.0194_wp,   &
         bet_dep = 1.6943),     &
         dep_imm_coeffs(        &  ! Spring with 95th percentile scaling factor
         nin_imm = 1.5684e5_wp * 17.82_wp, &
         alf_imm = 0.2466_wp,   &
         bet_imm = 1.2293_wp,   &
         nin_dep = 1.7836e5_wp * 5.87_wp, &
         alf_dep = 0.0075_wp,   &
         bet_dep = 2.0341_wp) /)

    LOGICAL   :: use_homnuc

    LOGICAL  :: ndiag_mask(kstart:kend)
    REAL(wp) :: nuc_n_a(kstart:kend)


    ice => ice_in ! ACCWA (nvhpc 22.7, IPSF, see above)


    nuc_typ = nuc_i_typ

    SELECT CASE (nuc_typ)
    CASE(0)
      ! Heterogeneous nucleation ONLY
      use_homnuc = .FALSE.
    CASE(1:9)
      ! Homog. and het. nucleation"
      use_homnuc = .TRUE.
    END SELECT

    IF (isdebug) THEN
      IF (.NOT.use_homnuc) THEN
        WRITE(txt,*) "ice_nucleation_homhet: Heterogeneous nucleation only"
      ELSE
        WRITE(txt,*) "ice_nucleation_homhet: Homogeneous and heterogeneous nucleation"
      END IF
      CALL message(routine,TRIM(txt))
    END IF

    ! switch for Hande et al. ice nucleation, if .true. this turns off Phillips scheme
    !use_hdcp2_het = (nuc_typ.le.5)

    !! Heterogeneous nucleation using Hande et al. scheme
    !IF (use_hdcp2_het) THEN
!#ifdef _OPENACC
    !  CALL finish('mo_2mom_mcrph_processes:','ice_nucleation_het_hdcp2 not available on GPU')
!#endif
    !  IF (nuc_typ < 1 .OR. nuc_typ > 5) THEN
    !    CALL finish(TRIM(routine), &
    !         & 'Error in two_moment_mcrph: Invalid value nuc_typ in case of &
    !         &use_hdcp2_het=.true.')
    !  END IF
    !  CALL ice_nucleation_het_hdcp2(kstart, kend, atmo, ice, cloud, &
    !       use_prog_in, hdcp2_nuc_coeffs(nuc_typ), n_inact, ndiag_mask, nuc_n_a)
    IF (luse_agi .AND. use_prog_in) THEN
        IF (iagi_param == 1) THEN
        CALL ice_nucleation_agi_dm95(kstart, kend, atmo, ice, cloud, n_inact, &
            &                       n_inagi, nuc_n_a)
        ELSE IF (iagi_param == 2) THEN
        CALL ice_nucleation_agi_m16(kstart, kend, atmo, ice, cloud, n_inact, &
            &                       n_inagi, nuc_n_a)
        END IF
        CALL ice_nucleation_het_inas(kstart, kend, atmo, cloud, ice, n_inact, nuc_typ,&
                  &                      use_prog_in, n_inpot, ndiag_mask, nuc_n_a)
    ELSE IF (nuc_typ < 5) THEN
        CALL ice_nucleation_het_inas(kstart, kend, atmo, cloud, ice, n_inact, nuc_typ, &
             &                       use_prog_in, n_inpot, ndiag_mask, nuc_n_a)
    ELSE
      ! Heterogeneous nucleation using Phillips et al. scheme
      IF (iphillips == 2010) THEN
        ! possible pre-defined choices
        IF (nuc_typ.EQ.6) THEN  ! with no organics and rather high soot, coming close to Meyers formula at -20 C
          na_dust  = 160.e4_wp    ! initial number density of dust [1/m3]
          na_soot  =  30.e6_wp    ! initial number density of soot [1/m3]
          na_orga  =   0.e0_wp    ! initial number density of organics [1/m3]
        ELSEIF (nuc_typ.EQ.7) THEN     ! with some organics and rather high soot,
          na_dust  = 160.e4_wp    !          coming close to Meyers formula at -20 C
          na_soot  =  25.e6_wp
          na_orga  =  30.e6_wp
        ELSE IF (nuc_typ.EQ.8) THEN     ! no organics, no soot, coming close to DeMott et al. 2010 at -20 C
          na_dust  =  70.e4_wp    ! i.e. roughly one order in magnitude lower than Meyers
          na_soot  =   0.e6_wp
          na_orga  =   0.e6_wp
        ELSE
          CALL finish(TRIM(routine),&
               & 'Error in two_moment_mcrph: Invalid value nuc_typ in case of use_hdcp2_het=.false.')
        END IF
      END IF
      CALL ice_nucleation_het_philips(kstart, kend, atmo, ice, cloud, &
           use_prog_in, n_inact, ndiag_mask, nuc_n_a, n_inpot)
    END IF

    IF (use_prog_in) THEN
      DO k = kstart, kend
        n_inpot(k) = MERGE(MAX(n_inpot(k) - nuc_n_a(k), 0.0_wp), &
             &               n_inpot(k), &
             &               ndiag_mask(k))
      END DO
    END IF

    ! Homogeneous nucleation using KHL06 approach
    IF (use_homnuc) THEN
      DO k = kstart,kend
        p_a  = atmo%p(k)
        T_a  = atmo%T(k)
        e_si = e_es(T_a)
        ssi  = atmo%qv(k) * R_d * T_a / e_si

        ! critical supersaturation for homogeneous nucleation
        scr  = 2.349 - T_a * (1.0_wp/ 259.00_wp)

        IF (ssi > scr .AND. T_a < 235.0 .AND. ice%n(k) < ni_hom_max ) THEN

          n_i = ice%n(k)
          q_i = ice%q(k)
          x_i = particle_meanmass(ice, q_i,n_i)
!            r_i = (x_i/(4./3.*pi*rho_ice))**(1./3.)
          r_i = EXP( (1.0_wp/3.0_wp)*LOG(x_i/(4.0_wp/3.0_wp*pi*rho_ice)) )

          v_th  = SQRT( 8.0_wp*k_b*T_a/(pi*ma_w) )
          flux  = alpha_d * v_th/4.
          n_sat = e_si / (k_b*T_a)

          ! coeffs of supersaturation equation
          acoeff(1) = (L_ed * grav) / (cp * R_d * T_a**2) - grav/(R_l * T_a)
          acoeff(2) = 1.0_wp/n_sat
          acoeff(3) = (L_ed**2 * M_w * ma_w)/(cp * p_a * T_a * M_a)

          ! coeffs of depositional growth equation
          bcoeff(1) = flux * svol * n_sat * (ssi - 1.0_wp)
          bcoeff(2) = flux / diffusivity(T_a,p_a)

          ! pre-existing ice crystals included as reduced updraft speed
          ri_dot = bcoeff(1) / (1.0_wp + bcoeff(2) * r_i)
          R_ik   = (4.0_wp * pi) / svol * n_i * r_i**2 * ri_dot
          w_pre  = (acoeff(2) + acoeff(3) * ssi)/(acoeff(1) * ssi) * R_ik  ! KHL06 Eq. 19
          w_pre  = MAX(w_pre,0.0_wp)

          IF (atmo%w(k) > w_pre) THEN   ! homogenous nucleation event

            ! timescales of freezing event (see KL02, RM05, KHL06)
            cool    = grav / cp * atmo%w(k)
            ctau    = T_a * ( 0.004_wp*T_a - 2.0_wp ) + 304.4_wp
            tau     = 1.0_wp / (ctau * cool)                       ! freezing timescale, eq. (5)
            delta   = (bcoeff(2) * r_0)                         ! dimless aerosol radius, eq.(4)
            phi     = acoeff(1)*ssi / ( acoeff(2) + acoeff(3)*ssi) * (atmo%w(k) - w_pre)

            ! monodisperse approximation following KHL06
            kappa   = 2.0_wp * bcoeff(1) * bcoeff(2) * tau / (1.0_wp+ delta)**2  ! kappa, Eq. 8 KHL06
            sqrtkap = SQRT(kappa)                                        ! root of kappa
            ren     = 3.0_wp * sqrtkap / ( 2.0_wp + SQRT(1.0_wp+9.0_wp*kappa/pi) )       ! analy. approx. of erfc by RM05
            R_imfc  = 4.0_wp * pi * bcoeff(1)/bcoeff(2)**2 / svol
            R_im    = R_imfc / (1.0_wp+ delta) * ( delta**2 - 1.0_wp &
                 & + (1.0_wp+0.5_wp*kappa*(1.0_wp+ delta)**2) * ren/sqrtkap)           ! RIM Eq. 6 KHL06

            ! number concentration and radius of ice particles
            ni_hom  = phi / R_im                                         ! ni Eq.9 KHL06
            ri_0    = 1.0_wp + 0.5_wp * sqrtkap * ren                           ! for Eq. 3 KHL06
            ri_hom  = (ri_0 * (1.0_wp + delta) - 1.0_wp ) / bcoeff(2)            ! Eq. 3 KHL06 * REN = Eq.23 KHL06
            mi_hom  = (4.0_wp/3.0_wp * pi * rho_ice) * ni_hom * ri_hom**3
            mi_hom  = MAX(mi_hom,ice%x_min)

            nuc_n = MAX(MIN(ni_hom, ni_hom_max), 0.0_wp)
            nuc_q = MIN(nuc_n * mi_hom, atmo%qv(k))

            ice%n(k) = ice%n(k) + nuc_n
            ice%q(k) = ice%q(k) + nuc_q
            atmo%qv(k)  = atmo%qv(k)  - nuc_q

          END IF
        END IF
      ENDDO
    END IF


  END SUBROUTINE ice_nucleation_homhet


  SUBROUTINE ice_nucleation_het_inas(kstart,kend,atmo,cloud,ice,ninact,inuc,use_prog_in,n_inpot,ndiag_mask,nuc_n_a)
    !*******************************************************************************
    ! INAS-based ice nucleation scheme                                            *
    !                                                                              *
    ! Ullrich et al. 2007, A new ice nucleation active site parameterization for   *
    ! desert dust and soot, J. Atmos. Sci, 2017                                    *
    !*******************************************************************************
    INTEGER, INTENT(in)             :: kstart, kend
    TYPE(atmosphere), INTENT(inout) :: atmo
    CLASS(particle),  INTENT(inout) :: cloud, ice
    INTEGER, INTENT(in)             :: inuc
    REAL(wp),INTENT(inout),OPTIONAL :: &
         &  ninact(:)             !< Number of already nucleated ice crystals, unit here: [m-3]

    INTEGER                         ::  &
         &  k,                          & !< Loop index
         &  nmodes,imodes,              & !< Number and counter of modes
         &  idx                           !< Indexing variable
    REAL(wp) ::               &
         &  ndust,               &  ! Number density of dust mode
         &  ddust,               &  ! Mean diameter of dust mode
         &  sdust,               &  ! Surface area of dust mode
         &  ndustmax(3),         & !< Maximum of number density of dust mode
         &  nsoot, dsoot,        & !< Number and diameter of soot
         &  norg, dorg,          & !< Number and diameter of organic particles
         &  sigdust1, sigdust2,  & !< Standard deviations of mineral dust distributions
         &  sigsoot,sigorg,      & !< Standard deviations of soot and organic particle distributions
         &  inas,                & !< Ice nucleation active surface site density in 1/m2
         &  e_v, ssw, tk,        & !< vapor pressure, supersaturation, temperature
         &  nuc_n,nuc_q            !< Nucleated number and mass
    REAL(wp)                  :: &
         &  nice, nhet,          & !< Number of heterogeneously formed ice crystals
         &  ssimax, sswmax, inpmax, qcmax, sfactor  !< Additional output and debugging

    REAL(wp), DIMENSION(kstart:kend) :: &
         & ssi,   &  ! supersaturation over ice
         & idust, &  ! total number density of mineral dust particles
         & inp       ! total number density of ice nucleating particles

    REAL(wp), PARAMETER, DIMENSION(1:3) ::            & ! constant values used
         & ndust_background = (/ 1e3, 1e3, 1e2/),     & ! as dust background
         & ddust_background = (/ 0.2_wp, 0.4_wp, 0.6_wp/) * 1e-6, &
         & sdust_background = (/ 1.7_wp, 1.6_wp, 1.5_wp/)

    REAL(wp), PARAMETER, DIMENSION(1:5) :: &
         & param_dust = (/ 286.0_wp, 0.017_wp, 256.7_wp, 0.080_wp, 200.75_wp/),   &   ! dust of Ullrich et al. (2007)
         & param_soot = (/  58.0_wp, 0.010_wp, 200.0_wp, 0.015_wp, 240.00_wp /)       ! sool of Ullrich

    REAL(wp), PARAMETER :: &
         & ssinuc = 0.02_wp  ! ice supersaturation threshold for heterogeneous nucleation

    LOGICAL :: lnuc_k  ! per-level nucleation-active mask (replaces the old i,k gather/compaction list)

    LOGICAL, PARAMETER :: use_ninact = .true.
    LOGICAL, PARAMETER :: debug = .false.
    !>>NO_03062022: prognostic INPs in INAS scheme
    LOGICAL, INTENT(in)   :: use_prog_in
    REAL(wp), DIMENSION(:), OPTIONAL :: n_inpot
    REAL(wp), INTENT(out) :: nuc_n_a(kstart:kend)
    LOGICAL, INTENT(out)  :: ndiag_mask(kstart:kend)
    LOGICAL :: lwrite_n_inpot
    !<<NO_03062022

    sfactor = 10**(inuc-1)

    IF (kstart.eq.50) THEN
      WRITE(txt,'(A,F6.1)') "ice_nucleation_het_inas, sfactor = ",sfactor ; CALL message(TRIM(routine),TRIM(txt))
    END IF

    ! scalar dummies for soot and organics
    nsoot   = 0.0_wp
    dsoot   = 10.0e-09_wp
    norg    = 0.0_wp
    dorg    = 1.0e-09_wp
    sigorg  = 1.5_wp
    sigsoot = 1.4_wp

    nmodes = 3
    sswmax = 0.0_wp

    ! .. Single-column direct level loop, replacing the original i,k
    !    gather/compaction index list (ii(j), kk(j)) which was a CPU/NEC
    !    vectorization trick, not physics, and is meaningless for one column.
    !    lnuc_k below reproduces the exact condition that used to gate
    !    inclusion in the index list.
    DO k = kstart, kend

      ssi(k) = atmo%qv(k)*atmo%T(k)*r_d/e_es(atmo%T(k)) - 1.0_wp

      lnuc_k = (ssi(k) > ssinuc .or. cloud%q(k) > 1e-20_wp) .and. atmo%T(k) < 265.0_wp .and. atmo%T(k) > 190.0_wp

      IF (lnuc_k) THEN

        ! .. calculate INPs for dust using INAS parameterization (loop over dust modes)
        inp(k) = 0.0_wp
        ndustmax(:) = 0.0_wp
        DO imodes = 1, nmodes

          sigdust2 = pi * EXP( 2._wp * LOG( sdust_background(imodes) )**2 )

          ! number, diameter and surface area of pre-defined dust modes
          ndust = ndust_background(imodes) * sfactor
          ddust = ddust_background(imodes)
          sdust = sigdust2 * ddust**2
          ndustmax(imodes) = max(ndust,ndustmax(imodes))

          ssw = atmo%qv(k)*atmo%T(k)*r_d/e_ws(atmo%T(k))
          sswmax = max(sswmax,ssw)

          IF (ssw > 0.99_wp .and.  atmo%T(k) > 235.0_wp ) THEN
            ! immersion freezing
            inas = EXP( 151.548_wp - 0.521_wp*atmo%T(k) )
            !>>NO_03062022: prognostic inps
            IF (use_prog_in) THEN
                inp(k) = n_inpot(k) + ndust * (1.0_wp - EXP(-MAX(MIN(inas*sdust,30.0_wp),0.0_wp)))
            ELSE
                inp(k) = inp(k) + ndust * (1.0_wp - EXP(-MAX(MIN(inas*sdust,30.0_wp),0.0_wp))) !
            END IF
            !<<NO_03062022
          ELSE
           ! deposition nucleation
           inas = het_icenuc_inas_depo(atmo%T(k),ssi(k),param_dust)
            !>>NO_03062022: prognostic inps
            IF (use_prog_in) THEN
                inp(k) =  n_inpot(k) + ndust * (1.0_wp - EXP(-MAX(MIN(inas*sdust,30.0_wp),0.0_wp)))
            ELSE
                inp(k) =  inp(k) + ndust * (1.0_wp - EXP(-MAX(MIN(inas*sdust,30.0_wp),0.0_wp)))
            END IF
            !<<NO_03062022

          END IF

        ENDDO

        ! .. actual update and time integration
        nhet  = MIN(inp(k), ni_het_max)
        if (use_ninact) then
          nuc_n = MAX(nhet - ninact(k),0.0_wp)
        else
          nuc_n = MAX(nhet - ice%n(k),0.0_wp)
        end if
        nuc_q = MIN(nuc_n * ice%x_min, atmo%qv(k))
        nuc_n = nuc_q / ice%x_min

        ice%n(k)   = ice%n(k)   + nuc_n
        ice%q(k)   = ice%q(k)   + nuc_q
        atmo%qv(k) = atmo%qv(k) - nuc_q
        ninact(k)  = ninact(k)  + nuc_n

        !>>NO_03062022: prognostic inps
        lwrite_n_inpot = use_prog_in .AND. inp(k) .GT. 1.0e-12_wp
        ndiag_mask(k) = lwrite_n_inpot

        nuc_n_a(k) = nuc_n
        !<<NO_03062022

      END IF

    END DO

    IF (debug) THEN
      qcmax  = MAXVAL(cloud%q(kstart:kend))
      ssimax = MAXVAL(ssi(kstart:kend))
      inpmax = MAXVAL(inp(kstart:kend))
      CALL message (TRIM(routine),TRIM("ice_nucleation_het_inas"))
      WRITE(txt,'(A,F6.1)')    "   sfactor = ",sfactor ; CALL message(TRIM(routine),TRIM(txt))
      WRITE(*,'(A,4(A,E10.3)))') "   max ssi = ",ssimax,", max ssw = ",sswmax, &
           &                                           " max inp = ",inpmax
    END IF

  END SUBROUTINE ice_nucleation_het_inas


  FUNCTION het_icenuc_inas_depo(tk,ssi,param) RESULT(inas)
    !
    ! INAS-based deposition nucleation formula, see Ullrich et al. (2007, JAS)
    !
    REAL(wp),INTENT(in)                :: tk           !< temperature in K
    REAL(wp),INTENT(in)                :: ssi          !< supersaturation over ice
    REAL(wp),INTENT(in)                :: param(5)     !< storage for constant parameters
    REAL(wp)                           :: inas         !< ice nucleating active site density in m^-2
    REAL(wp)                           :: acotan       !< Arc cotangent
    REAL(wp)                           :: temp

    temp   = MIN(MAX(tk,190.0_wp),260.0_wp)
    acotan = pi/2._wp - ATAN(param(4) * (temp - param(5)))

    inas = EXP( param(1)*EXP(0.25_wp*LOG(MIN(ssi,1.0))) * COS( (param(2) * (temp - param(3)))**2 ) * acotan / pi )
    inas = MAX(MIN(inas,1e15),1e5) ! paper recommends upper limit of 1e15

  END FUNCTION het_icenuc_inas_depo


  SUBROUTINE ice_nucleation_agi_dm95(kstart, kend,atmo,ice,cloud,n_inact,n_inagi,nuc_n_a)
    !**************************************************************************
    ! Ice nucleation parameterization for silver iodide (AgI)                 *
    !                                                                         *
    ! Xue et al., 2013: Implementation of a silver iodide cloud-seeding       *
    ! parameterization in WRF. Part I: Model description and idealized        *
    ! 2D sensitivity tests - based on DeMott 1995                             *
    !                                                                         *
    ! We discard deposition nucleation as its contribution to INPs            *  
    ! is negligible. We also discard also condensation freezing               *
    ! and contact freezing, as AgI from flares is so hygroscopic              *
    ! everything will freeze via immersion freezing.                          *  
    !**************************************************************************

    INTEGER, INTENT(in)             :: kstart, kend
    TYPE(atmosphere), INTENT(inout) :: atmo
    CLASS(particle),  INTENT(inout) :: cloud, ice 
    
    REAL(wp),INTENT(inout), DIMENSION(:) :: n_inact

    REAL(wp), INTENT(out) :: &
         nuc_n_a(kstart:kend)
    
    REAL(wp), DIMENSION(:), INTENT(inout), OPTIONAL :: n_inagi

    INTEGER                         ::  &
         &  i, k              
   
    REAL(wp), DIMENSION(kstart:kend) :: &
         & iagi              ! total number density of agi particles
      
    REAL(wp)  ::      &
         & fimm,      &      ! fraction of immersed particles in droplets
         & nuc_n,     &      ! number of nucleated ice cystals
         & nuc_q,     &      ! mass of nucleated ice crystals 
         & fimf,      &      ! fraction of immersion freezing
         & Tlow,      &      ! lower temperature bound, below this freezing is constant
         & Timf,      &      ! temperatuer immersion freezing sets in
         & a_coef,    &      ! a coefficient for immersion freezing
         & b_coef,    &      ! b coefficient for immersion freezing
         & T0,        &      ! T0 temperature increment needed in freezing
         & nhet              ! number of heterogenously nucleated particles

    
    IF (PRESENT(n_inagi)) THEN
        iagi(kstart:kend) = n_inagi(kstart:kend)
    ELSE
        CALL finish(TRIM(routine),'Error in n_inagi for AgI ice nucleation')
    END IF
    
    T0 = 10._wp
    fimm = 1._wp
    Tlow = 248._wp
    Timf = 268.2_wp
    a_coef = 0.0274_wp
    b_coef = 3.3_wp

    DO k = kstart, kend

      IF ( atmo%T(k) >= Timf ) THEN
        nuc_n_a(k) = 0.0_wp
      ELSE
          IF ( atmo%T(k) >= Tlow .AND. iagi(k) > 0.0_wp ) THEN
             fimf = a_coef * fimm * ((Timf - atmo%T(k))/T0) ** b_coef  ! Eq. 4 in Xue et al., 2013
          ELSE 
             fimf = a_coef * fimm * ((Timf - Tlow)/T0) ** b_coef  ! Eq. 4 in Xue et al., 2013
          END IF

          nhet = iagi(k) * fimf

          ! we scale the freezing by the available cloud droplets as we 
          ! have an immersion freezing process
          nuc_n = MIN(nhet, cloud%n(k))
          nuc_q = MIN(nuc_n * ice%x_min, atmo%qv(k))
          nuc_n = nuc_q / ice%x_min
              
          ice%n(k)   = ice%n(k)   + nuc_n
          ice%q(k)   = ice%q(k)   + nuc_q
          cloud%q(k) = cloud%q(k) - nuc_q
          cloud%n(k) = cloud%n(k) - nuc_n
          n_inact(k) = n_inact(k) + nuc_n 

          nuc_n_a(k) = nuc_n
          n_inagi(k) = n_inagi(k) - nuc_n
      END IF
    ENDDO

  END SUBROUTINE ice_nucleation_agi_dm95

  !<<NO_24062022

  !>>NO_24052023: implementation of AgI parameterization for seeding
  SUBROUTINE ice_nucleation_agi_m16(kstart, kend,atmo,ice,cloud,n_inact,n_inagi,nuc_n_a)
    !**************************************************************************
    ! Ice nucleation parameterization for silver iodide (AgI)                 *
    !                                                                         *
    ! Marcolli et al., 2016: Ice nucleation efficiency of AgI: review and     *
    ! insights                                                                * 
    !                                                                         *
    ! Based on the data in Figure 1 we fit a sigmoidal curve to the 400 nm    *
    ! particle size. Comparing it to DeMott 1995, a much higher activity is   *
    ! notable.                                                                *
    ! The equation has the form: y = -b / (1 + exp(-k(x-x0))) + b             *
    !**************************************************************************

    INTEGER, INTENT(in)             :: kstart, kend
    TYPE(atmosphere), INTENT(inout) :: atmo
    CLASS(particle),  INTENT(inout) :: cloud, ice 
    REAL(wp),INTENT(inout), DIMENSION(:) :: n_inact

    REAL(wp), INTENT(out) :: &
         nuc_n_a(kstart:kend)
    
    REAL(wp), DIMENSION(:), INTENT(inout), OPTIONAL :: n_inagi

    INTEGER                         ::  &
         &  i, k              
   
    REAL(wp), DIMENSION(kstart:kend) :: &
         & iagi,      &      ! total number density of agi particles
         & qc_old
      
    REAL(wp)  ::      &
         & nuc_n,     &      ! number of nucleated ice cystals
         & nuc_q,     &      ! mass of nucleated ice crystals 
         & fimf,      &      ! fraction of immersion freezing
         & nhet,      &      ! number of heterogenously nucleated particles
         & T0, m, b,  &      ! fitting parameters for sigmoid curve
         & lhm               ! latent heat computation at given tempetarure

    
    IF (PRESENT(n_inagi)) THEN
        iagi(kstart:kend) = n_inagi(kstart:kend)
    ELSE
        CALL finish(TRIM(routine),'Error in n_inagi for AgI ice nucleation')
    END IF
    
    T0 = 263.95_wp
    m  = 0.88_wp
    b  = 0.97_wp

    IF (kstart.eq.50) THEN
      WRITE(txt,'(A,F6.1)') "M16 agi param = ",m ; CALL message(TRIM(routine),TRIM(txt))
    END IF

    DO k = kstart, kend

      fimf = -b / (1 + EXP(-m * (atmo%T(k)-T0))) + b

      nhet = iagi(k) * fimf

      ! we scale the freezing by the available cloud droplets as we 
      ! have an immersion freezing process
      nuc_n = MIN(nhet, cloud%n(k))
      nuc_q = MIN(nuc_n * ice%x_min, atmo%qv(k))
      nuc_n = nuc_q / ice%x_min
          
      ice%n(k)   = ice%n(k)   + nuc_n
      ice%q(k)   = ice%q(k)   + nuc_q
      n_inact(k) = n_inact(k) + nuc_n 

      qc_old(k)  = cloud%q(k)
      cloud%q(k) = cloud%q(k) - nuc_q
          
      cloud%n(k) = cloud%n(k) - nuc_n

      nuc_n_a(k) = nuc_n
      n_inagi(k) = n_inagi(k) - nuc_n

    ENDDO

  END SUBROUTINE ice_nucleation_agi_m16


  SUBROUTINE ice_nucleation_het_philips(kstart, kend, atmo, ice, cloud, &
       use_prog_in, n_inact, ndiag_mask, nuc_n_a, n_inpot)
    INTEGER, INTENT(in) :: kstart, kend
    TYPE(atmosphere), INTENT(inout) :: atmo
    CLASS(particle), INTENT(in) :: ice, cloud
    LOGICAL, INTENT(in) :: use_prog_in
    REAL(wp), INTENT(inout), DIMENSION(:) :: n_inact
    REAL(wp), INTENT(out) :: &
         nuc_n_a(kstart:kend)
    LOGICAL, INTENT(out) :: &
         ndiag_mask(kstart:kend)
    REAL(wp), DIMENSION(:), OPTIONAL :: n_inpot

    REAL(wp)             :: nuc_n, nuc_q
    REAL(wp)             :: T_a, ssi, e_si
    REAL(wp)             :: ndiag, ndiag_dust, ndiag_all
    REAL(wp), PARAMETER  :: eps  = 1.0e-20_wp
    REAL(wp) :: infrac(3)
    LOGICAL :: lwrite_n_inpot

    ! variables for interpolation in look-up table (real is good enough here)
    REAL      :: xt,xs,ssr
    INTEGER   :: ss,tt

    INTEGER :: k


    DO k = kstart,kend
!NEC$ ivdep

      T_a  = atmo%T(k)
      e_si = e_es(T_a)
      ssi  = atmo%qv(k) * R_d * T_a / e_si

      IF (T_a < T_nuc .AND. T_a > 180.0_wp .AND. ssi > 1.0_wp  &
           & .AND. ( n_inact(k) < ni_het_max*cfg_params%in_fact ) ) THEN

        xt = (274.- REAL(atmo%T(k)))  / ttstep
        xt = MIN(xt,REAL(ttmax-1))
        tt = INT(xt)

        IF (cloud%q(k) > eps) THEN
          ! immersion freezing at water saturation
          ! Phillips scheme
          ! immersion freezing at water saturation
          infrac(1) = (AINT(xt) + 1.0_wp - xt) * afrac_dust(tt,99) &
               &        + (xt - AINT(xt)) * afrac_dust(tt+1,99)
          infrac(2) = (AINT(xt) + 1.0_wp - xt) * afrac_soot(tt,99) &
               &        + (xt - AINT(xt)) * afrac_soot(tt+1,99)
          infrac(3) = (AINT(xt) + 1.0_wp - xt) * afrac_orga(tt,99) &
               &        + (xt-AINT(xt)) * afrac_orga(tt+1,99)
        ELSE
          ! deposition nucleation below water saturation
          ! calculate indices used for 2D look-up tables
          xs = 100. * REAL(ssi-1.0_wp) / ssstep
          xs = MIN(xs,REAL(ssmax-1))
          ss = MAX(1,INT(xs))
          ssr = MAX(1.0, AINT(xs))
          ! bi-linear interpolation in look-up tables
          infrac(1) =   (AINT(xt) + 1.0_wp - xt) * (ssr + 1.0_wp - xs) &
               &        * afrac_dust(tt, ss) &
               &      + (xt - AINT(xt)) * (ssr + 1.0_wp - xs) &
               &        * afrac_dust(tt+1, ss) &
               &      + (AINT(xt) + 1.0_wp - xt) * (xs - ssr) &
               &        * afrac_dust(tt, ss+1) &
               &      + (xt - AINT(xt)) * (xs - ssr) &
               &        * afrac_dust(tt+1, ss+1)
          infrac(2) =   (AINT(xt) + 1.0_wp - xt) * (ssr + 1.0_wp - xs) &
               &        * afrac_soot(tt, ss) &
               &      + (xt - AINT(xt)) * (ssr + 1.0_wp - xs) &
               &        * afrac_soot(tt+1, ss  ) &
               &      + (AINT(xt) + 1.0_wp - xt) * (xs - ssr) &
               &        * afrac_soot(tt, ss + 1) &
               &      + (xt - AINT(xt)) * (xs - ssr) &
               &        * afrac_soot(tt+1, ss+1)
          infrac(3) = (AINT(xt) + 1.0_wp - xt) * (ssr + 1.0_wp - xs) &
               &        * afrac_orga(tt,ss) &
               &      + (xt - AINT(xt)) * (ssr + 1.0_wp - xs) &
               &        * afrac_orga(tt+1, ss) &
               &      + (AINT(xt) + 1.0_wp - xt) * (xs - ssr) &
               &        * afrac_orga(tt, ss+1) &
               &      + (xt - AINT(xt)) * (xs - ssr) &
               &        * afrac_orga(tt+1, ss+1)
        END IF
          
        ! sum up the three modes
        IF (use_prog_in) THEN
          ! n_inpot replaces na_dust, na_soot and na_orga are assumed to be constant
          ndiag  = n_inpot(k) * infrac(1) + na_soot * infrac(2) + na_orga * infrac(3)
          ndiag_dust = n_inpot(k) * infrac(1)
          ndiag_all = ndiag
        ELSE
          ! all aerosol species are diagnostic
          ndiag = na_dust * infrac(1) + na_soot * infrac(2) + na_orga * infrac(3)
          ndiag_dust = ndiag
          ndiag_all = ndiag
        END IF
        ndiag = MIN(ndiag,ni_het_max)

        nuc_n = MAX(ndiag*cfg_params%in_fact - n_inact(k),0.0_wp)
        nuc_q = MIN(nuc_n * ice%x_min, atmo%qv(k))
        nuc_n = nuc_q / ice%x_min

        ice%n(k)   = ice%n(k)   + nuc_n
        ice%q(k)   = ice%q(k)   + nuc_q
        atmo%qv(k) = atmo%qv(k) - nuc_q
        n_inact(k) = n_inact(k) + nuc_n

        lwrite_n_inpot = use_prog_in .AND. ndiag .GT. 1.0e-12_wp
        ndiag_mask(k) = lwrite_n_inpot

        IF (lwrite_n_inpot) THEN
          nuc_n = nuc_n * ndiag_dust / ndiag_all
        END IF
        nuc_n_a(k) = nuc_n

      ELSE
        nuc_n_a(k) = 0.0_wp
        ndiag_mask(k) = .FALSE.
      ENDIF

    END DO

  END SUBROUTINE ice_nucleation_het_philips


  SUBROUTINE vapor_dep_relaxation(kstart, kend, dt_local, &
       &               ice_coeffs, snow_coeffs, graupel_coeffs, hail_coeffs, &
       &               atmo, ice_in, snow_in, graupel_in, hail_in, dep_rate_ice, dep_rate_snow)
    !*******************************************************************************
    ! Deposition and sublimation                                                   *
    !*******************************************************************************
    INTEGER, INTENT(in) :: kstart, kend
    TYPE(atmosphere)    :: atmo
    CLASS(particle), INTENT(INOUT), TARGET :: ice_in, snow_in, graupel_in, hail_in
    CLASS(particle), POINTER :: ice, snow, graupel, hail ! ACCWA (nvhpc 22.7, IPSF, see above)
    CLASS(particle_sphere), INTENT(IN) :: ice_coeffs, snow_coeffs, graupel_coeffs, hail_coeffs
    REAL(wp), INTENT(IN) :: dt_local
    REAL(wp), INTENT(INOUT), DIMENSION(:) :: dep_rate_ice, dep_rate_snow

    REAL(wp), DIMENSION(SIZE(dep_rate_ice)) :: &
                                    & s_si,g_i,dep_ice,dep_snow,dep_graupel,dep_hail

    INTEGER :: k
    REAL(wp)            :: D_vtp
    REAL(wp)            :: zdt,qvsidiff,Xi_i,Xfac
    REAL(wp)            :: tau_i_i,tau_s_i,tau_g_i,tau_h_i
    REAL(wp), PARAMETER :: eps  = 1.0e-20_wp
    REAL(wp)            :: T_a
    REAL(wp)            :: e_si            !..saturation water pressure over ice
    REAL(wp)            :: e_d,p_a,dep_sum !,weight
    REAL(wp)            :: dep_ice_n,dep_snow_n,dep_graupel_n,dep_hail_n,x_i,x_s,x_g,x_h

    LOGICAL, PARAMETER  :: reduce_sublimation = .TRUE.
    REAL(wp), PARAMETER :: dep_n_fac = 0.5_wp  ! UB: if this new parameterization of n-reduction during sublimation
                                               !     really makes sense, move to a global constant or into the particle types

    IF (isdebug) CALL message(routine, "vapor_deposition_growth")

    ice => ice_in ! ACCWA (nvhpc 22.7, IPSF, see above)
    snow => snow_in
    graupel => graupel_in
    hail => hail_in



    DO k = kstart,kend
       p_a  = atmo%p(k)
       T_a  = atmo%T(k)
       IF (T_a < T_3) THEN
          e_d  = atmo%qv(k) * R_d * T_a
          e_si = e_es(T_a)
          s_si(k) = e_d / e_si - 1.0_wp    !..supersaturation over ice
          D_vtp = diffusivity(T_a,p_a)    !  D_v = 8.7602e-5 * T_a**(1.81) / p_a
          g_i(k) = 4.0_wp*pi / ( L_ed**2 / (K_T * R_d * T_a**2) + R_d * T_a / (D_vtp * e_si) )
       ELSE
          g_i(k)  = 0.0_wp
          s_si(k) = 0.0_wp
       ENDIF
    ENDDO


    CALL vapor_deposition_generic(kstart, kend, ice, ice_coeffs, g_i, s_si,dt_local, dep_ice)
    CALL vapor_deposition_generic(kstart, kend, snow, snow_coeffs, g_i, s_si, dt_local, dep_snow)
    CALL vapor_deposition_generic(kstart, kend, graupel, graupel_coeffs, g_i, s_si, dt_local, dep_graupel)
    CALL vapor_deposition_generic(kstart, kend, hail, hail_coeffs, g_i, s_si, dt_local, dep_hail)

    zdt = 1.0/dt_local

    DO k = kstart,kend

       T_a  = atmo%T(k)

       ! Deposition only below T_3, evaporation of melting particles at warmer T is treated elsewhere
       IF (T_a < T_3) THEN

          ! Depositional growth with relaxation time-scale approach based on:
          ! "A New Double-Moment Microphysics Parameterization for Application in Cloud and
          ! Climate Models. Part 1: Description" by H. Morrison, J.A.Curry, V.I. Khvorostyanov

          qvsidiff  = atmo%qv(k) - e_es(T_a)/(R_d*T_a)

          if (abs(qvsidiff).gt.eps) then

             ! deposition rates are already multiplied with dt_local, therefore divide them here
             tau_i_i  = zdt/qvsidiff*dep_ice(k)
             tau_s_i  = zdt/qvsidiff*dep_snow(k)
             tau_g_i  = zdt/qvsidiff*dep_graupel(k)
             tau_h_i  = zdt/qvsidiff*dep_hail(k)

             Xi_i = ( tau_i_i + tau_s_i + tau_g_i + tau_h_i )

             if (Xi_i.lt.eps) then
                Xfac = 0.0_wp
             else
                Xfac =  qvsidiff / Xi_i * (1.0_wp - EXP(- dt_local*Xi_i))
             end if

             dep_ice(k)     = Xfac * tau_i_i
             dep_snow(k)    = Xfac * tau_s_i
             dep_graupel(k) = Xfac * tau_g_i
             dep_hail(k)    = Xfac * tau_h_i

             ! this limiter should not be necessary
             IF (qvsidiff < 0.0_wp) THEN
                dep_ice(k)     = MAX(dep_ice(k),    -ice%q(k))
                dep_snow(k)    = MAX(dep_snow(k),   -snow%q(k))
                dep_graupel(k) = MAX(dep_graupel(k),-graupel%q(k))
                dep_hail(k)    = MAX(dep_hail(k),   -hail%q(k))
             END IF

             dep_sum = dep_ice(k) + dep_graupel(k) + dep_snow(k) + dep_hail(k)
                
             IF ( reduce_sublimation) THEN
               x_i = particle_meanmass(ice    , ice%    q(k), ice%    n(k))
               x_s = particle_meanmass(snow   , snow%   q(k), snow%   n(k))
               x_g = particle_meanmass(graupel, graupel%q(k), graupel%n(k))
               x_h = particle_meanmass(hail   , hail%   q(k), hail%   n(k))
             END IF

             ice%q(k)     = ice%q(k)     + dep_ice(k)
             snow%q(k)    = snow%q(k)    + dep_snow(k)
             graupel%q(k) = graupel%q(k) + dep_graupel(k)
             hail%q(k)    = hail%q(k)    + dep_hail(k)

             atmo%qv(k) = atmo%qv(k) - dep_sum

             ! .. If deposition rate is negative, parameterize the complete evaporation of some of the particles in a way
             !    that mean size is conserved times a tuning factor < 1:
             IF ( reduce_sublimation) THEN
               dep_ice_n      = MIN(dep_ice(k),0.0_wp) / x_i
               dep_snow_n     = MIN(dep_snow(k),0.0_wp) / x_s
               dep_graupel_n  = MIN(dep_graupel(k),0.0_wp) / x_g
               dep_hail_n     = MIN(dep_hail(k),0.0_wp) / x_h

               ice%n(k)     = MAX(ice%n(k)     + dep_n_fac*dep_ice_n    , 0.0_wp)
               snow%n(k)    = MAX(snow%n(k)    + dep_n_fac*dep_snow_n   , 0.0_wp)
               graupel%n(k) = MAX(graupel%n(k) + dep_n_fac*dep_graupel_n, 0.0_wp)
               hail%n(k)    = MAX(hail%n(k)    + dep_n_fac*dep_hail_n   , 0.0_wp)
             END IF
                
             dep_rate_ice(k)  = dep_rate_ice(k)  + dep_ice(k)
             dep_rate_snow(k) = dep_rate_snow(k) + dep_snow(k)

          END IF

       ENDIF
    ENDDO


  END SUBROUTINE vapor_dep_relaxation


  SUBROUTINE vapor_deposition_generic(kstart, kend, prtcl_in, coeffs, g_i, s_si, &
       dt, dep_q)
    INTEGER, INTENT(in) :: kstart, kend
    CLASS(particle), INTENT(in), TARGET :: prtcl_in
    CLASS(particle), POINTER :: prtcl ! ACCWA (nvhpc 22.7, IPSF, see above)
    CLASS(particle_coeffs), INTENT(in) :: coeffs
    REAL(wp), INTENT(in) :: g_i(:), s_si(:)
    REAL(wp), INTENT(in) :: dt
    REAL(wp), INTENT(out) :: dep_q(:)
    REAL(wp)            :: q,n,x,d,v,f_v
    INTEGER :: k

    prtcl => prtcl_in ! ACCWA (nvhpc 22.7, IPSF, see above)


    DO k = kstart,kend
      IF (prtcl%q(k) == 0.0_wp) THEN
        dep_q(k) = 0.0_wp
      ELSE
        n = prtcl%n(k)
        q = prtcl%q(k)

        x = particle_meanmass(prtcl,q,n)
        D = particle_diameter(prtcl,x)
        v = particle_velocity(prtcl,x) * prtcl%rho_v(k)

        !f_v = ( coeffs%a_f + coeffs%b_f * SQRT(D*v) ) * 2.0_wp
        f_v = coeffs%a_f + vent_coeff_b(prtcl,1) * (N_sc**n_f / sqrt(nu_l) * sqrt(D*v)) !* 3.0
        f_v = MAX(f_v,coeffs%a_f/prtcl%a_ven)
          
        dep_q(k) = g_i(k) * n * coeffs%c_i * d * f_v * s_si(k) * dt
      ENDIF
    ENDDO

  END SUBROUTINE vapor_deposition_generic


  SUBROUTINE setup_particle_coeffs(ptype,pcoeffs)
    CLASS(particle),        INTENT(in)    :: ptype
    CLASS(particle_coeffs), INTENT(inout) :: pcoeffs

    pcoeffs%c_i = 1.0 / ptype%cap
    pcoeffs%a_f = vent_coeff_a(ptype,1)
    pcoeffs%b_f = vent_coeff_b(ptype,1) * N_sc**n_f / sqrt(nu_l)
    pcoeffs%c_z = moment_gamma(ptype,2)

  END SUBROUTINE setup_particle_coeffs


  SUBROUTINE ice_melting(kstart, kend, atmo, ice_in, cloud, rain)
    !*******************************************************************************
    !                                                                              *
    !*******************************************************************************
    INTEGER, INTENT(in) :: kstart, kend
    TYPE(atmosphere), INTENT(inout) :: atmo
    CLASS(particle), INTENT(inout), TARGET :: ice_in
    CLASS(particle), POINTER :: ice ! ACCWA (nvhpc 22.7, IPSF, see above)
    CLASS(particle), INTENT(inout) :: cloud, rain
    INTEGER :: k
    REAL(wp)            :: q_i,x_i,n_i
    REAL(wp)            :: melt_q,melt_n

    IF (isdebug) CALL message(routine, "ice_melting")

    ice => ice_in ! ACCWA (nvhpc 22.7, IPSF, see above)


    DO k = kstart,kend

       q_i = ice%q(k)

       IF (atmo%T(k) > T_3 .AND. q_i > 0.0) THEN

         n_i = ice%n(k)
         x_i = particle_meanmass(ice, q_i,n_i)

         ! complete melting within this time step
         melt_q = q_i
         melt_n = n_i
         ice%q(k) = 0.0_wp
         ice%n(k) = 0.0_wp

         ! ice either melts into cloud droplets or rain depending on x_i
         IF (x_i > cloud%x_max) THEN
            rain%q(k)  = rain%q(k)  + melt_q
            rain%n(k)  = rain%n(k)  + melt_n
         ELSE
            cloud%q(k) = cloud%q(k) + melt_q
            cloud%n(k) = cloud%n(k) + melt_n
         ENDIF

       END IF
    END DO

  END SUBROUTINE ice_melting


 SUBROUTINE ccn_activation_hdcp2(kstart, kend, atmo, cloud)
   !*******************************************************************************
   !       Calculation of ccn activation                                          *
   !       using the approach of Hande et al 2015                                 *
   !*******************************************************************************
    IMPLICIT NONE
    INTEGER, INTENT(in) :: kstart, kend

    TYPE(atmosphere), INTENT(inout) :: atmo
    CLASS(particle), INTENT(inout)  :: cloud

    ! Locale Variablen

    INTEGER :: k, nuc_typ
    REAL(wp)           :: n_c,q_c
    REAL(wp)           :: nuc_n,nuc_q
    REAL(wp)           :: wcb,pres
    REAL(wp)           :: acoeff,bcoeff,ccoeff,dcoeff
    REAL(wp), PARAMETER:: eps = 1e-20_wp

    ! Data from HDCP2_CCN_params.txt for 20130417
    REAL(wp), PARAMETER :: &
         a_ccn(4) = (/  183230691.161_wp, 0.10147358938_wp, &
         &             -0.2922395814_wp, 229189886.226_wp /), &
         b_ccn(4) = (/ 0.0001984051994_wp, 4.473190485e-05_wp, &
         &             0.0001843225275_wp, 0.0001986158191_wp /), &
         c_ccn(4) = (/ 16.2420263911_wp, 3.22011836758_wp, &
         &             13.8499423719_wp, 16.2461600644_wp /), &
         d_ccn(4) = (/ 287736034.13_wp, 0.6258809883_wp, &
         &             0.8907491812_wp, 360848977.55_wp /)

#ifdef _OPENACC
    CALL finish(routine, 'ccn_activation_hdcp2 not available on GPU')
#endif


    nuc_typ = nuc_c_typ

    IF(isdebug) THEN
       WRITE(txt,*) "cloud_activation_hdcp2: nuc_typ = ",nuc_typ ; CALL message(routine,TRIM(txt))
    ENDIF

    DO k = kstart,kend

       nuc_q = 0.0d0
       nuc_n = 0.d0
       n_c   = cloud%n(k)
       q_c   = cloud%q(k)
       pres  = atmo%p(k)
       wcb   = atmo%w(k)

       if (q_c > eps .and. wcb > 0.0_wp) then

          ! Based on write-up of Luke Hande of 6 May 2015

          acoeff = a_ccn(1) * atan(b_ccn(1) * pres - c_ccn(1)) + d_ccn(1)
          bcoeff = a_ccn(2) * atan(b_ccn(2) * pres - c_ccn(2)) + d_ccn(2)
          ccoeff = a_ccn(3) * atan(b_ccn(3) * pres - c_ccn(3)) + d_ccn(3)
          dcoeff = a_ccn(4) * atan(b_ccn(4) * pres - c_ccn(4)) + d_ccn(4)

          nuc_n = acoeff * atan(bcoeff * log(wcb) + ccoeff) + dcoeff

          nuc_n = MAX(MAX(nuc_n,1.0e7_wp) - n_c,0.0_wp)

          nuc_q = MIN(nuc_n * cloud%x_min, atmo%qv(k))
          nuc_n = nuc_q / cloud%x_min

          cloud%n(k) = cloud%n(k) + nuc_n
          cloud%q(k) = cloud%q(k) + nuc_q
          atmo%qv(k) = atmo%qv(k) - nuc_q

       END IF

    END DO

  END SUBROUTINE ccn_activation_hdcp2


  SUBROUTINE ccn_activation_sk_4d(kstart, kend, ccn_coeffs, atmo, cloud, n_cn)
    !*******************************************************************************
    !       Calculation of cloud droplet nucleation                                *
    !       using the look-up tables by Segal and Khain 2006 (JGR, vol.11)         *
    !                                                                              *
    !       Difference to Heikes routine cloud_nucleation_SK()                     *
    !       Equidistant lookup table is used to enable better vectorization        *
    !       properties.                                                            *
    !*******************************************************************************

    INTEGER, INTENT(in), OPTIONAL :: kstart, kend

    ! parameters
    TYPE(aerosol_ccn),INTENT(in), OPTIONAL    :: ccn_coeffs

    ! 2mom variables
    TYPE(atmosphere), INTENT(inout), OPTIONAL :: atmo
    CLASS(particle),  INTENT(inout), OPTIONAL :: cloud
    REAL(wp), DIMENSION(:), OPTIONAL        :: n_cn

    ! local variables
    REAL(wp), PARAMETER :: nuc_eps = 1e-20_wp


    ! grid sizes of the original table:
    INTEGER, PARAMETER   :: n_r2 = 3, n_lsigs = 5, n_ncn = 8 , n_wcb = 4

    ! desired grid sizes of the new equidistant table:
    INTEGER, PARAMETER   :: nr2  = 3, nlsigs  = 5, nncn  = 129, nwcb  = 11

    ! more local variables
    REAL(wp)             :: n_c, q_c
    REAL(wp)             :: nuc_n, nuc_q
    REAL(wp)             :: ncn, n_cn0, lsigs, nccn, r2, wcb, wcb_min
    REAL(wp)             :: r2_loc, lsigs_loc, ncn_loc, wcb_loc
    REAL(wp)             :: z0_nccn, z1e_nccn, zf, etas
    INTEGER              :: k, kp1_fl
    INTEGER              :: iu, ju, ku, lu
    REAL(wp)             :: hilf1(2,2,2,2), hilf2(2,2,2), hilf3(2,2), hilf4(2)

    LOGICAL, PARAMETER   :: lincloud_nuc = .TRUE.

    ! call from init_2mom_scheme_once without arguments for initialization of tables
    IF (.NOT.PRESENT(kstart)) THEN
      CALL get_otab(n_r2,n_lsigs,n_ncn,n_wcb)     ! original look-up-table from Segal and Khain      
      CALL equi_table(nr2,nlsigs,nncn,nwcb)   ! construct the new equidistant table tab:
      RETURN
    END IF

    IF(isdebug) THEN
       WRITE(txt,*) "cloud_activation_sk_4d'"
       CALL message(routine, TRIM(txt))
    ENDIF

    !..parameter for exponential decrease of N_ccn with height:
    !  1) up to this height (m) constant unchanged value:
    !  2)  height interval at which N_ccn decreses by factor 1/e above z0_nccn:

    z0_nccn  = ccn_coeffs%z0
    z1e_nccn = ccn_coeffs%z1e
    n_cn0    = ccn_coeffs%Ncn0
    etas     = ccn_coeffs%etas
    wcb_min  = ccn_coeffs%wcb_min

    !..values for aerosol properties
    r2    = ccn_coeffs%R2
    lsigs = ccn_coeffs%lsigs

    

    DO k = kstart,kend
      kp1_fl = MIN(k+1,SIZE(atmo%rho))
!NEC$ ivdep

      ! hard upper limit for number conc that
      ! eliminates also unrealistic high value
      ! that would come from the dynamical core

      cloud%n(k) = MIN(cloud%n(k),n_cn0)

!!$ UB: determine wcb as in old COSMO version:
      ! determine vertical velocity for Segal&Khain nucleation parameterization:
      IF (lincloud_nuc) THEN
        ! ... incloud nucleation is allowed, look for height layers where qc (mass specific) increases with height and w is positive:
        !      (at the lowest model level, nucleation happens without the gradient check)
        IF ( cloud%q(k) > nuc_eps .AND. &
             ( k == kp1_fl .OR. cloud%q(k)/atmo%rho(k) > cloud%q(kp1_fl)/atmo%rho(kp1_fl) ) .AND. &
             atmo%w(k+1) > 0.0_wp ) THEN
          wcb = atmo%w(k+1)  ! take w of the lower cell face
        ELSE
          wcb = 0.0_wp ! set w for nucleation to 0.0, so that no new nucleation will take place below
        END IF
! We still miss the nucleation in fog situations during radiative cooling. This
! would require the inclusion of -cp/g*dT/dt|_diabatic in the effective
! nucleation velocity and allowing incloud nucleation everywhere, not only if qc
! increases with height.
      ELSE
        ! ... nucleation is allowed only in the model layer above cloud base:
        !      (the lowest model level always counts as cloud base if qc > nuc_eps and w > 0)
        IF ( (k == kp1_fl .OR. cloud%q(kp1_fl) <= nuc_eps) .AND. cloud%q(k) > nuc_eps .AND. atmo%w(k+1) > 0.0_wp) THEN
          wcb = atmo%w(k+1)    ! take w of the lower cell face
        ELSE
          wcb = 0.0_wp ! set w for nucleation to 0.0, so that no new nucleation will take place below
        END IF
      END IF

! previous formulation without the new lincloud_nuc mechanism:
!        IF (cloud%q(k) > eps .and. atmo%w(k) > 0.0_wp) THEN
! new formulation:
      IF ( wcb > 0.0_wp ) THEN

        nuc_q = 0.0_wp
        nuc_n = 0.0_wp
        n_c   = cloud%n(k)
        q_c   = cloud%q(k)
        wcb   = MAX(wcb, wcb_min)  ! enforce a minimal updraft for nucleation
 
        IF (PRESENT(n_cn)) THEN
          Ncn = n_cn(k) ! number of CN from prognostic variable
        ELSE
          zf = 0.5_wp*(atmo%zh(k)+atmo%zh(k+1))
          IF(zf > z0_nccn) THEN
            Ncn = n_cn0 * MIN(EXP((z0_nccn - zf)/z1e_nccn),1.0_wp)
          ELSE
            Ncn = n_cn0
          END IF
        END IF

        ! Interpolation of the look-up tables with respect to all 4 parameters:
        ! (clip values outside range to the marginal values)
        r2_loc    = MIN(MAX(r2,     tab%x1(1)), tab%x1(tab%n1))
        iu = MIN(FLOOR((r2_loc -    tab%x1(1)) * tab%odx1 ) + 1, tab%n1-1)
        lsigs_loc = MIN(MAX(lsigs,  tab%x2(1)), tab%x2(tab%n2))
        ju = MIN(FLOOR((lsigs_loc - tab%x2(1)) * tab%odx2 ) + 1, tab%n2-1)
        ncn_loc   = MIN(MAX(ncn,    tab%x3(1)), tab%x3(tab%n3))
        ku = MIN(FLOOR((ncn_loc -   tab%x3(1)) * tab%odx3 ) + 1, tab%n3-1)
        wcb_loc   = MIN(MAX(wcb,    tab%x4(1)), tab%x4(tab%n4))
        lu = MIN(FLOOR((wcb_loc -   tab%x4(1)) * tab%odx4 ) + 1, tab%n4-1)

        hilf1 = tab%ltable( iu:iu+1, ju:ju+1, ku:ku+1, lu:lu+1)
        hilf2 = hilf1(1,:,:,:) + (hilf1(2,:,:,:) - hilf1(1,:,:,:)) * tab%odx1 * ( r2_loc    - tab%x1(iu) )
        hilf3 = hilf2(1,:,:)   + (hilf2(2,:,:)   - hilf2(1,:,:)  ) * tab%odx2 * ( lsigs_loc - tab%x2(ju) )
        hilf4 = hilf3(1,:)     + (hilf3(2,:)     - hilf3(1,:)    ) * tab%odx3 * ( ncn_loc   - tab%x3(ku) )
        nccn  = hilf4(1)       + (hilf4(2)       - hilf4(1)      ) * tab%odx4 * ( wcb_loc   - tab%x4(lu) )

        ! If n_cn is outside the range of the lookup table values, resulting 
        ! NCCN are clipped to the margin values. For the case of these margin values
        ! beeing larger than n_cn (which happens sometimes, unfortunately), limit NCCN by n_cn:
        nccn = MIN(nccn, n_cn0)

        nuc_n = etas * nccn - n_c

        nuc_n = MAX(nuc_n,0.0d0)

        nuc_q = MIN(nuc_n * cloud%x_min,atmo%qv(k))
        nuc_n = nuc_q / cloud%x_min

        cloud%n(k) = cloud%n(k) + nuc_n
        cloud%q(k) = cloud%q(k) + nuc_q
        atmo%qv(k) = atmo%qv(k) - nuc_q

        IF (PRESENT(n_cn)) THEN
          n_cn(k) = n_cn(k) - MIN(Ncn,nuc_n)
        END IF

      END IF
    END DO

  END SUBROUTINE ccn_activation_sk_4d


  SUBROUTINE get_otab(n_r2,n_lsigs,n_ncn,n_wcb)

      INTEGER, INTENT(IN) :: n_r2,n_lsigs,n_ncn,n_wcb

      otab%n1 = n_r2
      otab%n2 = n_lsigs
      otab%n3 = n_ncn + 1
      otab%n4 = n_wcb + 1
      
      IF (.NOT. ASSOCIATED(otab%x1) ) THEN
        ALLOCATE( otab%x1(otab%n1) )
        ALLOCATE( otab%x2(otab%n2) )
        ALLOCATE( otab%x3(otab%n3) )
        ALLOCATE( otab%x4(otab%n4) )
        ALLOCATE( otab%ltable(otab%n1,otab%n2,otab%n3,otab%n4) )
      END IF
 
      ! original (non-)equidistant table vectors:
      ! r2:
      otab%x1  = (/0.02d0, 0.03d0, 0.04d0/)     ! in 10^(-6) m
      ! lsigs:
      otab%x2  = (/0.1d0, 0.2d0, 0.3d0, 0.4d0, 0.5d0/)
      ! n_cn: (UB: um 0.0 m**-3 ergaenzt zur linearen Interpolation zw. 0.0 und 50e6 m**-3)
      otab%x3  = (/0.0d6, 50.d06, 100.d06, 200.d06, 400.d06, 800.d06, 1600.d06, 3200.d06, 6400.d06/) ! in m**-3
      ! wcb: (UB: um 0.0 m/s ergaenzt zur linearen Interpolation zw. 0.0 und 0.5 m/s)
      otab%x4  = (/0.0d0, 0.5d0, 1.0d0, 2.5d0, 5.0d0/)

      ! look up table for NCCN activated at given R2, lsigs, Ncn and wcb:

      ! Ncn              50       100       200       400       800       1600      3200      6400
      ! table4a (R2=0.02mum, wcb=0.5m/s) (for Ncn=3200  and Ncn=6400 "extrapolated")
      otab%ltable(1,1,2:otab%n3,2) =  (/  42.2d06,  70.2d06, 112.2d06, 173.1d06, 263.7d06, 397.5d06, 397.5d06, 397.5d06/)
      otab%ltable(1,2,2:otab%n3,2) =  (/  35.5d06,  60.1d06, 100.0d06, 163.9d06, 264.5d06, 418.4d06, 418.4d06, 418.4d06/)
      otab%ltable(1,3,2:otab%n3,2) =  (/  32.6d06,  56.3d06,  96.7d06, 163.9d06, 272.0d06, 438.5d06, 438.5d06, 438.5d06/)
      otab%ltable(1,4,2:otab%n3,2) =  (/  30.9d06,  54.4d06,  94.6d06, 162.4d06, 271.9d06, 433.5d06, 433.5d06, 433.5d06/)
      otab%ltable(1,5,2:otab%n3,2) =  (/  29.4d06,  51.9d06,  89.9d06, 150.6d06, 236.5d06, 364.4d06, 364.4d06, 364.4d06/)
      ! table4b (R2=0.02mum, wcb=1.0m/s) (for Ncn=50 "interpolted" and Ncn=6400 extrapolated)
      otab%ltable(1,1,2:otab%n3,3) =  (/  45.3d06,  91.5d06, 158.7d06, 264.4d06, 423.1d06, 672.5d06, 397.5d06, 397.5d06/)
      otab%ltable(1,2,2:otab%n3,3) =  (/  38.5d06,  77.1d06, 133.0d06, 224.9d06, 376.5d06, 615.7d06, 418.4d06, 418.4d06/)
      otab%ltable(1,3,2:otab%n3,3) =  (/  35.0d06,  70.0d06, 122.5d06, 212.0d06, 362.1d06, 605.3d06, 438.5d06, 438.5d06/)
      otab%ltable(1,4,2:otab%n3,3) =  (/  32.4d06,  65.8d06, 116.4d06, 204.0d06, 350.6d06, 584.4d06, 433.5d06, 433.5d06/)
      otab%ltable(1,5,2:otab%n3,3) =  (/  31.2d06,  62.3d06, 110.1d06, 191.3d06, 320.6d06, 501.3d06, 364.4d06, 364.4d06/)
      ! table4c (R2=0.02mum, wcb=2.5m/s) (for Ncn=50 and Ncn=100 "interpolated")
      otab%ltable(1,1,2:otab%n3,4) =  (/  50.3d06, 100.5d06, 201.1d06, 373.1d06, 664.7d06,1132.8d06,1876.8d06,2973.7d06/)
      otab%ltable(1,2,2:otab%n3,4) =  (/  44.1d06,  88.1d06, 176.2d06, 314.0d06, 546.9d06, 941.4d06,1579.2d06,2542.2d06/)
      otab%ltable(1,3,2:otab%n3,4) =  (/  39.7d06,  79.5d06, 158.9d06, 283.4d06, 498.9d06, 865.9d06,1462.6d06,2355.8d06/)
      otab%ltable(1,4,2:otab%n3,4) =  (/  37.0d06,  74.0d06, 148.0d06, 264.6d06, 468.3d06, 813.3d06,1371.3d06,2137.2d06/)
      otab%ltable(1,5,2:otab%n3,4) =  (/  34.7d06,  69.4d06, 138.8d06, 246.9d06, 432.9d06, 737.8d06,1176.7d06,1733.0d06/)
      ! table4d (R2=0.02mum, wcb=5.0m/s) (for Ncn=50,100,200 "interpolated")
      otab%ltable(1,1,2:otab%n3,5) =  (/  51.5d06, 103.1d06, 206.1d06, 412.2d06, 788.1d06,1453.1d06,2585.1d06,4382.5d06/)
      otab%ltable(1,2,2:otab%n3,5) =  (/  46.6d06,  93.2d06, 186.3d06, 372.6d06, 657.2d06,1202.8d06,2098.0d06,3556.9d06/)
      otab%ltable(1,3,2:otab%n3,5) =  (/  70.0d06,  70.0d06, 168.8d06, 337.6d06, 606.7d06,1078.5d06,1889.0d06,3206.9d06/)
      otab%ltable(1,4,2:otab%n3,5) =  (/  42.2d06,  84.4d06, 166.4d06, 312.7d06, 562.2d06,1000.3d06,1741.1d06,2910.1d06/)
      otab%ltable(1,5,2:otab%n3,5) =  (/  36.5d06,  72.9d06, 145.8d06, 291.6d06, 521.0d06, 961.1d06,1551.1d06,2444.6d06/)
      ! table5a (R2=0.03mum, wcb=0.5m/s)  (for Ncn=3200  and Ncn=6400 "extrapolated")
      otab%ltable(2,1,2:otab%n3,2) =  (/  50.0d06,  95.8d06, 176.2d06, 321.6d06, 562.3d06, 835.5d06, 835.5d06, 835.5d06/)
      otab%ltable(2,2,2:otab%n3,2) =  (/  44.7d06,  81.4d06, 144.5d06, 251.5d06, 422.7d06, 677.8d06, 677.8d06, 677.8d06/)
      otab%ltable(2,3,2:otab%n3,2) =  (/  40.2d06,  72.8d06, 129.3d06, 225.9d06, 379.9d06, 606.5d06, 606.5d06, 606.5d06/)
      otab%ltable(2,4,2:otab%n3,2) =  (/  37.2d06,  67.1d06, 119.5d06, 206.7d06, 340.5d06, 549.4d06, 549.4d06, 549.4d06/)
      otab%ltable(2,5,2:otab%n3,2) =  (/  33.6d06,  59.0d06,  99.4d06, 150.3d06, 251.8d06, 466.0d06, 466.0d06, 466.0d06/)
      ! table5b (R2=0.03mum, wcb=1.0m/s) (Ncn=50 "interpolated", Ncn=6400 "extrapolated)
      otab%ltable(2,1,2:otab%n3,3) =  (/  50.7d06, 101.4d06, 197.6d06, 357.2d06, 686.6d06,1186.4d06,1892.2d06,1892.2d06/)
      otab%ltable(2,2,2:otab%n3,3) =  (/  46.6d06,  93.3d06, 172.2d06, 312.1d06, 550.7d06, 931.6d06,1476.6d06,1476.6d06/)
      otab%ltable(2,3,2:otab%n3,3) =  (/  42.2d06,  84.4d06, 154.0d06, 276.3d06, 485.6d06, 811.2d06,1271.7d06,1271.7d06/)
      otab%ltable(2,4,2:otab%n3,3) =  (/  39.0d06,  77.9d06, 141.2d06, 251.8d06, 436.7d06, 708.7d06,1117.7d06,1117.7d06/)
      otab%ltable(2,5,2:otab%n3,3) =  (/  35.0d06,  70.1d06, 123.9d06, 210.2d06, 329.9d06, 511.9d06, 933.4d06, 933.4d06/)
      ! table5c (R2=0.03mum, wcb=2.5m/s) (for Ncn=50 and Ncn=100 "interpolated")
      otab%ltable(2,1,2:otab%n3,4) =  (/  51.5d06, 103.0d06, 205.9d06, 406.3d06, 796.4d06,1524.0d06,2781.4d06,4609.3d06/)
      otab%ltable(2,2,2:otab%n3,4) =  (/  49.6d06,  99.1d06, 198.2d06, 375.5d06, 698.3d06,1264.1d06,2202.8d06,3503.6d06/)
      otab%ltable(2,3,2:otab%n3,4) =  (/  45.8d06,  91.6d06, 183.2d06, 339.5d06, 618.9d06,1105.2d06,1881.8d06,2930.9d06/)
      otab%ltable(2,4,2:otab%n3,4) =  (/  42.3d06,  84.7d06, 169.3d06, 310.3d06, 559.5d06, 981.7d06,1611.6d06,2455.6d06/)
      otab%ltable(2,5,2:otab%n3,4) =  (/  38.2d06,  76.4d06, 152.8d06, 237.3d06, 473.3d06, 773.1d06,1167.9d06,1935.0d06/)
      ! table5d (R2=0.03mum, wcb=5.0m/s) (for Ncn=50,100,200 "interpolated")
      otab%ltable(2,1,2:otab%n3,5) =  (/  51.9d06, 103.8d06, 207.6d06, 415.1d06, 819.6d06,1616.4d06,3148.2d06,5787.9d06/)
      otab%ltable(2,2,2:otab%n3,5) =  (/  50.7d06, 101.5d06, 203.0d06, 405.9d06, 777.0d06,1463.8d06,2682.6d06,4683.0d06/)
      otab%ltable(2,3,2:otab%n3,5) =  (/  47.4d06,  94.9d06, 189.7d06, 379.4d06, 708.7d06,1301.3d06,2334.3d06,3951.8d06/)
      otab%ltable(2,4,2:otab%n3,5) =  (/  44.0d06,  88.1d06, 176.2d06, 352.3d06, 647.8d06,1173.0d06,2049.7d06,3315.6d06/)
      otab%ltable(2,5,2:otab%n3,5) =  (/  39.7d06,  79.4d06, 158.8d06, 317.6d06, 569.5d06, 988.5d06,1615.6d06,2430.3d06/)
      ! table6a (R2=0.04mum, wcb=0.5m/s) (for Ncn=3200  and Ncn=6400 "extrapolated")
      otab%ltable(3,1,2:otab%n3,2) =  (/  50.6d06, 100.3d06, 196.5d06, 374.7d06, 677.3d06,1138.9d06,1138.9d06,1138.9d06/)
      otab%ltable(3,2,2:otab%n3,2) =  (/  48.4d06,  91.9d06, 170.6d06, 306.9d06, 529.2d06, 862.4d06, 862.4d06, 862.4d06/)
      otab%ltable(3,3,2:otab%n3,2) =  (/  44.4d06,  82.5d06, 150.3d06, 266.4d06, 448.0d06, 740.7d06, 740.7d06, 740.7d06/)
      otab%ltable(3,4,2:otab%n3,2) =  (/  40.9d06,  75.0d06, 134.7d06, 231.9d06, 382.1d06, 657.6d06, 657.6d06, 657.6d06/)
      otab%ltable(3,5,2:otab%n3,2) =  (/  34.7d06,  59.3d06,  93.5d06, 156.8d06, 301.9d06, 603.8d06, 603.8d06, 603.8d06/)
      ! table6b (R2=0.04mum, wcb=1.0m/s) (Ncn=50 "interpolated", Ncn=6400 "extrapolated)
      otab%ltable(3,1,2:otab%n3,3) =  (/  50.9d06, 101.7d06, 201.8d06, 398.8d06, 773.7d06,1420.8d06,2411.8d06,2411.8d06/)
      otab%ltable(3,2,2:otab%n3,3) =  (/  49.4d06,  98.9d06, 189.7d06, 356.2d06, 649.5d06,1117.9d06,1805.2d06,1805.2d06/)
      otab%ltable(3,3,2:otab%n3,3) =  (/  45.6d06,  91.8d06, 171.5d06, 314.9d06, 559.0d06, 932.8d06,1501.6d06,1501.6d06/)
      otab%ltable(3,4,2:otab%n3,3) =  (/  42.4d06,  84.7d06, 155.8d06, 280.5d06, 481.9d06, 779.0d06,1321.9d06,1321.9d06/)
      otab%ltable(3,5,2:otab%n3,3) =  (/  36.1d06,  72.1d06, 124.4d06, 198.4d06, 319.1d06, 603.8d06,1207.6d06,1207.6d06/)
      ! table6c (R2=0.04mum, wcb=2.5m/s) (for Ncn=50 and Ncn=100 "interpolated")
      otab%ltable(3,1,2:otab%n3,4) =  (/  51.4d06, 102.8d06, 205.7d06, 406.9d06, 807.6d06,1597.5d06,3072.2d06,5393.9d06/)
      otab%ltable(3,2,2:otab%n3,4) =  (/  50.8d06, 101.8d06, 203.6d06, 396.0d06, 760.4d06,1422.1d06,2517.4d06,4062.8d06/)
      otab%ltable(3,3,2:otab%n3,4) =  (/  48.2d06,  96.4d06, 193.8d06, 367.3d06, 684.0d06,1238.3d06,2087.3d06,3287.1d06/)
      otab%ltable(3,4,2:otab%n3,4) =  (/  45.2d06,  90.4d06, 180.8d06, 335.7d06, 611.2d06,1066.3d06,1713.4d06,2780.3d06/)
      otab%ltable(3,5,2:otab%n3,4) =  (/  38.9d06,  77.8d06, 155.5d06, 273.7d06, 455.2d06, 702.2d06,1230.7d06,2453.7d06/)
      ! table6d (R2=0.04mum, wcb=5.0m/s) (for Ncn=50,100,200 "interpolated")
      otab%ltable(3,1,2:otab%n3,5) =  (/  53.1d06, 106.2d06, 212.3d06, 414.6d06, 818.3d06,1622.2d06,3216.8d06,6243.9d06/)
      otab%ltable(3,2,2:otab%n3,5) =  (/  51.6d06, 103.2d06, 206.3d06, 412.5d06, 805.3d06,1557.4d06,2940.4d06,5210.1d06/)
      otab%ltable(3,3,2:otab%n3,5) =  (/  49.6d06,  99.2d06, 198.4d06, 396.7d06, 755.5d06,1414.5d06,2565.3d06,4288.1d06/)
      otab%ltable(3,4,2:otab%n3,5) =  (/  46.5d06,  93.0d06, 186.0d06, 371.9d06, 692.9d06,1262.0d06,2188.3d06,3461.2d06/)
      otab%ltable(3,5,2:otab%n3,5) =  (/  39.9d06,  79.9d06, 159.7d06, 319.4d06, 561.7d06, 953.9d06,1493.9d06,2464.7d06/)

      ! Additional values for wcb = 0.0 m/s, which are used for linear interpolation between
      ! wcb = 0.0 and 0.5 m/s. Values of 0.0 are reasonable here, because if no
      ! updraft is present, no new nucleation will take place:
      otab%ltable(:,:,:,1) = 0.0d0
      ! Additional values for n_cn = 0.0 m**-3, which are used for linear interpolation between
      ! n_cn = 0.0 and 50 m**-3. Values of 0.0 are reasonable, because if no aerosol
      ! particles are present, no nucleation will take place:
      otab%ltable(:,:,1,:) = 0.0d0

      !!! otab%dx1 ... otab%odx4 remain empty because this is a non-equidistant table.

    END SUBROUTINE get_otab

    
    SUBROUTINE equi_table(nr2,nlsigs,nncn,nwcb)
      
      INTEGER, INTENT(IN) :: nr2,nlsigs,nncn,nwcb

      INTEGER :: i, j, k, l, ii, iu, ju,ku, lu
      INTEGER, ALLOCATABLE, DIMENSION(:) :: iuv, juv, kuv, luv
      DOUBLE PRECISION :: odx1, odx2, odx3, odx4
      DOUBLE PRECISION :: hilf1(2,2,2,2), hilf2(2,2,2), hilf3(2,2), hilf4(2)

      tab%n1 = nr2
      tab%n2 = nlsigs
      tab%n3 = nncn
      tab%n4 = nwcb
      
      IF (.NOT. ASSOCIATED(tab%x1)) THEN
        ALLOCATE( tab%x1(tab%n1) )
        ALLOCATE( tab%x2(tab%n2) )
        ALLOCATE( tab%x3(tab%n3) )
        ALLOCATE( tab%x4(tab%n4) )
        ALLOCATE( tab%ltable(tab%n1,tab%n2,tab%n3,tab%n4) )
      END IF

      !===========================================================
      ! construct equidistant table:
      !===========================================================

      ! grid distances (also inverse):
      tab%dx1  = (otab%x1(otab%n1) - otab%x1(1)) / (tab%n1 - 1.0d0)  ! dr2
      tab%odx1 = 1.0d0 / tab%dx1
      tab%dx2  = (otab%x2(otab%n2) - otab%x2(1)) / (tab%n2 - 1.0d0)  ! dlsigs
      tab%odx2 = 1.0d0 / tab%dx2
      tab%dx3  = (otab%x3(otab%n3) - otab%x3(1)) / (tab%n3 - 1.0d0)  ! dncn
      tab%odx3 = 1.0d0 / tab%dx3
      tab%dx4  = (otab%x4(otab%n4) - otab%x4(1)) / (tab%n4 - 1.0d0)  ! dwcb
      tab%odx4 = 1.0d0 / tab%dx4

      ! grid vectors:
      DO i=1, tab%n1
        tab%x1(i) = otab%x1(1) + (i-1) * tab%dx1
      END DO
      DO i=1, tab%n2
        tab%x2(i) = otab%x2(1) + (i-1) * tab%dx2
      END DO
      DO i=1, tab%n3
        tab%x3(i) = otab%x3(1) + (i-1) * tab%dx3
      END DO
      DO i=1, tab%n4
        tab%x4(i) = otab%x4(1) + (i-1) * tab%dx4
      END DO
      
      ! Tetra-linear interpolation of the new equidistant lookuptable from
      ! the original non-equidistant table:

      ALLOCATE(iuv(tab%n1))
      ALLOCATE(juv(tab%n2))
      ALLOCATE(kuv(tab%n3))
      ALLOCATE(luv(tab%n4))

      DO l=1, tab%n1
        iuv(l) = 1
        DO ii=1, otab%n1 - 1
          IF (tab%x1(l) >= otab%x1(ii) .AND. tab%x1(l) <= otab%x1(ii+1)) THEN
            iuv(l) = ii
            EXIT
          END IF
        END DO
      END DO

      DO l=1, tab%n2
        juv(l) = 1
        DO ii=1, otab%n2 - 1
          IF (tab%x2(l) >= otab%x2(ii) .AND. tab%x2(l) <= otab%x2(ii+1)) THEN
            juv(l) = ii
            EXIT
          END IF
        END DO
      END DO

      DO l=1, tab%n3
        kuv(l) = 1
        DO ii=1, otab%n3 - 1
          IF (tab%x3(l) >= otab%x3(ii) .AND. tab%x3(l) <= otab%x3(ii+1)) THEN
            kuv(l) = ii
            EXIT
          END IF
        END DO
      END DO

      DO l=1, tab%n4
        luv(l) = 1
        DO ii=1, otab%n4 - 1
          IF (tab%x4(l) >= otab%x4(ii) .AND. tab%x4(l) <= otab%x4(ii+1)) THEN
            luv(l) = ii
            EXIT
          END IF
        END DO
      END DO

      ! Tetra-linear interpolation:

      DO l=1, tab%n4
        lu = luv(l)
        odx4 = 1.0d0 / ( otab%x4(lu+1) - otab%x4(lu) )
!NEC$ ivdep
        DO k=1, tab%n3
          ku = kuv(k)
          odx3 = 1.0d0 / ( otab%x3(ku+1) - otab%x3(ku) )
!NEC$ unroll_completely
          DO j=1, nlsigs ! It should be equal to tab%n2, but the variable is needed by the Vector compiler
            ju = juv(j)
            odx2 = 1.0d0 / ( otab%x2(ju+1) - otab%x2(ju) )
!NEC$ unroll_completely
            DO i=1, nr2 !  It should be equal to tab%n1, but the variable is needed by the Vector compiler
              iu = iuv(i)
              odx1 = 1.0d0 / ( otab%x1(iu+1) - otab%x1(iu) )
              hilf1 = otab%ltable( iu:iu+1, ju:ju+1, ku:ku+1, lu:lu+1)
              hilf2 = hilf1(1,1:2,1:2,1:2) + (hilf1(2,1:2,1:2,1:2) - hilf1(1,1:2,1:2,1:2)) * odx1 * ( tab%x1(i) - otab%x1(iu) )
              hilf3 = hilf2(1,1:2,1:2)     + (hilf2(2,1:2,1:2)     - hilf2(1,1:2,1:2)  )   * odx2 * ( tab%x2(j) - otab%x2(ju) )
              hilf4 = hilf3(1,1:2)         + (hilf3(2,1:2)         - hilf3(1,1:2)    )     * odx3 * ( tab%x3(k) - otab%x3(ku) )
              tab%ltable(i,j,k,l) = hilf4(1) +  ( hilf4(2) - hilf4(1) ) * odx4 * ( tab%x4(l) - otab%x4(lu) )
            END DO
          END DO
        END DO
      END DO

      ! clean up memory:
      DEALLOCATE(iuv,juv,kuv,luv)

      RETURN
    END SUBROUTINE equi_table


  !*******************************************************************************
  !       Set to a default number concentration in places with qnx = 0 and qx !=0*
  !       (implemented by Alberto de Lozar)                                      *
  !*******************************************************************************
  SUBROUTINE set_default_n(kstart, kend, cloud, ice, rain, snow, graupel, hail, n_cn)
    INTEGER, INTENT(in) :: kstart, kend
    CLASS(particle), INTENT(inout)      :: cloud
    CLASS(particle), INTENT(inout)      :: ice
    CLASS(particle), INTENT(inout)      :: rain
    CLASS(particle), INTENT(inout)      :: snow
    CLASS(particle), INTENT(inout)      :: graupel
    CLASS(particle), INTENT(inout)      :: hail
    REAL(wp), DIMENSION(:), OPTIONAL  :: n_cn
    LOGICAL                             :: n_cn_pres

    INTEGER :: k
    REAL(wp), PARAMETER :: eps = 1e-3_wp

    IF (PRESENT(n_cn)) THEN
      n_cn_pres = .TRUE.
    ELSE
      n_cn_pres = .FALSE.
    ENDIF


    DO k = kstart,kend

      IF ( .NOT. n_cn_pres) THEN
      IF ( cloud%q(k) > 0.0_wp .AND. cloud%n(k) < eps) THEN
        cloud%n(k) = set_qnc(cloud%q(k)) 
      END IF
      END IF

      IF ( ice%q(k) > 0.0_wp .AND. ice%n(k) < eps) THEN
        ice%n(k) = set_qni(ice%q(k)) 
      END IF

      IF ( rain%q(k) > 0.0_wp .AND. rain%n(k) < eps) THEN
        rain%n(k) = set_qnr(rain%q(k)) 
      END IF

      IF ( snow%q(k) > 0.0_wp .AND. snow%n(k) < eps) THEN
        snow%n(k) = set_qns(snow%q(k)) 
      END IF

      IF ( graupel%q(k) > 0.0_wp .AND. graupel%n(k) < eps) THEN
        graupel%n(k) = set_qng(graupel%q(k)) 
      END IF

      IF ( hail%q(k) > 0.0_wp .AND. hail%n(k) < eps) THEN
        hail%n(k) = set_qnh_expPSD_N0const(hail%q(k),750.0_wp,1.0e6_wp) 
      END IF

    END DO

  END SUBROUTINE set_default_n

END MODULE mo_2mom_mcrph_processes
