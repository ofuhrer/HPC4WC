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

! This module provides physical constants for the ICON general circulation models.
!
! Physical constants are grouped as follows:
! - Natural constants
! - Molar weights
! - Earth and Earth orbit constants
! - Thermodynamic constants for the dry and moist atmosphere
! - Constants used for the computation of lookup tables of the saturation
!    mixing ratio over liquid water (*c_les*) or ice(*c_ies*)
!    (to be shifted to the module that computes the lookup tables)

MODULE mo_physical_constants

  USE mo_kind,            ONLY: wp

  IMPLICIT NONE

  PUBLIC

  ! Natural constants
  ! -----------------
  !
!!$  ! ECHAM values
!!$  REAL(wp), PARAMETER :: avo   = 6.022045e23_wp   ! [1/mo]    Avogadro constant
!!$  REAL(wp), PARAMETER :: ak    = 1.380662e-23_wp  ! [J/K]     Boltzmann constant
!!$  REAL(wp), PARAMETER :: argas = 8.314409_wp      ! [J/K/mol] molar/universal/ideal gas constant
!!$  REAL(wp), PARAMETER :: stbo  = 5.67E-8_wp       ! [W/m2/K4] Stephan-Boltzmann constant
  ! WMO/SI values
  REAL(wp), PARAMETER :: avo   = 6.02214179e23_wp !> [1/mo]    Avogadro constant
  REAL(wp), PARAMETER :: ak    = 1.3806504e-23_wp !! [J/K]     Boltzmann constant
  REAL(wp), PARAMETER :: argas = 8.314472_wp      !! [J/K/mol] molar/universal/ideal gas constant
  REAL(wp), PARAMETER :: stbo  = 5.6704E-8_wp     !! [W/m2/K4] Stephan-Boltzmann constant


  !> Molar weights
  !! -------------
  !!
  !! Pure species
  REAL(wp), PARAMETER :: amco2 = 44.011_wp        !>[g/mol] CO2
  REAL(wp), PARAMETER :: amch4 = 16.043_wp        !! [g/mol] CH4
  REAL(wp), PARAMETER :: amo3  = 47.9982_wp       !! [g/mol] O3
  REAL(wp), PARAMETER :: amo2  = 31.9988_wp       !! [g/mol] O2
  REAL(wp), PARAMETER :: amn2o = 44.013_wp        !! [g/mol] N2O
  REAL(wp), PARAMETER :: amc11 =137.3686_wp       !! [g/mol] CFC11
  REAL(wp), PARAMETER :: amc12 =120.9140_wp       !! [g/mol] CFC12
  REAL(wp), PARAMETER :: amw   = 18.0154_wp       !! [g/mol] H2O
  REAL(wp), PARAMETER :: amo   = 15.9994_wp       !! [g/mol] O
  REAL(wp), PARAMETER :: amno  = 30.0061398_wp    !! [g/mol] NO
  REAL(wp), PARAMETER :: amn2  = 28.0134_wp       !! [g/mol] N2
  REAL(wp), PARAMETER :: amso4 = 96.0626_wp       !! [g/mol] SO4
  REAL(wp), PARAMETER :: ams   = 32.06_wp         !! [g/mol] S
  !
  !> Mixed species
  REAL(wp), PARAMETER :: amd   = 28.970_wp        !> [g/mol] dry air
  !
  !> Auxiliary constants
  ! ppmv2gg converts ozone from volume mixing ratio in ppmv
  ! to mass mixing ratio in g/g
  REAL(wp), PARAMETER :: ppmv2gg=1.e-6_wp*amo3/amd
  REAL(wp), PARAMETER :: o3mr2gg=amo3/amd

  !> Conversion factor from CO2 volume to mass mixing ratio [kg(CO2)/kg(Air)/(mol(CO2)/mol(Air))].
  REAL(wp), PARAMETER :: vmr_to_mmr_co2 = amco2 / amd
  !> Conversion factor from CH4 volume to mass mixing ratio [kg(CH4)/kg(Air)/(mol(CH4)/mol(Air))].
  REAL(wp), PARAMETER :: vmr_to_mmr_ch4 = amch4 / amd
  !> Conversion factor from N2O volume to mass mixing ratio [kg(N2O)/kg(Air)/(mol(N2O)/mol(Air))].
  REAL(wp), PARAMETER :: vmr_to_mmr_n2o = amn2o / amd
  !> Conversion factor from CFC11 volume to mass mixing ratio [kg(CFC11)/kg(Air)/(mol(CFC11)/mol(Air))].
  REAL(wp), PARAMETER :: vmr_to_mmr_c11 = amc11 / amd
  !> Conversion factor from CFC12 volume to mass mixing ratio [kg(CFC12)/kg(Air)/(mol(CFC12)/mol(Air))].
  REAL(wp), PARAMETER :: vmr_to_mmr_c12 = amc12 / amd

  !> Earth and Earth orbit constants
  !! -------------------------------
  !!
!   REAL(wp), PARAMETER :: re    = 6.371229e6_wp    !! [m]    average radius
!   REAL(wp), PARAMETER :: rre   = 1._wp/ph_re         !! [1/m]
  REAL(wp), PARAMETER :: earth_radius           = 6.371229e6_wp    !! [m]    average radius
  REAL(wp), PARAMETER :: inverse_earth_radius   = 1._wp/earth_radius         !! [1/m]
  REAL(wp), PARAMETER :: earth_angular_velocity = 7.29212e-5_wp    !! [rad/s]  angular velocity
  !
!!$  ! ECHAM values
!!  REAL(wp), PARAMETER :: grav  = 9.80616_wp       ! [m/s2] av. gravitational acceleration
  ! WMO/SI value
  REAL(wp), PARAMETER :: grav  = 9.80665_wp       !> [m/s2] av. gravitational acceleration
  !$ACC DECLARE COPYIN(grav)
  REAL(wp), PARAMETER :: rgrav = 1._wp/grav       !! [s2/m]
  !
  REAL(wp), PARAMETER :: rae   = 0.1277E-2_wp     !> [m/m]  ratio of atm. scale height
  !                                               !!        to Earth radius


  ! Thermodynamic constants for the dry and moist atmosphere
  ! --------------------------------------------------------
  !
  !> Dry air
  REAL(wp), PARAMETER :: rd    = 287.04_wp        !> [J/K/kg] gas constant
  !$ACC DECLARE COPYIN(rd)
  REAL(wp), PARAMETER :: cpd   = 1004.64_wp       !! [J/K/kg] specific heat at constant pressure
  !$ACC DECLARE COPYIN(cpd)
  REAL(wp), PARAMETER :: cvd   = cpd-rd           !! [J/K/kg] specific heat at constant volume
  !$ACC DECLARE COPYIN(cvd)
  REAL(wp), PARAMETER :: con_m = 1.50E-5_wp       !! [m^2/s]  kinematic viscosity of dry air
  REAL(wp), PARAMETER :: con_h = 2.20E-5_wp       !! [m^2/s]  scalar conductivity of dry air
  REAL(wp), PARAMETER :: con0_h= 2.40e-2_wp       !! [J/m/s/K]thermal conductivity of dry air
  REAL(wp), PARAMETER :: eta0d = 1.717e-5_wp      !! [N*s/m2] dyn viscosity of dry air at tmelt
  !
  !> H2O
  !! - gas
  REAL(wp), PARAMETER :: rv    = 461.51_wp        !> [J/K/kg] gas constant for water vapor
  !$ACC DECLARE COPYIN(rv)
  REAL(wp), PARAMETER :: cpv   = 1869.46_wp       !! [J/K/kg] specific heat at constant pressure
  REAL(wp), PARAMETER :: cvv   = cpv-rv           !! [J/K/kg] specific heat at constant volume
  !$ACC DECLARE COPYIN(cvv)
  REAL(wp), PARAMETER :: dv0   = 2.22e-5_wp       !! [m^2/s]  diff coeff of H2O vapor in dry air at tmelt
  !> - liquid / water
  REAL(wp), PARAMETER :: rhoh2o= 1000._wp         !> [kg/m3]  density of liquid water
  !> - solid / ice
  REAL(wp), PARAMETER :: rhoice=  916.7_wp        !> [kg/m3]  density of pure ice

 !REAL(wp), PARAMETER :: clw   = 4186.84_wp       !! [J/K/kg] specific heat of water
                                                  !!  see below
  REAL(wp), PARAMETER ::  cv_i =  2000.0_wp
  !> - phase changes
  REAL(wp), PARAMETER :: alv   = 2.5008e6_wp      !> [J/kg]   latent heat for vaporisation
  !$ACC DECLARE COPYIN(alv)
  REAL(wp), PARAMETER :: als   = 2.8345e6_wp      !! [J/kg]   latent heat for sublimation
  !$ACC DECLARE COPYIN(als)
  REAL(wp), PARAMETER :: alf   = als-alv          !! [J/kg]   latent heat for fusion
  !$ACC DECLARE COPYIN(alf)
  REAL(wp), PARAMETER :: tmelt = 273.15_wp        !! [K]      melting temperature of ice/snow
  REAL(wp), PARAMETER :: t3    = 273.16_wp        !! [K]      Triple point of water at 611hPa
  !
  !> Auxiliary constants
  REAL(wp), PARAMETER :: rdv   = rd/rv            !> [ ]
  !$ACC DECLARE COPYIN(rdv)
  REAL(wp), PARAMETER :: vtmpc1= rv/rd-1._wp      !! [ ]
  !$ACC DECLARE COPYIN(vtmpc1)
  REAL(wp), PARAMETER :: vtmpc2= cpv/cpd-1._wp    !! [ ]
  REAL(wp), PARAMETER :: rcpv  = cpd/cpv-1._wp    !! [ ]
  REAL(wp), PARAMETER :: alvdcp= alv/cpd          !! [K]
  REAL(wp), PARAMETER :: alsdcp= als/cpd          !! [K]
  REAL(wp), PARAMETER :: rcpd  = 1._wp/cpd        !! [K*kg/J]
  REAL(wp), PARAMETER :: rcvd  = 1._wp/cvd        !! [K*kg/J]
  REAL(wp), PARAMETER :: rcpl  =  3.1733_wp       !! cp_d / cp_l - 1
  !
  REAL(wp), PARAMETER :: clw   = (rcpl + 1.0_wp) * cpd !> specific heat capacity of liquid water
  !$ACC DECLARE COPYIN(clw)
  REAL(wp), PARAMETER :: cv_v  = (rcpv + 1.0_wp) * cpd - rv
  !
  REAL(wp), PARAMETER :: o_m_rdv  = 1._wp-rd/rv   !> [ ]
  !$ACC DECLARE COPYIN(o_m_rdv)
  REAL(wp), PARAMETER :: rd_o_cpd = rd/cpd        !! [ ]
  !$ACC DECLARE COPYIN(rd_o_cpd)
  REAL(wp), PARAMETER :: cvd_o_rd = cvd/rd        !! [ ]
  !$ACC DECLARE COPYIN(cvd_o_rd)
  !
  REAL(wp), PARAMETER :: p0ref     = 100000.0_wp   !> [Pa]  reference pressure for Exner function

  !> Variables for computing cloud cover in RH scheme
  REAL(wp), PARAMETER ::  uc1  = 0.8_wp
  REAL(wp), PARAMETER ::  ucl  = 1.00_wp

  !> U.S. standard atmosphere vertical tropospheric temperature gradient
  REAL(wp), PARAMETER :: dtdz_standardatm = -6.5e-3_wp    ! [ K/m ]

  !> constants for radiation module
  REAL(wp), PARAMETER :: zemiss_def = 0.996_wp  !> lw sfc default emissivity factor

  !> salinity factor for reduced saturation vapor pressure over oceans
  REAL(wp), PARAMETER :: salinity_fac = 0.981_wp

  !> dielectric constants at reference temperature 0 degree Celsius for radar reflectivity calculation:
  REAL(wp), PARAMETER :: K_w_0 = 0.93_wp
  REAL(wp), PARAMETER :: K_i_0 = 0.176_wp

!------------below are parameters for ocean model---------------
  ! coefficients in linear EOS
  REAL(wp), PARAMETER :: a_T = 2.55E-04_wp,     & ! thermal expansion coefficient (kg/m3/K)
    &                    b_S = 7.64E-01_wp         ! haline contraction coefficient (kg/m3/psu)

  ! density reference values, to be constant in Boussinesq ocean models
  REAL(wp), PARAMETER :: rho_ref = 1025.022_wp         ! reference density [kg/m^3]
  REAL(wp), PARAMETER :: rho_inv = 0.0009755881663_wp  ! inverse reference density [m^3/kg]
  REAL(wp), PARAMETER :: sal_ref = 35.0_wp             ! reference salinity [psu]

  REAL(wp), PARAMETER :: SItodBar = 1.0e-4_wp          !Conversion from pressure [p] to pressure [bar]
                                                       !used in ocean thermodynamics
  REAL(wp),PARAMETER :: sfc_press_pascal = 101300.0_wp
  REAL(wp),PARAMETER :: sfc_press_bar    = 101300.0_wp*SItodBar

  REAL(wp), PARAMETER :: p0sl_bg         = 101325._wp  ! [Pa]     sea level pressure

!----------below are parameters for sea-ice and lake model---------------
  REAL(wp), PARAMETER ::            &
    ks           = 0.31_wp,         & ! heat conductivity snow     [J  / (m s K)]
    ki           = 2.1656_wp,       & ! heat conductivity ice      [J  / (m s K)]
    rhoi         = 917.0_wp,        & ! density of sea ice         [kg / m**3]
    rhos         = 300.0_wp,        & ! density of snow            [kg / m**3]
    ci           = 2106.0_wp,       & ! Heat capacity of ice       [J / (kg K)]
    cs           = 2090._wp,        &!  Heat capacity of snow      [J / (kg K)]

    Tf           = -1.80_wp,        & ! Temperature ice bottom     [C]
    mu           = 0.054_wp,        & ! Constant in linear freezing-
                                      ! point relationship         [C/ppt]
                                      ! (aka melting) temperature) [C]
!   muS          = -(-0.0575 + 1.710523E-3*Sqrt(Sice) - 2.154996E-4*Sice) * Sice
    albedoW      = 0.07_wp,         & ! albedo of the ocean used in atmosphere


    fr_fac       = 1.1925_wp,       & ! Frank Roeske energy budget closing factor for OMIP
!   fr_fac       = 1.0_wp,          ! factor not active

! CCSM3 albedo scheme - not used for coupling
    alb_ice_vis  = 0.73_wp,         & ! Albedo of dry ice  (visible)
    alb_ice_nir  = 0.33_wp,         & ! Albedo of dry ice  (near-infrared)
    alb_sno_vis  = 0.96_wp,         & ! Albedo of dry snow (visible)
    alb_sno_nir  = 0.68_wp,         & ! Albedo of dry snow (near-infrared)
    !I_0          = 0.3             ! Ice-surface penetrating shortwave fraction
    I_0          = 0.17_wp            ! Ice-surface penetrating shortwave fraction
  !$ACC DECLARE COPYIN(ci)

!--------- parameters for NWP sea-ice model (we should agree on a single value)-----
!_cdm>
! The value of the salt-water freezing point is the same as in GME and COSMO (-1.7 dgr C).
! Note that a different value (Tf=-1.8 dgr C) is defined in "mo_physical_constants".
!_cdm<
  REAL (wp), PARAMETER ::                             &
    &  tf_salt      = 271.45_wp     !< salt-water freezing point [K]
                                    !< (note that it differs from Tf)

END MODULE mo_physical_constants
