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
! Provides various subroutines and functions for the two-moment microphysics

!NEC$ options "-finline-max-depth=3 -finline-max-function-size=1000"

MODULE mo_2mom_mcrph_util

  USE mo_kind,               ONLY: wp,sp,dp
  USE mo_exception,          ONLY: finish, message, txt => message_text
  USE mo_physical_constants, ONLY: &
       & rhoh2o,           & ! density of liquid water
       & T_3   => tmelt      ! melting temperature of ice
  USE mo_2mom_mcrph_types,   ONLY: particle, lookupt_1D, lookupt_4D
  
  IMPLICIT NONE

  PRIVATE


  PUBLIC :: &
       & rat2do3,                    & ! main
       & dyn_visc_sutherland,        & ! main
       & Dv_Rasmussen,               & ! main
       & ka_Rasmussen,               & ! main
       & lh_evap_RH87,               & ! main
       & lh_melt_RH87,               & ! main
       & set_qnc,                    &
       & set_qni,                    &
       & set_qnr,                    &
       & set_qns,                    &
       & set_qng,                    &
       & set_qnh_Dmean,              &
       & set_qnh_expPSD_N0const,     &
       & e_stick,                     &
       & init_estick_ltab_equi,      &
       & estick_ltab_equi


  CHARACTER(len=*), PARAMETER :: modname = 'mo_2mom_mcrph_util'

  ! In place of ICON's mo_math_constants: pi is the only constant this
  ! module needs from it.
  REAL(wp), PARAMETER :: pi = 3.14159265358979323846_wp


  ! Structure for holding the data of a lookup table for the incomplete gamma function:
  INTEGER, PARAMETER                     :: nlookup   = 2000    ! Internal number of bins (low res part)
  INTEGER, PARAMETER                     :: nlookuphr = 10000   ! Internal number of bins (high res part)

  ! dummy of internal number of bins (high res part) in case the high resolution part is not really needed:
  INTEGER, PARAMETER                     :: nlookuphr_dummy = 10

  ! Type to hold the lookup table for the incomplete gamma functions.
  ! The table is divided into a low resolution part, which spans the
  ! whole range of x-values up to the 99.5 % x-value, and a high resolution part for the
  ! smallest 1 % of these x-values, where the incomplete gamma function may increase
  ! very rapidly and nonlinearily, depending on paramter a.
  ! For some applications (e.g., Newtons Method in future subroutine
  ! graupel_hail_conv_wetgrowth_Dg_gamlook() ), this rapid change requires a much higher
  ! accuracy of the table lookup as compared to be achievable with the low resolution table.


CONTAINS

  !*******************************************************************************
  ! Special functions and utility functions like look-up tables
  !*******************************************************************************


  !*******************************************************************************
  !       Incomplete Gamma function
  !*******************************************************************************

  !*******************************************************************************
  ! 1) some helper functions:


  !*******************************************************************************
  ! Sticking efficiency of ice and snow as function of temperature
  !*******************************************************************************

  FUNCTION e_stick(T_a, istick) RESULT(e_i)

    REAL(wp), INTENT(IN) :: T_a     ! ambient T [K]
    INTEGER, INTENT(in)  :: istick  ! Flag to choose a specific parameterization

    REAL(wp) :: e_i, T_c

    T_c = T_a - T_3

    SELECT CASE (istick)
    CASE (1)
      ! temperature dependent sticking efficiency following Lin et al. (1983) which they
      ! use for rimed particles.
      ! (this relation is criticized by Ackerman et al. 2015, ACP, as being much too large at cold conditions)
      e_i = MIN(EXP(0.09_wp*T_c),1.0_wp)
    CASE (2)
      ! even higher sticking efficiency also suggested by Lin et al. (1983) and there used for unrimed particles.
      ! Also used by Spichtinger and Gierens, e.g., doi:10.5194/acp-13-9021-2013
      ! (similar to the values given in Pruppacher and Klett, Ch. 16.2, page 601, as used by Mitchell 1988, JAS)
      e_i = MIN(EXP(0.025_wp*T_c),1.0_wp) 
    CASE (3)
      ! as previous setting, but reduced sticking eff. below -40 C 
      IF ( (T_a-T_3) > -40_wp) THEN
        e_i = MIN(EXP(0.025_wp*T_c),1.0_wp) 
      ELSE
        e_i = 0.01_wp
      END IF
    CASE (4)
      ! piecewise defined sticking efficiency with maximum at -15 C,
      ! as given in Pruppacher and Klett, Ch. 16.2, page 601, as used by Mitchell 1988, JAS
      ! here extended below -20 C with values similar to Lin et al.
      IF ( T_c >= -4._wp ) THEN
        e_i = 0.1_wp
      ELSEIF ( T_c >= -6._wp ) THEN
        e_i = 0.6_wp
      ELSEIF ( T_c >= -9._wp ) THEN
        e_i = 0.1_wp
      ELSEIF ( T_c >= -12.5_wp ) THEN
        e_i = 0.4_wp
      ELSEIF ( T_c >= -17_wp ) THEN
        e_i = 1.0_wp
      ELSEIF ( T_c >= -20_wp ) THEN
        e_i = 0.40_wp
      ELSEIF ( T_c >= -30_wp ) THEN
        e_i = 0.25_wp
      ELSEIF ( T_c >= -40_wp ) THEN
        e_i = 0.10_wp
      ELSE
        e_i = 0.02_wp
      END IF
    CASE (5)
      ! piecewise linear sticking efficiency with maximum at -15 C,
      ! inspired by Figure 14 of Connolly et al. ACP 2012, doi:10.5194/acp-12-2055-2012
      ! Value at -40 C is based on Kajikawa and Heymsfield as cited by Philips et al. (2015, JAS)
      IF ( T_c >= 0_wp ) THEN
        e_i = 0.14_wp
      ELSEIF ( T_c >= -10_wp ) THEN
        e_i = -0.01_wp*(T_c+10_wp)+0.24_wp
      ELSEIF ( T_c >= -15_wp ) THEN
        e_i = -0.08_wp*(T_c+15_wp)+0.64_wp
      ELSEIF ( T_c >= -20_wp ) THEN
        e_i =  0.10_wp*(T_c+20_wp)+0.14_wp
      ELSEIF ( T_c >= -40_wp ) THEN
        e_i = 0.005_wp*(T_c+40_wp)+0.04_wp
      ELSE
        e_i =  0.04_wp
      END IF
    CASE (6)
      ! as option 5, but with factor 0.5, i.e., the lower range of Figure 14
      IF ( T_c >= 0_wp ) THEN
        e_i = 0.14_wp
      ELSEIF ( T_c >= -10_wp ) THEN
        e_i = -0.01_wp*(T_c+10_wp)+0.24_wp
      ELSEIF ( T_c >= -15_wp ) THEN
        e_i = -0.08_wp*(T_c+15_wp)+0.64_wp
      ELSEIF ( T_c >= -20_wp ) THEN
        e_i =  0.10_wp*(T_c+20_wp)+0.14_wp
      ELSEIF ( T_c >= -40_wp ) THEN
        e_i = 0.005_wp*(T_c+40_wp)+0.04_wp
      ELSE
        e_i =  0.04_wp
      END IF
      e_i = 0.5_wp * e_i
    CASE (7)
      e_i = 0.1_wp
    CASE (8)
      ! inspired by PK (option 4) and Connolly (option 5)
      IF ( T_c >= -1_wp ) THEN
        e_i = 1.0_wp
      ELSEIF (  T_c >= -6_wp ) THEN
        e_i = 0.19_wp*(T_c+6_wp)+0.24_wp
      ELSEIF (  T_c >= -10_wp ) THEN
        e_i = 0.24_wp
      ELSEIF (  T_c >= -12.5_wp ) THEN
        e_i = -0.304*(T_c+12.5_wp)+1.0_wp
      ELSEIF (  T_c >= -17_wp ) THEN
        e_i = 1.0_wp
      ELSEIF (  T_c >= -20_wp ) THEN
        e_i =  0.286666_wp*(T_c+20_wp)+0.14_wp
      ELSEIF (  T_c >= -40_wp ) THEN
        e_i = 0.005_wp*(T_c+40_wp)+0.04_wp
      ELSE
        e_i =  0.04_wp
      END IF
    CASE (9)
      ! as option 5, but with factor 0.75, i.e., the lower range of Figure 14
      IF ( T_c >= 0_wp ) THEN
        e_i = 0.14_wp
      ELSEIF ( T_c >= -10_wp ) THEN
        e_i = -0.01_wp*(T_c+10_wp)+0.24_wp
      ELSEIF ( T_c >= -15_wp ) THEN
        e_i = -0.08_wp*(T_c+15_wp)+0.64_wp
      ELSEIF ( T_c >= -20_wp ) THEN
        e_i =  0.10_wp*(T_c+20_wp)+0.14_wp
      ELSEIF ( T_c >= -40_wp ) THEN
        e_i = 0.005_wp*(T_c+40_wp)+0.04_wp
      ELSE
        e_i =  0.04_wp
      END IF
      e_i = 0.75_wp * e_i
    CASE (10)
      !.. Temperaturabhaengige Efficiency nach Cotton et al. (1986)
      !   (siehe auch Straka, 1989; S. 53)
      e_i = MIN(10**(0.035_wp*T_c-0.7_wp),0.2_wp)
    CASE default
      e_i = 1.0_wp
    END SELECT

    RETURN
  END FUNCTION e_stick

  SUBROUTINE init_estick_ltab_equi (ltab, istick, name)

    TYPE(lookupt_1D), INTENT(inout) :: ltab
    INTEGER, INTENT(in)             :: istick
    CHARACTER(len=*), INTENT(in)    :: name

    INTEGER  :: i, n
    REAL(wp) :: T_a
    CHARACTER(len=*), PARAMETER :: routine = TRIM(modname)//'::init_estick_ltab_equi'

    INTEGER, PARAMETER  :: ndT  = 801      ! Number of table nodes
    REAL(wp), PARAMETER :: Tmin = -70.0_wp ! Start T of table [deg C]
    REAL(wp), PARAMETER :: Tmax = 10.0_wp  ! End T of table [deg C]
    
    IF (.NOT. ltab%is_initialized) THEN

      ltab%name(:) = ' '
      ltab%name    = TRIM(name)
      ltab%iflag   = istick

      ltab%n1 = ndT
      ltab%dx1 = (Tmax - Tmin) / (ndT - 1.0_wp)
      ltab%odx1 = 1.0_wp / ltab%dx1

      NULLIFY (ltab%x1, ltab%ltable)
      
      ALLOCATE (ltab%x1(ltab%n1))
      ALLOCATE (ltab%ltable(ltab%n1))

      DO i = 1, ltab%n1
        T_a = Tmin + (i-1)*ltab%dx1 + T_3    ! [K]
        ltab%x1(i) = T_a                     ! equidistant table vector [K]
        ltab%ltable(i) = e_stick(T_a, istick)
      END DO
      
      ltab%is_initialized = .TRUE.

    ELSE

      IF (ltab%iflag /= istick) THEN
        ! conflicting initialization, don't know what to do:
        txt(:) = ' '
        WRITE(txt, '(a,i0,a,i0)') 'Conflicting re-initialization of LUT for '//TRIM(ltab%name)// &
             ', old istick=',ltab%iflag, ' / new istick=', istick
        CALL finish(TRIM(routine),TRIM(txt))
      END IF
    END IF

  END SUBROUTINE init_estick_ltab_equi

  FUNCTION estick_ltab_equi(T_a, ltab) RESULT (e_stick)

    !$ACC ROUTINE SEQ

    REAL(wp) :: e_stick
    REAL(wp), INTENT(in) :: T_a     ! ambient T [K]
    TYPE(lookupt_1D), INTENT(in) :: ltab

    INTEGER  :: iu, io
    REAL(wp) :: T_loc
    
    T_loc = MIN( MAX( T_a, ltab%x1(1)), ltab%x1(ltab%n1) )
    iu = MIN(FLOOR((T_loc - ltab%x1(1)) * ltab%odx1 ) + 1, ltab%n1-1)
    io = iu + 1

    e_stick = ltab%ltable(iu) + (ltab%ltable(io)-ltab%ltable(iu)) * ltab%odx1 * (T_loc-ltab%x1(iu))

  END FUNCTION estick_ltab_equi
  
  !*******************************************************************************
  ! 2D rational functions to evaluate bulk approximations                        *
  ! following Frick et al. (2013; cf. Eq. (31)), for n=2 and n=3                 *
  !*******************************************************************************

  REAL(wp) FUNCTION rat2do3(x,y,a,b)
    implicit none

    REAL(wp), INTENT(IN)                :: x,y
    REAL(wp), INTENT(IN), DIMENSION(10) :: a
    REAL(wp), INTENT(IN), DIMENSION(9)  :: b
    REAL(wp), PARAMETER :: eins = 1.0_wp
    REAL(wp)            :: p1,p2

    p1 = a(1)+a(2)*x+a(3)*y+a(4)*x*x+a(5)*x*y+a(6)*y*y &
         &   +a(7)*x*x*x+a(8)*x*x*y+a(9)*x*y*y+a(10)*y*y*y 
    p2 = eins+b(1)*x+b(2)*y+b(3)*x*x+b(4)*x*y+b(5)*y*y &
         &  + b(6)*x*x*x+b(7)*x*x*y+b(8)*x*y*y+b(9)*y*y*y 

    rat2do3 = p1/p2

    RETURN
  END FUNCTION rat2do3

  ELEMENTAL REAL(wp) FUNCTION dyn_visc_sutherland(Ta)
    !
    ! Calculate dynamic viscosity of air [kg m-1 s-1]
    ! following Sutherland's formula of an ideal
    ! gas with reference temp. T = 291.15 K
    !
    ! There is another alternative in P&K97 on
    ! page 417
    !
    IMPLICIT NONE
    REAL(wp), INTENT(in) :: Ta   ! ambient temp. [K]
    REAL(wp), PARAMETER :: &
         C = 120.d0      , &     ! Sutherland's constant (for air) [K]
         T0 = 291.15d0   , &     ! Reference temp. [K]
         eta0 = 1.827d-5         ! Reference dyn. visc. [kg m-1 s-1]
    REAL(wp) :: a, b

    a = T0 + C
    b = Ta + C
    dyn_visc_sutherland = eta0 * a/b * (Ta/T0)**(3.d0/2.d0)

    RETURN
  END FUNCTION dyn_visc_sutherland
  ! ---------------------------------------------------------------------
  ELEMENTAL REAL(wp) FUNCTION Dv_Rasmussen(Ta,pa)
    !
    ! Calculating the diffusivity of water vapor in air
    ! following Rasmussen et al. 1987, App. A, Tab. A1
    ! Changed: Units of D_v in m2 s-1
    !
    IMPLICIT NONE
    REAL(wp), INTENT(in) :: Ta, pa  ! Temp. and pressure in [K] and [Pa]
    REAL(wp), PARAMETER  :: p_0 = 1013.25e2_wp

    Dv_Rasmussen = 0.211d-4*(p_0/pa)*(Ta/T_3)**1.94
    RETURN
  END FUNCTION Dv_Rasmussen
  ! ---------------------------------------------------------------------
  ELEMENTAL REAL(wp) FUNCTION ka_Rasmussen(Ta)
    !
    ! Calculating the thermal conductivity of air
    ! following Rasmussen et al. 1987, App. A, Tab. A1
    !
    IMPLICIT NONE
    REAL(wp), INTENT(in)  :: Ta  ! ambient temp. [K]
    REAL(wp), PARAMETER :: &
         c_unit = 4.1840d2      ! for transforming units

    ! transform [cal cm-1 s-1 C-1] into [W m-1 K-1]
    ka_rasmussen = c_unit * (5.69 + 0.017*(Ta-T_3))*1.d-5
    RETURN
  END FUNCTION ka_Rasmussen
  ! ---------------------------------------------------------------------
  ELEMENTAL REAL(wp) FUNCTION lh_evap_RH87(T)
    !
    ! Calculating the latent heat of evaporation
    ! following the formulation of RH87a
    !
    IMPLICIT NONE
    REAL(wp), INTENT(in) :: T    ! ambient temp.
    REAL(wp) :: lh_e0, gam

    !.latent heat of evap. at T_3
    lh_e0 = 2.5008d6
    !.exponent for calculation
    gam = 0.167d0 + 3.67d-4 * T
    !.latent heat of evap. as a fct. of temp.
    lh_evap_RH87 = lh_e0 * (T_3 / T)**gam
    RETURN
  END FUNCTION lh_evap_RH87
  ! ---------------------------------------------------------------------
  ELEMENTAL REAL(wp) FUNCTION lh_melt_RH87(T)
    !
    ! Calculating the latent heat of melting
    ! following the formulation of RH87a
    !
    IMPLICIT NONE
    REAL(wp), INTENT(in) :: T    ! ambient temp.
    REAL(wp), PARAMETER :: &
         c_unit = 4.1840d3       ! constant to transform [cal g-1] to [J kg-1]

    !.latent heat of melt. as a fct. of temp.
    lh_melt_RH87 = c_unit * ( 79.7d0 + 0.485d0*(T-T_3) - 2.5d-3*(T-T_3)**2)
    RETURN
  END FUNCTION lh_melt_RH87

!==============================================================================

  REAL(wp) Function set_qnc(qc)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in)  :: qc  ! either [kg/kg] or [kg/m^3]
    REAL(wp), PARAMETER   :: Dmean = 10e-6_wp    ! Diameter of mean particle mass:

!    set_qnc = qc * 6.0_wp / (pi * rhoh2o * Dmean**3.0_wp)
    set_qnc = qc * 6.0_wp / (pi * rhoh2o * EXP(LOG(Dmean)*3.0_wp) )

  END FUNCTION set_qnc

  REAL(wp) Function set_qni(qi)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in)  :: qi  ! either [kg/kg] or [kg/m^3]

!    set_qni  = qi / 1e-10   !  qiin / ( ( Dmean / ageo) ** (1.0_wp / bgeo) )
    set_qni  = qi / 1e-10   !  qiin / ( exp(log(( Dmean / ageo)) * (1.0_wp / bgeo)) )
!     set_qni =  5.0E+0_wp * EXP(0.304_wp *  (T_3 - T))   ! FR: Cooper (1986) used by Greg Thompson(2008)
      
  END FUNCTION set_qni

  REAL(wp) Function set_qnr(qr)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in)  :: qr  ! has to be [kg/m^3]
    REAL(wp), PARAMETER   :: N0r = 8000.0e3_wp ! intercept of MP distribution

    !    set_qnr = N0r * ( qr * 6.0_wp / (pi * rhoh2o * N0r * gamma(4.0_wp)))**(0.25_wp)
    IF (qr >= 1e-20_wp) THEN
      set_qnr = N0r * EXP( LOG( qr * 6.0_wp / (pi * rhoh2o * N0r * GAMMA(4.0_wp))) * (0.25_wp) )
    ELSE
      set_qnr = 0.0_wp
    END IF

  END FUNCTION set_qnr

  REAL(wp) Function set_qns(qs)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in)  :: qs  ! has to be [kg/m^3]
    REAL(wp), PARAMETER   :: N0s = 800.0e3_wp
    REAL(wp), PARAMETER   :: ams = 0.038_wp  ! needs to be connected to snow-type
    REAL(wp), PARAMETER   :: bms = 2.0_wp

!    set_qns = N0s * ( qs / ( ams * N0s * gamma(bms+1.0_wp)))**( 1.0_wp/(1.0_wp+bms) )
    IF (qs >= 1e-20_wp) THEN
      set_qns = N0s * EXP( LOG( qs / ( ams * N0s * GAMMA(bms+1.0_wp))) * ( 1.0_wp/(1.0_wp+bms) ) )
    ELSE
      set_qns = 0.0_wp
    END IF
    
  END FUNCTION set_qns

  REAL(wp) Function set_qng(qg)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in)  :: qg  ! has to be [kg/m^3]
    REAL(wp), PARAMETER   :: N0g = 4000.0e3_wp
    REAL(wp), PARAMETER   :: amg = 169.6_wp     ! needs to be connected to graupel-type
    REAL(wp), PARAMETER   :: bmg = 3.1_wp

!    set_qng = N0g * ( qg / ( amg * N0g * gamma(bmg+1.0_wp)))**( 1.0_wp/(1.0_wp+bmg) )
    IF (qg >= 1e-20_wp) THEN
      set_qng = N0g * EXP( LOG ( qg / ( amg * N0g * GAMMA(bmg+1.0_wp))) * ( 1.0_wp/(1.0_wp+bmg) ) )
    ELSE
      set_qng = 0.0_wp
    END IF

  END FUNCTION set_qng

  FUNCTION set_qnh_Dmean(qh, rhobulk_hail, Dmean) RESULT (qnh)

    !$ACC ROUTINE SEQ

    REAL(wp), INTENT(in) :: qh           ! either [kg/kg] or [kg/m^3]
    REAL(wp), INTENT(in) :: rhobulk_hail ! assumed bulk density of hail [kg/m^3]
    REAL(wp), INTENT(in) :: Dmean        ! assumed mean mass diameter [m]

    REAL(wp) :: qnh

!    qnh = qh * 6.0_wp / (pi * rhobulk_hail * Dmean**3.0_wp)
    qnh = qh * 6.0_wp / (pi * rhobulk_hail * EXP(LOG(Dmean)*3.0_wp) )
    
  END FUNCTION set_qnh_Dmean

  FUNCTION set_qnh_expPSD_N0const(qh, rhobulk_hail, N0_h) RESULT (qnh)

    !$ACC ROUTINE SEQ

    ! .. Sets qnh based on assumption of an exponential PSD w.r.t. diameter D
    
    REAL(wp), INTENT(in) :: qh           ! has to be [kg/m^3] because of N0 held constant
    REAL(wp), INTENT(in) :: rhobulk_hail ! assumed bulk density of hail [kg/m^3]
    REAL(wp), INTENT(in) :: N0_h         ! assumed constant N0-parameter of expon. size distrib. [1/m^4]

    REAL(wp) :: qnh
 
!    set_qnh = N0_h * ( qh / ( pi * rhobulk_hail * N0_h) )**(0.25)
    IF (qh >= 1e-20_wp) THEN
      qnh = N0_h * EXP( LOG ( qh / ( pi * rhobulk_hail * N0_h) ) * ( 0.25_wp ) )
    ELSE
      qnh = 0.0_wp
    END IF
    
  END FUNCTION set_qnh_expPSD_N0const

 
END MODULE mo_2mom_mcrph_util
