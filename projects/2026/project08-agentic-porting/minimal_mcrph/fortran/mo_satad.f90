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

!  Description:
!  This module provides service utilities for meteorological calculations.
!
! Routines (module procedure)
!
!     - satad_V_3D
!       Corrects the temperature, the specific humidity and the cloud water
!      content for condensation/evaporation.
!
!    - qsat_rho
!      Specific humidity at water saturation (with respect to flat surface)
!      depending on the temperature "temp" and the total density "rhotot")
!
!    - dqsatdT_rho
!       Partial derivative of the specific humidity at water saturation with
!       respect to the temperature at constant total density.
!
!    the following functions should  later be replaced by lookup tables
!     from mo_convect_tables!
!     - pres_sat_water
!       Saturation water vapour pressure
!
!     - pres_sat_ice
!       Saturation water vapour pressure
!
!     - spec_humi
!       Specific humidity at saturation pressure

MODULE mo_satad


USE, INTRINSIC :: iso_fortran_env, ONLY: ireals => real64, iintegers =>  int32
USE mo_physical_constants, ONLY: r_v   => rv    , & !> gas constant for water vapour
                               rvd_m_o => vtmpc1 , & !! rv/rd-1._wp
                                 o_m_rdv        , & !! 1 - r_d/r_v
                                 rdv            , & !! r_d / r_v
                                 cvd            , & !!
                                 cl    => clw   , & !! specific heat of water
                                 lwd   => alv   , & !! latent heat of vaporization
                                 led   => als   , & !! latent heat of sublimation
                                 b3    => tmelt , & !!
                                 tmelt

USE mo_lookup_tables_constants, ONLY:  &
                                 b1    => c1es  , & !! constants for computing the sat. vapour
                                 b2w   => c3les , & !! pressure over water (l) and ice (i)
                                 b2i   => c3ies , & !!               -- " --
                                 b4w   => c4les , & !!               -- " --
                                 b4i   => c4ies , & !!               -- " --
                                 b234w => c5les , & !!               -- " --
                                 b234i => c5ies     !!               -- " --

  IMPLICIT NONE

  PRIVATE

  PUBLIC  :: satad_v_3D
  PUBLIC  :: satad_v_3D_gpu
  PUBLIC  :: spec_humi
  PUBLIC  :: sat_pres_water
  PUBLIC  :: sat_pres_ice
  PUBLIC  :: dqsatdT
  PUBLIC  :: dqsatdT_ice
  PUBLIC  :: qsat_rho
  PUBLIC  :: dqsatdT_rho
  PUBLIC  :: latent_heat_vaporization
  PUBLIC  :: latent_heat_sublimation
  PUBLIC  :: latent_heat_melting
  
  INTEGER, PARAMETER :: ipsat = 1    ! (1) Tetens (1930)
                                     ! (2) Murphy-Koop for liq and ice 

  REAL (KIND=ireals), PARAMETER :: cp_v = 1850._ireals ! specific heat of water vapor J
                                                       ! at constant pressure
                                                       ! (Landolt-Bornstein)
  REAL (KIND=ireals), PARAMETER :: ci = 2108.0_ireals  ! specific heat of ice
  
  real(ireals), parameter  ::      &
       &  c1i_mk = 9.550426,    &  ! coefficients in Murphy and Koop saturation vapor pressure
       &  c2i_mk = 5723.265,    &  ! over ice and over  liquid water
       &  c3i_mk = 3.53068,     &
       &  c4i_mk = 0.00728332,  &
       &  c1w_mk = 54.842763,   &
       &  c2w_mk = 6763.22,     &
       &  c3w_mk = 4.210,       &
       &  c4w_mk = 0.000367,    &
       &  c5w_mk = 53.878,      &
       &  c6w_mk = 1331.22,     &
       &  c7w_mk = 9.44523,     &
       &  c8w_mk = 0.014025,    &
       &  xi_mk  = 0.0415,      &
       &  t0_mk  = 218.8

CONTAINS

SUBROUTINE satad_v_3D (maxiter, tol, te, qve, qce,    & ! IN, INOUT
                       rhotot,                        & ! IN
                       kdim, klo, kup,                & ! IN
                       errstat)                         ! optional

  !-------------------------------------------------------------------------------
  !
  ! Description:
  !   This routine corrects the temperature (te), the specific humidity (qve),
  !   the cloud water content (qce) and the pressure (pe) for condensation/evaporation.
  !   Pressure adapts itself in ICON but has to be rediagnosed in COSMO
  !
  ! Method:
  !   Saturation adjustment at constant total density (adjustment of T and p accordingly)
  !   assuming chemical equilibrium of water and vapor. For the heat capacity of
  !   of the total system (dry air, vapor, and hydrometeors) the value of dry air
  !   is taken, which is a common approximation and introduces only a small error.
  !
  ! Inout fields: te, qve, qce (both COSMO and ICON); ppe (COSMO only)
  !
  ! Input only fields: rhotot (ICON only); qle, qie, p0e (COSMO only)
  !
  ! Outputs:  - count  :  number of iterations needed, maximum of all iterated gridpoints
  !           - errstat:  error status, /= 0 denotes an error; in this case, the calling routine
  !                       has to abort the model run immediately!!!
  !
  ! Description of input/inout - fields:
  !
  ! te:           abs. temperature (adjusted during satad)           [K]
  ! qve:          specific vapor content (adjusted during satad)     [-]
  ! qce:          specific cloud content (adjusted during satad)     [-]
  ! rhotot:       total density, assumed constant during satad       [kg/m^3]   (ICON only)
  ! qle:          non-adjusted liquid water (i.e., rain drops)       [-]        (COSMO only)
  ! qie:          non-adjusted total ice (i.e., sum of qi, qs, ...)  [-]        (COSMO only)
  ! p0e:          base state pressure                                [Pa]       (COSMO only)
  ! ppe:          pressure deviation                                 [Pa]       (COSMO only)
  !
  !-------------------------------------------------------------------------------

  IMPLICIT NONE

  ! Subroutine arguments:
  ! --------------------
  INTEGER (KIND=iintegers), INTENT (IN)    ::  &
       maxiter,                 & !  Max. number of iterations in the numerical scheme
       kdim,                    & !  Dimension of I/O-fields
       klo, kup                   !  start- and end-indices for the computations

  REAL    (KIND=ireals),    INTENT (IN) ::  &
       tol                 ! Desired abs. accuracy in K of the adjusted temperature

  REAL    (KIND=ireals),    INTENT (INOUT), DIMENSION(:) ::  &  !  dim (kdim)
       te      , & ! Temperature on input/ouput
       qve     , & ! Specific humidity on input/output
       qce         ! Specific cloud water content on input/output

  REAL    (KIND=ireals),    INTENT (IN),  DIMENSION(:) ::  &  !  dim (kdim)
       rhotot    ! density containing dry air and water constituents

!KF error status temporarly set to optional
  INTEGER (KIND=iintegers), INTENT (OUT),  OPTIONAL  ::  &
       errstat                ! Error status of the saturation adjustment
                                !(errstat /= 0 means an error)
  ! Local variables:
  ! -------------
  INTEGER (KIND=iintegers) ::  &
       k, count                 !  Loop index, iteration counter

  REAL    (KIND=ireals   ) ::  &
       Ttest, qtest, qw, qwd, qwa, dqwd, fT, dfT, & !, cvvmcl, qd,
       zqwmin              ! Minimum cloud water content for adjustment

  REAL    (KIND=ireals) ::  &
       lwdocvd  ! (Temperature-dependent) latent heat of vaporization over cv

  REAL (KIND=ireals) ::  &
       twork, tworkold

  LOGICAL :: iter_mask

  !------------ End of header ----------------------------------------------------

  !-------------------------------------------------------------------------------
  ! Begin Subroutine satad
  !-------------------------------------------------------------------------------

  ! Initialization

  IF (PRESENT(errstat)) errstat = 0_iintegers

  zqwmin = 1.0E-20_ireals

!!!=============================================================================================

  ! Single column: the original gather/scatter of "points needing the Newton
  ! iteration" into a compacted index list was a CPU/NEC vectorization trick
  ! (relevant when iterating over many horizontal points at once). With one
  ! column, at most one point per level needs it, so we just decide per level
  ! whether Newton iteration is needed and, if so, run it directly (mirroring
  ! the per-point-independent structure of satad_v_3D_gpu below).

  DO k = klo, kup

    ! total content of the species which are changed by the adjustment:
    qw = qve(k) + qce(k)

    ! check whether this level will still be subsaturated even
    ! if all the cloud water would have been evaporated.
    ! At such points, the Newton iteration is not necessary and the
    ! adjusted values of T, p, qv and qc can be obtained directly.

    lwdocvd = latent_heat_vaporization(te(k)) / cvd

    Ttest = te(k) - lwdocvd*qce(k)

    qtest = qsat_rho(Ttest, rhotot(k))

    iter_mask = .FALSE.
    twork = 0.0_ireals

    IF (qw <= qtest) THEN
      ! In this case, all the cloud water evaporates and there is still (sub)saturation.
      ! The resulting state depends only on the available cloud water and is
      ! not saturated, which enables direct computation of the adjusted variables:
      qve(k)  = qw
      qce(k)  = 0.0_ireals
      te(k)   = Ttest
    ELSE
      ! In this case, the Newton iteration is needed.
      ! Field for the iterated temperature, here set the starting value for the below iteration
      ! to the "old" temperature (as an alternative, the arithmetic mean
      ! between "old" temperature and dew point has been tested, but did
      ! not significantly increase the convergence speed of the iteration):
      twork  = te(k)

      ! And this is the storage variable for the "old" values in the below iteration:
      ! Add some nonesense increment to the starting, which is sufficient to trigger the
      ! iteration below:
      tworkold = twork + 10.0_ireals
      iter_mask = .TRUE.
    END IF

    ! Do the Newton iteration at this level, if needed:
    ! --------------------------------------------------
    count = 0
    DO WHILE (iter_mask .AND. ABS(twork-tworkold) > tol .AND. count < maxiter)
      tworkold = twork
      qwd  = qsat_rho(twork, rhotot(k))
      dqwd = dqsatdT_rho(qwd, twork, rhotot(k))
      ! Newton:
      fT = twork - te(k) + lwdocvd*(qwd - qve(k))
      dfT = 1.0_ireals + lwdocvd*dqwd
      twork = twork - fT / dfT
      count = count + 1
    END DO

    IF (iter_mask) THEN
      te(k) = twork

      ! We disregard here the extrapolation of qsat from the second-last iteration
      ! step, which is done in the original routine to exactly preserve the internal energy.
      ! This introduces a small error (see the COSMO-Documentation, Part III).
      qwa = qsat_rho(te(k), rhotot(k))
      qce(k) = MAX( qce(k) + qve(k) - qwa,zqwmin)
      qve(k) = qwa
    END IF

  END DO

!!!=============================================================================================

!KF
! since ICON re-diagnoses the Exner pressure after satad we do not need
! the following:

END SUBROUTINE satad_v_3D

SUBROUTINE satad_v_3D_gpu (maxiter, tol, te, qve, qce,    & ! IN, INOUT
                           rhotot,                        & ! IN
                           kdim, klo, kup,                & ! IN
                           errstat)                         ! optional

  !-------------------------------------------------------------------------------
  !
  ! Description:
  !   This routine corrects the temperature (te), the specific humidity (qve),
  !   the cloud water content (qce) and the pressure (pe) for condensation/evaporation.
  !   Pressure adapts itself in ICON but has to be rediagnosed in COSMO
  !
  ! Method:
  !   Saturation adjustment at constant total density (adjustment of T and p accordingly)
  !   assuming chemical equilibrium of water and vapor. For the heat capacity of
  !   of the total system (dry air, vapor, and hydrometeors) the value of dry air
  !   is taken, which is a common approximation and introduces only a small error.
  !
  ! Inout fields: te, qve, qce (both COSMO and ICON); ppe (COSMO only)
  !
  ! Input only fields: rhotot (ICON only); qle, qie, p0e (COSMO only)
  !
  ! Outputs:  - count  :  number of iterations needed, maximum of all iterated gridpoints
  !           - errstat:  error status, /= 0 denotes an error; in this case, the calling routine
  !                       has to abort the model run immediately!!!
  !
  ! Description of input/inout - fields:
  !
  ! te:           abs. temperature (adjusted during satad)           [K]
  ! qve:          specific vapor content (adjusted during satad)     [-]
  ! qce:          specific cloud content (adjusted during satad)     [-]
  ! rhotot:       total density, assumed constant during satad       [kg/m^3]   (ICON only)
  ! qle:          non-adjusted liquid water (i.e., rain drops)       [-]        (COSMO only)
  ! qie:          non-adjusted total ice (i.e., sum of qi, qs, ...)  [-]        (COSMO only)
  ! p0e:          base state pressure                                [Pa]       (COSMO only)
  ! ppe:          pressure deviation                                 [Pa]       (COSMO only)
  !
  !-------------------------------------------------------------------------------

  IMPLICIT NONE

  ! Subroutine arguments:
  ! --------------------
  INTEGER (KIND=iintegers), INTENT (IN)    ::  &
       maxiter,                 & !  Max. number of iterations in the numerical scheme
       kdim,                    & !  Dimension of I/O-fields
       klo, kup                   !  start- and end-indices for the computations

  REAL    (KIND=ireals),    INTENT (IN) ::  &
       tol                 ! Desired abs. accuracy in K of the adjusted temperature


  REAL    (KIND=ireals),    INTENT (INOUT), DIMENSION(:) ::  &  !  dim (kdim)
       te      , & ! Temperature on input/ouput
       qve     , & ! Specific humidity on input/output
       qce         ! Specific cloud water content on input/output

  REAL    (KIND=ireals),    INTENT (IN),  DIMENSION(:) ::  &  !  dim (kdim)
       rhotot    ! density containing dry air and water constituents

!KF error status temporarly set to optional
  INTEGER (KIND=iintegers), INTENT (OUT),  OPTIONAL  ::  &
       errstat                ! Error status of the saturation adjustment
                                !(errstat /= 0 means an error)
  ! Local variables:
  ! -------------
  INTEGER (KIND=iintegers) ::  &
       k, count                 !  Loop index, iteration counter

  REAL    (KIND=ireals   ) ::  &
       Ttest, qtest, qw, qwd, qwa, dqwd, fT, dfT, & !, cvvmcl, qd,
       zqwmin              ! Minimum cloud water content for adjustment

  REAL    (KIND=ireals) ::  &
       lwdocvd  ! (Temperature-dependent) latent heat of vaporization over cv

  REAL (KIND=ireals) ::  &
       twork, tworkold

  LOGICAL :: iter_mask

  REAL (KIND=ireals), PARAMETER :: cp_v = 1850._ireals ! specific heat of water vapor J
                                                       !at constant pressure
                                                       ! (Landolt-Bornstein)

  !------------ End of header ----------------------------------------------------

  !-------------------------------------------------------------------------------
  ! Begin Subroutine satad
  !-------------------------------------------------------------------------------

  ! Initialization

  IF (PRESENT(errstat)) errstat = 0_iintegers

  zqwmin = 1.0E-20_ireals

  ! KF old uncorrect version
  ! Some constants:
  ! lwdocvd = lwd / cvd

!!!=============================================================================================

  DO k = klo, kup
    ! total content of the species which are changed by the adjustment:
    qw = qve(k) + qce(k)

    ! check whether this level will still be subsaturated even
    ! if all the cloud water would have been evaporated.
    ! At such points, the Newton iteration is not necessary and the
    ! adjusted values of T, p, qv and qc can be obtained directly.

    lwdocvd = latent_heat_vaporization(te(k)) / cvd
    Ttest = te(k) - lwdocvd*qce(k)
    qtest = qsat_rho(Ttest, rhotot(k))

    iter_mask = .FALSE.
    twork = 0.0_ireals
    IF (qw <= qtest ) THEN
      ! In this case, all the cloud water evaporates and there is still (sub)saturation.
      ! The resulting state depends only on the available cloud water and is
      ! not saturated, which enables direct computation of the adjusted variables:
      qve(k)  = qw
      qce(k)  = 0.0_ireals
      te(k)   = Ttest
    ELSE
      ! Field for the iterated temperature, here set the starting value for the below iteration
      ! to the "old" temperature (as an alternative, the arithmetic mean
      ! between "old" temperature and dew point has been tested, but did
      ! not significantly increase the convergence speed of the iteration):
      twork  = te(k)
      ! And this is the storage variable for the "old" values in the below iteration:
      ! Add some nonesense increment to the starting, which is sufficient to trigger the
      ! iteration below:
      tworkold = twork + 10.0_ireals
      iter_mask = .TRUE.
    END IF

    ! Do the Newton iteration at this level, if needed:
    ! ---------------------------------------------------
    DO count = 1, maxiter
      IF (iter_mask) THEN
        IF (ABS(twork-tworkold) > tol) THEN
          ! Here we still have to iterate ...
          tworkold = twork
          qwd  = qsat_rho(twork, rhotot(k))
          dqwd = dqsatdT_rho(qwd, twork, rhotot(k) )
          ! Newton:
          fT = twork - te(k) + lwdocvd*(qwd - qve(k))
          dfT = 1.0_ireals + lwdocvd*dqwd
          twork = twork - fT / dfT
        END IF
      END IF
    END DO

    ! Distribute the results back:
    ! -----------------------------
    IF(iter_mask) THEN
      te (k) = twork
      ! We disregard here the extrapolation of qsat from the second-last iteration
      ! step, which is done in the original routine to exactly preserve the internal energy.
      ! This introduces a small error (see the COSMO-Documentation, Part III).
      qwa = qsat_rho(te(k), rhotot(k))
      qce(k) = MAX( qce(k) + qve(k) - qwa,zqwmin)
      qve(k) = qwa
    ENDIF

  ENDDO !k

!!!=============================================================================================
!!!=============================================================================================

END SUBROUTINE satad_v_3D_gpu

ELEMENTAL FUNCTION sat_pres_water(temp)
  IMPLICIT NONE

  REAL (KIND=ireals)              :: sat_pres_water
  REAL (KIND=ireals), INTENT(IN)  :: temp

  !$ACC ROUTINE SEQ

  IF (ipsat <= 1) THEN
    sat_pres_water = b1*EXP( b2w*(temp-b3)/(temp-b4w) )
  ELSEIF (ipsat == 2) THEN
    sat_pres_water = psatw_murphykoop(temp)
  END IF

END FUNCTION sat_pres_water

!==============================================================================

ELEMENTAL FUNCTION sat_pres_ice(temp)
  IMPLICIT NONE

  REAL (KIND=ireals)              :: sat_pres_ice
  REAL (KIND=ireals), INTENT(IN)  :: temp

  !$ACC ROUTINE SEQ

  IF (ipsat <= 1) THEN
    sat_pres_ice = b1*EXP( b2i*(temp-b3)/(temp-b4i) )
  ELSEIF (ipsat==2 .OR. ipsat==3) THEN
    sat_pres_ice = psati_murphykoop(temp)
  ENDIF

END FUNCTION sat_pres_ice

!==============================================================================

ELEMENTAL FUNCTION spec_humi(pvap,pres)
  IMPLICIT NONE

  REAL (KIND=ireals)              :: spec_humi
  REAL (KIND=ireals), INTENT(IN)  :: pres,pvap

  !$ACC ROUTINE SEQ

  spec_humi = rdv*pvap/( pres - o_m_rdv*pvap )

END FUNCTION spec_humi

! Routine with ACC ROUTINE SEQ cannot be elemental
#ifndef _OPENACC
ELEMENTAL &
#endif
FUNCTION qsat_rho(temp, rhotot)
  !$ACC ROUTINE SEQ

    !-------------------------------------------------------------------------------
    !
    ! Description:
    !   Specific humidity at water saturation (with respect to flat surface)
    !   depending on the temperature "temp" and the total density "rhotot")
    !
    !-------------------------------------------------------------------------------

  ! input variables: temperature [K], total density [kg/m^2]:
  REAL (KIND=ireals)             :: qsat_rho
  REAL (KIND=ireals), INTENT(IN) :: temp, rhotot

  qsat_rho   = sat_pres_water(temp) / (rhotot * r_v * temp)

END FUNCTION qsat_rho

! Routine with ACC ROUTINE SEQ cannot be elemental
#ifndef _OPENACC
ELEMENTAL &
#endif
FUNCTION dqsatdT_rho(zqsat, temp, rho)
  !$ACC ROUTINE SEQ

    !-------------------------------------------------------------------------------
    !
    ! Description:
    !   Partial derivative of the specific humidity at water saturation with
    !   respect to the temperature at constant total density.
    !   Depends on the temperature "temp" and the
    !   saturation specific humidity "zqsat".
    !
    !-------------------------------------------------------------------------------

  IMPLICIT NONE

  ! input variables: specific saturation humidity [-], temperature [K]:
  REAL (KIND=ireals)            :: dqsatdT_rho
  REAL (KIND=ireals), INTENT(IN):: zqsat, temp, rho

  ! local variables:
  REAL(kind=ireals) :: beta

  IF (ipsat == 1) THEN
    beta        = b234w/(temp-b4w)**2_iintegers - 1.0_ireals / temp
    dqsatdT_rho = beta * zqsat
  ELSEIF (ipsat == 2) THEN
    beta        = psatw_murphykoop_derivative(temp)
    dqsatdT_rho = beta/(r_v*rho*temp)- zqsat/temp
  END IF
  
END FUNCTION dqsatdT_rho

! UB_20100525<<

ELEMENTAL  FUNCTION dqsatdT (zqsat, temp)

    !-------------------------------------------------------------------------------
    !>
    !! Description:
    !!   Partial derivative of the specific humidity at water saturation with
    !!   respect to the temperature
    !-------------------------------------------------------------------------------

  IMPLICIT NONE

  ! input variables: specific saturation humidity [-], temperature [K]:
  REAL (KIND=ireals)            :: dqsatdT
  REAL (KIND=ireals), INTENT(IN):: zqsat, temp

  dqsatdT = b234w * ( 1.0_ireals + rvd_m_o*zqsat ) * zqsat &
                             / (temp-b4w)**2

END FUNCTION dqsatdT

ELEMENTAL  FUNCTION dqsatdT_ice (zqsat, temp)

    !-------------------------------------------------------------------------------
    !>
    !! Description:
    !!   Partial derivative of the specific humidity at ice saturation with
    !!   respect to the temperature
    !-------------------------------------------------------------------------------

  IMPLICIT NONE

  ! input variables: specific saturation humidity [-], temperature [K]:
  REAL (KIND=ireals)            :: dqsatdT_ice
  REAL (KIND=ireals), INTENT(IN):: zqsat, temp
  !$ACC ROUTINE SEQ

  dqsatdT_ice = b234i * ( 1.0_ireals + rvd_m_o*zqsat ) * zqsat &
                             / (temp-b4i)**2

END FUNCTION dqsatdT_ice

! Routine with ACC ROUTINE SEQ cannot be elemental
#ifndef _OPENACC
ELEMENTAL &
#endif
FUNCTION latent_heat_vaporization(temp)

    !-------------------------------------------------------------------------------
    !>
    !! Description:
    !!   Latent heat of vaporization as internal energy and taking into account
    !!   Kirchoff's relations
    !-------------------------------------------------------------------------------

  IMPLICIT NONE
  REAL(KIND=ireals)             :: latent_heat_vaporization
  REAL(KIND=ireals), INTENT(IN) :: temp
  !$ACC ROUTINE SEQ
  
  latent_heat_vaporization = lwd + (cp_v - cl)*(temp-tmelt) - r_v*temp
  
END FUNCTION latent_heat_vaporization

! Routine with ACC ROUTINE SEQ cannot be elemental
#ifndef _OPENACC
ELEMENTAL &
#endif
FUNCTION latent_heat_sublimation(temp)

    !-------------------------------------------------------------------------------
    !>
    !! Description:
    !!   Latent heat of sublimation as internal energy and taking into account
    !!   Kirchoff's relations
    !-------------------------------------------------------------------------------

  IMPLICIT NONE
  REAL(KIND=ireals)             :: latent_heat_sublimation
  REAL(KIND=ireals), INTENT(IN) :: temp
  !$ACC ROUTINE SEQ
  
  latent_heat_sublimation = led + (cp_v - ci)*(temp-tmelt) - r_v*temp
  
END FUNCTION latent_heat_sublimation

ELEMENTAL FUNCTION latent_heat_melting(temp)

    !-------------------------------------------------------------------------------
    !>
    !! Description:
    !!   Latent heat of sublimation as internal energy and taking into account
    !!   Kirchoff's relations
    !-------------------------------------------------------------------------------

  !$ACC ROUTINE SEQ

  IMPLICIT NONE
  REAL(KIND=ireals)             :: latent_heat_melting
  REAL(KIND=ireals), INTENT(IN) :: temp

  latent_heat_melting = lwd - led + (ci - cl)*(temp-tmelt)
  
END FUNCTION latent_heat_melting

ELEMENTAL FUNCTION psatw_murphykoop(tk)
  IMPLICIT NONE
  REAL(KIND=ireals)             :: psatw_murphykoop
  REAL(KIND=ireals), intent(IN) :: tk
    
  ! Eq. (10) of Murphy and Koop (2005)
  psatw_murphykoop = exp( c1w_mk - c2w_mk/tk - c3w_mk*log(tk) + c4w_mk*tk &
       & + tanh(xi_mk*(tk-t0_mk))*(c5w_mk-c6w_mk/tk-c7w_mk*log(tk)+c8w_mk*tk) )
    
END FUNCTION psatw_murphykoop

ELEMENTAL FUNCTION psati_murphykoop(tk)
  IMPLICIT NONE
  REAL(KIND=ireals)             :: psati_murphykoop
  REAL(KIND=ireals), intent(IN) :: tk
  
  ! Eq. (7) of Murphy and Koop (2005)
  psati_murphykoop = exp(c1i_mk - c2i_mk/tk + c3i_mk*log(tk) - c4i_mk*tk )
    
END FUNCTION psati_murphykoop
  
ELEMENTAL FUNCTION psatw_murphykoop_derivative(tk)
  IMPLICIT NONE
  REAL(KIND=ireals)             :: psatw_murphykoop_derivative
  REAL(KIND=ireals), intent(IN) :: tk
       
  ! Derivative of Eq. (10) of Murphy and Koop (2005)
  psatw_murphykoop_derivative = &
       & (c2w_mk/tk**2 - c3w_mk/tk + c4w_mk) * exp( c1w_mk - c2w_mk/tk - c3w_mk*log(tk) + c4w_mk*tk)  &
       & + cosh( xi_mk*(tk-t0_mk))**(-2) * (c5w_mk - c6w_mk/tk - c7w_mk*log(tk) + c8w_mk*tk) &
       & + tanh( xi_mk*(tk-t0_mk)) * (c6w_mk/tk**2 - c7w_mk/tk + c8w_mk)
    
END FUNCTION psatw_murphykoop_derivative

ELEMENTAL FUNCTION psati_murphykoop_derivative(tk)
  IMPLICIT NONE
  REAL(KIND=ireals)             :: psati_murphykoop_derivative
  REAL(KIND=ireals), intent(IN) :: tk
       
  ! Derivative of Eq. (7) of Murphy and Koop (2005)
  psati_murphykoop_derivative = (c2i_mk/tk**2 + c3i_mk/tk - c4i_mk) * psati_murphykoop(tk)
    
END FUNCTION psati_murphykoop_derivative
  

END MODULE mo_satad

