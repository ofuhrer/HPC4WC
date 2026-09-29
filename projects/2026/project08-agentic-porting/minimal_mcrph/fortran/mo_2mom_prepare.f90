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

MODULE mo_2mom_prepare

  USE mo_kind,            ONLY: wp
  USE mo_exception,       ONLY: finish, message, message_text
  USE mo_2mom_mcrph_main, ONLY: particle, particle_lwf, atmosphere

  IMPLICIT NONE
  PUBLIC :: prepare_twomoment, post_twomoment

  CHARACTER(len=*), PARAMETER :: routine = 'mo_2mom_prepare'

CONTAINS

  SUBROUTINE prepare_twomoment(atmo, cloud, rain, ice, snow, graupel, hail, &
       rho, rhocorr, rhocld, pres, w, tk, hhl, tke, &
       nccn, ninpot, ninagi, ninact, ssat,&
       qv, qc, qnc, qr, qnr, qi, qni, qs, qns, qg, qng, qh, qnh, qgl, qhl, &
       lprogccn, lprogin, luse_agi, lexpl_supersat, lprogmelt, kts, kte)

    TYPE(atmosphere), INTENT(inout)   :: atmo
    CLASS(particle),  INTENT(inout)   :: cloud, rain, ice, snow
    CLASS(particle),  INTENT(inout)   :: graupel, hail
    REAL(wp), TARGET, DIMENSION(:), INTENT(in) :: &
         rho, rhocorr, rhocld, pres, w, tk, hhl
    REAL(wp), POINTER, DIMENSION(:), INTENT(in) :: tke
    REAL(wp), DIMENSION(:), INTENT(inout) , TARGET :: &
         &               qv, qc, qnc, qr, qnr, qi, qni, qs, qns, qg, qng, qh, qnh
    LOGICAL, INTENT(in) :: lprogccn, lprogin, luse_agi, lexpl_supersat, lprogmelt
    REAL(wp), DIMENSION(:), INTENT(INOUT), TARGET, OPTIONAL :: &
         &               nccn, ninpot, ninagi, ninact, ssat
    REAL(wp), DIMENSION(:), INTENT(INOUT), TARGET, OPTIONAL :: &
         &               qgl,qhl
    INTEGER, INTENT(in) :: kts, kte
    INTEGER :: kk

    ! ... Transformation of microphysics variables to densities
    DO kk = kts, kte

      ! ... concentrations --> number densities
      qnc(kk) = rho(kk) * qnc(kk)
      qnr(kk) = rho(kk) * qnr(kk)
      qni(kk) = rho(kk) * qni(kk)
      qns(kk) = rho(kk) * qns(kk)
      qng(kk) = rho(kk) * qng(kk)
      qnh(kk) = rho(kk) * qnh(kk)

      ! ... mixing ratios -> mass densities
      qv(kk) = rho(kk) * qv(kk)
      qc(kk) = rho(kk) * qc(kk)
      qr(kk) = rho(kk) * qr(kk)
      qi(kk) = rho(kk) * qi(kk)
      qs(kk) = rho(kk) * qs(kk)
      qg(kk) = rho(kk) * qg(kk)
      qh(kk) = rho(kk) * qh(kk)

      ninact(kk)  = rho(kk) * ninact(kk)

      IF (lprogccn) THEN
        nccn(kk) = rho(kk) * nccn(kk)
      END IF
      IF (lprogin) THEN
        ninpot(kk)  = rho(kk) * ninpot(kk)
        if (luse_agi) then
            ninagi(kk) = rho(kk) * ninagi(kk)
        end if
      END IF
      if (lexpl_supersat) then
        if (PRESENT(ssat)) then
          ssat(kk) = rho(kk) * ssat(kk)
        else
          call finish(TRIM(routine),'Error, something wrong with ssat')
        end if
      end if
      IF (lprogmelt) THEN
        qgl(kk)  = rho(kk) * qgl(kk)
        qhl(kk)  = rho(kk) * qhl(kk)
      END IF

    END DO

    IF (lprogmelt.AND.(.not.PRESENT(qgl).or..not.PRESENT(qhl))) THEN
      CALL finish(TRIM(routine),'Error in prepare_twomoment, something wrong with qgl or qhl')
    END IF

    ! set pointers
    atmo%w   => w
    atmo%T   => tk
    atmo%p   => pres
    atmo%qv  => qv
    atmo%rho => rho
    atmo%zh  => hhl

    IF (ASSOCIATED(tke)) THEN
      atmo%tke => tke
    ELSE
      atmo%tke=>NULL()
    END IF

    cloud%rho_v   => rhocld
    rain%rho_v    => rhocorr
    ice%rho_v     => rhocorr
    graupel%rho_v => rhocorr
    snow%rho_v    => rhocorr
    hail%rho_v    => rhocorr

    cloud%q   => qc
    cloud%n   => qnc
    rain%q    => qr
    rain%n    => qnr
    ice%q     => qi
    ice%n     => qni
    snow%q    => qs
    snow%n    => qns
    graupel%q => qg
    graupel%n => qng
    hail%q    => qh
    hail%n    => qnh

    SELECT TYPE (graupel)
    CLASS IS (particle_lwf)
       graupel%l => qgl
    END SELECT

    SELECT TYPE (hail)
    CLASS IS (particle_lwf)
       hail%l    => qhl
    END SELECT

    ! enforce upper and lower bounds for number concentrations
    ! (may not be necessary or only at initial time)
    DO kk=kts,kte
      rain%n(kk) = MIN(rain%n(kk), rain%q(kk)/rain%x_min)
      rain%n(kk) = MAX(rain%n(kk), rain%q(kk)/rain%x_max)
      ice%n(kk) = MIN(ice%n(kk), ice%q(kk)/ice%x_min)
      ice%n(kk) = MAX(ice%n(kk), ice%q(kk)/ice%x_max)
      snow%n(kk) = MIN(snow%n(kk), snow%q(kk)/snow%x_min)
      snow%n(kk) = MAX(snow%n(kk), snow%q(kk)/snow%x_max)
      graupel%n(kk) = MIN(graupel%n(kk), graupel%q(kk)/graupel%x_min)
      graupel%n(kk) = MAX(graupel%n(kk), graupel%q(kk)/graupel%x_max)
      hail%n(kk) = MIN(hail%n(kk), hail%q(kk)/hail%x_min)
      hail%n(kk) = MAX(hail%n(kk), hail%q(kk)/hail%x_max)
    END DO

    DO kk=kts,kte
      IF(cloud%q(kk) <= 1.0e-12) cloud%n(kk) = 0.0_wp
      IF(rain%q(kk) <= 1.0e-12) rain%n(kk) = 0.0_wp
      IF(ice%q(kk) <= 1.0e-12) ice%n(kk) = 0.0_wp
      IF(snow%q(kk) <= 1.0e-12) snow%n(kk) = 0.0_wp
      IF(graupel%q(kk) <= 1.0e-12) graupel%n(kk) = 0.0_wp
      IF(hail%q(kk) <= 1.0e-12) hail%n(kk) = 0.0_wp
    END DO

  END SUBROUTINE prepare_twomoment

  SUBROUTINE post_twomoment(atmo, cloud, rain, ice, snow, graupel, hail, &
       rho_r, qnc, nccn, ninpot, ninagi, ninact, ssat, &
       qv, qc, qr, qnr, qi, qni, qs, qns, qg, qng, qh, qnh, qgl, qhl,  &
       lprogccn, lprogin, luse_agi, lexpl_supersat, lprogmelt, kts, kte)

    TYPE(atmosphere), INTENT(inout)   :: atmo
    CLASS(particle), INTENT(inout)    :: cloud, rain, ice, snow
    CLASS(particle), INTENT(inout)    :: graupel, hail
    REAL(wp), INTENT(in) :: rho_r(:)
    REAL(wp), DIMENSION(:), INTENT(inout) :: &
         &           qv, qc, qnc, qr, qnr, qi, qni, qs, qns, qg, qng, qh, qnh
    REAL(wp), DIMENSION(:), INTENT(INOUT), TARGET, OPTIONAL :: &
         &           nccn, ninpot, ninagi, ninact
    REAL(wp), DIMENSION(:), INTENT(INOUT), TARGET, OPTIONAL :: &
         &               qgl,qhl, ssat
    LOGICAL, INTENT(in) :: lprogccn, lprogin, luse_agi, lexpl_supersat, lprogmelt
    INTEGER, INTENT(in) :: kts, kte
    INTEGER :: kk
    REAL(wp) :: hlp

    IF (lprogmelt.AND.(.not.PRESENT(qgl).or..not.PRESENT(qhl))) THEN
      CALL finish(TRIM(routine),'Error in post_twomoment, something wrong with qgl or qhl')
    END IF

    ! nullify pointers
    atmo%w   => NULL()
    atmo%T   => NULL()
    atmo%p   => NULL()
    atmo%qv  => NULL()
    atmo%rho => NULL()
    atmo%zh  => NULL()
    atmo%tke => NULL()

    cloud%rho_v   => NULL()
    rain%rho_v    => NULL()
    ice%rho_v     => NULL()
    graupel%rho_v => NULL()
    snow%rho_v    => NULL()
    hail%rho_v    => NULL()

    cloud%q   => NULL()
    cloud%n   => NULL()
    rain%q    => NULL()
    rain%n    => NULL()
    ice%q     => NULL()
    ice%n     => NULL()
    snow%q    => NULL()
    snow%n    => NULL()
    graupel%q => NULL()
    graupel%n => NULL()
    hail%q    => NULL()
    hail%n    => NULL()

    SELECT TYPE (graupel)
    CLASS IS (particle_lwf)
      graupel%l => NULL()
    END SELECT

    SELECT TYPE (hail)
    CLASS IS (particle_lwf)
      hail%l    => NULL()
    END SELECT

    ! ... Transformation of variables back to ICON standard variables
    DO kk = kts, kte

      hlp = rho_r(kk)

      ! ... from mass densities back to mixing ratios
      qv(kk) = hlp * qv(kk)
      qc(kk) = hlp * qc(kk)
      qr(kk) = hlp * qr(kk)
      qi(kk) = hlp * qi(kk)
      qs(kk) = hlp * qs(kk)
      qg(kk) = hlp * qg(kk)
      qh(kk) = hlp * qh(kk)

      ! ... number concentrations
      qnc(kk) = hlp * qnc(kk)
      qnr(kk) = hlp * qnr(kk)
      qni(kk) = hlp * qni(kk)
      qns(kk) = hlp * qns(kk)
      qng(kk) = hlp * qng(kk)
      qnh(kk) = hlp * qnh(kk)

      ninact(kk)  = hlp * ninact(kk)

      IF (lprogccn) THEN
        nccn(kk) = hlp * nccn(kk)
      END IF
      IF (lprogin) THEN
        ninpot(kk)  = hlp * ninpot(kk)
        if (luse_agi) then
            ninagi(kk) = hlp * ninagi(kk)
        end if
      END IF
      if (lexpl_supersat) then
        if (PRESENT(ssat)) then
          ssat(kk) = hlp * ssat(kk)
        else
          call finish(TRIM(routine),'Error, something wrong with ssat')
        end if
      end if
      IF (lprogmelt) THEN
        qgl(kk)  = hlp * qgl(kk)
        qhl(kk)  = hlp * qhl(kk)
      END IF

    ENDDO

  END SUBROUTINE post_twomoment

END MODULE mo_2mom_prepare
