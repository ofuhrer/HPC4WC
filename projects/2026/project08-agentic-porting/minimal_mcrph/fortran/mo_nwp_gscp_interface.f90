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

! Single-column driver for the two-moment bulk microphysics scheme by
! Seifert and Beheng (2006) with prognostic cloud droplet number and
! aerosol/CCN/IN tracers (originally ICON's inwp_gscp == 5).
!
! This is a sandbox version of the original mo_nwp_gscp_interface.f90.
! It has been decoupled from ICON's block/grid-point/domain-nesting
! infrastructure (t_patch, prm_diag, atm_phy_nwp_config(jg), ...) so it
! operates on a single vertical column, passed in as plain arrays.

MODULE mo_nwp_gscp_interface

  USE mo_kind,                 ONLY: wp
  USE mo_exception,            ONLY: message
  USE mo_2mom_mcrph_driver,    ONLY: two_moment_mcrph
  USE mo_satad,                ONLY: satad_v_3D, satad_v_3D_gpu
  USE mo_stage_dump,           ONLY: dump_stage

  IMPLICIT NONE

  PRIVATE

  PUBLIC  ::  nwp_microphysics

CONTAINS
  !!
  !!-------------------------------------------------------------------------
  !!
  SUBROUTINE nwp_microphysics(  nlev, kstart,                 & !>in: column size, start level for moist physics
                            &   dt, lsatad,                    & !>in: time step, satad on/off
                            &   dz, hhl, rho, pres, w, tke,    & !>in: layer geometry, density, pressure, w, tke
                            &   tk,                            & !>inout: temperature
                            &   qv, ssat,                      & !>inout: humidity, supersaturation
                            &   qc, qnc, qr, qnr,               & !>inout: cloud, rain
                            &   qi, qni, qs, qns,               & !>inout: ice, snow
                            &   qg, qng, qh, qnh,               & !>inout: graupel, hail
                            &   nccn, ninpot, ninagi, ninact,   & !>inout: CCN / IN tracers
                            &   qrsflux,                        & !>inout: 3D precipitation flux (for LHN)
                            &   prec_r, prec_i, prec_s, prec_g, prec_h, & !>inout: precip rates
                            &   prec_gsp_rate,                  & !>out: combined surface precip rate
                            &   ithermo_water, ice_type,        & !>in: physics switches
                            &   luse_agi, iagi_param,           & !>in
                            &   lexpl_supersat,                 & !>in
                            &   msg_level,                      & !>in
                            &   lcompute_tt_lheat, tt_lheat     ) !>inout, optional: LHN temperature-tendency bookkeeping

    INTEGER,  INTENT(in)   :: nlev             !< number of vertical levels (column size)
    INTEGER,  INTENT(in)   :: kstart           !< first level with active moist physics

    REAL(wp), INTENT(in)   :: dt               !< time step for microphysics
    LOGICAL,  INTENT(in)   :: lsatad           !< saturation adjustment on/off

    REAL(wp), DIMENSION(nlev),   INTENT(in)    :: dz     !< layer thickness
    REAL(wp), DIMENSION(nlev+1), INTENT(in)    :: hhl    !< height of half levels
    REAL(wp), DIMENSION(nlev),   INTENT(in)    :: rho    !< density
    REAL(wp), DIMENSION(nlev),   INTENT(in)    :: pres   !< pressure
    REAL(wp), DIMENSION(nlev),   INTENT(inout) :: w      !< vertical wind speed
    REAL(wp), DIMENSION(:), INTENT(in), POINTER :: tke   !< turbulent kinetic energy (half levels), may be NULL()

    REAL(wp), DIMENSION(nlev), INTENT(inout) :: tk       !< temperature

    REAL(wp), DIMENSION(nlev), INTENT(inout) :: &
         &  qv, qc, qnc, qr, qnr, qi, qni, qs, qns, qg, qng, qh, qnh, ninact

    REAL(wp), DIMENSION(nlev), INTENT(inout), OPTIONAL :: &
         &  ssat, nccn, ninpot, ninagi

    REAL(wp), DIMENSION(nlev), INTENT(inout) :: qrsflux

    REAL(wp), INTENT(inout) :: prec_r, prec_i, prec_s, prec_g, prec_h
    REAL(wp), INTENT(out)   :: prec_gsp_rate    !< combined surface precip rate (rain+snow+graupel+hail; ice excluded, see below)

    INTEGER,  INTENT(in) :: ithermo_water       !< thermodynamic option for latent heat
    INTEGER,  INTENT(in) :: ice_type            !< ice nucleation parameterization choice
    LOGICAL,  INTENT(in) :: luse_agi            !< use AgI (cloud seeding) tracer
    INTEGER,  INTENT(in), OPTIONAL :: iagi_param
    LOGICAL,  INTENT(in) :: lexpl_supersat      !< explicit supersaturation prediction on/off

    INTEGER,  INTENT(in) :: msg_level           !< message/debug verbosity

    LOGICAL,  INTENT(in), OPTIONAL    :: lcompute_tt_lheat !< TRUE: accumulate temperature tendency for LHN
    REAL(wp), DIMENSION(nlev), INTENT(inout), OPTIONAL :: tt_lheat

    INTEGER :: jk

    LOGICAL :: lcompute_tt_lheat_now

    lcompute_tt_lheat_now = PRESENT(lcompute_tt_lheat) .AND. PRESENT(tt_lheat)
    IF (lcompute_tt_lheat_now) lcompute_tt_lheat_now = lcompute_tt_lheat

    ! tt_lheat to be used for LHN
    ! lateron the updated temperature is added again
    IF (lcompute_tt_lheat_now) THEN
      DO jk = 1, nlev
        tt_lheat(jk) = tt_lheat(jk) - tk(jk)
      ENDDO
    ENDIF

    ! saturation adjustment before microphysics
    ! - this is the first satad call
    ! - second satad call after two_moment_mcrph below

    IF (lsatad) THEN

      IF (msg_level >= 15) &
       & CALL message('mo_nwp_gscp_interface:', 'performing initial saturation adjustment')
#ifdef _OPENACC
      CALL satad_v_3d_gpu(                             &
#else
      CALL satad_v_3d(                                 &
#endif
           & maxiter  = 10                            ,& !> IN
           & tol      = 1.e-3_wp                      ,& !> IN
           & te       = tk                            ,& !> INOUT
           & qve      = qv                            ,& !> INOUT
           & qce      = qc                            ,& !> INOUT
           & rhotot   = rho                           ,& !> IN
           & kdim     = nlev                          ,& !> IN
           & klo      = kstart                        ,& !> IN
           & kup      = nlev                           & !> IN
           )

    ENDIF

    ! Output only, no-op unless the driver was given a dump directory. Mixing-ratio
    ! space -- this is before prepare_twomoment's conversion to densities.
    CALL dump_stage('satad_pre', nlev, rho, pres, w, tk, qv, &
         qc, qnc, qr, qnr, qi, qni, qs, qns, qg, qng, qh, qnh, &
         nccn, ninpot, ninact)

    ! two-moment scheme with prognostic cloud droplet number
    ! and budget equations for CCN and IN

    CALL two_moment_mcrph(                       &
                 ke     = nlev,                  &!in: number of levels
                 ks     = kstart,                &!in: start level
                 dt     = dt ,                   &!in: time step
                 dz     = dz,                     &!in: vertical layer thickness
                 hhl    = hhl,                    &!in: height of half levels
                 rho    = rho,                    &!in:  density
                 pres   = pres,                   &!in:  pressure
                 tke    = tke,                    &!in:  turbulent kinetic energy (half levels)
                 qv     = qv, &!inout: humidity
                 ssat   = ssat, &!inout: supersaturation with respect to liquid water
                 qc     = qc, &!inout: cloud water
                 qnc    = qnc,&!inout: cloud droplet number
                 qr     = qr, &!inout: rain
                 qnr    = qnr,&!inout: rain drop number
                 qi     = qi, &!inout: ice
                 qni    = qni,&!inout: cloud ice number
                 qs     = qs, &!inout: snow
                 qns    = qns,&!inout: snow number
                 qg     = qg, &!inout: graupel
                 qng    = qng,&!inout: graupel number
                 qh     = qh, &!inout: hail
                 qnh    = qnh,&!inout: hail number
                 nccn   = nccn,&!inout: CCN number
                 ninpot = ninpot, &!inout: IN number
                 ninagi = ninagi, &!inout: IN AgI number
                 ninact = ninact, &!inout: IN number
                 tk     = tk,                     &!inout: temp
                 w      = w,                      &!inout: w
                 prec_r = prec_r,  &!inout precp rate rain
                 prec_i = prec_i,   &!inout precp rate ice
                 prec_s = prec_s,  &!inout precp rate snow
                 prec_g = prec_g,&!inout precp rate graupel
                 prec_h = prec_h,   &!inout precp rate hail
                 qrsflux= qrsflux,      & !inout: 3D precipitation flux for LHN
                 msg_level = msg_level,    &
                 & l_cv=.TRUE.,    &
                 & ithermo_water=ithermo_water, & !< in: latent heat choice
                 & ice_type=ice_type, &
                 & luse_agi=luse_agi, &
                 & iagi_param=iagi_param, &
                 & lexpl_supersat=lexpl_supersat )

    !-------------------------------------------------------------------------
    !>
    !! Calculate surface precipitation
    !!
    !-------------------------------------------------------------------------

    ! note: ice is deliberately excluded here because it predominantly contains blowing snow
    prec_gsp_rate = prec_r + prec_s + prec_h + prec_g

    ! saturation adjustment after microphysics
    ! - this is the second satad call
    ! - first satad call before two_moment_mcrph above

    IF (lsatad) THEN

      IF (msg_level >= 15) &
       & CALL message('mo_nwp_gscp_interface:', 'performing final saturation adjustment')
#ifdef _OPENACC
      CALL satad_v_3d_gpu(                             &
#else
      CALL satad_v_3d(                                 &
#endif
           & maxiter  = 10                            ,& !> IN
           & tol      = 1.e-3_wp                      ,& !> IN
           & te       = tk                            ,& !> INOUT
           & qve      = qv                            ,& !> INOUT
           & qce      = qc                            ,& !> INOUT
           & rhotot   = rho                           ,& !> IN
           & kdim     = nlev                          ,& !> IN
           & klo      = kstart                        ,& !> IN
           & kup      = nlev                           & !> IN
           )

    ENDIF

    ! Update tt_lheat to be used in LHN
    IF (lcompute_tt_lheat_now) THEN
      DO jk = 1, nlev
        tt_lheat(jk) = tt_lheat(jk) + tk(jk)
      ENDDO
    ENDIF

  END SUBROUTINE nwp_microphysics

END MODULE mo_nwp_gscp_interface
