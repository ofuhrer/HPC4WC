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

! Two-moment mixed-phase bulk microphysics

!NEC$ options "-finline-max-depth=3 -finline-max-function-size=1000"

MODULE mo_2mom_mcrph_driver

!------------------------------------------------------------------------------
!
! Description:
!
!   The subroutines in the module "gscp" calculate the rates of change of
!   temperature, cloud condensate and water vapor due to cloud microphysical
!   processes related to the formation of grid scale clouds and precipitation.
!   In the COSMO model the microphysical subroutines are either
!   called from "organize_gscp" or from "organize_physics" itself.
!
! - uses intrinsic gamma function. In gfortran you may want to compile with
!   the option -fall-intrinsics
!
!==============================================================================
!
! Declarations:
!
! Modules used:
!------------------------------------------------------------------------------
! Microphysical constants and variables
!------------------------------------------------------------------------------

USE mo_kind,                 ONLY: wp
USE mo_physical_constants,   ONLY: &
    rhoh2o,           & ! density of liquid water
    alv,              & ! latent heat of vaporization
    als,              & ! latent heat of sublimation
    cpdr  => rcpd,    & ! (spec. heat of dry air at constant press)^-1
    cvdr  => rcvd,    & ! (spec. heat of dry air at const vol)^-1
    rho_ice => rhoice   ! density of pure ice

USE mo_satad,                ONLY: latent_heat_sublimation, latent_heat_melting

USE mo_exception,            ONLY: finish, message, message_text

USE mo_timer,                ONLY:                                                              &
                              timers_level, timer_start, timer_stop, timer_phys_2mom_dmin_init, &
                              timer_phys_2mom_prepost, timer_phys_2mom_proc, timer_phys_2mom_sedi


USE mo_reff_types,           ONLY: t_reff_calc
USE mo_stage_dump,           ONLY: dump_stage

USE mo_2mom_mcrph_config,    ONLY: t_cfg_2mom

USE mo_2mom_mcrph_main,      ONLY:                                &
     &                        clouds_twomoment,                   &
     &                        atmosphere, particle, particle_frozen, particle_lwf, &
     &                        rain_coeffs, ice_coeffs, snow_coeffs, graupel_coeffs, hail_coeffs, &
     &                        ccn_coeffs, in_coeffs,                     &
     &                        init_2mom_scheme, init_2mom_scheme_once,   &
     &                        qnc_const

USE mo_2mom_mcrph_processes,  ONLY:  q_crit, cfg_params

USE mo_2mom_mcrph_config_default, ONLY: cfg_2mom_default

! MINIMAL SCHEME: no wet-growth dmin lookup table -> nothing needed from
! mo_2mom_mcrph_util here.
USE mo_2mom_mcrph_types, ONLY: ltabdminwgg, ltabdminwgh

USE mo_2mom_prepare, ONLY: prepare_twomoment, post_twomoment
!==============================================================================

  IMPLICIT NONE
  PUBLIC

  CHARACTER(len=*), PARAMETER :: routine = 'mo_2mom_mcrph_driver'
  INTEGER,          PARAMETER :: dbg_level = 25                   ! level for debug prints

  ! In place of ICON's mo_run_config: this single-column sandbox has no LHN
  ! (latent-heat-nudging) data-assimilation cycle, so it is always off.
  LOGICAL, PARAMETER :: ldass_lhn = .FALSE.

  ! .. exponents for simple density of terminal fall velocity
  REAL(wp), PARAMETER :: rho_vel    = 0.4e0_wp    !..exponent for density correction
  REAL(wp), PARAMETER :: rho_vel_c  = 0.2e0_wp    !..for cloud droplets (in exact terms, this would be the ratio of dyn. visc. as function of T)
  REAL(wp), PARAMETER :: rho0       = 1.225_wp    !..reference air density

  INTEGER :: cloud_type, ccn_type

  LOGICAL :: lconstant_lh   ! Use constant latent heat (default .true.)

! UB: These settings should be converted into namelist parameters in the future!

!! Now in namelist phy_ctl!  INTEGER, PARAMETER :: i2mom_solver = 1  ! (0) explicit (1) semi-implicit solve
!!$  ! now this comes from cfg_params !  INTEGER, PARAMETER :: i2mom_solver = 1  ! (0) explicit (1) semi-implicit solve
  
  INTEGER, PARAMETER :: cloud_type_default_gscp4 = 2603, ccn_type_gscp4 = 7
  INTEGER, PARAMETER :: cloud_type_default_gscp5 = 2003, ccn_type_gscp5 = 8

  ! AS: For gscp=4 use 2103 with ccn_type = 1 (HDCP2 IN and CCN schemes)
  !     For gscp=5 use 2603 with ccn_type = 8 (PDA ice nucleation and Segal&Khain CCN activation)
  
  ! AS: Runs without hail, e.g, 1503 are buggy and give a segmentation fault.
  !     So far I was not able to identify the problem, needs more detailed debugging.
  
CONTAINS
  
  !==============================================================================
  !
  ! Two-moment mixed-phase bulk microphysics
  !
  ! original version by Axel Seifert, May 2003
  ! with modifications by Ulrich Blahak, August 2007

  !==============================================================================
  SUBROUTINE two_moment_mcrph(            &
                       ke,                & ! in: number of levels
                       ks,                & ! in: start index vertical , optional
                       dt,                & ! in: time step
                       dz,                & ! in: vertical layer thickness
                       hhl,               & ! in: height of half levels
                       rho,               & ! in: density
                       pres,              & ! in: pressure
                       tke,               & ! in: tke
                       qv,                & ! inout: specific humidity
                       ssat,              & ! inout: supersaturation w.r.t. liquid water
                       qc, qnc,           & ! inout: cloud water
                       qr, qnr,           & ! inout: rain
                       qi, qni,           & ! inout: ice
                       qs, qns,           & ! inout: snow
                       qg, qng, qgl,      & ! inout: graupel
                       qh, qnh, qhl,      & ! inout: hail
                       nccn,              & ! inout: ccn
                       ninpot,            & ! inout: potential ice nuclei
                       ninagi,            & ! inout: AgI tracer as additional potential IN
                       ninact,            & ! inout: activated ice nuclei
                       tk,                & ! inout: temp
                       w,                 & ! inout: w
                       prec_r,            & ! inout: precip rate rain
                       prec_i,            & ! inout: precip rate ice
                       prec_s,            & ! inout: precip rate snow
                       prec_g,            & ! inout: precip rate graupel
                       prec_h,            & ! inout: precip rate hail
                       qrsflux,           & ! inout: 3D total precipitation rate
                       dtemp,             & ! inout: opt. temp increment
                       msg_level,         & ! in: msg_level
                       l_cv,              & ! in: switch for cv/cp
                       ithermo_water,     & ! in: thermodynamic option
                       ice_type,          &
                       luse_agi,          &
                       iagi_param,        &
                       lexpl_supersat         )

    ! Declare variables in argument list

    INTEGER,            INTENT (IN)  :: ke    ! number of levels
    INTEGER,  OPTIONAL, INTENT (IN)  :: ks    ! start index vertical

    REAL(wp), INTENT (IN)            :: dt           ! time step

    ! Dynamical core variables
    REAL(wp), DIMENSION(:), INTENT(IN), TARGET :: dz, rho, pres, w

    ! Optional Dynamical core variables
    REAL(wp), DIMENSION(:), INTENT(IN), POINTER :: tke

    REAL(wp), DIMENSION(:), INTENT(IN), TARGET :: hhl

    REAL(wp), DIMENSION(:), INTENT(INOUT), TARGET :: tk

    ! Microphysics variables
    REAL(wp), DIMENSION(:), INTENT(INOUT) , TARGET :: &
         qv, qc, qnc, qr, qnr, qi, qni, qs, qns, qg, qng, qh, qnh, ninact

    REAL(wp), DIMENSION(:), INTENT(INOUT), TARGET, OPTIONAL :: &
         &               qgl, qhl, ssat

    REAL(wp), DIMENSION(:), INTENT(INOUT), TARGET, OPTIONAL :: &
         &               nccn, ninpot, ninagi

    ! Precip rates
    REAL(wp), INTENT (INOUT) :: &
         &               prec_r, prec_i, prec_s, prec_g, prec_h
    REAL(wp), DIMENSION(:), INTENT (INOUT) :: qrsflux

    REAL(wp), OPTIONAL, INTENT (INOUT)  :: dtemp(:)

    INTEGER,  INTENT (IN)             :: msg_level

    LOGICAL,  OPTIONAL,  INTENT (IN)  :: l_cv, luse_agi, lexpl_supersat

    INTEGER,  OPTIONAL,  INTENT (IN)  :: ithermo_water
    INTEGER,  OPTIONAL,  INTENT (IN)  :: ice_type, iagi_param

    ! ... Variables which are global in module_2mom_mcrph_main

    REAL(wp), TARGET, DIMENSION(ke) ::        &
         &  rhocorr,       & ! density dependency of particle fall speed
         &  rhocld           ! density dependency of particle fall speed for cloud droplets

    REAL(wp) :: q_liq_old(ke), q_vap_old(ke)  ! to store old values for latent heat calc

    INTEGER  :: kts,kte
    INTEGER  :: kk
    INTEGER  :: ntsedi_rain, ntsedi_graupel, ntsedi_hail     ! for sedimentation sub stepping

    REAL(wp) :: q_liq_new,q_vap_new
    REAL(wp) :: zf,hlp,dtemp_loc
    REAL(wp) :: convliq,convice,led,lwe
    REAL(wp), PARAMETER :: tau_inact =  600.  ! relaxation time scale for activated IN number density
    REAL(wp), PARAMETER :: tau_inpot = 1800.  ! relaxation time scale for potential IN number density
    REAL(wp) :: in_bgrd            ! background profile of IN number density
    REAL(wp) :: z_heat_cap_r       ! reciprocal of cpdr or cvdr (depending on l_cv)
    REAL(wp) :: rdz(ke), rho_r(ke)

    LOGICAL :: lprogccn, lprogin, lprogmelt, lexpl_supersat_loc, luse_agi_loc

    LOGICAL, PARAMETER :: debug     = .false.       !
    LOGICAL, PARAMETER :: clipping  = .true.        ! not really necessary, just for cleanup

    CHARACTER(len=*), PARAMETER :: routine = 'mo_2mom_mcrph_driver'

    ! These structures include the pointers to the model arrays (which are automatic arrays
    ! of this driver subroutine). These structures live only for one time step and are
    ! different for the OpenMP threads. In contrast, the types like rain_coeffs, ice_coeffs,
    ! etc. that are declared in mo_2mom_mcrph_main live for the whole runtime and may include
    ! coefficients that are calculated once during initialization (and there is only one per
    ! mpi thread).
    TYPE(atmosphere)           :: atmo

    ! These are the fundamental hydrometeor particle variables for the two-moment scheme
    ! as they are used in the various options of the scheme
    TYPE(particle), target          :: cloud_hyd, rain_hyd
    TYPE(particle_frozen), target   :: ice_frz, snow_frz, graupel_frz, hail_frz
    TYPE(particle_lwf), target      :: graupel_lwf, hail_lwf

    ! Pointers to the derived types that are actually needed
    CLASS(particle), pointer        :: cloud, rain
    CLASS(particle_frozen), pointer :: ice, snow, graupel, hail

    IF (PRESENT(lexpl_supersat)) THEN
        lexpl_supersat_loc = lexpl_supersat
    ELSE
        lexpl_supersat_loc = .FALSE.
    END IF                                  

    IF (PRESENT(luse_agi)) THEN
        luse_agi_loc = luse_agi
    ELSE
        luse_agi_loc = .FALSE.
    END IF                                  

    lprogccn  = PRESENT(nccn)
    lprogin   = PRESENT(ninpot)
    lprogmelt = PRESENT(qgl)

    IF (msg_level>5) CALL message (TRIM(routine), "called two_moment_mcrph")

    IF (PRESENT(ithermo_water)) THEN
       lconstant_lh = (ithermo_water == 0)
    ELSE  ! Default themodynamic is constant latent heat
       lconstant_lh = .true.
    END IF

    IF (lprogccn) THEN
       cloud_type = cloud_type_default_gscp5 + 10 * ccn_type
    ELSE
       cloud_type = cloud_type_default_gscp4 + 10 * ccn_type
    END IF

    IF (PRESENT(ice_type)) THEN
        cloud_type = cloud_type + 100 * ice_type
    ELSE
        cloud_type = cloud_type + 200
    END IF

    cloud => cloud_hyd
    rain  => rain_hyd
    ice   => ice_frz
    snow  => snow_frz

    IF (lprogmelt) THEN
       graupel => graupel_lwf   ! with prognostic melting 
       hail => hail_lwf         ! of graupel and hail
    ELSE
       graupel => graupel_frz   ! simple melting
       hail => hail_frz
    END IF

    ! start/end level
    IF (PRESENT(ks)) THEN
      kts = ks
    ELSE
      kts = 1
    END IF
    kte = ke

    IF (timers_level > 10) CALL timer_start(timer_phys_2mom_prepost)

    ! inverse of vertical layer thickness
    DO kk = kts,kte
      rdz(kk) = 1._wp / dz(kk)

      IF (clipping) THEN
            IF(qr(kk) < 0.0_wp) qr(kk) = 0.0_wp
            IF(qi(kk) < 0.0_wp) qi(kk) = 0.0_wp
            IF(qs(kk) < 0.0_wp) qs(kk) = 0.0_wp
            IF(qg(kk) < 0.0_wp) qg(kk) = 0.0_wp
            IF(qh(kk) < 0.0_wp) qh(kk) = 0.0_wp
      END IF
    ENDDO

    ! Initialize qrsflux for LHN:
    IF (ldass_lhn) THEN
      DO kk = kts,kte
        qrsflux(kk) = 0.0_wp
      ENDDO
    END IF

    IF (clipping) THEN
       IF (lprogmelt) THEN
          WHERE(qgl(kts:kte) < 0.0_wp) qgl(kts:kte) = 0.0_wp
          WHERE(qhl(kts:kte) < 0.0_wp) qhl(kts:kte) = 0.0_wp
       END IF
    END IF

    IF (PRESENT(l_cv)) THEN
      IF (l_cv) THEN
        z_heat_cap_r = cvdr
      ELSE
        z_heat_cap_r = cpdr
      ENDIF
    ELSE
      z_heat_cap_r = cpdr
    ENDIF

    IF (msg_level>dbg_level) CALL message(TRIM(routine),'')

    IF (msg_level>dbg_level)THEN
       WRITE (message_text,'(1X,A,I4,3(A,L2))') &
            & "cloud_type = ",cloud_type,", lprogccn = ",lprogccn,", lprogin = ",lprogin,", lprogmelt = ",lprogmelt
       CALL message(TRIM(routine),TRIM(message_text))
    END IF

    IF (msg_level>dbg_level) CALL message(TRIM(routine), "prepare variables for 2mom")

    DO kk = kts, kte

       ! ... 1/rho is used quite often
       rho_r(kk) = 1.0 / rho(kk)

       ! ... height dependency of terminal fall velocities
       hlp = LOG(MAX(rho(kk),1e-6_wp)/rho0)
       rhocorr(kk) = exp(-rho_vel*hlp)
       rhocld(kk)  = exp(-rho_vel_c*hlp)

    END DO

    ! .. set the particle types, but no calculations
    CALL init_2mom_scheme(cloud,rain,ice,snow,graupel,hail)

    ! .. convert to densities and set pointerns to two-moment module
    !    (pointers are used to avoid passing everything explicitly by argument and
    !     to avoid local allocates within the OpenMP-loop, and keep everything on stack)
    IF (msg_level>dbg_level) THEN
        WRITE(message_text,*) 'about to call prepare_twomoment, ssat min/max = ', MINVAL(ssat), MAXVAL(ssat)
        CALL message(routine, message_text)
        WRITE(message_text,*) 'lexpl_supersat = ', lexpl_supersat
        CALL message(routine, message_text)
    END IF
    CALL prepare_twomoment(atmo, cloud, rain, ice, snow, graupel, hail, &
         rho, rhocorr, rhocld, pres, w, tk, hhl, tke, &
         nccn, ninpot, ninagi, ninact, ssat, &
         qv, qc, qnc, qr, qnr, qi, qni, qs, qns, qg, qng, qh, qnh, qgl, qhl, &
         lprogccn, lprogin, luse_agi_loc, lexpl_supersat_loc, lprogmelt, kts, kte)
    IF (timers_level > 10) CALL timer_stop(timer_phys_2mom_prepost)

    ! Output only, no-op unless the driver was given a dump directory. The flat
    ! arrays are what prepare_twomoment converted in place (the particle pointers
    ! alias them), so dumping them here is the density-space state clouds_twomoment
    ! is about to see.
    CALL dump_stage('prepare', kte-kts+1, rho(kts:kte), pres(kts:kte), w(kts:kte), &
         tk(kts:kte), qv(kts:kte), qc(kts:kte), qnc(kts:kte), qr(kts:kte), qnr(kts:kte), &
         qi(kts:kte), qni(kts:kte), qs(kts:kte), qns(kts:kte), qg(kts:kte), qng(kts:kte), &
         qh(kts:kte), qnh(kts:kte), nccn(kts:kte), ninpot(kts:kte), ninact(kts:kte))

    IF (msg_level>dbg_level) CALL message(TRIM(routine)," calling clouds_twomoment")

    ! MINIMAL SCHEME: always the explicit path. The semi-implicit sedimentation
    ! solver has been removed; with no sedimentation the two solvers are
    ! equivalent, so cfg_params%i2mom_solver is ignored here.

      ! ... save old variables for latent heat calculation
      DO kk = kts, kte
        q_vap_old(kk) = qv(kk)

        IF (.NOT. lprogmelt) THEN
             q_liq_old(kk) = qc(kk) + qr(kk)
        END IF

      ENDDO

      IF (lprogmelt) THEN
         q_liq_old(kts:kte) = qc(kts:kte) + qr(kts:kte)  &
              &              + qgl(kts:kte) + qhl(kts:kte)
      END IF

       IF (timers_level > 10) CALL timer_start(timer_phys_2mom_proc)
       ! .. this subroutine calculates all the microphysical sources and sinks
       CALL clouds_twomoment(kts, kte, dt, lprogin, &
            atmo, cloud, rain, ice, snow, graupel, hail, ssat, lexpl_supersat_loc, ninact, nccn, ninpot, &
            ninagi, luse_agi_loc, iagi_param)

       IF (timers_level > 10) CALL timer_stop(timer_phys_2mom_proc)

       IF (lprogccn) THEN
        WHERE(qc(kts:kte) == 0.0_wp) cloud%n(kts:kte) = 0.0_wp
       END IF

       DO kk = kts, kte

            IF (lconstant_lh) THEN
              led = als
              lwe = (alv-als)
            ELSE
              led = latent_heat_sublimation(tk(kk))
              lwe = latent_heat_melting(tk(kk))
            END IF

            ! .. latent heat term for temperature equation
            convice = z_heat_cap_r * led
            convliq = z_heat_cap_r * lwe

            ! .. new variables
            q_vap_new = qv(kk)
            if (lprogmelt) then
              q_liq_new = qr(kk) + qc(kk) + qgl(kk) + qhl(kk)
            else
              q_liq_new = qr(kk) + qc(kk)
            end if

            ! .. update temperature
            dtemp_loc  = - convice * rho_r(kk) * (q_vap_new - q_vap_old(kk))  &
                 &       + convliq * rho_r(kk) * (q_liq_new - q_liq_old(kk))

            tk(kk) = tk(kk) + dtemp_loc

            IF(PRESENT(dtemp)) &
                 dtemp(kk) = dtemp_loc

       ENDDO

       ! MINIMAL SCHEME: no sedimentation. The retained processes act only on
       ! cloud-sized particles, so there is no precipitation flux and no vertical
       ! solver. (The full scheme would call sedimentation_explicit() here.)

    IF (timers_level > 10) CALL timer_start(timer_phys_2mom_prepost)    
    
    ! .. check for negative values
    IF (debug) CALL check_clouds()

    ! .. convert back and nullify two-moment pointers
    CALL post_twomoment(atmo, cloud, rain, ice, snow, graupel, hail, &
         rho_r, qnc, nccn, ninpot, ninagi, ninact, ssat, &
         qv, qc, qr, qnr, qi, qni, qs, qns, qg, qng, qh, qnh, qgl, qhl,  &
         lprogccn, lprogin, luse_agi_loc, lexpl_supersat_loc, lprogmelt, kts, kte)

    ! Back in mixing-ratio space, before the driver's own clipping and the
    ! nccn/ninpot background relaxation below -- that block is not part of
    ! clouds_twomoment, so it needs its own boundary to be checkable.
    CALL dump_stage('post', kte-kts+1, rho(kts:kte), pres(kts:kte), w(kts:kte), &
         tk(kts:kte), qv(kts:kte), qc(kts:kte), qnc(kts:kte), qr(kts:kte), qnr(kts:kte), &
         qi(kts:kte), qni(kts:kte), qs(kts:kte), qns(kts:kte), qg(kts:kte), qng(kts:kte), &
         qh(kts:kte), qnh(kts:kte), nccn(kts:kte), ninpot(kts:kte), ninact(kts:kte))

    IF (clipping) THEN
      IF (lprogmelt) THEN
        WHERE(qgl(kts:kte) < 0.0_wp) qgl(kts:kte) = 0.0_wp
        WHERE(qhl(kts:kte) < 0.0_wp) qhl(kts:kte) = 0.0_wp
      END IF
    END IF

    DO kk = kts,kte

      IF (clipping) THEN
        IF ( qr(kk) < 0.0_wp)  qr(kk) = 0.0_wp
        IF ( qi(kk) < 0.0_wp)  qi(kk) = 0.0_wp
        IF ( qs(kk) < 0.0_wp)  qs(kk) = 0.0_wp
        IF ( qg(kk) < 0.0_wp)  qg(kk) = 0.0_wp
        IF ( qh(kk) < 0.0_wp)  qh(kk) = 0.0_wp
        IF (qnr(kk) < 0.0_wp) qnr(kk) = 0.0_wp
        IF (qni(kk) < 0.0_wp) qni(kk) = 0.0_wp
        IF (qns(kk) < 0.0_wp) qns(kk) = 0.0_wp
        IF (qng(kk) < 0.0_wp) qng(kk) = 0.0_wp
        IF (qnh(kk) < 0.0_wp) qnh(kk) = 0.0_wp
      END IF

      IF (lprogccn) THEN
        zf = 0.5_wp*(hhl(kk)+hhl(kk+1))
        !..reset nccn for cloud-free grid points to background profile
        IF (qc(kk) .LE. q_crit) THEN
          IF(zf > ccn_coeffs%z0) THEN
            nccn(kk) = MAX(nccn(kk),ccn_coeffs%Ncn0 &
                 * EXP((ccn_coeffs%z0 - zf)*(1._wp/ccn_coeffs%z1e)))
          ELSE
            nccn(kk) = MAX(nccn(kk),ccn_coeffs%Ncn0)
          END IF
        END IF
      END IF

      !..relaxation of activated IN number density to zero
      IF(qi(kk) == 0) THEN
        ninact(kk) = ninact(kk) - ninact(kk)*(1._wp/tau_inact)*dt
      END IF

      IF (lprogin) THEN
        zf = 0.5_wp*(hhl(kk)+hhl(kk+1))
        !..relaxation of potential IN number density to background profile
        IF(zf > in_coeffs%z0) THEN
          in_bgrd = in_coeffs%N0*EXP((in_coeffs%z0 - zf)*(1._wp/in_coeffs%z1e))
        ELSE
          in_bgrd = in_coeffs%N0
        END IF
        ninpot(kk) = ninpot(kk) - (ninpot(kk)-in_bgrd)*(1._wp/tau_inpot)*dt
      END IF

      IF (lprogccn) THEN
        IF ( nccn(kk) < 35e6_wp ) nccn(kk) = 35e6_wp
      END IF

      IF(qc(kk) < 1.0e-12_wp) qnc(kk) = 0.0_wp

    ENDDO

    IF (msg_level>dbg_level) CALL message(TRIM(routine), "two moment mcrph ends!")

    IF (timers_level > 10) CALL timer_stop(timer_phys_2mom_prepost) 

    RETURN
    !
    ! end of driver routine, but many details are below in the contains-part of this subroutine
    !
  CONTAINS

    ! MINIMAL SCHEME: the semi-implicit sedimentation solver
    ! (clouds_twomoment_implicit) and the explicit sedimentation routine
    ! (sedimentation_explicit) have been removed — no precipitation-size
    ! particles means no vertical solver is needed.

    !
    ! check for negative values after microphysics
    !
    SUBROUTINE check_clouds()

      REAL(wp), PARAMETER :: meps = -1e-12

      IF (cloud_type.lt.2000) THEN
         IF (ANY(qh(kts:kte)>0._wp)) THEN
            qh(kts:kte)  = 0.0_wp
            WRITE (message_text,'(1X,A)') '  qh > 0, after cloud_twomoment for cloud_type < 2000'
            CALL message(routine,TRIM(message_text))
            CALL finish(TRIM(routine),'Error in two_moment_mcrph')
         END IF
         IF (ANY(qnh(kts:kte)>0._wp)) THEN
            qnh(kts:kte)  = 0.0_wp
            WRITE (message_text,'(1X,A)') '  qnh > 0, after cloud_twomoment for cloud_type < 2000'
            CALL message(routine,TRIM(message_text))
            CALL finish(TRIM(routine),'Error in two_moment_mcrph')
         END IF
      END IF
      IF (msg_level>dbg_level) CALL message(TRIM(routine), " test for negative values")
      IF (MINVAL(cloud%q(kts:kte)) < meps) THEN
         CALL finish(TRIM(routine),'Error in two_moment_mcrph, cloud%q < 0')
      ENDIF
      IF (MINVAL(rain%q(kts:kte)) < meps) THEN
         CALL finish(TRIM(routine),'Error in two_moment_mcrph, rain%q < 0')
      ENDIF
      IF (MINVAL(ice%q(kts:kte)) < meps) THEN
         CALL finish(TRIM(routine),'Error in two_moment_mcrph, ice%q < 0,')
      ENDIF
      IF (MINVAL(snow%q(kts:kte)) < meps) THEN
         CALL finish(TRIM(routine),'Error in two_moment_mcrph, snow%q < 0')
      ENDIF
      IF (MINVAL(graupel%q(kts:kte)) < meps) THEN
         CALL finish(TRIM(routine),'Error in two_moment_mcrph, graupel%q < 0')
      ENDIF
      IF (MINVAL(hail%q(kts:kte)) < meps) THEN
         CALL finish(TRIM(routine),'Error in two_moment_mcrph, hail%q < 0')
      ENDIF
      IF (MINVAL(cloud%n(kts:kte)) < meps) THEN
         CALL finish(TRIM(routine),'Error in two_moment_mcrph, cloud%n < 0')
      ENDIF
      IF (MINVAL(rain%n(kts:kte)) < meps) THEN
         CALL finish(TRIM(routine),'Error in two_moment_mcrph, rain%n < 0')
      ENDIF
      IF (MINVAL(ice%n(kts:kte)) < meps) THEN
         CALL finish(TRIM(routine),'Error in two_moment_mcrph, ice%n < 0')
      ENDIF
      IF (MINVAL(snow%n(kts:kte)) < meps) THEN
         CALL finish(TRIM(routine),'Error in two_moment_mcrph, snow%n < 0')
      ENDIF
      IF (MINVAL(graupel%n(kts:kte)) < meps) THEN
         CALL finish(TRIM(routine),'Error in two_moment_mcrph, graupel%n < 0')
      ENDIF
      IF (MINVAL(hail%n(kts:kte)) < meps) THEN
         CALL finish(TRIM(routine),'Error in two_moment_mcrph, hail%n < 0')
      ENDIF
    END subroutine check_clouds

  END SUBROUTINE two_moment_mcrph

  SUBROUTINE implicit_core(q_val,q_sum,q_impl,vsed_new,vsed_now,flux_new,flux_now,rdzdt)

    REAL(wp), INTENT(INOUT) :: &
         &    q_val,q_sum,q_impl,vsed_new,vsed_now,flux_new,flux_now,rdzdt

    REAL(wp) :: q_star, flux_sum

    ! new on r.h.s. is new value from level above
    vsed_new = 0.5 * (vsed_now + vsed_new)

    ! qflux_new, nflux_new are the updated flux values from the level above
    ! qflux_now, nflux_now are here the old (current time step) flux values from the level above
    ! In COSMO-Docu  {...} =  flux_(k-1),new + flux_(k-1),start
    flux_sum = flux_new + flux_now

    ! qflux_now, nflux_now are here overwritten with the current level
    flux_now = min(vsed_now * q_val,  flux_sum)    ! (loop dependency)
    flux_now = max(flux_now,0.0_wp)                ! maybe not necessary

    ! time integrated value without implicit weight
    q_sum  = q_val  + rdzdt * (flux_sum - flux_now)

    ! implicit weight
    q_impl = 1.0_wp/(1.0_wp + vsed_new * rdzdt)

    ! prepare for source term calculation
    q_star    = q_impl * q_sum
    q_val  = q_star                     ! source/sinks work on star-values
    q_sum  = q_sum - q_star

  END SUBROUTINE implicit_core

  SUBROUTINE implicit_time(q_val,q_sum,q_impl,vsed_new,vsed_now,flux_new)

    REAL(wp), INTENT(INOUT) :: &
         &    q_val,q_sum,q_impl,vsed_new,vsed_now,flux_new

    ! time integration
    q_val =   MAX( 0.0_wp, q_impl*(q_sum + q_val))

    ! prepare for next level
    flux_new = q_val * vsed_new     ! flux_(k),new
    vsed_new = vsed_now

  END SUBROUTINE implicit_time

  !===========================================================================================

  SUBROUTINE two_moment_mcrph_init(igscp,ice_type,N_cn0,z0_nccn,z1e_nccn,N_in0,z0_nin,z1e_nin,msg_level,cfg_2mom)

    INTEGER, INTENT(IN) :: igscp, msg_level, ice_type

    REAL(wp), OPTIONAL, INTENT(OUT) ::             & ! for CCN and IN in case of gscp=5
         & N_cn0,z0_nccn,z1e_nccn,    &
         & N_in0,z0_nin,z1e_nin

    TYPE(particle)        :: cloud, rain
    TYPE(particle_frozen) :: ice, snow, graupel, hail
    TYPE(particle_lwf)    :: graupel_lwf, hail_lwf

    TYPE(t_cfg_2mom), OPTIONAL, INTENT(in) :: cfg_2mom

    INTEGER        :: unitnr

    ! Transfer the configuration parameters to the 2mom internal type instance:
    IF (PRESENT(cfg_2mom)) THEN
      cfg_params = cfg_2mom
    ELSE
      cfg_params = cfg_2mom_default
    END IF

    IF (msg_level>5) THEN
      CALL message (TRIM(routine), " Initialization of two-moment microphysics scheme") 
      WRITE(message_text,'(A,I5)')   "   inwp_gscp    = ",igscp ; CALL message(TRIM(routine),TRIM(message_text))
      WRITE(message_text,'(A,I5)')   "   i2mom_solver = ",cfg_params%i2mom_solver ; CALL message(TRIM(routine),TRIM(message_text))
      WRITE(message_text,'(A,L5)'  ) "   lconstant_lh = ",lconstant_lh ; CALL message(TRIM(routine),TRIM(message_text))
    END IF

    IF (PRESENT(N_cn0)) THEN
      IF (PRESENT(cfg_2mom)) THEN
        IF (cfg_2mom%ccn_type > 0) THEN
          ccn_type   = cfg_2mom%ccn_type
        ELSE 
          ccn_type   = ccn_type_gscp5
        END IF
      ELSE
        ccn_type   = ccn_type_gscp5
      END IF
      cloud_type = cloud_type_default_gscp5 + 10 * ccn_type
    ELSE
      IF (PRESENT(cfg_2mom)) THEN
        IF (cfg_2mom%ccn_type > 0) THEN
          ccn_type   = cfg_2mom%ccn_type
        ELSE 
          ccn_type   = ccn_type_gscp4
        END IF
      ELSE
        ccn_type   = ccn_type_gscp4
      END IF
      cloud_type = cloud_type_default_gscp4 + 10 * ccn_type
    END IF

    cloud_type = cloud_type + 100 * ice_type

    ! .. set the particle types, and calculate some coefficients
    IF (igscp == 7) THEN
       CALL init_2mom_scheme_once(cloud,rain,ice,snow,graupel_lwf,hail_lwf,cloud_type)
    ELSE
       CALL init_2mom_scheme_once(cloud,rain,ice,snow,graupel,hail,cloud_type)
    END IF

    ! MINIMAL SCHEME: the graupel/hail wet-growth dmin lookup tables are not
    ! initialized. They are only consumed by graupel_hail_conv_wet_gamlook, which
    ! is not part of the reduced call tree, so building them (and the associated
    ! NetCDF file I/O) is skipped entirely.

    !..parameters for exponential decrease of N_ccn with height
    !  z0:  up to this height (m) constant unchanged value
    !  z1e: height interval at which N_ccn decreases by factor 1/e above z0_nccn
    
    ccn_coeffs%z0  = 4000.0_wp
    ccn_coeffs%z1e = 2000.0_wp

    ! min updraft speed for Segal&Khain activation
    ccn_coeffs%wcb_min = cfg_params%ccn_wcb_min

    ! characteristics of different kinds of CN
    ! (copied from COSMO 5.0 Segal & Khain nucleation subroutine)

    SELECT CASE(ccn_type)
    CASE(6)
      !... maritime case
      ccn_coeffs%Ncn0 = 100.0e6_wp   ! CN concentration at ground
      ccn_coeffs%Nmin =  35.0e6_wp   ! NOT relevant at the moment
      ccn_coeffs%lsigs = 0.4_wp      ! log(sigma_s)
      ccn_coeffs%R2    = 0.03_wp     ! in mum
      ccn_coeffs%etas  = 0.9_wp      ! soluble fraction
    CASE(7)
      !... intermediate case
      ccn_coeffs%Ncn0 = 250.0e6_wp
      ccn_coeffs%Nmin =  35.0e6_wp
      ccn_coeffs%lsigs = 0.4_wp
      ccn_coeffs%R2    = 0.03_wp       ! in mum
      ccn_coeffs%etas  = 0.8_wp        ! soluble fraction
    CASE(8)
      IF (cfg_params%tune_sbmccn < 1.0_wp) THEN
        !... maritime case
        ccn_coeffs%Ncn0 = 100.0e6_wp   ! CN concentration at ground
        ccn_coeffs%Nmin =  35.0e6_wp   ! NOT relevant at the moment
        ccn_coeffs%lsigs = 0.4_wp      ! log(sigma_s)
        ccn_coeffs%R2    = 0.03_wp     ! in mum
        ccn_coeffs%etas  = 0.9_wp      ! soluble fraction
      ELSE
        !... continental case
        ccn_coeffs%Ncn0 = 1700.0e6_wp
        ccn_coeffs%Nmin =   35.0e6_wp  ! NOT relevant at the moment
        ccn_coeffs%lsigs = 0.2_wp
        ccn_coeffs%R2    = 0.03_wp     ! in mum
        ccn_coeffs%etas  = 0.7_wp      ! soluble fraction
      END IF
    CASE(9)
      !... "polluted" continental
      ccn_coeffs%Ncn0 = 3200.0e6_wp
      ccn_coeffs%Nmin =   35.0e6_wp    ! NOT relevant at the moment
      ccn_coeffs%lsigs = 0.2_wp
      ccn_coeffs%R2    = 0.03_wp       ! in mum
      ccn_coeffs%etas  = 0.7_wp        ! soluble fraction
     CASE(1)
       !... dummy values
       ccn_coeffs%Ncn0  =  200.0e6_wp
       ccn_coeffs%Nmin  =   10.0e6_wp  ! NOT relevant at the moment
       ccn_coeffs%lsigs = 0.0_wp
       ccn_coeffs%R2    = 0.0_wp
       ccn_coeffs%etas  = 0.0_wp
    CASE DEFAULT
       CALL finish(TRIM(routine),'Error in two_moment_mcrph_init: Invalid value for ccn_type')
    END SELECT

    IF (cfg_params%ccn_Ncn0 > -900.0_wp) THEN
      ccn_coeffs%Ncn0 = cfg_params%ccn_Ncn0
    END IF

    IF (PRESENT(N_cn0)) THEN
      z0_nccn  = ccn_coeffs%z0
      z1e_nccn = ccn_coeffs%z1e
      N_cn0    = ccn_coeffs%Ncn0
    END IF
    
    WRITE(message_text,'(A)') "  CN properties:" ; CALL message(TRIM(routine),TRIM(message_text))
    WRITE(message_text,'(A,D10.3)') "    Ncn0 = ",ccn_coeffs%Ncn0 ; CALL message(TRIM(routine),TRIM(message_text))
    WRITE(message_text,'(A,D10.3)') "    z0   = ",ccn_coeffs%z0  ; CALL message(TRIM(routine),TRIM(message_text))
    WRITE(message_text,'(A,D10.3)') "    z1e  = ",ccn_coeffs%z1e ; CALL message(TRIM(routine),TRIM(message_text))

    IF (PRESENT(N_in0)) THEN

       in_coeffs%N0  = 200.0e6_wp ! this is currently just a scaling factor for the PDA scheme
       in_coeffs%z0  = 3000.0_wp
       in_coeffs%z1e = 1000.0_wp

       N_in0   = in_coeffs%N0
       z0_nin  = in_coeffs%z0
       z1e_nin = in_coeffs%z1e

       WRITE(message_text,'(A)') "  IN properties:" ; CALL message(TRIM(routine),TRIM(message_text))
       WRITE(message_text,'(A,D10.3)') "    Ncn0 = ",in_coeffs%N0  ; CALL message(TRIM(routine),TRIM(message_text))
       WRITE(message_text,'(A,D10.3)') "    z0   = ",in_coeffs%z0  ; CALL message(TRIM(routine),TRIM(message_text))
       WRITE(message_text,'(A,D10.3)') "    z1e  = ",in_coeffs%z1e ; CALL message(TRIM(routine),TRIM(message_text))
     END IF
     
    IF (msg_level>5) CALL message (TRIM(routine), " finished two_moment_mcrph_init successfully")
    !$ACC ENTER DATA COPYIN(ccn_coeffs, in_coeffs, cfg_params)
    !$ACC ENTER DATA COPYIN(ltabdminwgg, ltabdminwgh)
    !$ACC ENTER DATA COPYIN(ltabdminwgg%ltable, ltabdminwgg%x1, ltabdminwgg%x2, ltabdminwgg%x3, ltabdminwgg%x4) &
    !$ACC   COPYIN(ltabdminwgh%ltable, ltabdminwgh%x1, ltabdminwgh%x2, ltabdminwgh%x3, ltabdminwgh%x4)

  END SUBROUTINE two_moment_mcrph_init


  ! Subroutine that provides coefficients for the effective radius calculations
  ! consistent with two-moment microphysics
  SUBROUTINE two_mom_reff_coefficients( reff_calc ,return_fct)
    TYPE(t_reff_calc), INTENT(INOUT) ::  reff_calc                   ! Structure with options and coefficiencts
    LOGICAL          , INTENT(INOUT) ::  return_fct                  ! Return code of the subroutine

    ! These are the fundamental hydrometeor particle variables for the two-moment scheme
    TYPE(particle)        , TARGET   :: cloud_hyd, rain_hyd
    TYPE(particle_frozen) , TARGET   :: ice_frz, snow_frz, graupel_frz, hail_frz
    TYPE(particle_lwf)    , TARGET   :: graupel_lwf, hail_lwf

    ! Pointers to the derived types that are actually needed
    CLASS(particle)       , POINTER  :: cloud, rain
    CLASS(particle_frozen), POINTER  :: ice, snow, graupel, hail

    ! Parameters used in the paramaterization of reff (the same for all)
    CLASS(particle)       , POINTER  :: current_hyd
    REAL(wp)                         :: a_geo, b_geo, mu, nu
    REAL(wp)                         :: bf, bf2 
    LOGICAL                          :: monodisperse
        
    ! Check input return_fct
    IF (.NOT. return_fct) THEN
      WRITE (message_text,*) 'Reff: Function two_mom_provide_reff_coefficients entered with previous error'
      CALL message('',message_text)
      RETURN
    END IF

    ! We need to reinitiate the particles because they are deleted after every micro call
    cloud => cloud_hyd
    rain  => rain_hyd
    ice   => ice_frz
    snow  => snow_frz
    IF (reff_calc%microph_param == 7 ) THEN  ! Vivek Param. Frozen+Liquid
       graupel => graupel_lwf                ! gscp=7
       hail    => hail_lwf
    ELSE
       graupel => graupel_frz                ! gscp=4,5,6
       hail    => hail_frz
    END IF
    ! .. set the particle types, but no calculations
    CALL init_2mom_scheme(cloud,rain,ice,snow,graupel,hail)
   
    SELECT CASE ( reff_calc%hydrometeor )    ! Select Hydrometeor
    CASE (0)                                 ! Cloud water
      current_hyd => cloud
    CASE (1)  
      current_hyd => ice
    CASE (2)  
      current_hyd => rain
    CASE (3)  
      current_hyd => snow
    CASE (4)  
      current_hyd => graupel
    CASE (5)  
      current_hyd => hail
    END SELECT

    ! Extract properties of hydrometeor
    a_geo           = current_hyd%a_geo
    b_geo           = current_hyd%b_geo
    mu              = current_hyd%mu
    nu              = current_hyd%nu
    reff_calc%x_min = current_hyd%x_min
    reff_calc%x_max = current_hyd%x_max

    ! All DSD are polydisperse
    monodisperse    = .false.

    ! Overwrite monodisperse/polydisperse according to options
    SELECT CASE (reff_calc%dsd_type)
    CASE (1)
      monodisperse  = .true.
    CASE (2)
      monodisperse  = .false.
    END SELECT

    IF ( reff_calc%dsd_type == 2) THEN       ! Overwrite mu and nu coefficients
      mu            = reff_calc%mu
      nu            = reff_calc%nu
    END IF

    SELECT CASE ( reff_calc%reff_param )     ! Select Parameterization
    CASE(0)                                  ! Spheroids  Dge = c1 * x**[c2], which x = mean mass
      ! First calculate monodisperse
      reff_calc%reff_coeff(1)   = a_geo
      reff_calc%reff_coeff(2)   = b_geo

      ! Broadening for not monodisperse
      IF ( .NOT. monodisperse ) THEN 
        bf =  GAMMA( (3.0_wp * b_geo + nu + 1.0_wp)/ mu) / GAMMA( (2.0_wp * b_geo + nu + 1.0_wp)/ mu) * &
          & ( GAMMA( (nu + 1.0_wp)/ mu) / GAMMA( (nu + 2.0_wp)/ mu) )**b_geo

        reff_calc%reff_coeff(1) = reff_calc%reff_coeff(1)*bf        
      END IF      

    CASE (1)                                 ! Fu Random Hexagonal needles:  Dge = 1/(c1 * x**[c2] + c3 * x**[c4])
                                             ! Parameterization based on Fu, 1996; Fu et al., 1998; Fu ,2007
      ! First calculate monodisperse
      reff_calc%reff_coeff(1)   = SQRT( 3.0_wp *SQRT(3.0_wp) * rho_ice * a_geo / 8.0_wp )
      reff_calc%reff_coeff(2)   = (b_geo - 1.0_wp)/2.0_wp 
      reff_calc%reff_coeff(3)   = SQRT(3.0_wp)/4.0_wp/a_geo
      reff_calc%reff_coeff(4)   = -b_geo

      ! Broadening for not monodisperse. Generalized gamma distribution
      IF ( .NOT. monodisperse ) THEN 
        bf  =  GAMMA( ( b_geo + 2.0_wp * nu + 3.0_wp)/ mu/2.0_wp ) / GAMMA( (nu + 2.0_wp)/ mu) * &
           & ( GAMMA( (nu + 1.0_wp)/ mu) / GAMMA( (nu + 2.0_wp)/ mu) )**( (b_geo-1.0_wp)/2.0_wp)

        bf2 =  GAMMA( (-b_geo + nu + 2.0_wp)/ mu ) / GAMMA( (nu + 2.0_wp)/ mu) * &
           & ( GAMMA( (nu + 1.0_wp)/ mu) / GAMMA( (nu + 2.0_wp)/ mu) )**( -b_geo)

        reff_calc%reff_coeff(1) = reff_calc%reff_coeff(1)*bf
        reff_calc%reff_coeff(3) = reff_calc%reff_coeff(3)*bf2
      END IF

    END SELECT

  END SUBROUTINE two_mom_reff_coefficients

END MODULE mo_2mom_mcrph_driver
