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

USE mo_2mom_mcrph_config,    ONLY: t_cfg_2mom

USE mo_2mom_mcrph_main,      ONLY:                                &
     &                        clouds_twomoment,                   &
     &                        atmosphere, particle, particle_frozen, particle_lwf, &
     &                        rain_coeffs, ice_coeffs, snow_coeffs, graupel_coeffs, hail_coeffs, &
     &                        ccn_coeffs, in_coeffs,                     &
     &                        init_2mom_scheme, init_2mom_scheme_once,   &
     &                        qnc_const

USE mo_2mom_mcrph_processes,  ONLY:                                &
     &                         sedi_vel_rain, sedi_vel_sphere,     &
     &                         sedi_icon_rain, sedi_icon_sphere, sedi_icon_sphere_lwf, &
     &                         particle_meanmass, sedi_vel_lwf,&
     &                         q_crit, cfg_params

USE mo_2mom_mcrph_config_default, ONLY: cfg_2mom_default

USE mo_2mom_mcrph_util, ONLY:                            &
     &                       init_dmin_wg_gr_ltab_equi,  &
     &                       dmin_wetgrowth_fit_check, luse_dmin_wetgrowth_table, lprintout_comp_table_fit

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

    IF (msg_level>dbg_level) CALL message(TRIM(routine)," calling clouds_twomoment")

    IF (cfg_params%i2mom_solver.eq.0) THEN

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

       IF (timers_level > 10) CALL timer_start(timer_phys_2mom_sedi)

       IF (msg_level>dbg_level) CALL message(TRIM(routine)," calling sedimentation")

       ! .. if we solve explicitly, then sedimentation is done here after microphysics
       CALL sedimentation_explicit()
       IF (timers_level > 10) CALL timer_stop(timer_phys_2mom_sedi) 

    ELSE
      CALL clouds_twomoment_implicit ()
    END IF

    IF (timers_level > 10) CALL timer_start(timer_phys_2mom_prepost)    
    
    ! .. check for negative values
    IF (debug) CALL check_clouds()

    ! .. convert back and nullify two-moment pointers
    CALL post_twomoment(atmo, cloud, rain, ice, snow, graupel, hail, &
         rho_r, qnc, nccn, ninpot, ninagi, ninact, ssat, &
         qv, qc, qr, qnr, qi, qni, qs, qns, qg, qng, qh, qnh, qgl, qhl,  &
         lprogccn, lprogin, luse_agi_loc, lexpl_supersat_loc, lprogmelt, kts, kte)

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

    SUBROUTINE clouds_twomoment_implicit()
      !
      ! semi-implicit solver for sedimentation including microphysics, the same
      ! approach is used in the COSMO microphysics, e.g, hydci_pp
      ! (see COSMO documentation for details)
      !
      ! Single column: all per-point flux/sum/impl/velocity state below is now a
      ! plain scalar (it used to be a 1D array over the horizontal point range,
      ! recomputed fresh at every iteration of the level loop). The DO k=kts+1,kte
      ! loop remains the genuine sequential recurrence: each level's fluxes feed
      ! into the next level's implicit_core/implicit_time call via these scalars.

      real(wp) :: &
           & qr_flux_now,qr_flux_new,qr_sum,vr_sedq_new,vr_sedq_now,qr_impl,xr_now, &
           & nr_flux_now,nr_flux_new,nr_sum,vr_sedn_new,vr_sedn_now,nr_impl,        &
           & qs_flux_now,qs_flux_new,qs_sum,vs_sedq_new,vs_sedq_now,qs_impl,xs_now, &
           & ns_flux_now,ns_flux_new,ns_sum,vs_sedn_new,vs_sedn_now,ns_impl,        &
           & qg_flux_now,qg_flux_new,qg_sum,vg_sedq_new,vg_sedq_now,qg_impl,xg_now, &
           & ng_flux_now,ng_flux_new,ng_sum,vg_sedn_new,vg_sedn_now,ng_impl,        &
           & qh_flux_now,qh_flux_new,qh_sum,vh_sedq_new,vh_sedq_now,qh_impl,xh_now, &
           & nh_flux_now,nh_flux_new,nh_sum,vh_sedn_new,vh_sedn_now,nh_impl,        &
           & qi_flux_now,qi_flux_new,qi_sum,vi_sedq_new,vi_sedq_now,qi_impl,xi_now, &
           & ni_flux_now,ni_flux_new,ni_sum,vi_sedn_new,vi_sedn_now,ni_impl

      ! for lwf variables
      real(wp) :: &
           & lh_flux_now,lh_flux_new,lh_sum,vh_sedl_new,vh_sedl_now,lh_impl, &
           & lg_flux_now,lg_flux_new,lg_sum,vg_sedl_new,vg_sedl_now,lg_impl

      REAL(wp), DIMENSION(ke) :: rdzdt
      INTEGER :: k, kk

      logical, parameter :: lmicro_impl = .true.  ! microphysics within semi-implicit sedimentation loop?

      if (.not.lmicro_impl) then

        ! ... save old variables for latent heat calculation
        if (lprogmelt) then
          q_vap_old(kts:kte) = qv(kts:kte)
          q_liq_old(kts:kte) = qc(kts:kte) + qgl(kts:kte) &
               &              + qr(kts:kte) + qhl(kts:kte)
        else
          q_vap_old(kts:kte) = qv(kts:kte)
          q_liq_old(kts:kte) = qc(kts:kte) + qr(kts:kte)
        end if

        ! .. this subroutine calculates all the microphysical sources and sinks
        CALL clouds_twomoment(kts, kte, dt, lprogin, atmo, cloud, rain, &
             ice, snow, graupel, hail, ssat, lexpl_supersat_loc, ninact, nccn, ninpot, ninagi, luse_agi_loc, iagi_param)

        DO kk=kts,kte
          ! .. latent heat term for temperature equation
          IF (lconstant_lh) THEN
            led = als
            lwe = (alv-als)
          ELSE
            led = latent_heat_sublimation(tk(kk))
            lwe = latent_heat_melting(tk(kk))
          END IF

          convice = z_heat_cap_r * led
          convliq = z_heat_cap_r * lwe

          q_vap_new = qv(kk)
          if (lprogmelt) then
            q_liq_new = qr(kk) + qc(kk) + qgl(kk) + qhl(kk)
          else
            q_liq_new = qr(kk) + qc(kk)
          end if
          tk(kk) = tk(kk) - convice * rho_r(kk) * (q_vap_new - q_vap_old(kk))  &
               &          + convliq * rho_r(kk) * (q_liq_new - q_liq_old(kk))
        ENDDO

      end if

      ! clipping maybe not necessary
      DO k = kts,kte
        IF(qr(k) < 0.0_wp) qr(k) = 0.0_wp
        IF(qi(k) < 0.0_wp) qi(k) = 0.0_wp
        IF(qs(k) < 0.0_wp) qs(k) = 0.0_wp
        IF(qg(k) < 0.0_wp) qg(k) = 0.0_wp
        IF(qh(k) < 0.0_wp) qh(k) = 0.0_wp
        IF(qnr(k) < 0.0_wp) qnr(k) = 0.0_wp
        IF(qni(k) < 0.0_wp) qni(k) = 0.0_wp
        IF(qns(k) < 0.0_wp) qns(k) = 0.0_wp
        IF(qng(k) < 0.0_wp) qng(k) = 0.0_wp
        IF(qnh(k) < 0.0_wp) qnh(k) = 0.0_wp
      ENDDO

      if (lprogmelt) then
        WHERE(qgl(kts:kte) < 0.0_wp) qgl(kts:kte) = 0.0_wp
        WHERE(qhl(kts:kte) < 0.0_wp) qhl(kts:kte) = 0.0_wp
      end if

      DO k = kts,kte
          rdzdt(k) = 0.5_wp * rdz(k) * dt
      ENDDO

      qr_flux_now = 0.0_wp
      nr_flux_now = 0.0_wp
      qr_flux_new = 0.0_wp
      nr_flux_new = 0.0_wp

      qi_flux_now = 0.0_wp
      ni_flux_now = 0.0_wp
      qi_flux_new = 0.0_wp
      ni_flux_new = 0.0_wp

      qs_flux_now = 0.0_wp
      ns_flux_now = 0.0_wp
      qs_flux_new = 0.0_wp
      ns_flux_new = 0.0_wp

      qg_flux_now = 0.0_wp
      ng_flux_now = 0.0_wp
      qg_flux_new = 0.0_wp
      ng_flux_new = 0.0_wp

      qh_flux_now = 0.0_wp
      nh_flux_now = 0.0_wp
      qh_flux_new = 0.0_wp
      nh_flux_new = 0.0_wp

      if (lprogmelt) then
        lg_flux_now = 0.0_wp
        lg_flux_new = 0.0_wp
        lh_flux_now = 0.0_wp
        lh_flux_new = 0.0_wp
      end if

      vr_sedn_new = rain%vsedi_min
      vi_sedn_new = ice%vsedi_min
      vs_sedn_new = snow%vsedi_min
      vg_sedn_new = graupel%vsedi_min
      vh_sedn_new = hail%vsedi_min
      vr_sedq_new = rain%vsedi_min
      vi_sedq_new = ice%vsedi_min
      vs_sedq_new = snow%vsedi_min
      vg_sedq_new = graupel%vsedi_min
      vh_sedq_new = hail%vsedi_min

      if (lprogmelt) then
        vg_sedl_new = graupel%vsedi_min
        vh_sedl_new = hail%vsedi_min
      end if

      ! here we simply assume that there is no cloud or precip in the uppermost level
      ! i.e. we start from kts+1 going down in physical space

      DO k=kts+1,kte

        xr_now = particle_meanmass(rain, qr(k),qnr(k))
        xi_now = particle_meanmass(ice, qi(k),qni(k))
        xs_now = particle_meanmass(snow, qs(k),qns(k))
        xg_now = particle_meanmass(graupel, qg(k),qng(k))
        xh_now = particle_meanmass(hail, qh(k),qnh(k))

        call sedi_vel_rain(rain,rain_coeffs,qr(k),xr_now,rhocorr(k),vr_sedn_now,vr_sedq_now,qc(k))
        call sedi_vel_sphere(ice,ice_coeffs,qi(k),xi_now,rhocorr(k),vi_sedn_now,vi_sedq_now)
        call sedi_vel_sphere(snow,snow_coeffs,qs(k),xs_now,rhocorr(k),vs_sedn_now,vs_sedq_now)
        if (lprogmelt) then
          call sedi_vel_lwf(graupel_lwf,graupel_coeffs,  &
               & qg(k),qgl(k),xg_now,rhocorr(k),vg_sedn_now,vg_sedq_now,vg_sedl_now)
          call sedi_vel_lwf(hail_lwf,hail_coeffs,        &
               & qh(k),qhl(k),xh_now,rhocorr(k),vh_sedn_now,vh_sedq_now,vh_sedl_now)
        else
          call sedi_vel_sphere(graupel,graupel_coeffs,qg(k),xg_now,rhocorr(k),vg_sedn_now,vg_sedq_now)
          call sedi_vel_sphere(hail,hail_coeffs,qh(k),xh_now,rhocorr(k),vh_sedn_now,vh_sedq_now)
        end if

        call implicit_core(qr(k), qr_sum,qr_impl,vr_sedq_new,vr_sedq_now,qr_flux_new,qr_flux_now,rdzdt(k))
        call implicit_core(qnr(k),nr_sum,nr_impl,vr_sedn_new,vr_sedn_now,nr_flux_new,nr_flux_now,rdzdt(k))
        call implicit_core(qi(k), qi_sum,qi_impl,vi_sedq_new,vi_sedq_now,qi_flux_new,qi_flux_now,rdzdt(k))
        call implicit_core(qni(k),ni_sum,ni_impl,vi_sedn_new,vi_sedn_now,ni_flux_new,ni_flux_now,rdzdt(k))
        call implicit_core(qs(k), qs_sum,qs_impl,vs_sedq_new,vs_sedq_now,qs_flux_new,qs_flux_now,rdzdt(k))
        call implicit_core(qns(k),ns_sum,ns_impl,vs_sedn_new,vs_sedn_now,ns_flux_new,ns_flux_now,rdzdt(k))
        call implicit_core(qg(k), qg_sum,qg_impl,vg_sedq_new,vg_sedq_now,qg_flux_new,qg_flux_now,rdzdt(k))
        call implicit_core(qng(k),ng_sum,ng_impl,vg_sedn_new,vg_sedn_now,ng_flux_new,ng_flux_now,rdzdt(k))
        call implicit_core(qh(k), qh_sum,qh_impl,vh_sedq_new,vh_sedq_now,qh_flux_new,qh_flux_now,rdzdt(k))
        call implicit_core(qnh(k),nh_sum,nh_impl,vh_sedn_new,vh_sedn_now,nh_flux_new,nh_flux_now,rdzdt(k))

        if (lprogmelt) then
          call implicit_core(qgl(k),lg_sum,lg_impl,vg_sedl_new,vg_sedl_now,lg_flux_new,lg_flux_now,rdzdt(k))
          call implicit_core(qhl(k),lh_sum,lh_impl,vh_sedl_new,vh_sedl_now,lh_flux_new,lh_flux_now,rdzdt(k))
        end if

        ! do microphysics on this k-level only (using the star-values)
        IF (lmicro_impl) THEN

          ! .. save old variables for latent heat calculation
          q_vap_old(k) = qv(k)
          if (lprogmelt) then
            q_liq_old(k) = qr(k) + qc(k) + qgl(k) + qhl(k)
          else
            q_liq_old(k) = qc(k) + qr(k)
          end if

          CALL clouds_twomoment(k, k, dt, lprogin, &
               atmo, cloud, rain, ice, snow, graupel, hail, ssat, lexpl_supersat_loc, &
               ninact, nccn, ninpot, ninagi, luse_agi_loc, iagi_param)

          ! .. latent heat term for temperature equation
          IF (lconstant_lh) THEN
            led = als
            lwe = (alv-als)
          ELSE
            led = latent_heat_sublimation(tk(k))
            lwe = latent_heat_melting(tk(k))
          END IF

          convice = z_heat_cap_r * led
          convliq = z_heat_cap_r * lwe

          q_vap_new  = qv(k)
          if (lprogmelt) then
            q_liq_new = qr(k) + qc(k) + qgl(k) + qhl(k)
          else
            q_liq_new = qr(k) + qc(k)
          end if
          tk(k)   = tk(k) - convice * rho_r(k) * (q_vap_new - q_vap_old(k))  &
               &          + convliq * rho_r(k) * (q_liq_new - q_liq_old(k))

        END IF

        call implicit_time(qr(k), qr_sum,qr_impl,vr_sedq_new,vr_sedq_now,qr_flux_new)
        call implicit_time(qnr(k),nr_sum,nr_impl,vr_sedn_new,vr_sedn_now,nr_flux_new)
        call implicit_time(qi(k), qi_sum,qi_impl,vi_sedq_new,vi_sedq_now,qi_flux_new)
        call implicit_time(qni(k),ni_sum,ni_impl,vi_sedn_new,vi_sedn_now,ni_flux_new)
        call implicit_time(qs(k), qs_sum,qs_impl,vs_sedq_new,vs_sedq_now,qs_flux_new)
        call implicit_time(qns(k),ns_sum,ns_impl,vs_sedn_new,vs_sedn_now,ns_flux_new)
        call implicit_time(qg(k), qg_sum,qg_impl,vg_sedq_new,vg_sedq_now,qg_flux_new)
        call implicit_time(qng(k),ng_sum,ng_impl,vg_sedn_new,vg_sedn_now,ng_flux_new)
        call implicit_time(qh(k), qh_sum,qh_impl,vh_sedq_new,vh_sedq_now,qh_flux_new)
        call implicit_time(qnh(k),nh_sum,nh_impl,vh_sedn_new,vh_sedn_now,nh_flux_new)

        if (lprogmelt) then
          call implicit_time(qgl(k),lg_sum,lg_impl,vg_sedl_new,vg_sedl_now,lg_flux_new)
          call implicit_time(qhl(k),lh_sum,lh_impl,vh_sedl_new,vh_sedl_now,lh_flux_new)
        end if

        IF (ldass_lhn) THEN
          IF (lprogmelt) THEN
            qrsflux(k) = qr_flux_new + qi_flux_new + qs_flux_new + qg_flux_new + qh_flux_new + &
                         lg_flux_new + lh_flux_new
          ELSE
            qrsflux(k) = qr_flux_new + qi_flux_new + qs_flux_new + qg_flux_new + qh_flux_new
          END IF
        END IF

      END DO

      IF (lprogmelt) THEN
        ! implicit solver for LWF-scheme still has some issues
        prec_g = MAX( qg_flux_new + lg_flux_new, 0.0_wp )
        prec_h = MAX( qh_flux_new + lh_flux_new, 0.0_wp )
        prec_r = qr_flux_new
        prec_i = qi_flux_new
        prec_s = qs_flux_new
      ELSE
        prec_r = qr_flux_new
        prec_i = qi_flux_new
        prec_s = qs_flux_new
        prec_g = qg_flux_new
        prec_h = qh_flux_new
      END IF

      IF (ldass_lhn) THEN
        qrsflux(kte) = qr_flux_new + qi_flux_new + qs_flux_new + qg_flux_new + qh_flux_new
      END IF

    END SUBROUTINE clouds_twomoment_implicit
   
   !
   ! sedimentation for explicit solver, i.e., sedimentation is done with an explicit
   ! flux-form semi-lagrangian scheme after the microphysics.
   !
   SUBROUTINE sedimentation_explicit()
    ! D.Rieger: the parameter lfullyexplicit needs to be set false, otherwise the nproma/mpi tests of buildbot are not passed
    LOGICAL, PARAMETER :: lfullyexplicit = .FALSE.
    REAL(wp) :: cmax, rdzmaxdt
    REAL(wp) :: prec3D_tmp(ke)
    INTEGER :: ii, kk

    cmax = 0.0_wp
    ! Use for sub-stepping of hydrometeors, lfullyexplicit needs to be set to TRUE
    IF (lfullyexplicit) THEN
      rdzmaxdt = maxval(rdz(kts:kte)) * dt
      ntsedi_rain = ceiling(rain%vsedi_max*rdzmaxdt)
      ntsedi_graupel = ceiling(graupel%vsedi_max*rdzmaxdt)
      ntsedi_hail = ceiling(hail%vsedi_max*rdzmaxdt)
    ELSE
      ntsedi_rain = 1
      ntsedi_graupel = 1
      ntsedi_hail = 1
    ENDIF

     prec_r = 0.0_wp
     prec_i = 0.0_wp
     prec_s = 0.0_wp
     prec_g = 0.0_wp
     prec_h = 0.0_wp

     ! Skip a species entirely if it's absent from the whole column (cheap short-circuit,
     ! harmless for a single column; kept from the original CPU optimization).
     IF (ANY(qr(kts:kte)>0._wp)) THEN
      IF (ldass_lhn) THEN
        DO kk = kts,kte
          prec3D_tmp(kk) = 0.0_wp
        ENDDO
      ENDIF
      DO ii=1,ntsedi_rain
        CALL sedi_icon_rain(rain,rain_coeffs,qr,qnr,prec_r,prec3D_tmp,qc,rhocorr, &
          & rdz,dt/ntsedi_rain,kts,kte,cmax)
      END DO
      IF (ldass_lhn) THEN
        DO kk = kts,kte
          qrsflux(kk) = qrsflux(kk) + prec3D_tmp(kk)
        ENDDO
      ENDIF
     END IF

     IF (ANY(qi(kts:kte)>0._wp)) THEN
      IF (ldass_lhn) THEN
        DO kk = kts,kte
          prec3D_tmp(kk) = 0.0_wp
        ENDDO
      ENDIF
      CALL sedi_icon_sphere(ice,ice_coeffs,qi,qni,prec_i,prec3D_tmp,rhocorr,rdz,dt,kts,kte)
      IF (ldass_lhn) THEN
        DO kk = kts,kte
          qrsflux(kk) = qrsflux(kk) + prec3D_tmp(kk)
        ENDDO
      ENDIF
     END IF

     IF (ANY(qs(kts:kte)>0._wp)) THEN
      IF (ldass_lhn) THEN
        DO kk = kts,kte
          prec3D_tmp(kk) = 0.0_wp
        ENDDO
      ENDIF
      CALL sedi_icon_sphere(snow,snow_coeffs,qs,qns,prec_s,prec3D_tmp,rhocorr,rdz,dt,kts,kte)
      IF (ldass_lhn) THEN
        DO kk = kts,kte
          qrsflux(kk) = qrsflux(kk) + prec3D_tmp(kk)
        ENDDO
      ENDIF
     END IF

     IF (ANY(qg(kts:kte)>0._wp)) THEN
      IF (ldass_lhn) THEN
        DO kk = kts,kte
          prec3D_tmp(kk) = 0.0_wp
        ENDDO
      ENDIF
       IF (lprogmelt) THEN
         DO ii=1,ntsedi_graupel
           call sedi_icon_sphere_lwf(graupel_lwf,graupel_coeffs,qg,qng,qgl,&
                &                    prec_g,prec3D_tmp,rhocorr,rdz,dt/ntsedi_graupel,kts,kte,cmax)
         END DO
       ELSE
         DO ii=1,ntsedi_graupel
           CALL sedi_icon_sphere(graupel,graupel_coeffs,qg,qng,prec_g,prec3D_tmp,rhocorr,rdz,dt/ntsedi_graupel, &
             & kts,kte,cmax)
         END DO
       END IF
      IF (ldass_lhn) THEN
        DO kk = kts,kte
          qrsflux(kk) = qrsflux(kk) + prec3D_tmp(kk)
        ENDDO
      ENDIF
     END IF

     IF (ANY(qh(kts:kte)>0._wp)) THEN
      IF (ldass_lhn) THEN
        DO kk = kts,kte
          prec3D_tmp(kk) = 0.0_wp
        ENDDO
      ENDIF
       IF (lprogmelt) THEN
         DO ii=1,ntsedi_hail
           call sedi_icon_sphere_lwf(hail_lwf,hail_coeffs,qh,qnh,qhl,&
                &                    prec_h,prec3D_tmp,rhocorr,rdz,dt/ntsedi_hail,kts,kte,cmax)
         END DO
       ELSE
         DO ii=1,ntsedi_hail
           call sedi_icon_sphere(hail,hail_coeffs,qh,qnh,prec_h,prec3D_tmp,rhocorr,rdz,dt/ntsedi_hail, &
             & kts,kte,cmax)
         END DO
       END IF
      IF (ldass_lhn) THEN
        DO kk = kts,kte
          qrsflux(kk) = qrsflux(kk) + prec3D_tmp(kk)
        ENDDO
      ENDIF
     END IF

     IF (msg_level > 100)THEN
       WRITE (message_text,'(1X,A,f8.2)') ' sedimentation_explicit  cmax = ',cmax
       CALL message(routine, message_text)
     END IF

   END SUBROUTINE sedimentation_explicit

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

    IF (timers_level > 10) CALL timer_start(timer_phys_2mom_dmin_init)
    IF (luse_dmin_wetgrowth_table .OR. lprintout_comp_table_fit) THEN
      unitnr = 11
      IF (msg_level>5) CALL message (TRIM(routine), " Looking for dmin_wetgrowth table file for "//TRIM(graupel%name))
      CALL init_dmin_wg_gr_ltab_equi('dmin_wetgrowth_lookup', graupel, &
           unitnr, 61, ltabdminwgg, msg_level)
      IF (msg_level>5) CALL message (TRIM(routine), " Looking for dmin_wetgrowth table file for "//TRIM(hail%name))
      CALL init_dmin_wg_gr_ltab_equi('dmin_wetgrowth_lookup', hail, &
           unitnr, 61, ltabdminwgh, msg_level)
    END IF
    IF (.NOT. luse_dmin_wetgrowth_table) THEN
      ! check whether 4d-fit is consistent with graupel parameters
      IF (dmin_wetgrowth_fit_check(graupel)) THEN 
        CALL message (TRIM(routine), " Using 4d-fit for dmin_wetgrowth for "//TRIM(graupel%name))
      ELSE
        CALL finish(TRIM(routine),&
             & 'Error: luse_dmin_wetgrowth_table=.false., so 4D-fit should be used, '// &
             & 'but graupel parameters inconsistent with 4d-fit')
      END IF
    END IF
    
    IF (msg_level>dbg_level) CALL message (TRIM(routine), ' finished init_dmin_wetgrowth for '// &
         TRIM(graupel%name)//' and '//TRIM(hail%name))
    !..parameters for CCN and IN are set here. The 3D fields for prognostic CCN are then
    !  initialized in mo_nwp_phy_init.
    IF (timers_level > 10) CALL timer_stop(timer_phys_2mom_dmin_init)

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
