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

MODULE mo_nonhydrostatic_config

  USE mo_kind,                    ONLY: wp
  USE mo_impl_constants,          ONLY: max_dom, MAX_NTRACER
  USE mo_exception,               ONLY: message, message_text
  USE mo_vertical_coord_table,    ONLY: vct_a
  USE mo_run_config,              ONLY: msg_level
  USE mo_name_list_output_config, ONLY: is_variable_in_output

  IMPLICIT NONE

  PUBLIC

  !>
  !!----------------------------------------------------------------------------
  !! Derived type containing control variables specific to the nonhydrostatic 
  !! atm model
  !!----------------------------------------------------------------------------
!  TYPE :: t_nonhydrostatic_config

    INTEGER :: itime_scheme             !< Choice of time stepping scheme

    INTEGER :: ndyn_substeps            !< number of dynamics substeps per fast-physics step
    REAL(wp):: vcfl_threshold           ! threshold for vertical advection CFL number at which the adaptive time step reduction
                                        ! (increase of ndyn_substeps w.r.t. the fixed fast-physics time step) is triggered

    ! related runtime control variables for adaptive ndyn_substeps:
    INTEGER :: ndyn_substeps_max        ! maximum number of dynamics substeps per fast-physics step
    INTEGER :: ndyn_substeps_var(max_dom)! current (variable) number of dynamics substeps per fast-physics step
    INTEGER :: nlev_hcfl(max_dom)       ! number of model levels (counted from top) for which the horizontal CFL number is monitored in addition
    INTEGER :: cfl_monitoring_freq      ! monitoring frequency for CFL number (in units of fast-physics time steps of domain 1)

    LOGICAL :: lextra_diffu             ! if true: apply additional diffusion at grid points close
                                        ! to the CFL stability limit for vertical advection

    REAL(wp):: divdamp_fac              ! Scaling factor for divergence damping at height divdamp_z and below
    REAL(wp):: divdamp_fac2             ! Scaling factor for divergence damping at height divdamp_z2
    REAL(wp):: divdamp_fac3             ! Scaling factor for divergence damping at height divdamp_z3
    REAL(wp):: divdamp_fac4             ! Scaling factor for divergence damping at height divdamp_z4 and above
    REAL(wp):: divdamp_z                ! Height up to which divdamp_fac is used, start of linear profile
    REAL(wp):: divdamp_z2               ! Height of divdamp_fac2, end of linear and start of quadratic profile
    REAL(wp):: divdamp_z3               ! Height of divdamp_fac3, to define quadratic profile
    REAL(wp):: divdamp_z4               ! Height from which divdamp_fac4, end of quadratic profile

    REAL(wp):: divdamp_fac_o2           ! Scaling factor for second-order divergence damping
                                        ! (derived variable; used if divdamp_order = 2 or 24)
    INTEGER :: divdamp_order            ! Order of divergence damping
    INTEGER :: divdamp_type             ! Type of divergence damping (2D or 3D divergence)
    REAL(wp):: divdamp_trans_start      ! Lower bound of transition zone between 2D and 3D div damping in case of divdamp_type = 32
    REAL(wp):: divdamp_trans_end        ! Upper bound of transition zone between 2D and 3D div damping in case of divdamp_type = 32
    INTEGER :: ivctype                  ! Type of vertical coordinate (Gal-Chen / SLEVE)
    REAL(wp):: htop_moist_proc          ! Top height (in m) of the part of the model domain
                                        ! where processes related to moist physics are computed
    REAL(wp):: hbot_qvsubstep           ! Bottom height (in m) down to which water vapor is 
                                        ! advected with internal substepping (to circumvent CFL 
                                        ! instability in the stratopause region).
    REAL(wp):: htop_aero_proc           ! Top height (in m) of the model domain where (ART) tracers
                                        ! are being transported/diffused/modified
    INTEGER :: ih_clch(max_dom)         ! end index for levels contributing to high-level clouds, clch
    INTEGER :: ih_clcm(max_dom)         ! end index for levels contributing to mid-level clouds, clcm



    INTEGER :: rayleigh_type    ! type of Rayleigh damping (1: CLASSIC, 2: Klemp (2008))
    REAL(wp):: damp_height(max_dom)    ! height at which w-damping and sponge layer start
    REAL(wp):: rayleigh_coeff(max_dom) ! Rayleigh damping coefficient in w-equation
    REAL(wp):: vwind_offctr     ! Off-centering in vertical wind solver
    REAL(wp):: rhotheta_offctr  ! Off-centering for density and potential temperature at interface levels
    REAL(wp):: veladv_offctr    ! Off-centering for velocity advection
    INTEGER :: iadv_rhotheta    ! Advection scheme used for density and pot. temperature
    INTEGER :: igradp_method    ! Method for computing the horizontal presure gradient
    REAL(wp):: exner_expol      ! Temporal extrapolation of Exner for computation of
                                ! horizontal pressure gradient
    LOGICAL :: l_zdiffu_t       ! .true.: apply truly horizontal temperature diffusion 
                                ! over steep slopes
    REAL(wp):: thslp_zdiffu     ! threshold slope above which temperature diffusion is applied
    REAL(wp):: thhgtd_zdiffu    ! threshold height difference between adjacent model grid points
                                ! above which temperature diffusion is applied

    ! derived variables
    !
    INTEGER :: kstart_dd3d(max_dom)     ! start level for 3D divergence damping terms
    INTEGER :: kstart_moist(max_dom)    ! related flow control variable
    INTEGER :: kstart_tracer(max_dom,MAX_NTRACER) ! start level for (ART) tracers
    INTEGER :: kend_qvsubstep(max_dom)  ! related flow control variable
    LOGICAL :: lcalc_dpsdt              !< TRUE: compute dpsdt for output even if a
                                        !  low message level (<= 10) is selected

!  END TYPE t_nonhydrostatic_config 
  !>
  !!
!  TYPE(t_nonhydrostatic_config) :: nonhydrostatic_config(max_dom) ! config state 


CONTAINS

  !! Setup of additional nonhydrostatic control variables depending on the 
  !! nonhydrostatic-NAMELIST and potentially other namelists. This routine is 
  !! called, after all namelists have been read and a synoptic consistency 
  !! check has been done.
  !!
  SUBROUTINE configure_nonhydrostatic(jg, nlev, nshift_total)
  !
    INTEGER,  INTENT(IN) :: jg           !< patch 
    INTEGER,  INTENT(IN) :: nlev         !< number of full vertical levels 
    INTEGER,  INTENT(IN) :: nshift_total 

    INTEGER :: jk, jk1

    REAL(wp), PARAMETER :: hbase_clch = 7185.44_wp  ! height in m of 400 hPa level in US standard atmosphere
    REAL(wp), PARAMETER :: hbase_clcm = 1948.99_wp  ! height in m of 800 hPa level in US standard atmosphere


    CHARACTER(len=*), PARAMETER ::  &
      &  routine = 'mo_nonhydrostatic_config:configure_nonhydrostatic'

    !-----------------------------------------------------------------------

    ! Determine start level for moist physics processes (specified by htop_moist_proc)
    kstart_moist(jg) = 1
    DO jk = 1, nlev
      jk1 = jk + nshift_total
      IF (0.5_wp*(vct_a(jk1)+vct_a(jk1+1)) < htop_moist_proc) THEN
        kstart_moist(jg) = jk
        EXIT
      ENDIF
    ENDDO

    IF ( kstart_moist(jg) >= 1 ) THEN
      WRITE(message_text,'(2(a,i4))') 'Domain', jg, &
        '; computation of moist physics processes starts in layer ', kstart_moist(jg)
      CALL message(routine, message_text)
    ENDIF


    ! Determine end level for qv-advection substepping (specified by hbot_qvsubstep)
    kend_qvsubstep(jg) = 0
    DO jk = nlev, 1, -1
      jk1 = jk + nshift_total
      IF (0.5_wp*(vct_a(jk1)+vct_a(jk1+1)) > hbot_qvsubstep) THEN
        kend_qvsubstep(jg) = jk
        EXIT
      ENDIF
    ENDDO


    ! Initialize kstart_tracer; will be overwritten for ART aerosol tracers
    ! (specified by htop_aero_proc)
    ! Is set in the module mo_art_tracer of the ART external.
    kstart_tracer(jg,:) = 1


    ! height indices for cloud classification
    !
    ih_clch(jg) = kstart_moist(jg)
    DO jk = 1, nlev
      jk1 = jk + nshift_total
      IF (0.5_wp*(vct_a(jk1)+vct_a(jk1+1)) < hbase_clch) THEN
        ih_clch(jg) = jk
        EXIT
      ENDIF
    ENDDO
    ih_clcm(jg) = ih_clch(jg) + 1
    DO jk = ih_clch(jg) + 1, nlev
      jk1 = jk + nshift_total
      IF (0.5_wp*(vct_a(jk1)+vct_a(jk1+1)) < hbase_clcm) THEN
        ih_clcm(jg) = jk
        EXIT
      ENDIF
    ENDDO
    WRITE(message_text,'(2(a,i4),i4)') 'Domain', jg, &
      '; high- and mid-level clouds in layers above ', ih_clch(jg), ih_clcm(jg)
    CALL message(routine, message_text)

    ! initialization of control variables derived from ndyn_substeps
    ndyn_substeps_max    = ndyn_substeps + 7
    ndyn_substeps_var(:) = ndyn_substeps !** needs to be saved in restart attributes **


    ! check whether dpsdt should be computed for output purposes
    lcalc_dpsdt = (is_variable_in_output(var_name='ddt_pres_sfc')) &
      &           .OR. (msg_level >= 11)

  END SUBROUTINE configure_nonhydrostatic


END MODULE mo_nonhydrostatic_config
