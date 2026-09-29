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

! Setup of config-state for NWP physics package

MODULE mo_atm_phy_nwp_config

  USE mo_kind,                ONLY: wp, i8
  USE mo_grid_config,         ONLY: l_limited_area, start_time, end_time,      &
    &                               DEFAULT_ENDTIME
  USE mo_run_config,          ONLY: msg_level, timers_level
  USE mo_parallel_config,     ONLY: nproma
  USE mo_io_units,            ONLY: filename_max
  USE mo_impl_constants,      ONLY: max_dom, itconv, itccov,  &
    &                               itrad, itradheat, itsso, itgscp, itsatad,  &
    &                               itturb, itsfc, itgwd, itfastphy,           &
    &                               iphysproc, iphysproc_short, ismag,         &
    &                               iprog, SUCCESS, ivdiff
  USE mo_math_constants,      ONLY: dbl_eps, pi_2, deg2rad
  USE mo_exception,           ONLY: message, message_text, finish
  USE mo_model_domain,        ONLY: t_patch
  USE mo_vertical_coord_table,ONLY: vct_a
  USE mo_time_config,         ONLY: t_time_config
  USE mo_radiation_config,    ONLY: irad_o3
#ifndef __NO_ICON_LES__
  USE mo_les_config,          ONLY: configure_les, les_config
#endif
  USE mo_limarea_config,      ONLY: configure_latbc
  USE mo_initicon_config,     ONLY: timeshift
  USE mo_nwp_tuning_config,   ONLY: itune_o3
  USE mtime,                  ONLY: datetime, timedelta, newTimedelta, event, newEvent, no_Error,     &
    &                               getPTStringFromMS, MAX_TIMEDELTA_STR_LEN, datetimeToString,       &
    &                               deallocateTimedelta, OPERATOR(+), OPERATOR(>), timedeltaToString, &
    &                               MAX_DATETIME_STR_LEN, MAX_MTIME_ERROR_STR_LEN, mtime_strerror
  USE mo_util_table,          ONLY: t_table, initialize_table, add_table_column, &
    &                               set_table_entry, print_table, finalize_table
  USE mo_mpi,                 ONLY: my_process_is_stdio
  USE mo_phy_events,          ONLY: t_phyProcFast, t_phyProcSlow, t_phyProcGroup
  USE mo_nudging_config,      ONLY: configure_nudging, nudging_config
  USE mo_name_list_output_config, ONLY: is_variable_in_output
  USE mo_io_config,           ONLY: dt_lpi, dt_celltracks, dt_radar_dbz, dt_hailcast
  USE mo_2mom_mcrph_config,   ONLY: t_cfg_2mom

  IMPLICIT NONE

  PRIVATE

  CHARACTER(LEN = *), PARAMETER :: modname = 'mo_atm_phy_nwp_config'

  ! TYPES
  PUBLIC :: t_atm_phy_nwp_config

  ! SUBROUTINES
  PUBLIC :: configure_atm_phy_nwp

  ! VARS
  PUBLIC :: atm_phy_nwp_config, dt_phy
  PUBLIC :: lrtm_filename
  PUBLIC :: cldopt_filename
  PUBLIC :: ltuning_kessler
  PUBLIC :: icpl_aero_conv
  PUBLIC :: icpl_o3_tp
  PUBLIC :: iprog_aero
  PUBLIC :: setup_nwp_diag_events
  PUBLIC :: icpl_aero_ice
  PUBLIC :: lcuda_graph_turb_tran

  !!--------------------------------------------------------------------------
  !! Basic configuration setup for nwp physics
  !!--------------------------------------------------------------------------
  TYPE :: t_atm_phy_nwp_config

    ! namelist variables

    INTEGER ::  inwp_gscp        !> microphysics
    TYPE(t_cfg_2mom) :: cfg_2mom !> config parameters of 2-mom cloud microphysics (inwp_gscp = 4...7)
    INTEGER ::  inwp_satad       !! saturation adjustment
    LOGICAL ::  lexpl_supersat   !! explicit supersaturation and cloud droplet condensational mass growth/evaporation
    INTEGER ::  inwp_convection  !! convection
    LOGICAL ::  lshallowconv_only !! use shallow convection only
    LOGICAL  :: lstoch_expl        !! use explicit stochastic shallow convection
    LOGICAL  :: lstoch_sde         !! use stochastic differential equations for shallow convection
    LOGICAL  :: lstoch_deep        !! use stochastic deep convection (SDE scheme)
    LOGICAL  :: lvvcouple          !! use vertical velocity at 650hPa as criterion to couple shallow convection
                                   !! shallow convection with resolved deep convection 
    LOGICAL  :: lvv_shallow_deep   !! use vertical velocity at 650hPa to distinguish between shallow and 
                                   !! deep convection within convection routines (instead of cloud depth)
    LOGICAL ::  lstoch_spinup      !! spinup stochastic cloud ensemble into equilibrium at first time step
    LOGICAL ::  lrestune_off       !! switch off all resolution-dependent tuning in convection setup
    LOGICAL ::  lmflimiter_off     !! switch off mass flux limiters in convection
    INTEGER ::  nclds              !! max number of clouds in stochastic cloud ensemble
    LOGICAL ::  lgrayzone_deepconv !! use grayzone tuning for deep convection
    LOGICAL ::  ldetrain_conv_prec !! detrain convective rain and snow
    LOGICAL ::  lsgs_cond          !! subgrid-scale condensation related to cloud cover
    INTEGER ::  inwp_radiation   !! radiation
    INTEGER ::  inwp_sso         !! sso
    INTEGER ::  inwp_gwd         !! non-orographic gravity wave drag
    INTEGER ::  inwp_cldcover    !! cloud cover
    INTEGER ::  inwp_turb        !! turbulence
    INTEGER ::  inwp_surface     !! surface including soil, ocean, ice,lake
    INTEGER  :: itype_z0         !! type of roughness length data
    REAL(wp) :: dt_conv          !> time step for convection
    REAL(wp) :: dt_ccov          !! time step for subscale cloud cover
    REAL(wp) :: dt_rad           !! "-"                     radiation
    REAL(wp) :: dt_sso           !! "-"  for subscale orographic gravity waves
    REAL(wp) :: dt_gwd           !! "-"  for subscale gravity waves
    REAL(wp) :: dt_fastphy       !! time step for fast physics processes
                                 !! microphysics, saturation adjustment, turbulence, 
                                 !! surface (in addition: update and radheat)
    ! hydci_pp                   
    REAL(wp) :: mu_rain          !! parameter in gamma distribution for rain
    REAL(wp) :: mu_snow          !! ...for snow
    REAL(wp) :: rain_n0_factor   !! tuning factor for intercept parameter of raindrop size distribution
    LOGICAL ::  lsbm_warm_full    !! false: Piggy Backing with 2M, true: full warm-phase SBM
    REAL(wp) :: qi0, qc0

    INTEGER  :: icpl_aero_gscp     !! type of aerosol-microphysics coupling
    LOGICAL  :: lscale_cdnc        !! switch to activate the scaling of MODIS CDNCs

    REAL(wp) :: ustart_raylfric    !! velocity at which extra Rayleigh friction starts
    REAL(wp) :: efdt_min_raylfric  !! e-folding time corresponding to maximum relaxation 
                                   !! coefficient
    LOGICAL  :: latm_above_top     !! use extra layer above model top for radiation 
                                   !! (reduced grid only)
    INTEGER  :: icalc_reff         !! type of effective radius calculation
    INTEGER  :: icpl_rad_reff      !! couplig of radiation and effective radius
    INTEGER  :: ithermo_water      !! thermodynamic of water
    INTEGER  :: i2mom_icenucleation
    LOGICAL  :: luse_agi
    INTEGER  :: iagi_param
    CHARACTER(LEN=max_datetime_str_len)  :: sd_agi
    CHARACTER(LEN=max_datetime_str_len)  :: ed_agi
    ! upper atmosphere
    LOGICAL ::  lupatmo_phy        !! use upper atmosphere physics

    ! Derived variables

    LOGICAL :: lenabled(iphysproc) !> contains information about status of 
                                   !! corresponding physical process
                                   !! enabled: TRUE; disabled: FALSE

    LOGICAL, ALLOCATABLE :: &      !> ith physics package must be called at the current time step
      &  lcall_phy(:)              !! TRUE/FALSE, time dependent

    LOGICAL :: lcalc_acc_avg       ! TRUE: calculate accumulated and averaged quantities

    LOGICAL :: lcalc_extra_avg     ! TRUE: calculate aditional temporally averaged fields, which normally 
                                   !       are not computed in operational runs.
                                   !       lcalc_extra_avg is set to true automatically, if any of the 
                                   !       non-standard fields is specified in the output namelist.

    LOGICAL :: lhave_graupel       ! Flag if microphysics scheme has a prognostic variable for graupel
    LOGICAL :: l2moment            ! Flag if 2-moment microphysics scheme is used 
    LOGICAL :: lsbm                ! Flag if sbm microphysics scheme is used    
    LOGICAL :: lhydrom_read_from_fg(1:20)  ! Flag for each hydrometeor tracer, if it has been read from fg file
    LOGICAL :: lhydrom_read_from_ana(1:20) ! Flag for each hydrometeor tracer, if it has been read from ana file

    LOGICAL :: luse_clc_rad

#ifndef __NO_ICON_LES__
    LOGICAL :: is_les_phy          !>TRUE is turbulence is 3D 
                                   !>FALSE otherwise
#endif
    INTEGER :: nclass_gscp         !> number of hydrometeor classes for 
                                   ! chosen grid scale microphysics

    LOGICAL :: l_3d_rad_fluxes     ! logical to determine if 3d radiative flux variable are allocated

    LOGICAL :: l_3d_turb_fluxes    ! logical to determine if 3d turbulent flux variable are allocated


    ! NWP events
    TYPE(t_phyProcGroup) :: phyProcs        !> physical processes event group
    TYPE(t_phyProcFast)  :: phyProc_satad   !> saturation adjustment
    TYPE(t_phyProcFast)  :: phyProc_turb    !> turbulence
    TYPE(t_phyProcFast)  :: phyProc_gscp    !> grid-scale microphysics
    TYPE(t_phyProcFast)  :: phyProc_sfc     !> land/surface
    TYPE(t_phyProcFast)  :: phyProc_radheat !> radiative heating

    TYPE(t_phyProcSlow)  :: phyProc_conv    !> convection (deep+shallow)
    TYPE(t_phyProcSlow)  :: phyProc_ccov    !> cloud cover
    TYPE(t_phyProcSlow)  :: phyProc_rad     !> radiation (SW+LW)
    TYPE(t_phyProcSlow)  :: phyProc_sso     !> sub-gridscale orographic wave drag
    TYPE(t_phyProcSlow)  :: phyProc_gwd     !> non-orographic wave drag


    ! Tuning variables

    REAL(wp), ALLOCATABLE :: fac_ozone(:)         ! vertical profile funtion for ozone tuning
    REAL(wp), ALLOCATABLE :: shapefunc_ozone(:,:) ! horizontal profile funtion for ozone tuning
    REAL(wp)              :: ozone_maxinc         ! maximum allowed change of ozone mixing ratio

  CONTAINS
    !
    ! finalization routine
    PROCEDURE  :: finalize => atm_phy_nwp_config_finalize
  END TYPE t_atm_phy_nwp_config

  !>
  !!
  TYPE(t_atm_phy_nwp_config), TARGET :: atm_phy_nwp_config(max_dom) !< shape: (n_dom)


  !> NetCDF file containing longwave absorption coefficients and other data
  !> for RRTMG_LW k-distribution model ('rrtmg_lw.nc')
  CHARACTER(LEN=filename_max) :: lrtm_filename

  !> NetCDF file with RRTM Cloud Optical Properties for ECHAM6
  CHARACTER(LEN=filename_max) :: cldopt_filename

  INTEGER  :: icpl_aero_conv     !! type of coupling between aerosols and convection scheme
  INTEGER  :: iprog_aero         !! type of prognostic aerosol
  INTEGER  :: icpl_o3_tp         !! type of coupling between ozone and the tropopause
  INTEGER  :: icpl_aero_ice      !! type of coupling between aersols and ice nucleation

  REAL(wp) ::  &                       !> Field of calling-time interval (seconds) for
    &  dt_phy(max_dom,iphysproc_short) !! each domain and phys. process

  ! Optimization
  LOGICAL :: lcuda_graph_turb_tran   !! activate CUDA GRAPH in turbulent transfer
  
  !!--------------------------------------------------------------------------
  !! Tuning parameters for physics
  !!--------------------------------------------------------------------------
  
  ! convection:
  ! GZ, 2013-09-13: tuning to reduce drizzle (may be overridden by icpl_aero_conv=1)
  LOGICAL,  PARAMETER :: ltuning_kessler  = .TRUE.

  ! profile parameters for ozone tuning:
  REAL(wp) :: tune_ozone_ztop
  REAL(wp) :: tune_ozone_zmid, tune_ozone_zmid2
  REAL(wp) :: tune_ozone_zbot
  REAL(wp) :: tune_ozone_fac
  REAL(wp) :: tune_ozone_lat
  REAL(wp) :: tune_ozone_maxinc
  INTEGER  :: ozone_shapemode

CONTAINS

  !-------------------------------------------------------------------------
  !>
  !! Setup NWP physics
  !!
  !! Read namelist for NWP physics and setup physics time control.
  !!
  SUBROUTINE configure_atm_phy_nwp( n_dom, p_patch, time_config )

    TYPE(t_patch),       INTENT(IN) :: p_patch(:)
    INTEGER,             INTENT(IN) :: n_dom
    TYPE(t_time_config), INTENT(IN) :: time_config       !< time and date information


    ! local
    INTEGER :: jg, jk, jk_shift, jb, jc
    INTEGER :: error
    CHARACTER(len=*), PARAMETER ::  &
      &      routine = modname//":configure_atm_phy_nwp"
    REAL(wp) :: z_mc_ref
    REAL(wp) :: &                             ! time-intervals for calling various 
      &  dt_phy_orig(max_dom,iphysproc_short) ! physical processes. Original values as 
                                              ! provided by user

    CHARACTER(LEN=MAX_TIMEDELTA_STR_LEN) :: td_start_str
    CHARACTER(LEN=MAX_TIMEDELTA_STR_LEN) :: td_end_str
    CHARACTER(LEN=MAX_TIMEDELTA_STR_LEN) :: td_dt_str
    TYPE(datetime)           :: domStartDate, domEndDate
    TYPE(datetime)           :: eventStartDate, eventEndDate
    TYPE(timedelta), POINTER :: td_start, td_end, td_dt   => NULL()

  !-------------------------------------------------------------------------


    !$ACC ENTER DATA CREATE(atm_phy_nwp_config)

    ! for each fast physics process the time interval is set
    ! equal to the time interval for advection.
    DO jg = 1,n_dom
      atm_phy_nwp_config(jg)%dt_fastphy = time_config%get_model_timestep_sec(p_patch(jg)%nest_level)
    ENDDO


    ! store original (user defined) dt_phy values for LOG-output (see below)
    DO jg = 1,n_dom
      dt_phy_orig(jg,:)        = 0._wp  ! init
      dt_phy_orig(jg,itconv)   = atm_phy_nwp_config(jg)% dt_conv
      dt_phy_orig(jg,itsso)    = atm_phy_nwp_config(jg)% dt_sso
      dt_phy_orig(jg,itgwd)    = atm_phy_nwp_config(jg)% dt_gwd
      dt_phy_orig(jg,itrad)    = atm_phy_nwp_config(jg)% dt_rad
      dt_phy_orig(jg,itccov)   = atm_phy_nwp_config(jg)% dt_ccov
      dt_phy_orig(jg,itfastphy)= atm_phy_nwp_config(jg)% dt_fastphy
    ENDDO


    DO jg = 1,n_dom

      ! initialize lenabled
      atm_phy_nwp_config(jg)%lenabled(1:iphysproc)    = .FALSE.


      ! Fill derived variable lenabled
 
      IF ( atm_phy_nwp_config(jg)%inwp_satad > 0 )               &
        &  atm_phy_nwp_config(jg)%lenabled(itsatad)   = .TRUE. 

      IF ( atm_phy_nwp_config(jg)%inwp_convection > 0 )          &
        &  atm_phy_nwp_config(jg)%lenabled(itconv)    = .TRUE. 

      IF ( atm_phy_nwp_config(jg)%inwp_cldcover > 0 )            &
        &  atm_phy_nwp_config(jg)%lenabled(itccov)    = .TRUE.

      IF ( atm_phy_nwp_config(jg)%inwp_radiation > 0 )           &
        &  atm_phy_nwp_config(jg)%lenabled(itrad)     = .TRUE.

      IF ( atm_phy_nwp_config(jg)%inwp_sso > 0 )                 &
        &  atm_phy_nwp_config(jg)%lenabled(itsso)     = .TRUE.

      IF ( atm_phy_nwp_config(jg)%inwp_gscp > 0 )                &
        &  atm_phy_nwp_config(jg)%lenabled(itgscp)    = .TRUE.

      IF ( atm_phy_nwp_config(jg)%inwp_turb > 0 )                &
        &  atm_phy_nwp_config(jg)%lenabled(itturb)    = .TRUE.

      IF ( atm_phy_nwp_config(jg)%inwp_radiation > 0 )           &
        &  atm_phy_nwp_config(jg)%lenabled(itradheat) = .TRUE.

      IF ( atm_phy_nwp_config(jg)%inwp_surface > 0 )             &
        &  atm_phy_nwp_config(jg)%lenabled(itsfc)     = .TRUE.

      IF ( atm_phy_nwp_config(jg)%inwp_gwd > 0 )                 &
        &  atm_phy_nwp_config(jg)%lenabled(itgwd)     = .TRUE.


      ! Set flags for the microphysics schemes:
      atm_phy_nwp_config(jg)%lsbm = .FALSE.
      SELECT CASE (atm_phy_nwp_config(jg)%inwp_gscp)
      CASE (2)
        atm_phy_nwp_config(jg)%lhave_graupel = .TRUE.
        atm_phy_nwp_config(jg)%l2moment = .FALSE.
      CASE (4,5,6,7)
        atm_phy_nwp_config(jg)%lhave_graupel = .TRUE.
        atm_phy_nwp_config(jg)%l2moment = .TRUE.
      CASE (8)
        atm_phy_nwp_config(jg)%lhave_graupel = .TRUE.
        atm_phy_nwp_config(jg)%lsbm = .TRUE.
      CASE DEFAULT
        atm_phy_nwp_config(jg)%lhave_graupel = .FALSE.
        atm_phy_nwp_config(jg)%l2moment = .FALSE.
      END SELECT
      atm_phy_nwp_config(jg)%lhydrom_read_from_fg(:) = .FALSE.
      atm_phy_nwp_config(jg)%lhydrom_read_from_ana(:) = .FALSE.

      IF (atm_phy_nwp_config(jg)%icalc_reff > 0 .AND. &
             atm_phy_nwp_config(jg)%icpl_rad_reff == 1 .AND. &
             atm_phy_nwp_config(jg)%icalc_reff /= 101 ) THEN
        atm_phy_nwp_config(jg)%luse_clc_rad = .TRUE.
      ELSE
        atm_phy_nwp_config(jg)%luse_clc_rad = .FALSE.
      END IF

      ! Switch off stochastic convection for horizontal resolution greater than 20km
      IF ((atm_phy_nwp_config(jg)%lstoch_sde .or. atm_phy_nwp_config(jg)%lstoch_expl) .and. &
           & p_patch(jg)%geometry_info%mean_characteristic_length .GT. 2.e4_wp) THEN
        atm_phy_nwp_config(jg)%lstoch_expl      = .FALSE.
        atm_phy_nwp_config(jg)%lstoch_sde       = .FALSE.
        atm_phy_nwp_config(jg)%lrestune_off     = .FALSE.
        atm_phy_nwp_config(jg)%lmflimiter_off   = .FALSE.
        atm_phy_nwp_config(jg)%lvvcouple        = .FALSE.
        atm_phy_nwp_config(jg)%lvv_shallow_deep = .FALSE.
        atm_phy_nwp_config(jg)%lstoch_spinup    = .FALSE.
        WRITE(message_text,'(a,i2)') 'Resolution greater than 20km, stochastic shallow convection has been switched off for domain ',jg
        CALL message(TRIM(routine), TRIM(message_text))
      ENDIF

      !$ACC ENTER DATA COPYIN(atm_phy_nwp_config(jg)%lhydrom_read_from_fg, atm_phy_nwp_config(jg)%lhydrom_read_from_ana)

      ! check for contradicting convection settings
      IF (atm_phy_nwp_config(jg)%lshallowconv_only .AND. atm_phy_nwp_config(jg)%lgrayzone_deepconv) THEN
        CALL finish('configure_atm_phy_nwp', "lshallowconv_only and lgrayzone_deepconv are mutually exclusive")
      ENDIF

#ifndef __NO_ICON_LES__
      ! Configure LES physics (if activated)
      !
      atm_phy_nwp_config(jg)%is_les_phy = .FALSE. 
    
      IF(ANY( (/ismag,iprog/)  == atm_phy_nwp_config(jg)%inwp_turb ) )THEN
        CALL configure_les(jg, dtime = time_config%get_model_timestep_sec(p_patch(jg)%nest_level))
        atm_phy_nwp_config(jg)%is_les_phy = .TRUE. 
      END IF 

      !sanity check
      IF( atm_phy_nwp_config(jg)%inwp_surface>0 .AND. les_config(jg)%isrfc_type>1)THEN
         WRITE(message_text,'(a,i2,a)') 'isrfc_type = ',les_config(jg)%isrfc_type, &
            ' enables idealized surface which needs inwp_surface = 0. Check simulation configuration!'
        CALL finish(routine, message_text)
      END IF

      IF( atm_phy_nwp_config(jg)%is_les_phy ) THEN

        ! convection should be turned off for LES
        IF(atm_phy_nwp_config(jg)%inwp_convection>0)THEN
          CALL message(routine, 'Turning off convection for LES!')
          atm_phy_nwp_config(jg)%inwp_convection  = 0
          atm_phy_nwp_config(jg)%lenabled(itconv) = .FALSE.
        END IF

        ! SSO should be turned off for LES
        IF(atm_phy_nwp_config(jg)%inwp_sso>0)THEN
          CALL message(routine, 'Turning off SSO scheme for LES!')
          atm_phy_nwp_config(jg)%inwp_sso = 0
          atm_phy_nwp_config(jg)%lenabled(itsso)= .FALSE.
        END IF

        ! GWD should be turned off for LES
        IF(atm_phy_nwp_config(jg)%inwp_gwd>0)THEN
          CALL message(routine, 'Turning off GWD scheme for LES!')
          atm_phy_nwp_config(jg)%inwp_gwd = 0
          atm_phy_nwp_config(jg)%lenabled(itgwd) =.FALSE.
        END IF

      ENDIF ! is_les_phy
#endif

      !$ACC ENTER DATA COPYIN(atm_phy_nwp_config(jg)%lenabled)

      ! Check, whether the user-defined slow-physics timesteps adhere 
      ! to ICON-internal rules. If not, adapt the timesteps accordingly.
      ! RULES:
      !  I) Every slow-physics timestep must be an integer multiple  
      !     of the advection/fast-physics timestep.
      !     If not, the slow physics timestep is rounded up to the 
      !     next integer multiple.
      ! II) Special rules for cloud cover time step are set 
      !     when using no convection scheme (see below)
      !III) The radiation timestep must be an integer multiple of the 
      !     cloud-cover timestep and the convection. If not, the radiation timestep is 
      !     rounded up to the next integer multiple.


      ! These rules are applied for every patch. As a result, timesteps for a 
      ! particular process may differ from patch to patch.

      ! RULE (I)
      IF (isModulo(atm_phy_nwp_config(jg)%dt_conv,atm_phy_nwp_config(jg)%dt_fastphy)) THEN
        WRITE(message_text,'(a,i2,a)') 'DOM ',jg, &
          &                            ': Convection timestep is not a multiple of advection step => rounded up!'
        CALL message(routine, message_text)
        atm_phy_nwp_config(jg)%dt_conv = roundToNextMultiple(atm_phy_nwp_config(jg)%dt_conv,  &
          &                                                  atm_phy_nwp_config(jg)%dt_fastphy)
      ENDIF
      !
      IF (isModulo(atm_phy_nwp_config(jg)%dt_ccov,atm_phy_nwp_config(jg)%dt_fastphy)) THEN
        WRITE(message_text,'(a,i2,a)') 'DOM ',jg, &
          &                            ': Cloud-cover timestep is not a multiple of advection step => rounded up!'
        CALL message(routine, message_text)
        atm_phy_nwp_config(jg)%dt_ccov = roundToNextMultiple(atm_phy_nwp_config(jg)%dt_ccov,  &
          &                                                  atm_phy_nwp_config(jg)%dt_fastphy)
      ENDIF
      !
      IF (isModulo(atm_phy_nwp_config(jg)%dt_sso,atm_phy_nwp_config(jg)%dt_fastphy)) THEN
        WRITE(message_text,'(a,i2,a)') 'DOM ',jg, &
          &                            ': SSO timestep is not a multiple of advection step => rounded up!'
        CALL message(routine, message_text)
        atm_phy_nwp_config(jg)%dt_sso = roundToNextMultiple(atm_phy_nwp_config(jg)%dt_sso,    &
          &                                                 atm_phy_nwp_config(jg)%dt_fastphy)
      ENDIF
      !
      IF (isModulo(atm_phy_nwp_config(jg)%dt_gwd,atm_phy_nwp_config(jg)%dt_fastphy)) THEN
        WRITE(message_text,'(a,i2,a)') 'DOM ',jg, &
          &                            ': GWD timestep is not a multiple of advection step => rounded up!'
        CALL message(routine, message_text)
        atm_phy_nwp_config(jg)%dt_gwd = roundToNextMultiple(atm_phy_nwp_config(jg)%dt_gwd,    &
                                                            atm_phy_nwp_config(jg)%dt_fastphy)
      ENDIF
      !
      IF (isModulo(atm_phy_nwp_config(jg)%dt_rad,atm_phy_nwp_config(jg)%dt_fastphy)) THEN
        WRITE(message_text,'(a,i2,a)') 'DOM ',jg, &
          &                            ': Radiation timestep is not a multiple of advection step => rounded up!'
        CALL message(routine, message_text)
        atm_phy_nwp_config(jg)%dt_rad = roundToNextMultiple(atm_phy_nwp_config(jg)%dt_rad,    &
          &                                                 atm_phy_nwp_config(jg)%dt_fastphy)
      ENDIF


      ! RULE (II)
      !
      ! When using no convection scheme, users may not be aware that setting a convection or cloud cover time step
      ! is relevant. We thus prevent the cloud cover time step from being unreasonably large by limiting
      ! it to six times the fast physics time step
      IF ( atm_phy_nwp_config(jg)%inwp_convection == 0 .AND. atm_phy_nwp_config(jg)%inwp_cldcover > 0 .AND. &
           atm_phy_nwp_config(jg)%dt_ccov > 6._wp*atm_phy_nwp_config(jg)%dt_fastphy ) THEN
        WRITE(message_text,'(a)') 'No convection scheme selected => Reduce dt_ccov to 6*dt_fastphy.'
        CALL message(routine, message_text)
        atm_phy_nwp_config(jg)%dt_ccov = 6._wp*atm_phy_nwp_config(jg)%dt_fastphy
      ENDIF
      !
      ! In addition, dt_conv and dt_ccov are reset to the fast-physics time step when the respective parameterization
      ! is turned off (to avoid side effects on the radiation time step)
      IF ( atm_phy_nwp_config(jg)%inwp_convection == 0) atm_phy_nwp_config(jg)%dt_conv = atm_phy_nwp_config(jg)%dt_fastphy
      IF ( atm_phy_nwp_config(jg)%inwp_cldcover == 0)   atm_phy_nwp_config(jg)%dt_ccov = atm_phy_nwp_config(jg)%dt_fastphy


      ! RULE (III)
      IF (isModulo(atm_phy_nwp_config(jg)%dt_rad,atm_phy_nwp_config(jg)%dt_ccov)) THEN
        WRITE(message_text,'(a,i2,a)') 'DOM ',jg, &
          &                            ': Radiation timestep is not a multiple of cloud-cover step => rounded up!'
        CALL message(routine, message_text)
        atm_phy_nwp_config(jg)%dt_rad = roundToNextMultiple(atm_phy_nwp_config(jg)%dt_rad,    &
          &                                                 atm_phy_nwp_config(jg)%dt_ccov)
      ENDIF

      IF (isModulo(atm_phy_nwp_config(jg)%dt_rad,atm_phy_nwp_config(jg)%dt_conv)) THEN
        WRITE(message_text,'(a,i2,a)') 'DOM ',jg, &
          &                            ': Radiation timestep is not a multiple of convection step => rounded up!'
        CALL message(routine, message_text)
        atm_phy_nwp_config(jg)%dt_rad = roundToNextMultiple(atm_phy_nwp_config(jg)%dt_rad,    &
          &                                                 atm_phy_nwp_config(jg)%dt_conv)
      ENDIF

      ! Check if the radiation time step is still a multiple of dt_ccov. Otherwise finish
      IF (isModulo(atm_phy_nwp_config(jg)%dt_rad,atm_phy_nwp_config(jg)%dt_ccov)) THEN
        WRITE(message_text,'(a,i2,a)') 'DOM ',jg, &
          & ': Time steps for cloud cover and convection need to be set such that the radiation step can be a multiple of both'
        CALL finish(routine, message_text)
      ENDIF


      ! Fill dt_phy with final timesteps
      !
      dt_phy(jg,:) = 0._wp  ! init

      ! Slow physics time steps
      !
      dt_phy(jg,itconv) = atm_phy_nwp_config(jg)% dt_conv    ! sec

      dt_phy(jg,itsso)  = atm_phy_nwp_config(jg)% dt_sso     ! sec

      dt_phy(jg,itgwd)  = atm_phy_nwp_config(jg)% dt_gwd     ! sec

      dt_phy(jg,itrad)  = atm_phy_nwp_config(jg)% dt_rad     ! sec

      dt_phy(jg,itccov) = atm_phy_nwp_config(jg)% dt_ccov    ! sec

      ! Fast physics time step
      !
      dt_phy(jg,itfastphy) = atm_phy_nwp_config(jg)%dt_fastphy ! sec


      ! screen printout of chosen physics timesteps
      IF ( msg_level>=7 ) THEN
        CALL phy_nwp_print_dt (atm_phy_nwp_config = atm_phy_nwp_config(jg), &
          &                    dt_phy_orig        = dt_phy_orig(jg,:),      &
          &                    dt_phy             = dt_phy(jg,:),           &
          &                    pid                = jg                      )
      ENDIF

    ENDDO  ! jg loop




    ! Configure lateral boundary condition for limited area model (or global nudging)
    IF( l_limited_area .OR. ANY(nudging_config(1:n_dom)%lnudging) ) THEN
      CALL configure_latbc()
    END IF
    ! Configure nudging (primary domain only)
    CALL configure_nudging(p_patch(1)%nlev, p_patch(1:)%nshift_total, n_dom, msg_level, timers_level) 

    ! Settings for ozone tuning, depending on option for ozone climatology
    SELECT CASE (irad_o3)
    CASE (7)  ! GEMS climatology
      tune_ozone_ztop   = 30000.0_wp
      tune_ozone_zmid2  = 15000.0_wp
      tune_ozone_zmid   = 15000.0_wp
      tune_ozone_zbot   = 10000.0_wp
      ozone_shapemode   = 1        ! tuning is applied at low latitudes only
      tune_ozone_lat    = 45._wp   ! tuning ends at 45 deg
      IF (itune_o3 == 1) THEN
        CALL message(routine, 'Use GEMS ozone climatology with tuning')
        tune_ozone_fac    = 0.5_wp
        tune_ozone_maxinc = 2.e-6_wp ! maximum absolute change of O3 mixing ratio
                                     ! this value is about 12% of the climatological maximum in the tropics
      ELSE
        tune_ozone_fac    = 0._wp
        tune_ozone_maxinc = 0._wp
      ENDIF
    CASE (79,97) ! Blending between GEMS and MACC climatologies
      IF (itune_o3 > 0) CALL message(routine, 'Use blending between GEMS and MACC ozone climatologies with tuning')
      IF (itune_o3 >= 2) THEN
        tune_ozone_ztop   = 29000.0_wp
        tune_ozone_zmid2  = 24000.0_wp
      ELSE
        tune_ozone_ztop   = 29000.0_wp
        tune_ozone_zmid2  = 26000.0_wp
      ENDIF
      tune_ozone_zmid   = 19000.0_wp
      tune_ozone_zbot   = 16000.0_wp
      ozone_shapemode   = 2
      tune_ozone_lat    = 30._wp
      SELECT CASE (itune_o3)
      CASE (1,2)
        tune_ozone_fac    = 0.25_wp
      CASE (3)
        tune_ozone_fac    = 0.35_wp
      CASE (4)
        tune_ozone_fac    = 0.1_wp
      CASE DEFAULT
        tune_ozone_fac    = 0.0_wp
      END SELECT
      IF (itune_o3 == 3) THEN
        tune_ozone_maxinc = 1.75e-6_wp
      ELSE
        tune_ozone_maxinc = 1.25e-6_wp
      ENDIF
    CASE DEFAULT
      IF (itune_o3 /= 0) THEN
        CALL message(routine, 'itune_o3 is reset to 0 because irad_o3 is not 7, 79 or 97')
        itune_o3 = 0
      ENDIF
      tune_ozone_ztop   = 30000.0_wp
      tune_ozone_zmid2  = 15000.0_wp
      tune_ozone_zmid   = 15000.0_wp
      tune_ozone_zbot   = 10000.0_wp
      tune_ozone_fac    = 0._wp
      ozone_shapemode   = 1
      tune_ozone_lat    = -1._wp
      tune_ozone_maxinc = 0._wp
    END SELECT

    ! Ozone tuning function, applied to the ozone climatology as 
    ! o3clim_tuned = o3clim*(1.+fac_ozone*shapefunc_ozone)
    DO jg = 1, n_dom
      atm_phy_nwp_config(jg)%ozone_maxinc = tune_ozone_maxinc
      !$ACC UPDATE DEVICE(atm_phy_nwp_config(jg)%ozone_maxinc) ASYNC(1)
      ALLOCATE(atm_phy_nwp_config(jg)%fac_ozone(p_patch(jg)%nlev), &
               atm_phy_nwp_config(jg)%shapefunc_ozone(nproma,p_patch(jg)%nblks_c) )
      ! Vertical profile function
      DO jk = 1,p_patch(jg)%nlev
        jk_shift = jk+p_patch(jg)%nshift_total
        z_mc_ref = 0.5_wp*(vct_a(jk_shift)+vct_a(jk_shift+1))
        IF ( z_mc_ref > tune_ozone_zbot .AND. z_mc_ref < tune_ozone_ztop ) THEN
          IF ( z_mc_ref < tune_ozone_zmid ) THEN
            atm_phy_nwp_config(jg)%fac_ozone(jk) = tune_ozone_fac *                        &
                 & sin((z_mc_ref-tune_ozone_zbot)/(tune_ozone_zmid-tune_ozone_zbot)*pi_2)**2
          ELSE IF ( z_mc_ref < tune_ozone_zmid2 ) THEN
            atm_phy_nwp_config(jg)%fac_ozone(jk) = tune_ozone_fac 
          ELSE
            atm_phy_nwp_config(jg)%fac_ozone(jk) = tune_ozone_fac *                        &
                 & cos((z_mc_ref-tune_ozone_zmid2)/(tune_ozone_ztop-tune_ozone_zmid2)*pi_2)**2
          END IF
        ELSE
          atm_phy_nwp_config(jg)%fac_ozone(jk) = 0.0_wp
        ENDIF
      ENDDO
      !$ACC ENTER DATA COPYIN(atm_phy_nwp_config(jg)%fac_ozone)
      ! Horizontal profile function for fac_ozone
      DO jb = 1, p_patch(jg)%nblks_c
        DO jc = 1, nproma          
          IF (ozone_shapemode == 1 .AND. tune_ozone_lat > 0._wp) THEN
            IF (ABS(p_patch(jg)%cells%center(jc,jb)%lat) < tune_ozone_lat * deg2rad) THEN
              atm_phy_nwp_config(jg)%shapefunc_ozone(jc,jb) = &
                COS(p_patch(jg)%cells%center(jc,jb)%lat * 90._wp/tune_ozone_lat)**2
            ELSE
              atm_phy_nwp_config(jg)%shapefunc_ozone(jc,jb) = 0._wp
            END IF
          ELSE IF (ozone_shapemode == 2 .AND. tune_ozone_lat > 0._wp) THEN
            IF (ABS(p_patch(jg)%cells%center(jc,jb)%lat) < tune_ozone_lat * deg2rad) THEN
              atm_phy_nwp_config(jg)%shapefunc_ozone(jc,jb) = &
                1._wp - 1.0_wp*(COS(p_patch(jg)%cells%center(jc,jb)%lat * 90._wp/tune_ozone_lat))**0.25_wp
            ELSE
              atm_phy_nwp_config(jg)%shapefunc_ozone(jc,jb) = 1._wp
            END IF
          ELSE
            atm_phy_nwp_config(jg)%shapefunc_ozone(jc,jb) = 1.0_wp
          END IF
        ENDDO
      ENDDO
      !$ACC ENTER DATA COPYIN(atm_phy_nwp_config(jg)%shapefunc_ozone)
    ENDDO

    !$ACC ENTER DATA COPYIN(atm_phy_nwp_config(jg)%fac_ozone, atm_phy_nwp_config(jg)%shapefunc_ozone)



    !
    ! initialize event-management for parameterized physical processes
    !
    ! - Event start/end dates are set equal to the domain start/end dates
    ! - For the computation of domain start/end dates we have to distinguish 
    !   between global/master domains (i.e. DOM(1)) and the nests.
    !   Start dates:
    !   * DOM(1): dom_start_date = tc_exp_startdate + timeshift%dt_shift 
    !             where dt_shift is a possible timeshift due to IAU.
    !   * DOM(i>1): dom_start_date = tc_exp_startdate + start_time(i) 
    !   End dates:
    !   * DOM(1)  : dom_end_date = tc_exp_stopdate
    !   * DOM(i>1): dom_end_date = tc_exp_startdate +  end_time(i)
    !   If end_time(i) is not set, tc_exp_stopdate is used.
    !
    ! - for each domain, the event start dates are set to
    !   event_start_date(i) =  dom_start_date(i) +  dt_fastphy(i)
    !   * adding dt_fastphy is necessitated by the fact, that the model time 
    !     is updated at the beginning of a timestep.
    !   * by doing so, care is taken that all physics processes are called during the 
    !   first integration step of the given patch.
    !
    ! - initialization calls are not controlled by mtime events. 
    !   It is decided upon the physical process TYPE metainformation, 
    !   whether an intialization call should be issued or not.
    ! 
    DO jg = 1,n_dom


      ! compute event start date
      IF (jg > 1) THEN
        CALL getPTStringFromMS(INT(start_time(jg)*1000._wp,i8), td_start_str)
        td_start => newTimedelta(td_start_str)
        domStartDate = time_config%tc_exp_startdate + td_start
        CALL deallocateTimedelta(td_start)
      ELSE
        domStartDate = time_config%tc_exp_startdate
        ! take care of possibe IAU-Timeshift
        IF (timeshift%dt_shift < 0._wp) THEN
          domStartDate = domStartDate + timeshift%mtime_shift
        ENDIF
      ENDIF
      ! Note that the model time is updated at the beginning of a timestep.
      !
      ! by adding td_dt we make sure, that all events are triggered 
      ! during the first integration step of the given patch.
      CALL getPTStringFromMS(INT(atm_phy_nwp_config(jg)%dt_fastphy*1000._wp,i8), td_dt_str)
      td_dt => newTimedelta(td_dt_str)
      !
      eventStartDate = domStartDate + td_dt
      CALL deallocateTimedelta(td_dt)


      ! compute event end date
      IF (jg > 1) THEN
        IF (end_time(jg) /= DEFAULT_ENDTIME) THEN
          CALL getPTStringFromMS(INT(end_time(jg)*1000._wp,i8), td_end_str)
          td_end => newTimedelta(td_end_str)
          domEndDate = time_config%tc_exp_startdate + td_end
          CALL deallocateTimedelta(td_end)
          ! make sure that eventEndDate<=tc_exp_stopdate
          IF (domEndDate > time_config%tc_exp_stopdate) &
            &  domEndDate = time_config%tc_exp_stopdate
        ELSE
          domEndDate = time_config%tc_exp_stopdate
        ENDIF
      ELSE
        domEndDate = time_config%tc_exp_stopdate
      ENDIF
      eventEndDate = domEndDate


      ! Setup NWP physics event group for domain jg
      CALL setupEventsNwp(atm_phy_nwp_config = atm_phy_nwp_config(jg),      & !inout
        &                 pid                = jg,                          & !in
        &                 grpName            = 'phyNwpEventGroup',          & !in
        &                 eventStartDate     = eventStartDate,              & !in
        &                 eventEndDate       = eventEndDate,                & !in
        &                 dt_phy             = dt_phy(jg,:)                 ) !in


      ! allocate lcall_phy to be of the same size as phyProcs%proc(:),  
      ! since lcall_phy shall contain one entry per physical process.
      ALLOCATE(atm_phy_nwp_config(jg)%lcall_phy(          &
        &  SIZE(atm_phy_nwp_config(jg)%phyProcs%proc, 1)  &
        &  ), STAT = error)
      IF(error /= SUCCESS) CALL finish(routine, "memory allocation failure")

      ! initialize lcall_phy (will be updated by mo_phy_events:mtime_ctrl_physics)
      atm_phy_nwp_config(jg)%lcall_phy(:) = .FALSE.

      !$ACC ENTER DATA COPYIN(atm_phy_nwp_config(jg)%lcall_phy)


      ! 3d radiative flux output: only allocate and write variable if at least one is requested as output
      atm_phy_nwp_config(jg)%l_3d_rad_fluxes &
        =    is_variable_in_output(var_name="group:all")    &
        .OR. is_variable_in_output(var_name="lwflx_dn")     &
        .OR. is_variable_in_output(var_name="swflx_dn")     &
        .OR. is_variable_in_output(var_name="lwflx_up")     &
        .OR. is_variable_in_output(var_name="swflx_up")     &
        .OR. is_variable_in_output(var_name="lwflx_dn_clr") &
        .OR. is_variable_in_output(var_name="swflx_dn_clr") &
        .OR. is_variable_in_output(var_name="lwflx_up_clr") &
        .OR. is_variable_in_output(var_name="swflx_up_clr")

      ! output of turbulent fluxes in column
      atm_phy_nwp_config(jg)%l_3d_turb_fluxes  =                 &
               is_variable_in_output( var_name="tetfl_turb")     &
          .OR. is_variable_in_output( var_name="vapfl_turb")     &
          .OR. is_variable_in_output( var_name="liqfl_turb")     

    ENDDO  ! jg

  END SUBROUTINE configure_atm_phy_nwp



  !>
  !! Setup event group for NWP physics
  !!
  !! Setup event group for NWP physics.
  !! Makes use of mtime events
  !!
  !! Creating and adding a new physics event works as follows:
  !! 1) Add a new variable X of type t_phyProcFast/t_phyProcSlow 
  !!    to the config state. Initialize the new physics event by calling 
  !!    X%initialize() and pass details describing the new event via the 
  !!    argument list.
  !! 2) Add the new variable to the existing physics event group G by calling
  !!    G%addToGroup(X). You may alternatively add X to another group. 
  !!    A new group, say G2, can be constructed by calling G2%construct() just 
  !!    before G2%addToGroup(X).
  !! 3) A single physics event and/or an entire group can be queried with 
  !!    the help of various type-bound procedures listed in atm_phy_nwp:mo_phy_events.
  !! 4) Calling the routine atm_phy_nwp:mtime_ctrl_physics will provide you with 
  !!    an array of logicals of the same size as your group. This array tells you 
  !!    whether a specific physics event is due at the current time step, or not.  
  !!
  SUBROUTINE setupEventsNwp(atm_phy_nwp_config, pid, grpName, eventStartDate, eventEndDate, dt_phy)

    TYPE(t_atm_phy_nwp_config), TARGET, INTENT(INOUT)  :: atm_phy_nwp_config  !< config state
    !
    INTEGER       , INTENT(IN)      :: pid                  !< patch ID
    CHARACTER(len=*)                :: grpName              !< group name 
    TYPE(datetime), INTENT(IN)      :: eventStartDate
    TYPE(datetime), INTENT(IN)      :: eventEndDate
    REAL(wp)      , INTENT(IN)      :: dt_phy(:)            !< physics time intervals

    ! local
    TYPE(timedelta), POINTER        :: eventInterval    => NULL()
    TYPE(datetime)                  :: eventEndDate_proc     ! process-specific end date
                                                             ! set to startDate, if process is disabled
    TYPE(timedelta), POINTER        :: plusSlack    => NULL()
    CHARACTER(LEN=MAX_TIMEDELTA_STR_LEN) :: dt_phy_str       ! physics timestep (PT-format)

    LOGICAL :: requires_init


    ! construct physics group for given patch
    CALL atm_phy_nwp_config%phyProcs%construct(grpName=TRIM(grpName), pid=pid, grpSize=iphysproc)

    ! Events are triggered between [actual_trigger_time, actual_trigger_time + plus_slack]
    plusSlack =>newTimedelta("PT0S")


    ! convection
    eventEndDate_proc = MERGE(eventEndDate, eventStartDate, atm_phy_nwp_config%lenabled(itconv))
    !
    CALL getPTStringFromMS(INT(dt_phy(itconv)*1000._wp,i8), dt_phy_str)
    eventInterval=>newTimedelta(dt_phy_str)
    !
    CALL atm_phy_nwp_config%phyProc_conv%initialize(                            &
      &                     name        = 'conv',                               & !in
      &                     id          = itconv,                               & !in
      &                     is_enabled  = atm_phy_nwp_config%lenabled(itconv),  & !in
      &                     startDate   = eventStartDate,                       & !in
      &                     endDate     = eventEndDate_proc,                    & !in
      &                     dt          = eventInterval,                        & !in
      &                     plusSlack   = plusSlack,                            & !in
      &                     optReqInit  = .TRUE.,                               & !in
      &                     optInclStart= .TRUE.                                ) !in
    ! add to physics group
    CALL atm_phy_nwp_config%phyProcs%addToGroup(                          &
      &                     phyProc = atm_phy_nwp_config%phyProc_conv)
    CALL deallocateTimedelta(eventInterval)

    !
    ! cloudcover
    eventEndDate_proc = MERGE(eventEndDate, eventStartDate, atm_phy_nwp_config%lenabled(itccov))
    !
    CALL getPTStringFromMS(INT(dt_phy(itccov)*1000._wp,i8), dt_phy_str)
    eventInterval=>newTimedelta(dt_phy_str)
    !
    CALL atm_phy_nwp_config%phyProc_ccov%initialize(                            &
      &                     name        = 'ccov',                               & !in
      &                     id          = itccov,                               & !in
      &                     is_enabled  = atm_phy_nwp_config%lenabled(itccov),  & !in
      &                     startDate   = eventStartDate,                       & !in
      &                     endDate     = eventEndDate_proc,                    & !in
      &                     dt          = eventInterval,                        & !in
      &                     plusSlack   = plusSlack,                            & !in
      &                     optReqInit  = .TRUE.,                               & !in
      &                     optInclStart= .TRUE.                                ) !in
    ! add to physics group
    CALL atm_phy_nwp_config%phyProcs%addToGroup(                          &
      &                     phyProc = atm_phy_nwp_config%phyProc_ccov)
    CALL deallocateTimedelta(eventInterval)


    !
    ! radiation (SW+LW)
    eventEndDate_proc = MERGE(eventEndDate, eventStartDate, atm_phy_nwp_config%lenabled(itrad))
    !
    CALL getPTStringFromMS(INT(dt_phy(itrad)*1000._wp,i8), dt_phy_str)
    eventInterval=>newTimedelta(dt_phy_str)
    !
    CALL atm_phy_nwp_config%phyProc_rad%initialize(                             &
      &                     name        = 'rad',                                & !in
      &                     id          = itrad,                                & !in
      &                     is_enabled  = atm_phy_nwp_config%lenabled(itrad),   & !in
      &                     startDate   = eventStartDate,                       & !in
      &                     endDate     = eventEndDate_proc,                    & !in
      &                     dt          = eventInterval,                        & !in
      &                     plusSlack   = plusSlack,                            & !in
      &                     optReqInit  = .TRUE.,                               & !in
      &                     optInclStart= .TRUE.                                ) !in
    ! add to physics group
    CALL atm_phy_nwp_config%phyProcs%addToGroup(                          &
      &                     phyProc = atm_phy_nwp_config%phyProc_rad)
    CALL deallocateTimedelta(eventInterval)


    !
    ! sub-gridscale orographic drag
    eventEndDate_proc = MERGE(eventEndDate, eventStartDate, atm_phy_nwp_config%lenabled(itsso))
    !
    CALL getPTStringFromMS(INT(dt_phy(itsso)*1000._wp,i8), dt_phy_str)
    eventInterval=>newTimedelta(dt_phy_str)
    !
    CALL atm_phy_nwp_config%phyProc_sso%initialize(                             &
      &                     name        = 'sso',                                & !in
      &                     id          = itsso,                                & !in
      &                     is_enabled  = atm_phy_nwp_config%lenabled(itsso),   & !in
      &                     startDate   = eventStartDate,                       & !in
      &                     endDate     = eventEndDate_proc,                    & !in
      &                     dt          = eventInterval,                        & !in
      &                     plusSlack   = plusSlack,                            & !in
      &                     optReqInit  = .TRUE.,                               & !in
      &                     optInclStart= .TRUE.                                ) !in
    ! add to physics group
    CALL atm_phy_nwp_config%phyProcs%addToGroup(                          &
      &                     phyProc = atm_phy_nwp_config%phyProc_sso)
    CALL deallocateTimedelta(eventInterval)


    !
    ! non-orographic gravity wave drag
    eventEndDate_proc = MERGE(eventEndDate, eventStartDate, atm_phy_nwp_config%lenabled(itgwd))
    !
    CALL getPTStringFromMS(INT(dt_phy(itgwd)*1000._wp,i8), dt_phy_str)
    eventInterval=>newTimedelta(dt_phy_str)
    !
    CALL atm_phy_nwp_config%phyProc_gwd%initialize(                             &
      &                     name        = 'gwd',                                & !in
      &                     id          = itgwd,                                & !in
      &                     is_enabled  = atm_phy_nwp_config%lenabled(itgwd),   & !in
      &                     startDate   = eventStartDate,                       & !in
      &                     endDate     = eventEndDate_proc,                    & !in
      &                     dt          = eventInterval,                        & !in
      &                     plusSlack   = plusSlack,                            & !in
      &                     optReqInit  = .TRUE.,                               & !in
      &                     optInclStart= .TRUE.                                ) !in
    ! add to physics group
    CALL atm_phy_nwp_config%phyProcs%addToGroup(                          &
      &                     phyProc = atm_phy_nwp_config%phyProc_gwd)
    CALL deallocateTimedelta(eventInterval)


    !
    ! saturation adjustment
    eventEndDate_proc = MERGE(eventEndDate, eventStartDate, atm_phy_nwp_config%lenabled(itsatad))
    !
    CALL getPTStringFromMS(INT(dt_phy(itfastphy)*1000._wp,i8), dt_phy_str)
    eventInterval=>newTimedelta(dt_phy_str)
    !
    CALL atm_phy_nwp_config%phyProc_satad%initialize(                           &
      &                     name        = 'satad',                              & !in
      &                     id          = itsatad,                              & !in
      &                     is_enabled  = atm_phy_nwp_config%lenabled(itsatad), & !in
      &                     startDate   = eventStartDate,                       & !in
      &                     endDate     = eventEndDate_proc,                    & !in
      &                     dt          = eventInterval,                        & !in
      &                     plusSlack   = plusSlack,                            & !in
      &                     optInclStart= .TRUE.                                ) !in
    ! add to physics group
    CALL atm_phy_nwp_config%phyProcs%addToGroup(                          &
      &                     phyProc = atm_phy_nwp_config%phyProc_satad)
    CALL deallocateTimedelta(eventInterval)


    !
    ! turbulence
    eventEndDate_proc = MERGE(eventEndDate, eventStartDate, atm_phy_nwp_config%lenabled(itturb))
    !
    CALL getPTStringFromMS(INT(dt_phy(itfastphy)*1000._wp,i8), dt_phy_str)
    eventInterval=>newTimedelta(dt_phy_str)
    !
    ! VDIFF runs initialization step with slow physics on GPU, not with fast physics on host.
    requires_init = (atm_phy_nwp_config%inwp_turb == ivdiff)
    !
    CALL atm_phy_nwp_config%phyProc_turb%initialize(                            &
      &                     name        = 'turb',                               & !in
      &                     id          = itturb,                               & !in
      &                     is_enabled  = atm_phy_nwp_config%lenabled(itturb),  & !in
      &                     startDate   = eventStartDate,                       & !in
      &                     endDate     = eventEndDate_proc,                    & !in
      &                     dt          = eventInterval,                        & !in
      &                     plusSlack   = plusSlack,                            & !in
      &                     optReqInit  = requires_init,                        & !in
      &                     optInclStart= .TRUE.                                ) !in
    ! add to physics group
    CALL atm_phy_nwp_config%phyProcs%addToGroup(                          &
      &                     phyProc = atm_phy_nwp_config%phyProc_turb)
    CALL deallocateTimedelta(eventInterval)


    !
    ! grid-scale microphysics
    eventEndDate_proc = MERGE(eventEndDate, eventStartDate, atm_phy_nwp_config%lenabled(itgscp))
    !
      CALL getPTStringFromMS(INT(dt_phy(itfastphy)*1000._wp,i8), dt_phy_str)
    eventInterval=>newTimedelta(dt_phy_str)
    !
    CALL atm_phy_nwp_config%phyProc_gscp%initialize(                            &
      &                     name        = 'gscp',                               & !in
      &                     id          = itgscp,                               & !in
      &                     is_enabled  = atm_phy_nwp_config%lenabled(itgscp),  & !in
      &                     startDate   = eventStartDate,                       & !in
      &                     endDate     = eventEndDate_proc,                    & !in
      &                     dt          = eventInterval,                        & !in
      &                     plusSlack   = plusSlack,                            & !in
      &                     optInclStart= .TRUE.                                ) !in
    ! add to physics group
    CALL atm_phy_nwp_config%phyProcs%addToGroup(                          &
      &                     phyProc = atm_phy_nwp_config%phyProc_gscp)
    CALL deallocateTimedelta(eventInterval)


    !
    ! land/surface
    eventEndDate_proc = MERGE(eventEndDate, eventStartDate, atm_phy_nwp_config%lenabled(itsfc))
    !
    CALL getPTStringFromMS(INT(dt_phy(itfastphy)*1000._wp,i8), dt_phy_str)
    eventInterval=>newTimedelta(dt_phy_str)
    !
    CALL atm_phy_nwp_config%phyProc_sfc%initialize(                             &
      &                     name        = 'sfc',                                & !in
      &                     id          = itsfc,                                & !in
      &                     is_enabled  = atm_phy_nwp_config%lenabled(itsfc),   & !in
      &                     startDate   = eventStartDate,                       & !in
      &                     endDate     = eventEndDate_proc,                    & !in
      &                     dt          = eventInterval,                        & !in
      &                     plusSlack   = plusSlack,                            & !in
      &                     optInclStart= .TRUE.                                ) !in
    ! add to physics group
    CALL atm_phy_nwp_config%phyProcs%addToGroup(                          &
      &                     phyProc = atm_phy_nwp_config%phyProc_sfc)
    CALL deallocateTimedelta(eventInterval)


    !
    ! radheat
    eventEndDate_proc = MERGE(eventEndDate, eventStartDate, atm_phy_nwp_config%lenabled(itradheat))
    !
    CALL getPTStringFromMS(INT(dt_phy(itfastphy)*1000._wp,i8), dt_phy_str)
    eventInterval=>newTimedelta(dt_phy_str)
    !
    CALL atm_phy_nwp_config%phyProc_radheat%initialize(                           &
      &                     name        = 'radheat',                              & !in
      &                     id          = itradheat,                              & !in
      &                     is_enabled  = atm_phy_nwp_config%lenabled(itradheat), & !in
      &                     startDate   = eventStartDate,                         & !in
      &                     endDate     = eventEndDate_proc,                      & !in
      &                     dt          = eventInterval,                          & !in
      &                     plusSlack   = plusSlack,                              & !in
      &                     optReqInit  = .TRUE.,                                 & !in
      &                     optInclStart= .TRUE.                                  ) !in 
    ! add to physics group
    CALL atm_phy_nwp_config%phyProcs%addToGroup(                          &
      &                     phyProc = atm_phy_nwp_config%phyProc_radheat)
    CALL deallocateTimedelta(eventInterval)


    ! Debug output
    IF (msg_level >= 10) THEN
      CALL atm_phy_nwp_config%phyProcs%printSetup()
    ENDIF


  END SUBROUTINE setupEventsNwp



  !>
  !! Setup mtime events for optional NWP diagnostics
  !!
  !! Shifted from nh_stepping in order to improve code structure
  !!
  SUBROUTINE setup_nwp_diag_events(time_config, lpi_max_Event, celltracks_Event, dbz_Event, hail_Event)

    TYPE(t_time_config),  INTENT(IN   ) :: time_config       !< time and date information
    TYPE(event), POINTER, INTENT(INOUT) :: lpi_max_Event, celltracks_Event, dbz_Event, hail_Event

    ! local
    TYPE(timedelta), POINTER               :: eventInterval    => NULL()
    CHARACTER(LEN=MAX_TIMEDELTA_STR_LEN)   :: td_string
    CHARACTER(LEN=MAX_DATETIME_STR_LEN)    :: dt_string
    CHARACTER(LEN=MAX_MTIME_ERROR_STR_LEN) :: errstring

    INTEGER   :: ierr


  ! --- create Event for LPI_MAX maximization:
  CALL getPTStringFromMS(INT(dt_lpi*1000._wp,i8), td_string) ! default 3 mins
  eventInterval => newTimedelta(td_string)
  lpi_max_Event => newEvent( 'lpi_max', time_config%tc_exp_startdate,  &   ! "anchor date"
       &                     time_config%tc_exp_startdate,             &   ! start
       &                     time_config%tc_exp_stopdate,              &
       &                     eventInterval, errno=ierr )
  IF (ierr /= no_Error) THEN
    ! give an elaborate error message:
    CALL datetimeToString( time_config%tc_exp_startdate, dt_string)
    WRITE (0,*) "event reference date: ", dt_string
    CALL datetimeToString( time_config%tc_exp_startdate, dt_string)
    WRITE (0,*) "event start date    : ", dt_string
    CALL datetimeToString( time_config%tc_exp_stopdate,  dt_string)
    WRITE (0,*) "event end date      : ", dt_string
    CALL timedeltaToString(eventInterval, td_string)
    WRITE (0,*) "event interval      : ", td_string
    CALL mtime_strerror(ierr, errstring)
    CALL finish('setup_nwp_diag_events', "event 'lpi_max': "//errstring)
  ENDIF


  ! --- create Event for celltrack variables (i.e. tcond/tcond10, uh, vorw_ct, w_ct) maximization:
  CALL getPTStringFromMS(INT(dt_celltracks*1000._wp,i8), td_string) ! default 2 mins
  eventInterval => newTimedelta(td_string)
  celltracks_Event => newEvent( 'celltracks', time_config%tc_exp_startdate,  &   ! "anchor date"
       &                       time_config%tc_exp_startdate,             &   ! start
       &                       time_config%tc_exp_stopdate,              &
       &                       eventInterval, errno=ierr )
  IF (ierr /= no_Error) THEN
    ! give an elaborate error message:
    CALL datetimeToString( time_config%tc_exp_startdate, dt_string)
    WRITE (0,*) "event reference date: ", dt_string
    CALL datetimeToString( time_config%tc_exp_startdate, dt_string)
    WRITE (0,*) "event start date    : ", dt_string
    CALL datetimeToString( time_config%tc_exp_stopdate,  dt_string)
    WRITE (0,*) "event end date      : ", dt_string
    CALL timedeltaToString(eventInterval, td_string)
    WRITE (0,*) "event interval      : ", td_string
    CALL mtime_strerror(ierr, errstring)
    CALL finish('setup_nwp_diag_events', "event 'celltracks': "//errstring)
  ENDIF

  ! --- create Event for DBZ maximisations:
  CALL getPTStringFromMS(INT(dt_radar_dbz*1000._wp,i8), td_string) ! default 2 mins
  eventInterval => newTimedelta(td_string)
  dbz_Event     => newEvent( 'dbz_max', time_config%tc_exp_startdate,  &   ! "anchor date"
       &                       time_config%tc_exp_startdate,             &   ! start
       &                       time_config%tc_exp_stopdate,              &
       &                       eventInterval, errno=ierr )
  IF (ierr /= no_Error) THEN
    ! give an elaborate error message:
    CALL datetimeToString( time_config%tc_exp_startdate, dt_string)
    WRITE (0,*) "event reference date: ", dt_string
    CALL datetimeToString( time_config%tc_exp_startdate, dt_string)
    WRITE (0,*) "event start date    : ", dt_string
    CALL datetimeToString( time_config%tc_exp_stopdate,  dt_string)
    WRITE (0,*) "event end date      : ", dt_string
    CALL timedeltaToString(eventInterval, td_string)
    WRITE (0,*) "event interval      : ", td_string
    CALL mtime_strerror(ierr, errstring)
    CALL finish('setup_nwp_diag_events', "event 'dbz_max': "//errstring)
  ENDIF

  ! --- create Event for hailcast
  CALL getPTStringFromMS(INT(dt_hailcast*1000._wp,i8), td_string)
  eventInterval => newTimedelta(td_string)
  hail_Event => newEvent( 'hail_max', time_config%tc_exp_startdate,  &   ! "anchor date"
       &                     time_config%tc_exp_startdate,             &   ! start
       &                     time_config%tc_exp_stopdate,              &
       &                     eventInterval, errno=ierr )
  IF (ierr /= no_Error) THEN
    ! give an elaborate error message:
    CALL datetimeToString( time_config%tc_exp_startdate, dt_string)
    WRITE (0,*) "event reference date: ", dt_string
    CALL datetimeToString( time_config%tc_exp_startdate, dt_string)
    WRITE (0,*) "event start date    : ", dt_string
    CALL datetimeToString( time_config%tc_exp_stopdate,  dt_string)
    WRITE (0,*) "event end date      : ", dt_string
    CALL timedeltaToString(eventInterval, td_string)
    WRITE (0,*) "event interval      : ", td_string
    CALL mtime_strerror(ierr, errstring)
    CALL finish('setup_nwp_diag_events', "event 'hail_max': "//errstring)
  ENDIF


  END SUBROUTINE setup_nwp_diag_events

  !>
  !! Checks, whether the modulo operation remainder is above a certain threshold.
  !!
  !! Checks, whether the modulo operation results in a remainder 
  !! which is above a certain threshold. The threshold can be given 
  !! as an optional argument. If nothing is specified the threshold 
  !! is set to 10._wp*dbl_eps.
  !!
  LOGICAL FUNCTION isModulo (dividend, divisor, optThresh)

    REAL(wp)          , INTENT(IN) :: dividend
    REAL(wp)          , INTENT(IN) :: divisor
    REAL(wp), OPTIONAL, INTENT(IN) :: optThresh  !< optional threshold value

    ! local
    REAL(wp) :: thresh    ! threshold value
  !-------------------------------------------------------------------------

    IF (PRESENT(optThresh)) THEN
      thresh = optThresh
    ELSE
      thresh = 10._wp*dbl_eps
    ENDIF
    isModulo = MOD(dividend,divisor) > thresh

  END FUNCTION isModulo


  !>
  !! Round up to the next integer multiple
  !!
  !! The input value is rounded up to the next integer multiple
  !!  of the divisor
  !!
  REAL(wp) FUNCTION roundToNextMultiple (inval, divisor)

    REAL(wp)          , INTENT(IN) :: inval
    REAL(wp)          , INTENT(IN) :: divisor

  !-------------------------------------------------------------------------

    roundToNextMultiple = REAL(FLOOR(inval/divisor+1),wp) * divisor

  END FUNCTION roundToNextMultiple



  !>
  !! Screen print out of physics timesteps
  !!
  !! Screen print out of physics timesteps.
  !! Printout in any case, if the respective timestep was modified by ICON
  !! Conditional printout (msg_lev>10), if the respective timestep was not modified. 
  !!
  SUBROUTINE phy_nwp_print_dt (atm_phy_nwp_config, dt_phy_orig, dt_phy, pid)
    !
    TYPE(t_atm_phy_nwp_config), INTENT(IN) :: atm_phy_nwp_config  !< object for which the setup will be printed
    REAL(wp)                  , INTENT(IN) :: dt_phy_orig(:)      !< calling intervals as defined by user
    REAL(wp)                  , INTENT(IN) :: dt_phy(:)           !< final calling intervals
    INTEGER                   , INTENT(IN) :: pid                 !< patch ID 

    ! local variables
    TYPE(t_table)   :: table
    INTEGER         :: irow            ! row to fill
    INTEGER         :: i               ! loop index
    CHARACTER(LEN=64) :: dt_str, dt_str_orig
    INTEGER, PARAMETER :: idx_arr(iphysproc_short) &
         = (/itfastphy,itconv,itccov,itrad,itsso,itgwd/)
    CHARACTER(LEN=7), PARAMETER :: proc_names(iphysproc_short) &
      &                     = (/ "conv   ", &
      &                          "ccov   ", &
      &                          "rad    ", &
      &                          "sso    ", &
      &                          "gwd    ", &
      &                          "fastphy" /)
    !--------------------------------------------------------------------------

    ! will only be executed by stdio process
    IF(.NOT. my_process_is_stdio()) RETURN

    ! Initialize index-arrax and string-array
    !

    ! could this be transformed into a table header?
    write(0,*) "Time intervals for calling NWP physics on patch ", pid

    ! table-based output
    CALL initialize_table(table)
    ! the latter is no longer mandatory
    CALL add_table_column(table, "Process")
    CALL add_table_column(table, "dt user [=> final]")

    irow = 0

    DO i=1,SIZE(idx_arr)

      IF (atm_phy_nwp_config%lenabled(i)) THEN
        irow=irow+1
        CALL set_table_entry(table,irow,"Process", TRIM(proc_names(i)))
        IF (dt_phy(i) /= dt_phy_orig(i)) THEN
          WRITE(dt_str,'(f7.2,a,f7.2)') dt_phy_orig(i), ' => ', dt_phy(i)
        ELSE
          WRITE(dt_str,'(f7.2)') dt_phy(i)
        ENDIF
        CALL set_table_entry(table,irow,"dt user [=> final]", TRIM(dt_str))
      ENDIF

    ENDDO


    CALL print_table(table, opt_delimiter=' | ')
    CALL finalize_table(table)

    WRITE (0,*) " " ! newline
  END SUBROUTINE phy_nwp_print_dt


  !! Finalize atm_phy_nwp_config state
  !!
  SUBROUTINE atm_phy_nwp_config_finalize (me)
    CLASS(t_atm_phy_nwp_config), INTENT(INOUT) :: me       !< passed-object dummy argument

    ! local
    CHARACTER(LEN=*), PARAMETER :: routine = modname//":t_atm_phy_nwp_config_finalize"
  !-----------------------------------------------------------------

    !$ACC WAIT(1)
    !$ACC EXIT DATA DELETE(me%lcall_phy) IF(ALLOCATED(me%lcall_phy))
    !$ACC EXIT DATA DELETE(me%fac_ozone) IF(ALLOCATED(me%fac_ozone))
    !$ACC EXIT DATA DELETE(me%shapefunc_ozone) IF(ALLOCATED(me%shapefunc_ozone))
    IF (ALLOCATED(me%lcall_phy))          DEALLOCATE(me%lcall_phy) 
    IF (ALLOCATED(me%fac_ozone)) THEN
      !$ACC EXIT DATA DELETE(me%fac_ozone)
      DEALLOCATE(me%fac_ozone)
    ENDIF
    IF (ALLOCATED(me%shapefunc_ozone)) THEN
      !$ACC EXIT DATA DELETE(me%shapefunc_ozone)
      DEALLOCATE(me%shapefunc_ozone)
    ENDIF

    CALL me%phyProcs%finalize()

  END SUBROUTINE atm_phy_nwp_config_finalize


END MODULE mo_atm_phy_nwp_config
