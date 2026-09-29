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

! Description:  Contains the data structures
!  to store the physical model state and other auxiliary variables
!  in order to run the ECHAM physics.
!  This module should be an analogon to 'mo_nonhydro_types.f90'
!
!  TODO/To think about:
!     - should physics be called before or after dynamics?
!     - allocate fluxes at edges instead at the centers?
!     - horizontal/vertical tracer flux (reconstruct q'v_n' into q'u' and q'v') ?
!     - provide the "virt_inc" with meaning
!     - where to provide the lat/lon info for radiation?
!     - how to implement the echam-modules - rewriting them or "capsulate"?
!     - revision of fields if there are needed or tp be replaced
!     - fill the physics tendency construction/destruction subroutine
!     - later implement already calculated icon gradients for echam physics
!     - think about variables for flexible time steps

MODULE mo_nwp_phy_types

  USE mo_kind,                ONLY: wp, vp
  USE mo_fortran_tools,       ONLY: t_ptr_2d3d,t_ptr_tracer

  USE mo_nwp_vdiff_types, ONLY: t_nwp_vdiff_state

  IMPLICIT NONE
  PRIVATE

  !public interface
  !
  !types
  PUBLIC :: t_nwp_phy_diag
  PUBLIC :: t_nwp_phy_tend
  PUBLIC :: t_ptr_cloud_ensemble
  PUBLIC :: t_nwp_phy_stochconv

  !> derived data type for synthetic satellite images
  TYPE t_rttov_image
    REAL(wp), POINTER :: p(:,:)      ! pointer to 2D image
  END TYPE t_rttov_image


  !
  !!data structure defining model states
  !
  !!diagnostic variables
  !

  TYPE t_nwp_phy_diag

    TYPE(t_ptr_2d3d),ALLOCATABLE :: tot_ptr(:)  !< pointer array: one pointer for each tot var (grid+subgrid)
    TYPE(t_ptr_2d3d),ALLOCATABLE :: tci_ptr(:)  !< pointer array: total column-integrated values

    TYPE(t_ptr_2d3d),ALLOCATABLE :: cfm_ptr(:)  !< pointer array: average of cfm
    TYPE(t_ptr_2d3d),ALLOCATABLE :: cfh_ptr(:)  !< pointer array: average of cfh
    TYPE(t_ptr_2d3d),ALLOCATABLE :: albdif_t_ptr(:)   !< pointer array: tile-specific albedo (shortwave)
    TYPE(t_ptr_2d3d),ALLOCATABLE :: albvisdif_t_ptr(:)!< pointer array: tile-specific albedo (UV/visible)
    TYPE(t_ptr_2d3d),ALLOCATABLE :: albnirdif_t_ptr(:)!< pointer array: tile-specific albedo (NIR)
    TYPE(t_ptr_2d3d),ALLOCATABLE :: swflxsfc_t_ptr(:) !< pointer array: shortwave net flux at surface
    TYPE(t_ptr_2d3d),ALLOCATABLE :: lwflxsfc_t_ptr(:) !< pointer array: longwave net flux at surface
    TYPE(t_ptr_2d3d),ALLOCATABLE :: tcm_t_ptr(:) !< pointer array: turbulent transfer coefficients for momentum
    TYPE(t_ptr_2d3d),ALLOCATABLE :: tch_t_ptr(:) !< pointer array: turbulent transfer coefficients for heat
    TYPE(t_ptr_2d3d),ALLOCATABLE :: tfv_t_ptr(:) !< pointer array: laminar reduction factor for evaporation
    TYPE(t_ptr_2d3d),ALLOCATABLE :: tvm_t_ptr(:) !< pointer array: turbulent transfer velocity for momentum
    TYPE(t_ptr_2d3d),ALLOCATABLE :: tvh_t_ptr(:) !< pointer array: turbulent transfer velocity for heat
    TYPE(t_ptr_2d3d),ALLOCATABLE :: tkr_t_ptr(:) !< pointer array: turbulent reference surface diffusion coefficient
    TYPE(t_ptr_2d3d),ALLOCATABLE :: gz0_t_ptr(:) !< pointer array: roughness length * gravity
    TYPE(t_ptr_2d3d),ALLOCATABLE :: rlamh_fac_ptr(:) !< pointer array: scaling factor for rlam_heat

    TYPE(t_ptr_2d3d),ALLOCATABLE :: tvs_s_t_ptr(:)  !< pointer array: turbulent velocity scale at surface
    TYPE(t_ptr_2d3d),ALLOCATABLE :: tkvm_s_t_ptr(:) !< pointer array: exchange coefficient for momentum at surface
    TYPE(t_ptr_2d3d),ALLOCATABLE :: tkvh_s_t_ptr(:) !< pointer array: exchange coefficient for heat at surface
    TYPE(t_ptr_2d3d),ALLOCATABLE :: rcld_s_t_ptr(:) !< pointer array: standard deviation of the saturation deficit at surface
    TYPE(t_ptr_2d3d),ALLOCATABLE :: u_10m_t_ptr(:)  !< pointer array: zonal wind at 10m
    TYPE(t_ptr_2d3d),ALLOCATABLE :: v_10m_t_ptr(:)  !< pointer array: meridional wind at 10m
    TYPE(t_ptr_2d3d),ALLOCATABLE :: shfl_s_t_ptr(:) !< pointer array: surface sensible heat flux 
    TYPE(t_ptr_2d3d),ALLOCATABLE :: lhfl_s_t_ptr(:) !< pointer array: surface latent heat flux
    TYPE(t_ptr_2d3d),ALLOCATABLE :: umfl_s_t_ptr(:) !< pointer array: u-momentum flux at the surface
    TYPE(t_ptr_2d3d),ALLOCATABLE :: vmfl_s_t_ptr(:) !< pointer array: v-momentum flux at the surface
    TYPE(t_ptr_2d3d),ALLOCATABLE :: qhfl_s_t_ptr(:) !< pointer array: surface moisture flux
    TYPE(t_ptr_2d3d),ALLOCATABLE :: lhfl_bs_t_ptr(:)!< pointer array: lhf from bare soil
    TYPE(t_ptr_2d3d),ALLOCATABLE :: lhfl_pl_t_ptr(:)!< pointer array: lhf from plants
    TYPE(t_ptr_2d3d),ALLOCATABLE :: aerosol_ptr(:)  !< pointer array: prognostic vertically integrated aerosol optical depth
    TYPE(t_ptr_2d3d),ALLOCATABLE :: uh_max_ptr(:)   !< pointer array: max. updraft helicity in time interval

    REAL(wp), POINTER, CONTIGUOUS :: &
      &   acdnc(:,:,:),        & !! cloud droplet number concentration                   [1/m**3]
      &   cape    (:,:),       & !! convective available energy
      &   cloud_num(:,:),      & !! 2D cloud droplet number concentration for simple aerosol-cloud coupling [1/m**3]
      &   cloud_num_fac(:,:),  & !! scaling factor for cloud_num, can be used for icpl_aero_gscp = 3 and lscale_cdnc = true
      &   conv_eis(:,:),       & !! estimated inversion strength
      &   con_gust(:,:),       & !! convective gusts near surface
      &   con_udd(:,:,:,:),    & !!(nproma,nlev,nblks,8) convective up/downdraft fields
                                 !! 1= convective updraft mass flux (pmfu)
                                 !! 2= convective downdraft mass flux (pmfd)
                                 !! 3= updraft   detrainment rate  (pmfude_rate)
                                 !! 4= downdraft   detrainment rate (pmfdde_rate)
                                 !! 5= temperature in updraft region (ptu)
                                 !! 6= humidity in updraft region (pqu)
                                 !! 7= condensate in updraft region (plu)
                                 !! 8= updraft core fraction
      &  rain_upd(:,:),        & !! total precipitation produced in updrafts [kg/m2/s]
      &  hzerocl(:,:),         & !! height of 0 deg C level [m]
      &  shfl_s(:,:),          & !! sensible heat flux (surface) ( W/m2)
      &  shfl_s_t(:,:,:),      & !! sensible heat flux (surface) ( W/m2)
      &  lhfl_s(:,:),          & !! latent   heat flux (surface) ( W/m2)
      &  lhfl_s_t(:,:,:),      & !! latent   heat flux (surface) ( W/m2)
      &  lhfl_bs(:,:),         & !! latent heat flux from bare soil evap. (surface) ( W/m2)
      &  lhfl_bs_t(:,:,:),     & !! latent heat flux from bare soil evap. (surface) ( W/m2)
      &  lhfl_pl(:,:,:),       & !! latent heat flux from plants                    ( W/m2)
      &  lhfl_pl_t(:,:,:,:),   & !! latent heat flux from plants                    ( W/m2)
      &  qhfl_s(:,:),          & !!      moisture flux (surface) ( Kg/m2/s)
                                 !!      = evaporation rate at surface
      &  qhfl_s_t(:,:,:),      & !! moisture flux (surface)                         ( Kg/m2/s)
                                 !!      = evaporation rate at surface
      &  ashfl_s(:,:),         & !! average or accumulated since model start of shfl_s [W/m2]
      &  alhfl_s(:,:),         & !! average or accumulated since model start of lhfl_s [W/m2]
      &  aqhfl_s(:,:),         & !! average since model start of qhfl_s ( Kg/m2/s) 
                                 !! = average of evaporation rate at surface
      &  alhfl_bs(:,:),        & !! average or accumulated since model start of lhfl_bs [W/m2]
      &  alhfl_pl(:,:,:),      & !! average or accumulated since model start of lhfl_pl [W/m2]
      &  tetfl_turb(:,:,:),    & !! vertical turbulent theta flux [K/m^2s]
      &  vapfl_turb(:,:,:),    & !! vertical turbulent water vapour flux [kg/m^2s]
      &  liqfl_turb(:,:,:),    & !! vertical turbulent liquid water flux [kg/m^2s]
      &  clc_rad(:,:,:),       & !! cloud cover used in radiation schemes and RTTOV, if reff and qr, qs, qg are active in radiation
      &  clc(:,:,:),           & !! cloud cover used otherwise in radiation and in other parameterizations and diagnostics
      &  clct(:,:),            & !! total cloud cover  
      &  clch(:,:),            & !! cloud cover of high-level clouds
      &  clcm(:,:),            & !! cloud cover of mid-level clouds
      &  clcl(:,:),            & !! cloud cover of low-level clouds
      &  cldepth(:,:),         & !! modified cloud depth for media
      &  clct_mod(:,:),        & !! modified total cloud cover for media
      &  fac_ccqc(:,:),        & !! tuning factor (for ensemble perturbations) for CLC-QC relationship in cloud cover scheme
      &  fac_entrorg(:,:),     & !! tuning factor (for ensemble perturbations) for entrainment parameter
      &  fac_rmfdeps(:,:),     & !! tuning factor (for ensemble perturbations) for downdraft mass flux
      &  hbas_con(:,:),        & !! height of base of convection [m]
      &  htop_con(:,:),        & !! height of top of convection [m]
      &  htop_dc(:,:),         & !! height above msl of the top of dry convection [m]
      &  tot_cld(:,:,:,:),     & !! total cloud variables (qv,qc,qi)
      &  tot_cld_vi(:,:,:),    & !! vertically integrated tot_cld (qv,qc,qi), including vertical 
                                 !! integrals of qr and qs 
      &  cosmu0(:,:),          & !! cosine of solar zenith angle
      &  albdif(:,:),          & !! Shortwave albedo for diffuse radiation  (0.3-5.0um)
      &  albvisdif(:,:),       & !! UV visible albedo for diffuse radiation (0.3-0.7um)
      &  albvisdir(:,:),       & !! UV visible albedo for direct radiation  (0.3-0.7um)
      &  albnirdif(:,:),       & !! near IR albedo for diffuse radiation    (0.7-5.0um)
      &  albnirdir(:,:),       & !! near IR albedo for direct radiation     (0.7-5.0um)
      &  albdif_t(:,:,:),      & !! tile-based shortwave albedo for diffuse radiation  (0.3-5.0um)
      &  albvisdif_t(:,:,:),   & !! tile-based UV visible albedo for diffuse radiation (0.3-0.7um)
      &  albnirdif_t(:,:,:),   & !! tile-based near IR albedo for diffuse radiation (0.3-0.7um)
      &  lw_emiss(:,:),        & !! Longwave emissivity with corrections for deserts and snow cover
      &  snowalb_fac(:,:),     & !! Factor for adaptive snow albedo tuning (coupled to DA increments for T)
      &  landalb_inc(:,:),     & !! Increment for adaptive land albedo tuning
      &  heatcond_fac(:,:),    & !! Factor for adaptive soil heat conductivity tuning (coupled to DA increments for T)
      &  heatcap_fac(:,:),     & !! Factor for adaptive soil heat capacity tuning (coupled to DA increments for T)
      &  hydiffu_fac(:,:),     & !! Factor for adaptive tuning of hydraulic diffusivity
      &  snowfrac_fac(:,:),    & !! Factor for adaptive tuning of snow-cover fraction diagnosis
      &  sfcfric_fac(:,:),     & !! Factor for adaptive surface friction tuning (coupled to DA increments for V_abs)
      &  hflux_si_fac(:,:),    & !! Factor for adaptive tuning of seaice bottom heat flux (coupled to DA increments for T)
      &  vio3(:,:),            & !! vertically integrated ozone amount (Pa O3)
      &  hmo3(:,:),            & !! height of O3 maximum (Pa)
      &  flxdwswtoa(:,:),      & !! downward shortwave flux at TOA [W/m2]
      &  tsfctrad(:,:),        & !! surface temperature at trad [K]

      &  lwflx_up(:,:,:),      & !! longwave  3D upward   flux            [W/m2]
      &  lwflx_dn(:,:,:),      & !! longwave  3D downward flux            [W/m2]
      &  swflx_up(:,:,:),      & !! shortwave 3D upward   flux            [W/m2]
      &  swflx_dn(:,:,:),      & !! shortwave 3D downward flux            [W/m2]
      &  lwflx_up_clr(:,:,:),  & !! longwave  3D upward   flux clear-sky  [W/m2]
      &  lwflx_dn_clr(:,:,:),  & !! longwave  3D downward flux clear-sky  [W/m2]
      &  swflx_up_clr(:,:,:),  & !! shortwave 3D upward   flux clear-sky  [W/m2]
      &  swflx_dn_clr(:,:,:),  & !! shortwave 3D downward flux clear-sky  [W/m2]

      &  lwflxall(:,:,:),      & !! longwave net flux           [W/m2]
      &  lwflxsfc(:,:),        & !! longwave net flux at surface [W/m2]
      &  lwflx_up_sfc(:,:),    & !! longwave upward flux at surface [W/m2]
      &  lwflx_up_sfc_rs(:,:), & !! longwave upward flux at surface calculated at radiation time steps [W/m2]
      &  lwflxsfc_t(:,:,:),    & !! tile-based longwave net flux at surface [W/m2]
      &  trsolall(:,:,:),      & !! shortwave net transmissivity (i.e. net flux normalized by irradiance) []
      &  trsolclr_sfc(:,:),    & !! clear-sky shortwave net transmissivity at the surface
      &  swflxclr_sfc(:,:),    & !! clear-sky shortwave net flux at the surface
      &  lwflxclr_sfc(:,:),    & !! clear-sky longwave net flux at the surface
      &  trsol_up_toa(:,:),    & !! shortwave upward transmissivity at the top of the atmosphere
      &  trsol_up_sfc(:,:),    & !! shortwave upward transmissivity at the surface
      &  trsol_nir_sfc(:,:),   & !! downward near-infrared transmissivity at the surface
      &  trsol_vis_sfc(:,:),   & !! downward visible transmissivity at the surface
      &  trsol_par_sfc(:,:),   & !! downward photosynthetically active transmissivity at the surface
      &  trsol_dn_sfc_diff(:,:),& !! shortwave diffuse downward radiative transmissivity at the surface
      &  swflx_up_toa(:,:),    & !! shortwave upward flux at the top of the atmosphere [W/m2]
      &  swflx_up_sfc(:,:),    & !! shortwave upward flux at the surface [W/m2]
      &  swflx_up_sfc_os(:,:), & !! shortwave upward flux at the surface incl. orographic shading [W/m2]
      &  swflx_up_sfc_tan_os(:,:), & !! shortwave upward flux at the surface incl. slope-dependent and orographic shading [W/m2]
      &  swflx_nir_sfc(:,:),   & !! shortwave downward near-infrared flux at the surface [W/m2]
      &  swflx_vis_sfc(:,:),   & !! shortwave downward visible flux at the surface [W/m2]
      &  swflx_par_sfc(:,:),   & !! shortwave downward photosynthetically active flux at the surface [W/m2]
      &  swflx_par_sfc_tan_os(:,:), & !! shortwave downward photosynthetically active flux at the surface incl. slope-dependent and orographic shading [W/m2] 
      &  fr_nir_sfc_diff(:,:), & !! diffuse fraction of downward near-infrared flux at the surface
      &  fr_vis_sfc_diff(:,:), & !! diffuse fraction of downward visible flux at the surface
      &  fr_par_sfc_diff(:,:), & !! diffuse fraction of downward photosynthetically active flux at the surface
      &  aswflx_par_sfc(:,:),  & !! shortwave downward photosynthetically active flux at the surface [W/m2]
      &  aswflx_par_sfc_tan_os(:,:),  & !! shortwave downward photosynthetically active flux at the surface [W/m2]
                                 !! accumulated or mean since model start
      &  swflx_dn_sfc_diff(:,:),& !! shortwave diffuse downward radiative flux at the surface [W/m2]
      &  swflxsfc(:,:),        & !! shortwave net flux at surface [W/m2]
      &  swflxsfc_os(:,:),     & !! shortwave net flux at surface incl. orographic shading [W/m2]
      &  swflxsfc_tan_os(:,:), & !! shortwave net flux at surface incl. slope-dependent and orographic shading [W/m2]
      &  swflxsfc_t(:,:,:),    & !! tile-based shortwave net flux at surface [W/m2]
      &  swflxtoa(:,:),        & !! shortwave net flux at toa [W/m2]
      &  lwflxtoa(:,:),        & !! thermal net flux at toa [W/m2]
      &  lwflxsfc_a(:,:),      & !! Surface net thermal radiation [W/m2], accumulated or mean since model start
      &  swflxsfc_a(:,:),      & !! Surface net solar radiation [W/m2], accumulated or mean since model start
      &  swflxsfc_a_os(:,:),   & !! Surface net solar radiation incl. orographic shading [W/m2], accumulated or mean since model start
      &  swflxsfc_a_tan_os(:,:),& !! Surface net solar radiation incl. slope-dependent and orographic shading [W/m2], accumulated or mean since model start
      &  lwflxclrsfc_a(:,:),   & !! Clear-sky surface net thermal radiation [W/m2], accumulated or mean since model start
      &  swflxclrsfc_a(:,:),   & !! Clear-sky surface net solar radiation [W/m2], accumulated or mean since model start
      &  lwflxtoa_a(:,:),      & !! TOA net thermal radiation [W/m2], accumulated or mean since model start
      &  swflxtoa_a(:,:),      & !! shortwave net flux at toa [W/m2], accumulated or mean since model start
      &  dursun_m(:,:),        & !! maximum duration of sunshine [s]
      &  dursun_r(:,:),        & !! relative duration of sunshine [s]
      &  dursun(:,:),          & !! duration of sunshine [s]
      &  asod_t    (:,:),      & !! Top down solar radiation  [W/m2], accumulated or mean since model start
      &  asou_t    (:,:),      & !! Top up solar radiation  [W/m2], accumulated or mean since model start
      &  athd_s    (:,:),      & !! Surface down thermal radiation [W/m2], accumulated or mean since model start
      &  athu_s    (:,:),      & !! Surface up thermal radiation [W/m2], accumulated or mean since model start
      &  asod_s    (:,:),      & !! Surface down solar rad. [W/m2], accumulated or mean since model start 
      &  asod_s_os    (:,:),     & !! Surface down solar rad. uncorr. [W/m2], accumulated or mean since model start 
      &  asod_s_tan_os    (:,:), & !! Surface down solar rad. uncorr. [W/m2], accumulated or mean since model start 
      &  asodird_s (:,:),      & !! Surface down solar direct rad. [W/m2], accumulated or mean since model start 
      &  asodird_s_os (:,:),   & !! Surface down solar direct rad. incl. orographic shading [W/m2], accumulated or mean since model start
      &  asodird_s_tan_os (:,:),& !! Surface down solar direct rad. incl. slope-dependent and orographic shading [W/m2], accumulated or mean since model start
      &  asodifd_s (:,:),      & !! Surface down solar diff. rad. [W/m2], accumulated or mean since model start 
      &  asodifu_s (:,:),      & !! Surface up solar diff. rad. [W/m2], accumulated or mean since model start 
                                 !! _a means average values if lflux_avg=.TRUE.
                                 !! and accumulated values if lflux_avg=.FALSE., default is .FALSE.
      &  asodifu_s_os(:,:),    & !! Surface up solar diff. rad. incl. orographic shading [W/m2], accumulated or mean since model start 
      &  asodifu_s_tan_os(:,:),& !! Surface up solar diff. rad. incl. slope-dependent and orographic shading [W/m2], accumulated or mean since model start 
      &  snowlmt     (:,:),    & !! height of snowfall limit above MSL
      &  drag_u_grid (:,:),    & !! zonal resolved surface stress [N/m2]
      &  drag_v_grid (:,:),    & !! meridional resolved surface stress [N/m2]
      &  adrag_u_grid(:,:),    & !! zonal resolved surface stress, accumulated or mean since model start
      &  adrag_v_grid(:,:),    & !! meridional resolved surface stress, accumulated or mean since model start
      &  str_u_sso   (:,:),    & !! zonal sso surface stress [N/m2]
      &  str_v_sso   (:,:),    & !! meridional sso surface stress [N/m2]
      &  astr_u_sso  (:,:),    & !! zonal sso surface stress, accumulated or mean since model start
      &  astr_v_sso  (:,:),    & !! meridional sso surface stress, accumulated or mean since model start
      &  lhn_diag (:,:,:),     & !! diagnostic output fields of LHN
      &  tt_lheat (:,:,:),     & !! latent heat release
      &  ttend_lhn (:,:,:),    & !! temperature increment of LHN
      &  qvtend_lhn (:,:,:),   & !! moisture increment of LHN
      &  qrs_flux (:,:,:),     & !! precipitation flux
      &  mf_b(:,:),            & !! bulk cloud-base mass-flux  
      &  mf_p(:,:),            & !! perturbed cloud-base mass-flux 
      &  mf_num(:,:)             !! number of clouds per grid box


    REAL(vp), POINTER, CONTIGUOUS :: &
      &  qc_sgs(:,:,:)           !! subgrid-scale cloud water from cloud cover diagnostic

    !> Precipitation fields
    REAL(wp), POINTER, CONTIGUOUS :: &
      !  Instantaneuous precipitation rates [kg/m2/s]
      !  grid scale
      &  rain_gsp_rate    (:,:),  & !! grid-scale surface rain rate                    [kg/m2/s]
      &  snow_gsp_rate    (:,:),  & !! grid_scale surface snow rate                    [kg/m2/s]
      &  ice_gsp_rate     (:,:),  & !! grid_scale surface ice rate                     [kg/m2/s]
      &  graupel_gsp_rate (:,:),  & !! grid_scale surface graupel rate                 [kg/m2/s]
      &  hail_gsp_rate    (:,:),  & !! grid_scale surface hail rate                    [kg/m2/s]
      !  convective
      &  rain_con_rate_corr(:,:), & !! convective surface rain rate (water-conserving) [kg/m2/s]
      &  rain_con_rate    (:,:),  & !! convective surface rain rate (next time step)   [kg/m2/s]
      &  snow_con_rate_corr(:,:), & !! convective surface snow_rate (water-conserving) [kg/m2/s]
      &  snow_con_rate    (:,:),  & !! convective surface snow rate (next time step)   [kg/m2/s]
      &  rain_con_rate_3d (:,:,:),& !! 3d convective rain rate (convection scheme)     [kg/m2/s]
      &  snow_con_rate_3d (:,:,:),& !! 3d convective snow_rate (convection scheme)     [kg/m2/s]
      !
      ! Instantaneous grid scale precipitation rate [kg/m2/s] (sum over gsp hydromets):
      &  prec_gsp_rate    (:,:),  & !! total surface precipitation rate                [kg/m2/s]
      !
      ! Instantaneous total precipitation rate [kg/m2/s] (sum of gsp + con hydromets):
      &  tot_prec_rate    (:,:),  & !! total surface precipitation rate                [kg/m2/s]
      !
      !  Integrated instantaneous rates since model start (precipitation amount) [kg/m2]
      !  grid scale
      &  rain_gsp         (:,:),  & !! accumulated grid-scale surface rain             [kg/m2]
      &  snow_gsp         (:,:),  & !! accumulated grid_scale surface snow             [kg/m2]
      &  ice_gsp          (:,:),  & !! accumulated grid_scale surface ice              [kg/m2]
      &  hail_gsp         (:,:),  & !! accumulated grid_scale surface hail             [kg/m2]
      &  graupel_gsp      (:,:),  & !! accumulated grid_scale surface graupel          [kg/m2]
      &  prec_gsp         (:,:),  & !! accumulated grid scale precipitation            [kg/m2]
      &  prec_gsp_d       (:,:),  & !! accumulated grid scale precipitation            [kg/m2]
                                    !!  (reset after "tpotprec_d_interval" seconds)
      !  convective
      &  rain_con         (:,:),  & !! accumulated convective surface rain             [kg/m2]
      &  snow_con         (:,:),  & !! accumulated convective surface snow             [kg/m2]
      &  prec_con         (:,:),  & !! accumulated convective precipitation            [kg/m2]
      &  prec_con_d       (:,:),  & !! accumulated convective precipitation            [kg/m2]
                                    !!  (reset after "tpotprec_d_interval" seconds)
      !  total
      &  tot_prec         (:,:),  & !! accumulated total precipitation                 [kg/m2]
                                    !!  (grid-scale plus convective)
      &  tot_prec_d       (:,:),  & !! accumulated total precipitation over a time interval [kg/m2]
                                    !!  (grid-scale plus convective; reset after "tpotprec_d_interval" seconds)
      !
      !  Time averaged precipitation rates since model start [kg/m2/s]
      &  prec_con_rate_avg(:,:),  & !! time averaged convective precipitation rate    [kg/m2/s]
      &  prec_gsp_rate_avg(:,:),  & !! time averaged grid-scale precipitation rate    [kg/m2/s]
      &  tot_prec_rate_avg(:,:),  & !! time averaged total precipitation rate         [kg/m2/s]

      !  Auxiliary variables for ww_diagnostics
      !  Precipitation variables *0 are accumulated only to the previous call of ww_diagnostics
      &  rain_gsp0        (:,:),  & !! accumulated grid-scale surface rain            [kg/m2]
      &  snow_gsp0        (:,:),  & !! accumulated grid_scale surface snow            [kg/m2]
      &  rain_con0        (:,:),  & !! accumulated convective surface rain            [kg/m2]
      &  snow_con0        (:,:)     !! accumulated convective surface snow            [kg/m2]


    !> Parameter fields for turbulence
    REAL(wp), POINTER, CONTIGUOUS :: &
      rcld(:,:,:)     ,    & !> standard deviation of the saturation deficit    --
      tcm(:,:)        ,    & !! turbulent transfer coefficients for momentum    --
      tch(:,:)        ,    & !! turbulent transfer coefficients for heat        --
      tfm(:,:)        ,    & !! factor of laminar transfer of momentum          --
      tfh(:,:)        ,    & !! factor of laminar transfer of scalars           --
      tfv(:,:)        ,    & !! laminar reduction factor for evaporation        --
      tvm(:,:)        ,    & !! turbulent transfer velocity for momentum      (m/s)
      tvh(:,:)        ,    & !! factor of laminar transfer of scalars           --
      tkred_sfc(:,:)  ,    & !! reduction factor for minimum diffusion coefficients near the surface
                             !! (affects heat and momentum; used for EPS perturbations)
      tkred_sfc_h(:,:),    & !! reduction factor for minimum diffusion coefficient for heat near the surface
                             !! (used for model-DA coupling)
      pat_len(:,:)    ,    & !! length scale of sub-grid scale roughness elements (m)
      rlamh_fac_t(:,:,:),  & !! tuning factor for laminar transfer resistance (rlam_heat)
      gz0(:,:),            & !! roughness length * g of the vertically not
                             !! resolved canopy                               (m2/s2)
      z0_waves(:,:),       & !! wave-dependent roughness length               (  m  )
      tkvm(:,:,:),         & !! turbulent diffusion coefficients for momentum (m/s2 )
      tkvh(:,:,:),         & !! turbulent diffusion coefficients for heat     (m/s2 )
      tprn(:,:,:),         & !! turbulent Prandtl-number                        --   
      t_2m(:,:)       ,    & !! temperature in 2m                             (  K  )
      t_2m_land(:,:)  ,    & !! temperature in 2m (land tiles only)           (  K  )
      tmax_2m(:,:)    ,    & !! maximum temperature in 2m (for specified timerange) ( K )
      tmin_2m(:,:)    ,    & !! minimum temperature in 2m (for specified timerange) ( K )
      t_tilemax_inst_2m(:,:), & !! instantaneous 2m temperature; maximum over tiles (  K  )
      t_tilemin_inst_2m(:,:), & !! instantaneous 2m temperature; minimum over tiles (  K  )
      qv_2m (:,:)     ,    & !! specific water vapor content in 2m            (kg/kg)
      td_2m (:,:)     ,    & !! dew-point in 2m                               (  K  )
      rh_2m (:,:)     ,    & !! relative humidity in 2m                       (  %  )
      td_2m_land (:,:),    & !! dew-point in 2m (land tiles only)             (  K  )
      rh_2m_land (:,:),    & !! relative humidity in 2m  (land tiles only)    (  %  )
      u_10m (:,:)     ,    & !! zonal wind in 10m                             ( m/s )
      v_10m (:,:)     ,    & !! meridional wind in 10m                        ( m/s )
      u_10m_a (:,:)   ,    & !! time-averaged zonal wind in 10m               ( m/s )
      v_10m_a (:,:)   ,    & !! time-averaged meridional wind in 10m          ( m/s )
      tcm_a (:,:)     ,    & !! time-averaged momentum transfer coefficient   ( --  )
      gust_lim(:,:)   ,    & !! upper limit on gust speed                     ( m/s )
      sp_10m(:,:)     ,    & !! wind speed in 10m                             ( m/s )
      dyn_gust(:,:)   ,    & !! dynamic gust at 10m                           ( m/s )
      gust10(:,:)     ,    & !! max. gust at 10m                              ( m/s )
      edr   (:,:,:)    ,   & !! eddy dissipation rate
      tcm_t(:,:,:)     ,   & !! turbulent transfer coefficients for momentum    --
      tch_t(:,:,:)     ,   & !! turbulent transfer coefficients for heat        --
      tfv_t(:,:,:)     ,   & !! laminar reduction factor for evaporation        --
      tvm_t(:,:,:)     ,   & !! turbulent transfer velocity for momentum      ( m/s )
      tvh_t(:,:,:)     ,   & !! turbulent transfer velocity for heat          ( m/s )
      tkr_t(:,:,:)     ,   & !! turbulent reference surface diffusion coeff.  ( m2/s) (Ustar*kap*z0)
      gz0_t(:,:,:)     ,   & !! roughness length * g                          (m2/s2)
      tvs_s_t(:,:,:)   ,   & !! surface turbulence velocity scale (SQRT(2*TKE)) (m/s)
                             !! (tile based)
      tkvm_s_t(:,:,:)  ,   & !! surface turbulent diffusion coefficients for momentum (m/s2)
                             !! (tile based)
      tkvh_s_t(:,:,:)  ,   & !! surface turbulent diffusion coefficients for heat (m/s2)
                             !! (tile based)
      rcld_s_t(:,:,:)  ,   & !! standard deviation of the saturation deficit at surface --
                             !! (tile based)
      u_10m_t(:,:,:)   ,   & !! zonal wind at 10m                             ( m/s )
      v_10m_t(:,:,:)   ,   & !! meridional wind at 10m                        ( m/s )
      umfl_s_t(:,:,:)  ,   & !! u-momentum flux at the surface (tile based)    (N/m2)
      vmfl_s_t(:,:,:)  ,   & !! v-momentum flux at the surface (tile based)    (N/m2)
      umfl_s(:,:)      ,   & !! u-momentum flux at the surface                 (N/m2)
      vmfl_s(:,:)      ,   & !! v-momentum flux at the surface                 (N/m2)
      aumfl_s(:,:)     ,   & !! u-momentum flux at the surface (N/m2), accumulated or mean since model start
      avmfl_s(:,:)     ,   & !! v-momentum flux at the surface (N/m2), accumulated or mean since model start
                             !! a means average values if lflux_avg=.TRUE.
                             !! and accumulated values if lflux_avg=.FALSE., default is .FALSE.
      qcfl_s(:,:)      ,   & !! cloud water turbulent deposition flux         (kg/m2/s)
      qifl_s(:,:)      ,   & !! cloud ice turbulent deposition flux           (kg/m2/s)
      reff_qc(:,:,:)   ,   & !! effective radius of cloud water               (m)
      reff_qi(:,:,:)   ,   & !! effective radius of cloud ice                 (m)
      reff_qr(:,:,:)   ,   & !! effective radius of cloud rain                (m)
      reff_qs(:,:,:)   ,   & !! effective radius of cloud snow                (m)
      reff_qg(:,:,:)   ,   & !! effective radius of cloud graupel             (m)
      reff_qh(:,:,:)         !! effective radius of cloud hail                (m)

    REAL(wp) :: prev_v10mavg_reset  !! storage for previous reset of averaged v10m field

    !> Diagnostics for LES turbulence
    REAL(wp), POINTER, CONTIGUOUS :: &
      z_pbl(:,:)     ,     & !> Boundary layer height  (m) (LES)
      bruvais(:,:,:) ,     & !> Brunt Vaisala Frequency
      mech_prod(:,:,:),    & !> Mechanical production/loss term in TKE equation
      t_cbase(:,:),        & !>cloud base temperature
      p_cbase(:,:),        & !>cloud base pressure
      t_ctop(:,:),         & !>cloud top temperature
      p_ctop(:,:) !         & !>cloud top pressure
      !cld_opt_thck(

    ! time-interpolated values for Tegen aerosol climatology (needed as state fields for coupling with microphysics and convection)
    REAL(wp), POINTER, CONTIGUOUS :: &
      & pref_aerdis(:,:),   &
      & aercl_ss  (:,:),    &
      & aercl_or  (:,:),    &
      & aercl_bc  (:,:),    &
      & aercl_su  (:,:),    &
      & aercl_du  (:,:),    &
      & aerosol   (:,:,:)

    INTEGER, POINTER, CONTIGUOUS :: &
      &  mbas_con(:,:),     & !< cloud base level index
      &  mtop_con(:,:),     & !< cloud top  level index
      &  ktype   (:,:),     & !< Type of convection
      &  k650    (:,:),     & !< level index that corrsponds to the height
                              !< of the standard atmosphere 650hPa level above ground
      &  k850    (:,:),     & !< level index that corrsponds to the height 
                              !< of the standard atmosphere 850hPa level above ground
      &  k950    (:,:),     & !< level index that corresponds to the height 
                              !< of the standard atmosphere 950hPa level above ground
      &  k800    (:,:),     & !< level index that corresponds to the height 
                              !< of the standard atmosphere 800hPa level above ground
      &  k400    (:,:),     & !< level index that corresponds to the height 
                              !< of the standard atmosphere 400hPa level above ground
      &  k700    (:,:),     & !< level index that corresponds to the height 
                              !< of the standard atmosphere 700hPa level above ground
      &  ktop_envel(:,:),   & !< level index of upper boundary of SSO envelope layer
      &  iww     (:,:),     & !< significant weather
      &  wup_mask(:,:)        ! mask for tracking of strong updrafts

    REAL(wp), POINTER :: tropics_mask(:,:)      !< mask field that is 1 in the tropics and 0 in the extratropics
    REAL(wp), POINTER :: innertropics_mask(:,:) !< mask field that is 1 in the inner tropics and 0 elsewhere
    REAL(wp), POINTER :: sso_lat_mask(:,:)      !< mask field that is used for latitude-dependent SSO tuning parameters

    LOGICAL, POINTER, CONTIGUOUS :: &
      & locum     (:,:),    & !< convective  activity indicator
      & ldshcv    (:,:)       !< shallow convection indicator




    !> (Optional:) Additional diagnostic fields:
    REAL(wp), POINTER ::   &
      rh(:,:,:),           & !> relative humidity
      pv(:,:,:),           & !> potential vorticity
      sdi2(:,:),           & !> supercell detection index (SDI2)
      dhail(:,:,:),        & !> expected hail diameter at the ground
      dhail_mx(:,:),       & !> maximum expected hail diameter at the ground
      dhail_av(:,:),       & !> average expected hail diameter at the ground
      dhail_sd(:,:),       & !> standard deviation of hail diameter at the ground
      wdur(:,:),           & !> duration of strong updraft in a grid column
      lpi(:,:),            & !> lightning potential index (LPI)
      lpi_max(:,:),        & !> lightning potential index, maximum (LPI_MAX)
      koi(:,:),            & !> KOI (stability measure - equivalent potential temperature difference
      lpi_con(:,:),        & !> LPI computed with convection scheme variables
      lpi_con_max(:,:),    & !> Maximum of LPI
      mlpi_con(:,:),       & !> modified LPI (making use of KOI)
      mlpi_con_max(:,:),   & !> maximum of modified LPI (making use of KOI)
      lfd_con(:,:),        & !> lightening flash density computed with convection scheme variables
      lfd_con_max(:,:),    & !> maximum of LFD
      ceiling_height(:,:), & !> ceiling height
      vis(:,:),            & !> near surface visibility [meters]
      inversion_height(:,:),& !> lowest inversion height
      low_ent_zone(:,:),   &  !> entreinment zone from lowest inversion 
      hbas_sc(:,:),        & !> height of base above MSL from shallow convection parameterization
      htop_sc(:,:),        & !> height of top  above MSL from shallow convection parameterization
      twater(:,:),         & !> Total column integrated water
      q_sedim(:,:,:),      & !> Specific content of precipitation particles
      mconv(:,:),          & !> Low level horizontal moisture convergence (0-1000 m AGL average) div.(q_v*v_h) [1/s]
      tcond_max(:,:),      & !< Total column-integrated condensate
      tcond10_max(:,:),    & !< Total column-integrated condensate above z(T=-10 degC) 
      uh_max_3d(:,:,:),    & !< Updraft helicity (integrated over different vertical layers)
      vorw_ctmax(:,:),     & !< Maximum low level rotation amplitude: Time-max amplitude (positive or negative) of mean 0-3000 m MSL (or 1500 m AGL, whichever is higher) vorticity
      w_ctmax(:,:),        & !< Maximum updraft track
      dbz3d_lin(:,:,:),    & !< Radar reflectivity 3D in linear units mm^6/m^3
      dbz_850(:,:),        & !< Radar reflectivity in approx. 850 hPa
      dbzlmx_low(:,:),     & !< Radar reflectivity layer maximum [500,2500] m AGL
      dbz_cmax(:,:),       & !< Column maximum radar reflectivity
      dbz_ctmax(:,:),      & !< Column and time maximum radar reflectivity
      echotop(:,:,:),      & !< Echotop pressure in p
      echotopinm(:,:,:),   & !< Echotop altitude in m MSL
      wshear_u(:,:,:),     & !< U-component of vertical wind shear vector between some heights AGL and lowest model level
      wshear_v(:,:,:),     & !< V-component of vertical wind shear vector between some heights AGL and lowest model level
      lapse_rate(:,:),     & !< T(500hPa) - T(850hPa) with a correction if 850 hPa is below the surface
      cape_mu (:,:),       & !< Most unstable convective available energy
      cin_mu(:,:),         & !< Most unstable convective inhibition
      cape_ml (:,:),       & !< convective available energy of mean surface layer parcel
      si      (:,:),       & !< Showalter Index SI
      sli     (:,:),       & !< Surface Lifted Index SLI
      swiss12 (:,:),       & !< SWISS12 Index
      swiss00 (:,:),       & !< SWISS00 Index
      cin_ml  (:,:),       & !< convective inhibition of mean surface layer parcel
      lcl_ml  (:,:),       & !< Lifted Condensation Level of mean surface layer parcel
      lfc_ml  (:,:),       & !< Level of Free Convection of mean surface layer parcel
      cape_3km (:,:),      & !< convective available energy of mean surface layer parcel with endpoint 3km.
      cin_3km(:,:),        & !< convective inhibition of mean surface layer parcel with endpoint 3km.
      cloudtop(:,:),       & !< Cloud Top
      srh(:,:,:),          & !< Storm relative helicity with right-moving storm motion after Bunkers et al. (2000)
      tot_pr_max(:,:),     & !< Time maximum total precipitation rate
      hpbl(:,:),           & !< Boundary layer height  (m)
      aod_550nm(:,:)         !< aerosol optical depth visible 550 nm (spectral band 25)

    ! Buffer field needed when vertical nesting is combined with a reduced radiation
    ! grid and latm_above_top = .TRUE.
    REAL(wp), POINTER :: buffer_rrg(:,:,:)

    ! Buffer field needed for RTTOV calculations on a vertical nested grid
    REAL(wp), POINTER :: buffer_rttov(:,:,:)

    ! pointer to satellite images (all images in one array):
    REAL(wp), POINTER    :: synsat_arr(:,:,:)

    ! pointers to satellite images (list of 2D slices)
    TYPE (t_rttov_image), ALLOCATABLE :: synsat_image(:)

    !> Special 1D and 0D diagnostics for LES runs
    REAL(wp), ALLOCATABLE :: &
      turb_diag_1dvar(:,:), turb_diag_0dvar(:)  

    TYPE(t_nwp_vdiff_state) :: nwp_vdiff_state

    ! vars for global diagnostics based on src/atm_phy_echam/mo_echam_phy_memory.f90
    REAL(wp),POINTER ::       &
      !
      & tas_gmean    (:)=>NULL(),      &!< [K] global mean 2m-temperature
      & rsdt_gmean   (:)=>NULL(),      &!< [W/m2] global mean toa incident shortwave radiation
      & rsut_gmean   (:)=>NULL(),      &!< [W/m2] global mean toa outgoing shortwave radiation
      & rlut_gmean   (:)=>NULL(),      &!< [W/m2] global mean toa outgoing longwave radiation
      & prec_gmean   (:)=>NULL(),      &!< [kg/m2/s] global mean precipitation flux
      & evap_gmean   (:)=>NULL(),      &!< [kg/m2/s] global mean evaporation flux
      & pme_gmean    (:)=>NULL(),      &!< [kg/m2/s] global mean P-E
      & radtop_gmean (:)=>NULL()        !< [W/m2] global mean toa total radiation, derived variable

  END TYPE t_nwp_phy_diag
  !
  ! !---tendencies of type global!
  !
  TYPE t_nwp_phy_tend

    REAL(wp), POINTER, CONTIGUOUS :: &
      ddt_temp_radsw  (:,:,:)  ,& !! Temp-tendency from shortwave radiation
      ddt_temp_radlw  (:,:,:)  ,& !! Temp-tendency from longwave radiation
      ddt_temp_turb   (:,:,:)  ,& !! Temp-tendency from turbulence
      ddt_temp_gscp   (:,:,:)  ,& !! Temp-tendency from microphysics
      ddt_u_turb      (:,:,:)  ,& !! ZonalW-tendency from turbulence
      ddt_u_pconv     (:,:,:)  ,& !! ZonalW-tendency from convective prec
      ddt_v_turb      (:,:,:)  ,& !! MeridW-tendency from turbulence
      ddt_w_turb      (:,:,:)  ,& !! VertW-tendency from turbulence
      ddt_v_pconv     (:,:,:)  ,& !! MeridW-tendency from convective prec
      ddt_tracer_turb (:,:,:,:),& !! Hydromet-tendency from turbulence
      ddt_tracer_pconv(:,:,:,:),& !! Hydromet-tendency from convective prec
      ddt_tke_pconv   (:,:,:)  ,& !! TKE tendency from convective prec
      ddt_tke_hsh     (:,:,:)  ,& !! TKE tendency from horizontal shear
      ddt_tracer_gscp (:,:,:,:),& !! Hydromet-tendency from microphysics
      ddt_tke         (:,:,:)     !! tendency for turbulent velocity scale [m/s^2]

    REAL(vp), POINTER, CONTIGUOUS :: &
      ddt_temp_drag   (:,:,:)  ,& !! Temp-tendency from sso + gravity-wave drag + Rayleigh friction
      ddt_temp_pconv  (:,:,:)  ,& !! Temp-tendency from convective prec
      ddt_temp_clcov  (:,:,:)  ,& !! Temp-tendency from cloud cover scheme
      ddt_u_gwd       (:,:,:)  ,& !! ZonalW-tendency from gravity wave drag
      ddt_u_sso       (:,:,:)  ,& !! ZonalW-tendency from sso drag
      ddt_v_gwd       (:,:,:)  ,& !! MeridW-tendency from gravity wave drag
      ddt_v_sso       (:,:,:)     !! MeridW-tendency from sso drag


    !Anurag Dipankar, MPIM (2013-May-31)
    !Large-scale tendencies for idealized testcases (nlev)
    REAL(wp), ALLOCATABLE ::  &
      ddt_u_ls        (:),    &   !! LS tendency for u 
      ddt_v_ls        (:),    &   !! LS tendency for v 
      ddt_temp_ls     (:),    &   !! LS tendency for temp 
      ddt_tracer_ls   (:,:),  &   !! LS tendency for tracer
      ddt_temp_subs_ls(:),    &   !! Christopher Moseley: 7 LS tendencies for profile output
      ddt_qv_subs_ls  (:),    &   !! LS tendency for water vapor from subsidence
      ddt_temp_adv_ls (:),    &   !! LS tendency for temperature from advection
      ddt_qv_adv_ls   (:),    &   !! LS tendency for water vapor from advection
      ddt_u_adv_ls    (:),    &   !! LS tendency for u-wind from advection
      ddt_v_adv_ls    (:),    &   !! LS tendency for v-wind from advection
      ddt_temp_nud_ls (:),    &   !! LS tendency for temperature from nudging
      ddt_qv_nud_ls   (:),    &   !! LS tendency for water vapor from nudging
      wsub            (:),    &   !! subsidence [m/s]
      temp_nudge      (:),    &   !! T nudging profile [K]
      q_nudge         (:,:),  &   !! qv/qc/qi nudging profile [kg/kg]
      u_nudge         (:),    &   !! u nudging profile [m/s]
      v_nudge         (:)         !! v nudging profile [m/s]

    !variables for surface boundary conditions for SCM cases
    !defined in mo_ls_forcing/apply_ls_forcing, used in mo_nh_torus_exp/set_scm_bnd
    REAL(wp)              ::  &
      fc_sfc_lat_flx,         &   !! latent heat flux
      fc_sfc_sens_flx,        &   !! sensible heat flux
      fc_ts,                  &   !! surface temperature
      fc_tg,                  &   !! ground temperature
      fc_qvs,                 &   !! surface water vapor mixing ratio
      fc_Ch,                  &   !! surface exchange coefficient for heat
      fc_Cq,                  &   !!   ... for water vapor
      fc_Cm,                  &   !!   ... for momentum
      fc_ustar,               &   !! friction velocity
      fc_umfl_s,              &   !! u momentum flux
      fc_vmfl_s                   !! v momentum flux

    !tendencies from simplified radiation for DYCOMS stratocumulus case
    REAL(wp),ALLOCATABLE  ::  &
      ddt_temp_sim_rad(:,:,:)     !! temperature tendency from simplified radiation


    TYPE(t_ptr_2d3d),ALLOCATABLE ::  &
      &  tracer_turb_ptr(:)    ,& !< pointer array: one pointer for each component
      &  tracer_conv_ptr(:)    ,& !< pointer array: one pointer for each component
      &  tracer_gscp_ptr(:)       !< pointer array: one pointer for each component

    TYPE(t_ptr_tracer), ALLOCATABLE :: conv_tracer_tend(:,:) !< pointer for chemical tracer conv. tend.

    TYPE(t_ptr_tracer), ALLOCATABLE :: turb_tracer_tend(:,:) !< pointer for chemical tracer turb. tend.

  END TYPE t_nwp_phy_tend

  TYPE t_nwp_phy_stochconv
     
! Variables for SDE stochastic convection schemes
     REAL(wp), POINTER, CONTIGUOUS :: &
      & clnum_a        (:,:)     ,& ! number density of active convective clouds    ( - )
      & clmf_a         (:,:)     ,& ! cloud-base mass flux for active conv. clouds  (kg/m**2s)
      & clnum_p        (:,:)     ,& ! number density of passive convective clouds   ( - )
      & clmf_p         (:,:)     ,& ! cloud-base mass flux for passive conv. clouds (kg/m**2s)
      & clnum_d        (:,:)     ,& ! number density of deep convective clouds      ( - )
      & clmf_d         (:,:)        ! cloud-base mass flux for deep conv. clouds    (kg/m**2s)
     
! Variables for explicit stochastic convection scheme
     REAL(wp), POINTER, CONTIGUOUS :: &
      & mf_i        (:,:,:)     ,& ! number density of active convective clouds    ( - )
      & time_i      (:,:,:)     ,& ! time since birth of cloud
      & life_i      (:,:,:)     ,& ! expected lifetime of cloud
      & area_i      (:,:,:)     ,& ! cloud-base mass flux for passive conv. clouds (kg/m**2s)
      & type_i      (:,:,:)     ,& ! number density of deep convective clouds      ( - )
      & ktype_i     (:,:,:)        ! cloud-base mass flux for deep conv. clouds    (kg/m**2s)
     
    INTEGER, POINTER, CONTIGUOUS :: &
      & depth_i     (:,:,:)     ,& ! number density of passive convective clouds   ( - )
      & base_i      (:,:,:)     ,& ! cloud-base mass flux for passive conv. clouds (kg/m**2s)
      & used_cell   (:,:,:)        ! number density of deep convective clouds      ( - )

  END TYPE t_nwp_phy_stochconv
    
  TYPE t_ptr_cloud_ensemble
     ! Pointer to variables for explicit stochastic convection scheme
     ! (nproma,nlev) dimension only
     REAL(wp), POINTER, CONTIGUOUS :: &
      & mf_i        (:,:)     ,& ! number density of active convective clouds    ( - )
      & time_i      (:,:)     ,& ! time since birth of cloud
      & life_i      (:,:)     ,& ! expected lifetime of cloud
      & area_i      (:,:)     ,& ! cloud-base mass flux for passive conv. clouds (kg/m**2s)
      & type_i      (:,:)     ,& ! number density of deep convective clouds      ( - )
      & ktype_i     (:,:)        ! cloud-base mass flux for deep conv. clouds    (kg/m**2s)
     
     INTEGER, POINTER, CONTIGUOUS :: &
      & depth_i     (:,:)     ,& ! number density of passive convective clouds   ( - )
      & base_i      (:,:)     ,& ! cloud-base mass flux for passive conv. clouds (kg/m**2s)
      & used_cell   (:,:)        ! number density of deep convective clouds      ( - )

  END TYPE t_ptr_cloud_ensemble

  
END MODULE mo_nwp_phy_types
