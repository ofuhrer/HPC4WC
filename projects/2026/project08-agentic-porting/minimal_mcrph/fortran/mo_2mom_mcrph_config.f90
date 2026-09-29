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

! @brief Setup for 2-moment cloud microphysics scheme
!
! configuration setup for synthetic radar data on the model grid

MODULE mo_2mom_mcrph_config
  
  USE mo_kind, ONLY: wp

  IMPLICIT NONE
  PUBLIC

  !--------------------------------------------------------------------------
  ! Namelist parameters
  !--------------------------------------------------------------------------

  ! Container for some configuration parameters of the Seifert-Beheng 2-moment cloud microphysical scheme,
  ! which may be changed by namelist:
  TYPE t_cfg_2mom

    !-----------------------
    ! .. General parameters:
    !-----------------------
    INTEGER  :: i2mom_solver ! 0) explicit (1) semi-implicit solver

    !---------------------------------
    ! .. Parameters for cloud droplets:
    !----------------------------------
    INTEGER  :: ccn_type     ! if not set by namelist, the ccn_type_gscp4 or ccn_type_gscp5 will win
    REAL(wp) :: ccn_Ncn0     ! CN concentration at ground
    REAL(wp) :: ccn_wcb_min  ! min updraft speed for Segal&Khain cloud nucleation
    REAL(wp) :: tune_sbmccn  ! < 1: maritime CCN case, >= 1: continental CCN case for ccn_type 8 (Segal&Khain)

    !------------------------
    ! .. Parameters for rain:
    !------------------------
    LOGICAL  :: luse_mu_Dm_rain ! if the mu-Dm-Relation of Seifert (2008) should be applied outside the cloud cores
    REAL(wp) :: nu_r         ! nu for rain, N(x) = N0 * D^nu * exp(-lambda*x^mu). 
                             ! This nu is used incloud for luse_mu_Dm_rain = .true. or everywhere when .false.
    REAL(wp) :: rain_cmu0    ! asymptotic mue-value for small D_m in the mu-Dm-Relation of Seifert (2008)
    REAL(wp) :: rain_cmu1    ! asymptotic mue-value for large D_m in the mu-Dm-Relation of Seifert (2008)
    REAL(wp) :: rain_cmu3    ! D_br: equilibrium diameter for breakup and selfcollection
    REAL(wp) :: rain_cmu4       

    !-----------------------------
    ! .. Parameters for cloud ice:
    !-----------------------------
    REAL(wp) :: nu_i      ! nu for ice, N(x) = N0 * D^nu * exp(-lambda*x^mu)
    REAL(wp) :: mu_i      ! mu for ice, N(x) = N0 * D^nu * exp(-lambda*x^mu)
    REAL(wp) :: ageo_i    ! ageo for ice, D = ageo*x^bgeo
    REAL(wp) :: bgeo_i    ! bgeo for ice, D = ageo*x^bgeo
    REAL(wp) :: avel_i    ! avel for ice, v = avel*x^bvel
    REAL(wp) :: bvel_i    ! bvel for ice, v = avel*x^bvel
    REAL(wp) :: cap_ice   ! capacitance for ice deposition/sublimation
    REAL(wp) :: in_fact   ! factor for tuning IN concentration for heterogenous ice nucleation

    !------------------------
    ! .. Parameters for snow:
    !------------------------
    REAL(wp) :: nu_s      ! nu for snow, N(x) = N0 * D^nu * exp(-lambda*x^mu)
    REAL(wp) :: mu_s      ! mu for snow, N(x) = N0 * D^nu * exp(-lambda*x^mu)
    REAL(wp) :: ageo_s    ! ageo for snow, D = ageo*x^bgeo
    REAL(wp) :: bgeo_s    ! bgeo for snow, D = ageo*x^bgeo
    REAL(wp) :: avel_s    ! avel for snow, v = avel*x^bvel
    REAL(wp) :: bvel_s    ! bvel for snow, v = avel*x^bvel
    REAL(wp) :: cap_snow  ! capacitance for snow deposition/sublimation
    REAL(wp) :: vsedi_max_s ! max fallspeed limit for snow

    !------------------------
    ! .. Parameters for graupel:
    !------------------------
    REAL(wp) :: nu_g      ! nu for graupel, N(x) = N0 * D^nu * exp(-lambda*x^mu)
    REAL(wp) :: mu_g      ! mu for graupel, N(x) = N0 * D^nu * exp(-lambda*x^mu)
    REAL(wp) :: ageo_g    ! ageo for graupel, D = ageo*x^bgeo
    REAL(wp) :: bgeo_g    ! bgeo for graupel, D = ageo*x^bgeo
    REAL(wp) :: avel_g    ! avel for graupel, v = avel*x^bvel
    REAL(wp) :: bvel_g    ! bvel for graupel, v = avel*x^bvel
    REAL(wp) :: melt_g_tune_fak ! Factor multiplying melting of graupel

    !------------------------
    ! .. Parameters for hail:
    !------------------------
    REAL(wp) :: nu_h      ! nu for hail, N(x) = N0 * D^nu * exp(-lambda*x^mu)
    REAL(wp) :: mu_h      ! mu for hail, N(x) = N0 * D^nu * exp(-lambda*x^mu)
    REAL(wp) :: ageo_h    ! ageo for hail, D = ageo*x^bgeo
    REAL(wp) :: bgeo_h    ! bgeo for hail, D = ageo*x^bgeo
    REAL(wp) :: avel_h    ! avel for hail, v = avel*x^bvel
    REAL(wp) :: bvel_h    ! bvel for hail, v = avel*x^bvel
    REAL(wp) :: melt_h_tune_fak ! Factor to increase/decrease hail melting rate

    !------------------------------------------
    ! .. Parameters for conversions/collisions:
    !------------------------------------------
    LOGICAL  :: lturb_enhc   ! Enhancement of collisons by turbulence (only for warm microphysics)
    REAL(wp) :: turb_len     ! lturb_len:  Turbulent lenght scale (dummy, later overtaken by the TKE scheme, NOT a namelist parameter)
    INTEGER  :: iice_stick    ! Formulation for sticking efficiency of cloud ice
    INTEGER  :: isnow_stick   ! Formulation for sticking efficiency of snow
    INTEGER  :: iparti_stick  ! Formulation for sticking efficiency of frozen inter-categorical collisions
    REAL(wp) :: alpha_spacefilling  !..factor involved in the conversion of ice/snow to graupel by riming
    REAL(wp) :: D_conv_ii    ! D-threshold for conversion to snow ice_selfcollection [m]
    REAL(wp) :: D_rainfrz_ig ! rain --> ice oder graupel [m]
    REAL(wp) :: D_rainfrz_gh ! rain --> graupel oder hail [m]
    REAL(wp) :: ecoll_gg        ! Collision efficiency for graupel autoconversion (dry graupel)
    REAL(wp) :: ecoll_gg_wet    ! Collision efficiency for graupel autoconversion (wet graupel)
    REAL(wp) :: Tcoll_gg_wet ! Temperature threshold for switching to wet graupel autoconversion
    INTEGER  :: iicephase    ! (0) warm-phase 2M, (1) mixed-phase 2M
    INTEGER  :: itype_shedding_gh ! Choice of shedding parameterization during collisions of graupel and hail with water droplets: 0=off, 1=simple, 2=more physical
    REAL(wp) :: D_shed_gh    ! Shedding happens if:
                             ! itype_shedding_gh = 1: D_meanmass > D_shed_gh
                             ! itype_shedding_gh = 2: in the PSD-part where D > MAX(D_wetgr,D_shed_gh) - that is
                             !                        for wet growth but not below the Rasmussen&Heymsfield stable diameter
    ! .. Parameters for the limitation of graupel production by riming of ice/snow:
    REAL(wp) :: Tmax_gr_rime    ! Allow formation of graupel by riming ice/snow only at T < this threshold [K]
    LOGICAL  :: llim_gr_prod_rain_riming ! whether to limit the graupel production by rain riming of ice/snow by
                             ! a bulk-density-based criterion on the mean-mass-particles
    REAL(wp) :: wgt_D_coll_limgrprod ! weight for the collided mean-mass-particle's diameter D_coll: how much does the
                                     ! smaller collision partner contribute to the overall diameter?
    REAL(wp) :: wgt_rho_coll_limgrprod ! weight for the limit of the collided mean-mass-particle's bulk density:
                             ! how near should it be to the bulk density of graupel in order to convert it to graupel?

  END TYPE t_cfg_2mom


END MODULE mo_2mom_mcrph_config
