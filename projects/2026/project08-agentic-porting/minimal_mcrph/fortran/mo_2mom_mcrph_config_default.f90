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

! Setup for 2-moment cloud microphysics scheme
!
! default configuration setup for synthetic radar data on the model grid

MODULE mo_2mom_mcrph_config_default
  
  USE mo_kind, ONLY: wp
  USE mo_2mom_mcrph_config, ONLY: t_cfg_2mom

  IMPLICIT NONE
  PUBLIC

  !--------------------------------------------------------------------------
  ! Namelist parameters
  !--------------------------------------------------------------------------

  !.. Type instance to hold the defaults for the config params:
  TYPE(t_cfg_2mom), PARAMETER :: cfg_2mom_default = t_cfg_2mom ( &

       !-----------------------
       ! .. General parameters:
       !-----------------------
       &            1, & ! i2mom_solver: 0) explicit (1) semi-implicit solver

       !----------------------------------
       ! .. Parameters for cloud droplets:
       !----------------------------------
       &           -1, & ! ccn_type: 6,7,8,9; if not set by namelist, the ccn_type_gscp4 or ccn_type_gscp5 will win
       &   -999.99_wp, & ! CN concentration at ground; if > -900 will override Ncn0 of ccn_type, but will use other configs of ccn_type
       &       0.1_wp, & ! min updraft speed [m/s] for Segal&Khain cloud nucleation
       &            1.0_wp, & ! tune_sbmccn: < 1 maritime, >= 1 continental CCN case for ccn_type 8 (Segal&Khain)

       ! .. Parameters for rain:
       !------------------------
       &            .FALSE.,  &   ! luse_mu_Dm_rain
       &          -999.99_wp, &   ! nu for rain, N(x) = N0 * D^nu * exp(-lambda*x^mu)
       &            6.0_wp, &     ! rain_cmu0
       &            30.0_wp, &    ! rain_cmu1
       &            1.1e-3_wp, &  ! rain_cmu3 = D_br
       &            1.0_wp, &     ! rain_cmu4

       !-----------------------------
       ! .. Parameters for cloud ice:
       !-----------------------------
       &            -999.99_wp, & ! nu_i for ice, N(x) = N0 * D^nu * exp(-lambda*x^mu) - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! mu_i for ice, N(x) = N0 * D^nu * exp(-lambda*x^mu) - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! ageo_i for ice, D = ageo*x^bgeo - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! bgeo_i for ice, D = ageo*x^bgeo - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! avel_i for ice, v = avel*x^bvel - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! bvel_i for ice, v = avel*x^bvel - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! cap_ice capacitance for ice deposition/sublimation - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &               1.0_wp,  & ! in_fact: factor for tuning IN concentration for heterogenous ice nucleation

       !------------------------
       ! .. Parameters for snow:
       !------------------------
       &            -999.99_wp, & ! nu_s for snow, N(x) = N0 * D^nu * exp(-lambda*x^mu) - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! mu_s for snow, N(x) = N0 * D^nu * exp(-lambda*x^mu) - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! ageo_s for snow, D = ageo*x^bgeo - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! bgeo_s for snow, D = ageo*x^bgeo - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! avel_s for snow, v = avel*x^bvel - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! bvel_s for snow, v = avel*x^bvel - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! cap_snow capacitance for snow deposition/sublimation - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! vsedi_max_s max fallspeed limit for snow - in this case the background value in mo_2mom_mcrph_main.f90 will win

       !---------------------------
       ! .. Parameters for graupel:
       !---------------------------
       &            -999.99_wp, & ! nu_g for graupel, N(x) = N0 * D^nu * exp(-lambda*x^mu) - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! mu_g for graupel, N(x) = N0 * D^nu * exp(-lambda*x^mu) - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! ageo_g for graupel, D = ageo*x^bgeo - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! bgeo_g for graupel, D = ageo*x^bgeo - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! avel_g for graupel, v = avel*x^bvel - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! bvel_g for graupel, v = avel*x^bvel - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            1.0_wp,   &   ! melt_g_tune_fac: factor multiplying melting rate of graupel 

       !------------------------
       ! .. Parameters for hail:
       !------------------------
       &            -999.99_wp, & ! nu_h for hail, N(x) = N0 * D^nu * exp(-lambda*x^mu) - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! mu_h for hail, N(x) = N0 * D^nu * exp(-lambda*x^mu) - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! ageo_h for hail, D = ageo*x^bgeo - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! bgeo_h for hail, D = ageo*x^bgeo - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! avel_h for hail, v = avel*x^bvel - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            -999.99_wp, & ! bvel_h for hail, v = avel*x^bvel - in this case the background value in mo_2mom_mcrph_main.f90 will win
       &            1.0_wp, &     ! melt_h_tune_fac: factor multiplying melting rate of hail

       !------------------------------------------
       ! .. Parameters for conversions/collisions:
       !------------------------------------------
       &            .FALSE.,   &   ! lturb_enhc: Turbulent enhancement of collisons
       &            300.0_wp, &   ! lturb_len:  Turbulent lenght scale (dummy, later overtaken by the TKE scheme, NOT a namelist parameter)
       &            10,        &  ! iice_stick: sticking efficiency of cloud ice
       &            5,        &   ! isnow_stick: sticking efficiency of snow
       &            5,        &   ! iparti_stick: sticking efficiency of frozen inter-categorical collisions
       &            0.01_wp,  &   ! alpha_spacefilling
       &            75.0e-6_wp, & ! D_conv_ii: D-threshold for conversion to snow ice_selfcollection: newly created snowflakes have at least this mean mass diameter
       &            0.50e-3_wp, & ! D_rainfrz_ig
       &            1.25e-3_wp, & ! D_rainfrz_gh
       &            0.10_wp,  &   ! Collision efficiency for graupel autoconversion (dry graupel) 
       &            0.40_wp,  &   ! Collision efficiency for graupel autoconversion (wet graupel)
       &            270.16_wp, &  ! Temperature threshold for switching to wet graupel autoconversion
       &            1, &          ! iicephase: (0) warm-phase 2M (1) mixed-phase 2M
       &            0, &       ! itype_shedding_gh: choice of shedding parameterization during collisions of graupel and hail with water droplets: 0=off, 1=simple, 2=more physical
       &            9e-3_wp, & ! D_shed_gh: Shedding happens if:
                               ! itype_shedding_gh = 1: D_meanmass > D_shed_gh
                               ! itype_shedding_gh = 2: in the PSD-part where D > MAX(D_wetgr,D_shed_gh) - that is
                               !                        for wet growth but not below the Rasmussen&Heymsfield stable diameter
       ! .. Parameters for the limitation of graupel production by rain riming of ice/snow:
       &            270.16_wp, &  ! Tmax_gr_rime
       &            .FALSE., & ! llim_gr_prod_rain_riming: whether to limit the graupel production by rain riming of ice/snow by
                               ! a bulk-density-based criterion on the mean-mass-particles

       &            0.5_wp, & ! wgt_D_coll_limgrprod: weight for the collided-particle's diameter D_coll: how much does the
                              ! smaller collision partner contribute to the overall diameter?
       &            0.5_wp & ! wgt_rho_coll_limgrprod: weight for the limit of the collided-particle's bulk density:
                             ! how near should it be to the bulk density of graupel in order to convert it to graupel?
       &            )

END MODULE mo_2mom_mcrph_config_default
