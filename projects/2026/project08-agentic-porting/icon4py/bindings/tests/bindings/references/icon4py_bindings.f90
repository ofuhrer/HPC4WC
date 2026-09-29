module icon4py_bindings
   use, intrinsic :: iso_c_binding
   implicit none

   public :: diffusion_init

   public :: diffusion_run

   public :: grid_init

   public :: solve_nh_init

   public :: solve_nh_run

   interface

      function diffusion_init_wrapper(theta_ref_mc, &
                                      theta_ref_mc_size_0, &
                                      theta_ref_mc_size_1, &
                                      wgtfac_c, &
                                      wgtfac_c_size_0, &
                                      wgtfac_c_size_1, &
                                      e_bln_c_s, &
                                      e_bln_c_s_size_0, &
                                      e_bln_c_s_size_1, &
                                      geofac_div, &
                                      geofac_div_size_0, &
                                      geofac_div_size_1, &
                                      geofac_grg_x, &
                                      geofac_grg_x_size_0, &
                                      geofac_grg_x_size_1, &
                                      geofac_grg_y, &
                                      geofac_grg_y_size_0, &
                                      geofac_grg_y_size_1, &
                                      geofac_n2s, &
                                      geofac_n2s_size_0, &
                                      geofac_n2s_size_1, &
                                      nudgecoeff_e, &
                                      nudgecoeff_e_size_0, &
                                      rbf_vec_coeff_v, &
                                      rbf_vec_coeff_v_size_0, &
                                      rbf_vec_coeff_v_size_1, &
                                      rbf_vec_coeff_v_size_2, &
                                      zd_cellidx, &
                                      zd_cellidx_size_0, &
                                      zd_cellidx_size_1, &
                                      zd_vertidx, &
                                      zd_vertidx_size_0, &
                                      zd_vertidx_size_1, &
                                      zd_intcoef, &
                                      zd_intcoef_size_0, &
                                      zd_intcoef_size_1, &
                                      zd_diffcoef, &
                                      zd_diffcoef_size_0, &
                                      ndyn_substeps, &
                                      diffusion_type, &
                                      hdiff_w, &
                                      hdiff_vn, &
                                      hdiff_smag_w, &
                                      zdiffu_t, &
                                      type_t_diffu, &
                                      type_vn_diffu, &
                                      hdiff_efdt_ratio, &
                                      hdiff_w_efdt_ratio, &
                                      smagorinski_scaling_factor, &
                                      smagorinski_scaling_factor2, &
                                      smagorinski_scaling_factor3, &
                                      smagorinski_scaling_factor4, &
                                      smagorinski_scaling_height, &
                                      smagorinski_scaling_height2, &
                                      smagorinski_scaling_height3, &
                                      smagorinski_scaling_height4, &
                                      hdiff_temp, &
                                      denom_diffu_v, &
                                      nudge_max_coeff, &
                                      itype_sher, &
                                      iforcing, &
                                      a_hshr, &
                                      loutshs, &
                                      backend, &
                                      on_gpu) bind(c, name="diffusion_init_wrapper") result(rc)
         import :: c_int, c_long, c_float, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: theta_ref_mc

         integer(c_int), value :: theta_ref_mc_size_0

         integer(c_int), value :: theta_ref_mc_size_1

         type(c_ptr), value, target :: wgtfac_c

         integer(c_int), value :: wgtfac_c_size_0

         integer(c_int), value :: wgtfac_c_size_1

         type(c_ptr), value, target :: e_bln_c_s

         integer(c_int), value :: e_bln_c_s_size_0

         integer(c_int), value :: e_bln_c_s_size_1

         type(c_ptr), value, target :: geofac_div

         integer(c_int), value :: geofac_div_size_0

         integer(c_int), value :: geofac_div_size_1

         type(c_ptr), value, target :: geofac_grg_x

         integer(c_int), value :: geofac_grg_x_size_0

         integer(c_int), value :: geofac_grg_x_size_1

         type(c_ptr), value, target :: geofac_grg_y

         integer(c_int), value :: geofac_grg_y_size_0

         integer(c_int), value :: geofac_grg_y_size_1

         type(c_ptr), value, target :: geofac_n2s

         integer(c_int), value :: geofac_n2s_size_0

         integer(c_int), value :: geofac_n2s_size_1

         type(c_ptr), value, target :: nudgecoeff_e

         integer(c_int), value :: nudgecoeff_e_size_0

         type(c_ptr), value, target :: rbf_vec_coeff_v

         integer(c_int), value :: rbf_vec_coeff_v_size_0

         integer(c_int), value :: rbf_vec_coeff_v_size_1

         integer(c_int), value :: rbf_vec_coeff_v_size_2

         type(c_ptr), value, target :: zd_cellidx

         integer(c_int), value :: zd_cellidx_size_0

         integer(c_int), value :: zd_cellidx_size_1

         type(c_ptr), value, target :: zd_vertidx

         integer(c_int), value :: zd_vertidx_size_0

         integer(c_int), value :: zd_vertidx_size_1

         type(c_ptr), value, target :: zd_intcoef

         integer(c_int), value :: zd_intcoef_size_0

         integer(c_int), value :: zd_intcoef_size_1

         type(c_ptr), value, target :: zd_diffcoef

         integer(c_int), value :: zd_diffcoef_size_0

         integer(c_int), value, target :: ndyn_substeps

         integer(c_int), value, target :: diffusion_type

         logical(c_bool), value, target :: hdiff_w

         logical(c_bool), value, target :: hdiff_vn

         logical(c_bool), value, target :: hdiff_smag_w

         logical(c_bool), value, target :: zdiffu_t

         integer(c_int), value, target :: type_t_diffu

         integer(c_int), value, target :: type_vn_diffu

         real(c_double), value, target :: hdiff_efdt_ratio

         real(c_double), value, target :: hdiff_w_efdt_ratio

         real(c_double), value, target :: smagorinski_scaling_factor

         real(c_double), value, target :: smagorinski_scaling_factor2

         real(c_double), value, target :: smagorinski_scaling_factor3

         real(c_double), value, target :: smagorinski_scaling_factor4

         real(c_double), value, target :: smagorinski_scaling_height

         real(c_double), value, target :: smagorinski_scaling_height2

         real(c_double), value, target :: smagorinski_scaling_height3

         real(c_double), value, target :: smagorinski_scaling_height4

         logical(c_bool), value, target :: hdiff_temp

         real(c_double), value, target :: denom_diffu_v

         real(c_double), value, target :: nudge_max_coeff

         integer(c_int), value, target :: itype_sher

         integer(c_int), value, target :: iforcing

         real(c_double), value, target :: a_hshr

         logical(c_bool), value, target :: loutshs

         integer(c_int), value, target :: backend

         logical(c_bool), value :: on_gpu

      end function diffusion_init_wrapper

      function diffusion_run_wrapper(w, &
                                     w_size_0, &
                                     w_size_1, &
                                     vn, &
                                     vn_size_0, &
                                     vn_size_1, &
                                     exner, &
                                     exner_size_0, &
                                     exner_size_1, &
                                     theta_v, &
                                     theta_v_size_0, &
                                     theta_v_size_1, &
                                     rho, &
                                     rho_size_0, &
                                     rho_size_1, &
                                     hdef_ic, &
                                     hdef_ic_size_0, &
                                     hdef_ic_size_1, &
                                     div_ic, &
                                     div_ic_size_0, &
                                     div_ic_size_1, &
                                     dwdx, &
                                     dwdx_size_0, &
                                     dwdx_size_1, &
                                     dwdy, &
                                     dwdy_size_0, &
                                     dwdy_size_1, &
                                     dtime, &
                                     linit, &
                                     on_gpu) bind(c, name="diffusion_run_wrapper") result(rc)
         import :: c_int, c_long, c_float, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: w

         integer(c_int), value :: w_size_0

         integer(c_int), value :: w_size_1

         type(c_ptr), value, target :: vn

         integer(c_int), value :: vn_size_0

         integer(c_int), value :: vn_size_1

         type(c_ptr), value, target :: exner

         integer(c_int), value :: exner_size_0

         integer(c_int), value :: exner_size_1

         type(c_ptr), value, target :: theta_v

         integer(c_int), value :: theta_v_size_0

         integer(c_int), value :: theta_v_size_1

         type(c_ptr), value, target :: rho

         integer(c_int), value :: rho_size_0

         integer(c_int), value :: rho_size_1

         type(c_ptr), value, target :: hdef_ic

         integer(c_int), value :: hdef_ic_size_0

         integer(c_int), value :: hdef_ic_size_1

         type(c_ptr), value, target :: div_ic

         integer(c_int), value :: div_ic_size_0

         integer(c_int), value :: div_ic_size_1

         type(c_ptr), value, target :: dwdx

         integer(c_int), value :: dwdx_size_0

         integer(c_int), value :: dwdx_size_1

         type(c_ptr), value, target :: dwdy

         integer(c_int), value :: dwdy_size_0

         integer(c_int), value :: dwdy_size_1

         real(c_double), value, target :: dtime

         logical(c_bool), value, target :: linit

         logical(c_bool), value :: on_gpu

      end function diffusion_run_wrapper

      function grid_init_wrapper(cell_starts, &
                                 cell_starts_size_0, &
                                 cell_ends, &
                                 cell_ends_size_0, &
                                 vertex_starts, &
                                 vertex_starts_size_0, &
                                 vertex_ends, &
                                 vertex_ends_size_0, &
                                 edge_starts, &
                                 edge_starts_size_0, &
                                 edge_ends, &
                                 edge_ends_size_0, &
                                 c2e, &
                                 c2e_size_0, &
                                 c2e_size_1, &
                                 e2c, &
                                 e2c_size_0, &
                                 e2c_size_1, &
                                 c2e2c, &
                                 c2e2c_size_0, &
                                 c2e2c_size_1, &
                                 e2c2e, &
                                 e2c2e_size_0, &
                                 e2c2e_size_1, &
                                 e2v, &
                                 e2v_size_0, &
                                 e2v_size_1, &
                                 v2e, &
                                 v2e_size_0, &
                                 v2e_size_1, &
                                 v2c, &
                                 v2c_size_0, &
                                 v2c_size_1, &
                                 e2c2v, &
                                 e2c2v_size_0, &
                                 e2c2v_size_1, &
                                 c2v, &
                                 c2v_size_0, &
                                 c2v_size_1, &
                                 c_owner_mask, &
                                 c_owner_mask_size_0, &
                                 e_owner_mask, &
                                 e_owner_mask_size_0, &
                                 v_owner_mask, &
                                 v_owner_mask_size_0, &
                                 c_glb_index, &
                                 c_glb_index_size_0, &
                                 e_glb_index, &
                                 e_glb_index_size_0, &
                                 v_glb_index, &
                                 v_glb_index_size_0, &
                                 tangent_orientation, &
                                 tangent_orientation_size_0, &
                                 inverse_primal_edge_lengths, &
                                 inverse_primal_edge_lengths_size_0, &
                                 inv_dual_edge_length, &
                                 inv_dual_edge_length_size_0, &
                                 inv_vert_vert_length, &
                                 inv_vert_vert_length_size_0, &
                                 edge_areas, &
                                 edge_areas_size_0, &
                                 f_e, &
                                 f_e_size_0, &
                                 cell_center_lat, &
                                 cell_center_lat_size_0, &
                                 cell_center_lon, &
                                 cell_center_lon_size_0, &
                                 cell_areas, &
                                 cell_areas_size_0, &
                                 primal_normal_vert_x, &
                                 primal_normal_vert_x_size_0, &
                                 primal_normal_vert_x_size_1, &
                                 primal_normal_vert_y, &
                                 primal_normal_vert_y_size_0, &
                                 primal_normal_vert_y_size_1, &
                                 dual_normal_vert_x, &
                                 dual_normal_vert_x_size_0, &
                                 dual_normal_vert_x_size_1, &
                                 dual_normal_vert_y, &
                                 dual_normal_vert_y_size_0, &
                                 dual_normal_vert_y_size_1, &
                                 primal_normal_cell_x, &
                                 primal_normal_cell_x_size_0, &
                                 primal_normal_cell_x_size_1, &
                                 primal_normal_cell_y, &
                                 primal_normal_cell_y_size_0, &
                                 primal_normal_cell_y_size_1, &
                                 dual_normal_cell_x, &
                                 dual_normal_cell_x_size_0, &
                                 dual_normal_cell_x_size_1, &
                                 dual_normal_cell_y, &
                                 dual_normal_cell_y_size_0, &
                                 dual_normal_cell_y_size_1, &
                                 edge_center_lat, &
                                 edge_center_lat_size_0, &
                                 edge_center_lon, &
                                 edge_center_lon_size_0, &
                                 primal_normal_x, &
                                 primal_normal_x_size_0, &
                                 primal_normal_y, &
                                 primal_normal_y_size_0, &
                                 vct_a, &
                                 vct_a_size_0, &
                                 lowest_layer_thickness, &
                                 model_top_height, &
                                 stretch_factor, &
                                 flat_height, &
                                 rayleigh_damping_height, &
                                 mean_cell_area, &
                                 comm_id, &
                                 num_vertices, &
                                 num_cells, &
                                 num_edges, &
                                 vertical_size, &
                                 limited_area, &
                                 backend, &
                                 on_gpu) bind(c, name="grid_init_wrapper") result(rc)
         import :: c_int, c_long, c_float, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: cell_starts

         integer(c_int), value :: cell_starts_size_0

         type(c_ptr), value, target :: cell_ends

         integer(c_int), value :: cell_ends_size_0

         type(c_ptr), value, target :: vertex_starts

         integer(c_int), value :: vertex_starts_size_0

         type(c_ptr), value, target :: vertex_ends

         integer(c_int), value :: vertex_ends_size_0

         type(c_ptr), value, target :: edge_starts

         integer(c_int), value :: edge_starts_size_0

         type(c_ptr), value, target :: edge_ends

         integer(c_int), value :: edge_ends_size_0

         type(c_ptr), value, target :: c2e

         integer(c_int), value :: c2e_size_0

         integer(c_int), value :: c2e_size_1

         type(c_ptr), value, target :: e2c

         integer(c_int), value :: e2c_size_0

         integer(c_int), value :: e2c_size_1

         type(c_ptr), value, target :: c2e2c

         integer(c_int), value :: c2e2c_size_0

         integer(c_int), value :: c2e2c_size_1

         type(c_ptr), value, target :: e2c2e

         integer(c_int), value :: e2c2e_size_0

         integer(c_int), value :: e2c2e_size_1

         type(c_ptr), value, target :: e2v

         integer(c_int), value :: e2v_size_0

         integer(c_int), value :: e2v_size_1

         type(c_ptr), value, target :: v2e

         integer(c_int), value :: v2e_size_0

         integer(c_int), value :: v2e_size_1

         type(c_ptr), value, target :: v2c

         integer(c_int), value :: v2c_size_0

         integer(c_int), value :: v2c_size_1

         type(c_ptr), value, target :: e2c2v

         integer(c_int), value :: e2c2v_size_0

         integer(c_int), value :: e2c2v_size_1

         type(c_ptr), value, target :: c2v

         integer(c_int), value :: c2v_size_0

         integer(c_int), value :: c2v_size_1

         type(c_ptr), value, target :: c_owner_mask

         integer(c_int), value :: c_owner_mask_size_0

         type(c_ptr), value, target :: e_owner_mask

         integer(c_int), value :: e_owner_mask_size_0

         type(c_ptr), value, target :: v_owner_mask

         integer(c_int), value :: v_owner_mask_size_0

         type(c_ptr), value, target :: c_glb_index

         integer(c_int), value :: c_glb_index_size_0

         type(c_ptr), value, target :: e_glb_index

         integer(c_int), value :: e_glb_index_size_0

         type(c_ptr), value, target :: v_glb_index

         integer(c_int), value :: v_glb_index_size_0

         type(c_ptr), value, target :: tangent_orientation

         integer(c_int), value :: tangent_orientation_size_0

         type(c_ptr), value, target :: inverse_primal_edge_lengths

         integer(c_int), value :: inverse_primal_edge_lengths_size_0

         type(c_ptr), value, target :: inv_dual_edge_length

         integer(c_int), value :: inv_dual_edge_length_size_0

         type(c_ptr), value, target :: inv_vert_vert_length

         integer(c_int), value :: inv_vert_vert_length_size_0

         type(c_ptr), value, target :: edge_areas

         integer(c_int), value :: edge_areas_size_0

         type(c_ptr), value, target :: f_e

         integer(c_int), value :: f_e_size_0

         type(c_ptr), value, target :: cell_center_lat

         integer(c_int), value :: cell_center_lat_size_0

         type(c_ptr), value, target :: cell_center_lon

         integer(c_int), value :: cell_center_lon_size_0

         type(c_ptr), value, target :: cell_areas

         integer(c_int), value :: cell_areas_size_0

         type(c_ptr), value, target :: primal_normal_vert_x

         integer(c_int), value :: primal_normal_vert_x_size_0

         integer(c_int), value :: primal_normal_vert_x_size_1

         type(c_ptr), value, target :: primal_normal_vert_y

         integer(c_int), value :: primal_normal_vert_y_size_0

         integer(c_int), value :: primal_normal_vert_y_size_1

         type(c_ptr), value, target :: dual_normal_vert_x

         integer(c_int), value :: dual_normal_vert_x_size_0

         integer(c_int), value :: dual_normal_vert_x_size_1

         type(c_ptr), value, target :: dual_normal_vert_y

         integer(c_int), value :: dual_normal_vert_y_size_0

         integer(c_int), value :: dual_normal_vert_y_size_1

         type(c_ptr), value, target :: primal_normal_cell_x

         integer(c_int), value :: primal_normal_cell_x_size_0

         integer(c_int), value :: primal_normal_cell_x_size_1

         type(c_ptr), value, target :: primal_normal_cell_y

         integer(c_int), value :: primal_normal_cell_y_size_0

         integer(c_int), value :: primal_normal_cell_y_size_1

         type(c_ptr), value, target :: dual_normal_cell_x

         integer(c_int), value :: dual_normal_cell_x_size_0

         integer(c_int), value :: dual_normal_cell_x_size_1

         type(c_ptr), value, target :: dual_normal_cell_y

         integer(c_int), value :: dual_normal_cell_y_size_0

         integer(c_int), value :: dual_normal_cell_y_size_1

         type(c_ptr), value, target :: edge_center_lat

         integer(c_int), value :: edge_center_lat_size_0

         type(c_ptr), value, target :: edge_center_lon

         integer(c_int), value :: edge_center_lon_size_0

         type(c_ptr), value, target :: primal_normal_x

         integer(c_int), value :: primal_normal_x_size_0

         type(c_ptr), value, target :: primal_normal_y

         integer(c_int), value :: primal_normal_y_size_0

         type(c_ptr), value, target :: vct_a

         integer(c_int), value :: vct_a_size_0

         real(c_double), value, target :: lowest_layer_thickness

         real(c_double), value, target :: model_top_height

         real(c_double), value, target :: stretch_factor

         real(c_double), value, target :: flat_height

         real(c_double), value, target :: rayleigh_damping_height

         real(c_double), value, target :: mean_cell_area

         integer(c_int), value, target :: comm_id

         integer(c_int), value, target :: num_vertices

         integer(c_int), value, target :: num_cells

         integer(c_int), value, target :: num_edges

         integer(c_int), value, target :: vertical_size

         logical(c_bool), value, target :: limited_area

         integer(c_int), value, target :: backend

         logical(c_bool), value :: on_gpu

      end function grid_init_wrapper

      function solve_nh_init_wrapper(c_lin_e, &
                                     c_lin_e_size_0, &
                                     c_lin_e_size_1, &
                                     c_intp, &
                                     c_intp_size_0, &
                                     c_intp_size_1, &
                                     e_flx_avg, &
                                     e_flx_avg_size_0, &
                                     e_flx_avg_size_1, &
                                     geofac_grdiv, &
                                     geofac_grdiv_size_0, &
                                     geofac_grdiv_size_1, &
                                     geofac_rot, &
                                     geofac_rot_size_0, &
                                     geofac_rot_size_1, &
                                     pos_on_tplane_e_1, &
                                     pos_on_tplane_e_1_size_0, &
                                     pos_on_tplane_e_1_size_1, &
                                     pos_on_tplane_e_2, &
                                     pos_on_tplane_e_2_size_0, &
                                     pos_on_tplane_e_2_size_1, &
                                     rbf_vec_coeff_e, &
                                     rbf_vec_coeff_e_size_0, &
                                     rbf_vec_coeff_e_size_1, &
                                     e_bln_c_s, &
                                     e_bln_c_s_size_0, &
                                     e_bln_c_s_size_1, &
                                     rbf_vec_coeff_v, &
                                     rbf_vec_coeff_v_size_0, &
                                     rbf_vec_coeff_v_size_1, &
                                     rbf_vec_coeff_v_size_2, &
                                     geofac_div, &
                                     geofac_div_size_0, &
                                     geofac_div_size_1, &
                                     geofac_n2s, &
                                     geofac_n2s_size_0, &
                                     geofac_n2s_size_1, &
                                     geofac_grg_x, &
                                     geofac_grg_x_size_0, &
                                     geofac_grg_x_size_1, &
                                     geofac_grg_y, &
                                     geofac_grg_y_size_0, &
                                     geofac_grg_y_size_1, &
                                     nudgecoeff_e, &
                                     nudgecoeff_e_size_0, &
                                     mask_prog_halo_c, &
                                     mask_prog_halo_c_size_0, &
                                     rayleigh_w, &
                                     rayleigh_w_size_0, &
                                     exner_exfac, &
                                     exner_exfac_size_0, &
                                     exner_exfac_size_1, &
                                     exner_ref_mc, &
                                     exner_ref_mc_size_0, &
                                     exner_ref_mc_size_1, &
                                     wgtfac_c, &
                                     wgtfac_c_size_0, &
                                     wgtfac_c_size_1, &
                                     wgtfacq_c, &
                                     wgtfacq_c_size_0, &
                                     wgtfacq_c_size_1, &
                                     inv_ddqz_z_full, &
                                     inv_ddqz_z_full_size_0, &
                                     inv_ddqz_z_full_size_1, &
                                     rho_ref_mc, &
                                     rho_ref_mc_size_0, &
                                     rho_ref_mc_size_1, &
                                     theta_ref_mc, &
                                     theta_ref_mc_size_0, &
                                     theta_ref_mc_size_1, &
                                     vwind_expl_wgt, &
                                     vwind_expl_wgt_size_0, &
                                     d_exner_dz_ref_ic, &
                                     d_exner_dz_ref_ic_size_0, &
                                     d_exner_dz_ref_ic_size_1, &
                                     ddqz_z_half, &
                                     ddqz_z_half_size_0, &
                                     ddqz_z_half_size_1, &
                                     theta_ref_ic, &
                                     theta_ref_ic_size_0, &
                                     theta_ref_ic_size_1, &
                                     d2dexdz2_fac1_mc, &
                                     d2dexdz2_fac1_mc_size_0, &
                                     d2dexdz2_fac1_mc_size_1, &
                                     d2dexdz2_fac2_mc, &
                                     d2dexdz2_fac2_mc_size_0, &
                                     d2dexdz2_fac2_mc_size_1, &
                                     rho_ref_me, &
                                     rho_ref_me_size_0, &
                                     rho_ref_me_size_1, &
                                     theta_ref_me, &
                                     theta_ref_me_size_0, &
                                     theta_ref_me_size_1, &
                                     ddxn_z_full, &
                                     ddxn_z_full_size_0, &
                                     ddxn_z_full_size_1, &
                                     zdiff_gradp, &
                                     zdiff_gradp_size_0, &
                                     zdiff_gradp_size_1, &
                                     zdiff_gradp_size_2, &
                                     vertidx_gradp, &
                                     vertidx_gradp_size_0, &
                                     vertidx_gradp_size_1, &
                                     vertidx_gradp_size_2, &
                                     pg_edgeidx, &
                                     pg_edgeidx_size_0, &
                                     pg_vertidx, &
                                     pg_vertidx_size_0, &
                                     pg_exdist, &
                                     pg_exdist_size_0, &
                                     ddqz_z_full_e, &
                                     ddqz_z_full_e_size_0, &
                                     ddqz_z_full_e_size_1, &
                                     ddxt_z_full, &
                                     ddxt_z_full_size_0, &
                                     ddxt_z_full_size_1, &
                                     wgtfac_e, &
                                     wgtfac_e_size_0, &
                                     wgtfac_e_size_1, &
                                     wgtfacq_e, &
                                     wgtfacq_e_size_0, &
                                     wgtfacq_e_size_1, &
                                     vwind_impl_wgt, &
                                     vwind_impl_wgt_size_0, &
                                     hmask_dd3d, &
                                     hmask_dd3d_size_0, &
                                     scalfac_dd3d, &
                                     scalfac_dd3d_size_0, &
                                     coeff1_dwdz, &
                                     coeff1_dwdz_size_0, &
                                     coeff1_dwdz_size_1, &
                                     coeff2_dwdz, &
                                     coeff2_dwdz_size_0, &
                                     coeff2_dwdz_size_1, &
                                     coeff_gradekin, &
                                     coeff_gradekin_size_0, &
                                     coeff_gradekin_size_1, &
                                     c_owner_mask, &
                                     c_owner_mask_size_0, &
                                     itime_scheme, &
                                     iadv_rhotheta, &
                                     igradp_method, &
                                     rayleigh_type, &
                                     divdamp_order, &
                                     divdamp_type, &
                                     l_vert_nested, &
                                     ldeepatmo, &
                                     iau_init, &
                                     extra_diffu, &
                                     rhotheta_offctr, &
                                     veladv_offctr, &
                                     nudge_max_coeff, &
                                     divdamp_fac, &
                                     divdamp_fac2, &
                                     divdamp_fac3, &
                                     divdamp_fac4, &
                                     divdamp_z, &
                                     divdamp_z2, &
                                     divdamp_z3, &
                                     divdamp_z4, &
                                     nflat_gradp, &
                                     backend, &
                                     on_gpu) bind(c, name="solve_nh_init_wrapper") result(rc)
         import :: c_int, c_long, c_float, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: c_lin_e

         integer(c_int), value :: c_lin_e_size_0

         integer(c_int), value :: c_lin_e_size_1

         type(c_ptr), value, target :: c_intp

         integer(c_int), value :: c_intp_size_0

         integer(c_int), value :: c_intp_size_1

         type(c_ptr), value, target :: e_flx_avg

         integer(c_int), value :: e_flx_avg_size_0

         integer(c_int), value :: e_flx_avg_size_1

         type(c_ptr), value, target :: geofac_grdiv

         integer(c_int), value :: geofac_grdiv_size_0

         integer(c_int), value :: geofac_grdiv_size_1

         type(c_ptr), value, target :: geofac_rot

         integer(c_int), value :: geofac_rot_size_0

         integer(c_int), value :: geofac_rot_size_1

         type(c_ptr), value, target :: pos_on_tplane_e_1

         integer(c_int), value :: pos_on_tplane_e_1_size_0

         integer(c_int), value :: pos_on_tplane_e_1_size_1

         type(c_ptr), value, target :: pos_on_tplane_e_2

         integer(c_int), value :: pos_on_tplane_e_2_size_0

         integer(c_int), value :: pos_on_tplane_e_2_size_1

         type(c_ptr), value, target :: rbf_vec_coeff_e

         integer(c_int), value :: rbf_vec_coeff_e_size_0

         integer(c_int), value :: rbf_vec_coeff_e_size_1

         type(c_ptr), value, target :: e_bln_c_s

         integer(c_int), value :: e_bln_c_s_size_0

         integer(c_int), value :: e_bln_c_s_size_1

         type(c_ptr), value, target :: rbf_vec_coeff_v

         integer(c_int), value :: rbf_vec_coeff_v_size_0

         integer(c_int), value :: rbf_vec_coeff_v_size_1

         integer(c_int), value :: rbf_vec_coeff_v_size_2

         type(c_ptr), value, target :: geofac_div

         integer(c_int), value :: geofac_div_size_0

         integer(c_int), value :: geofac_div_size_1

         type(c_ptr), value, target :: geofac_n2s

         integer(c_int), value :: geofac_n2s_size_0

         integer(c_int), value :: geofac_n2s_size_1

         type(c_ptr), value, target :: geofac_grg_x

         integer(c_int), value :: geofac_grg_x_size_0

         integer(c_int), value :: geofac_grg_x_size_1

         type(c_ptr), value, target :: geofac_grg_y

         integer(c_int), value :: geofac_grg_y_size_0

         integer(c_int), value :: geofac_grg_y_size_1

         type(c_ptr), value, target :: nudgecoeff_e

         integer(c_int), value :: nudgecoeff_e_size_0

         type(c_ptr), value, target :: mask_prog_halo_c

         integer(c_int), value :: mask_prog_halo_c_size_0

         type(c_ptr), value, target :: rayleigh_w

         integer(c_int), value :: rayleigh_w_size_0

         type(c_ptr), value, target :: exner_exfac

         integer(c_int), value :: exner_exfac_size_0

         integer(c_int), value :: exner_exfac_size_1

         type(c_ptr), value, target :: exner_ref_mc

         integer(c_int), value :: exner_ref_mc_size_0

         integer(c_int), value :: exner_ref_mc_size_1

         type(c_ptr), value, target :: wgtfac_c

         integer(c_int), value :: wgtfac_c_size_0

         integer(c_int), value :: wgtfac_c_size_1

         type(c_ptr), value, target :: wgtfacq_c

         integer(c_int), value :: wgtfacq_c_size_0

         integer(c_int), value :: wgtfacq_c_size_1

         type(c_ptr), value, target :: inv_ddqz_z_full

         integer(c_int), value :: inv_ddqz_z_full_size_0

         integer(c_int), value :: inv_ddqz_z_full_size_1

         type(c_ptr), value, target :: rho_ref_mc

         integer(c_int), value :: rho_ref_mc_size_0

         integer(c_int), value :: rho_ref_mc_size_1

         type(c_ptr), value, target :: theta_ref_mc

         integer(c_int), value :: theta_ref_mc_size_0

         integer(c_int), value :: theta_ref_mc_size_1

         type(c_ptr), value, target :: vwind_expl_wgt

         integer(c_int), value :: vwind_expl_wgt_size_0

         type(c_ptr), value, target :: d_exner_dz_ref_ic

         integer(c_int), value :: d_exner_dz_ref_ic_size_0

         integer(c_int), value :: d_exner_dz_ref_ic_size_1

         type(c_ptr), value, target :: ddqz_z_half

         integer(c_int), value :: ddqz_z_half_size_0

         integer(c_int), value :: ddqz_z_half_size_1

         type(c_ptr), value, target :: theta_ref_ic

         integer(c_int), value :: theta_ref_ic_size_0

         integer(c_int), value :: theta_ref_ic_size_1

         type(c_ptr), value, target :: d2dexdz2_fac1_mc

         integer(c_int), value :: d2dexdz2_fac1_mc_size_0

         integer(c_int), value :: d2dexdz2_fac1_mc_size_1

         type(c_ptr), value, target :: d2dexdz2_fac2_mc

         integer(c_int), value :: d2dexdz2_fac2_mc_size_0

         integer(c_int), value :: d2dexdz2_fac2_mc_size_1

         type(c_ptr), value, target :: rho_ref_me

         integer(c_int), value :: rho_ref_me_size_0

         integer(c_int), value :: rho_ref_me_size_1

         type(c_ptr), value, target :: theta_ref_me

         integer(c_int), value :: theta_ref_me_size_0

         integer(c_int), value :: theta_ref_me_size_1

         type(c_ptr), value, target :: ddxn_z_full

         integer(c_int), value :: ddxn_z_full_size_0

         integer(c_int), value :: ddxn_z_full_size_1

         type(c_ptr), value, target :: zdiff_gradp

         integer(c_int), value :: zdiff_gradp_size_0

         integer(c_int), value :: zdiff_gradp_size_1

         integer(c_int), value :: zdiff_gradp_size_2

         type(c_ptr), value, target :: vertidx_gradp

         integer(c_int), value :: vertidx_gradp_size_0

         integer(c_int), value :: vertidx_gradp_size_1

         integer(c_int), value :: vertidx_gradp_size_2

         type(c_ptr), value, target :: pg_edgeidx

         integer(c_int), value :: pg_edgeidx_size_0

         type(c_ptr), value, target :: pg_vertidx

         integer(c_int), value :: pg_vertidx_size_0

         type(c_ptr), value, target :: pg_exdist

         integer(c_int), value :: pg_exdist_size_0

         type(c_ptr), value, target :: ddqz_z_full_e

         integer(c_int), value :: ddqz_z_full_e_size_0

         integer(c_int), value :: ddqz_z_full_e_size_1

         type(c_ptr), value, target :: ddxt_z_full

         integer(c_int), value :: ddxt_z_full_size_0

         integer(c_int), value :: ddxt_z_full_size_1

         type(c_ptr), value, target :: wgtfac_e

         integer(c_int), value :: wgtfac_e_size_0

         integer(c_int), value :: wgtfac_e_size_1

         type(c_ptr), value, target :: wgtfacq_e

         integer(c_int), value :: wgtfacq_e_size_0

         integer(c_int), value :: wgtfacq_e_size_1

         type(c_ptr), value, target :: vwind_impl_wgt

         integer(c_int), value :: vwind_impl_wgt_size_0

         type(c_ptr), value, target :: hmask_dd3d

         integer(c_int), value :: hmask_dd3d_size_0

         type(c_ptr), value, target :: scalfac_dd3d

         integer(c_int), value :: scalfac_dd3d_size_0

         type(c_ptr), value, target :: coeff1_dwdz

         integer(c_int), value :: coeff1_dwdz_size_0

         integer(c_int), value :: coeff1_dwdz_size_1

         type(c_ptr), value, target :: coeff2_dwdz

         integer(c_int), value :: coeff2_dwdz_size_0

         integer(c_int), value :: coeff2_dwdz_size_1

         type(c_ptr), value, target :: coeff_gradekin

         integer(c_int), value :: coeff_gradekin_size_0

         integer(c_int), value :: coeff_gradekin_size_1

         type(c_ptr), value, target :: c_owner_mask

         integer(c_int), value :: c_owner_mask_size_0

         integer(c_int), value, target :: itime_scheme

         integer(c_int), value, target :: iadv_rhotheta

         integer(c_int), value, target :: igradp_method

         integer(c_int), value, target :: rayleigh_type

         integer(c_int), value, target :: divdamp_order

         integer(c_int), value, target :: divdamp_type

         logical(c_bool), value, target :: l_vert_nested

         logical(c_bool), value, target :: ldeepatmo

         logical(c_bool), value, target :: iau_init

         logical(c_bool), value, target :: extra_diffu

         real(c_double), value, target :: rhotheta_offctr

         real(c_double), value, target :: veladv_offctr

         real(c_double), value, target :: nudge_max_coeff

         real(c_double), value, target :: divdamp_fac

         real(c_double), value, target :: divdamp_fac2

         real(c_double), value, target :: divdamp_fac3

         real(c_double), value, target :: divdamp_fac4

         real(c_double), value, target :: divdamp_z

         real(c_double), value, target :: divdamp_z2

         real(c_double), value, target :: divdamp_z3

         real(c_double), value, target :: divdamp_z4

         integer(c_int), value, target :: nflat_gradp

         integer(c_int), value, target :: backend

         logical(c_bool), value :: on_gpu

      end function solve_nh_init_wrapper

      function solve_nh_run_wrapper(rho_now, &
                                    rho_now_size_0, &
                                    rho_now_size_1, &
                                    rho_new, &
                                    rho_new_size_0, &
                                    rho_new_size_1, &
                                    exner_now, &
                                    exner_now_size_0, &
                                    exner_now_size_1, &
                                    exner_new, &
                                    exner_new_size_0, &
                                    exner_new_size_1, &
                                    w_now, &
                                    w_now_size_0, &
                                    w_now_size_1, &
                                    w_new, &
                                    w_new_size_0, &
                                    w_new_size_1, &
                                    theta_v_now, &
                                    theta_v_now_size_0, &
                                    theta_v_now_size_1, &
                                    theta_v_new, &
                                    theta_v_new_size_0, &
                                    theta_v_new_size_1, &
                                    vn_now, &
                                    vn_now_size_0, &
                                    vn_now_size_1, &
                                    vn_new, &
                                    vn_new_size_0, &
                                    vn_new_size_1, &
                                    w_concorr_c, &
                                    w_concorr_c_size_0, &
                                    w_concorr_c_size_1, &
                                    ddt_vn_apc_ntl1, &
                                    ddt_vn_apc_ntl1_size_0, &
                                    ddt_vn_apc_ntl1_size_1, &
                                    ddt_vn_apc_ntl2, &
                                    ddt_vn_apc_ntl2_size_0, &
                                    ddt_vn_apc_ntl2_size_1, &
                                    ddt_w_adv_ntl1, &
                                    ddt_w_adv_ntl1_size_0, &
                                    ddt_w_adv_ntl1_size_1, &
                                    ddt_w_adv_ntl2, &
                                    ddt_w_adv_ntl2_size_0, &
                                    ddt_w_adv_ntl2_size_1, &
                                    theta_v_ic, &
                                    theta_v_ic_size_0, &
                                    theta_v_ic_size_1, &
                                    rho_ic, &
                                    rho_ic_size_0, &
                                    rho_ic_size_1, &
                                    exner_pr, &
                                    exner_pr_size_0, &
                                    exner_pr_size_1, &
                                    exner_dyn_incr, &
                                    exner_dyn_incr_size_0, &
                                    exner_dyn_incr_size_1, &
                                    ddt_exner_phy, &
                                    ddt_exner_phy_size_0, &
                                    ddt_exner_phy_size_1, &
                                    grf_tend_rho, &
                                    grf_tend_rho_size_0, &
                                    grf_tend_rho_size_1, &
                                    grf_tend_thv, &
                                    grf_tend_thv_size_0, &
                                    grf_tend_thv_size_1, &
                                    grf_tend_w, &
                                    grf_tend_w_size_0, &
                                    grf_tend_w_size_1, &
                                    mass_fl_e, &
                                    mass_fl_e_size_0, &
                                    mass_fl_e_size_1, &
                                    ddt_vn_phy, &
                                    ddt_vn_phy_size_0, &
                                    ddt_vn_phy_size_1, &
                                    grf_tend_vn, &
                                    grf_tend_vn_size_0, &
                                    grf_tend_vn_size_1, &
                                    vn_ie, &
                                    vn_ie_size_0, &
                                    vn_ie_size_1, &
                                    vt, &
                                    vt_size_0, &
                                    vt_size_1, &
                                    vn_incr, &
                                    vn_incr_size_0, &
                                    vn_incr_size_1, &
                                    rho_incr, &
                                    rho_incr_size_0, &
                                    rho_incr_size_1, &
                                    exner_incr, &
                                    exner_incr_size_0, &
                                    exner_incr_size_1, &
                                    mass_flx_me, &
                                    mass_flx_me_size_0, &
                                    mass_flx_me_size_1, &
                                    mass_flx_ic, &
                                    mass_flx_ic_size_0, &
                                    mass_flx_ic_size_1, &
                                    vol_flx_ic, &
                                    vol_flx_ic_size_0, &
                                    vol_flx_ic_size_1, &
                                    vn_traj, &
                                    vn_traj_size_0, &
                                    vn_traj_size_1, &
                                    dtime, &
                                    max_vcfl_size1_array, &
                                    max_vcfl_size1_array_size_0, &
                                    lprep_adv, &
                                    at_initial_timestep, &
                                    divdamp_fac_o2, &
                                    ndyn_substeps_var, &
                                    idyn_timestep, &
                                    is_iau_active, &
                                    iau_wgt_dyn, &
                                    on_gpu) bind(c, name="solve_nh_run_wrapper") result(rc)
         import :: c_int, c_long, c_float, c_double, c_bool, c_ptr
         integer(c_int) :: rc  ! Stores the return code

         type(c_ptr), value, target :: rho_now

         integer(c_int), value :: rho_now_size_0

         integer(c_int), value :: rho_now_size_1

         type(c_ptr), value, target :: rho_new

         integer(c_int), value :: rho_new_size_0

         integer(c_int), value :: rho_new_size_1

         type(c_ptr), value, target :: exner_now

         integer(c_int), value :: exner_now_size_0

         integer(c_int), value :: exner_now_size_1

         type(c_ptr), value, target :: exner_new

         integer(c_int), value :: exner_new_size_0

         integer(c_int), value :: exner_new_size_1

         type(c_ptr), value, target :: w_now

         integer(c_int), value :: w_now_size_0

         integer(c_int), value :: w_now_size_1

         type(c_ptr), value, target :: w_new

         integer(c_int), value :: w_new_size_0

         integer(c_int), value :: w_new_size_1

         type(c_ptr), value, target :: theta_v_now

         integer(c_int), value :: theta_v_now_size_0

         integer(c_int), value :: theta_v_now_size_1

         type(c_ptr), value, target :: theta_v_new

         integer(c_int), value :: theta_v_new_size_0

         integer(c_int), value :: theta_v_new_size_1

         type(c_ptr), value, target :: vn_now

         integer(c_int), value :: vn_now_size_0

         integer(c_int), value :: vn_now_size_1

         type(c_ptr), value, target :: vn_new

         integer(c_int), value :: vn_new_size_0

         integer(c_int), value :: vn_new_size_1

         type(c_ptr), value, target :: w_concorr_c

         integer(c_int), value :: w_concorr_c_size_0

         integer(c_int), value :: w_concorr_c_size_1

         type(c_ptr), value, target :: ddt_vn_apc_ntl1

         integer(c_int), value :: ddt_vn_apc_ntl1_size_0

         integer(c_int), value :: ddt_vn_apc_ntl1_size_1

         type(c_ptr), value, target :: ddt_vn_apc_ntl2

         integer(c_int), value :: ddt_vn_apc_ntl2_size_0

         integer(c_int), value :: ddt_vn_apc_ntl2_size_1

         type(c_ptr), value, target :: ddt_w_adv_ntl1

         integer(c_int), value :: ddt_w_adv_ntl1_size_0

         integer(c_int), value :: ddt_w_adv_ntl1_size_1

         type(c_ptr), value, target :: ddt_w_adv_ntl2

         integer(c_int), value :: ddt_w_adv_ntl2_size_0

         integer(c_int), value :: ddt_w_adv_ntl2_size_1

         type(c_ptr), value, target :: theta_v_ic

         integer(c_int), value :: theta_v_ic_size_0

         integer(c_int), value :: theta_v_ic_size_1

         type(c_ptr), value, target :: rho_ic

         integer(c_int), value :: rho_ic_size_0

         integer(c_int), value :: rho_ic_size_1

         type(c_ptr), value, target :: exner_pr

         integer(c_int), value :: exner_pr_size_0

         integer(c_int), value :: exner_pr_size_1

         type(c_ptr), value, target :: exner_dyn_incr

         integer(c_int), value :: exner_dyn_incr_size_0

         integer(c_int), value :: exner_dyn_incr_size_1

         type(c_ptr), value, target :: ddt_exner_phy

         integer(c_int), value :: ddt_exner_phy_size_0

         integer(c_int), value :: ddt_exner_phy_size_1

         type(c_ptr), value, target :: grf_tend_rho

         integer(c_int), value :: grf_tend_rho_size_0

         integer(c_int), value :: grf_tend_rho_size_1

         type(c_ptr), value, target :: grf_tend_thv

         integer(c_int), value :: grf_tend_thv_size_0

         integer(c_int), value :: grf_tend_thv_size_1

         type(c_ptr), value, target :: grf_tend_w

         integer(c_int), value :: grf_tend_w_size_0

         integer(c_int), value :: grf_tend_w_size_1

         type(c_ptr), value, target :: mass_fl_e

         integer(c_int), value :: mass_fl_e_size_0

         integer(c_int), value :: mass_fl_e_size_1

         type(c_ptr), value, target :: ddt_vn_phy

         integer(c_int), value :: ddt_vn_phy_size_0

         integer(c_int), value :: ddt_vn_phy_size_1

         type(c_ptr), value, target :: grf_tend_vn

         integer(c_int), value :: grf_tend_vn_size_0

         integer(c_int), value :: grf_tend_vn_size_1

         type(c_ptr), value, target :: vn_ie

         integer(c_int), value :: vn_ie_size_0

         integer(c_int), value :: vn_ie_size_1

         type(c_ptr), value, target :: vt

         integer(c_int), value :: vt_size_0

         integer(c_int), value :: vt_size_1

         type(c_ptr), value, target :: vn_incr

         integer(c_int), value :: vn_incr_size_0

         integer(c_int), value :: vn_incr_size_1

         type(c_ptr), value, target :: rho_incr

         integer(c_int), value :: rho_incr_size_0

         integer(c_int), value :: rho_incr_size_1

         type(c_ptr), value, target :: exner_incr

         integer(c_int), value :: exner_incr_size_0

         integer(c_int), value :: exner_incr_size_1

         type(c_ptr), value, target :: mass_flx_me

         integer(c_int), value :: mass_flx_me_size_0

         integer(c_int), value :: mass_flx_me_size_1

         type(c_ptr), value, target :: mass_flx_ic

         integer(c_int), value :: mass_flx_ic_size_0

         integer(c_int), value :: mass_flx_ic_size_1

         type(c_ptr), value, target :: vol_flx_ic

         integer(c_int), value :: vol_flx_ic_size_0

         integer(c_int), value :: vol_flx_ic_size_1

         type(c_ptr), value, target :: vn_traj

         integer(c_int), value :: vn_traj_size_0

         integer(c_int), value :: vn_traj_size_1

         real(c_double), value, target :: dtime

         type(c_ptr), value, target :: max_vcfl_size1_array

         integer(c_int), value :: max_vcfl_size1_array_size_0

         logical(c_bool), value, target :: lprep_adv

         logical(c_bool), value, target :: at_initial_timestep

         real(c_double), value, target :: divdamp_fac_o2

         integer(c_int), value, target :: ndyn_substeps_var

         integer(c_int), value, target :: idyn_timestep

         logical(c_bool), value, target :: is_iau_active

         real(c_double), value, target :: iau_wgt_dyn

         logical(c_bool), value :: on_gpu

      end function solve_nh_run_wrapper

   end interface

contains

   subroutine diffusion_init(theta_ref_mc, &
                             wgtfac_c, &
                             e_bln_c_s, &
                             geofac_div, &
                             geofac_grg_x, &
                             geofac_grg_y, &
                             geofac_n2s, &
                             nudgecoeff_e, &
                             rbf_vec_coeff_v, &
                             zd_cellidx, &
                             zd_vertidx, &
                             zd_intcoef, &
                             zd_diffcoef, &
                             ndyn_substeps, &
                             diffusion_type, &
                             hdiff_w, &
                             hdiff_vn, &
                             hdiff_smag_w, &
                             zdiffu_t, &
                             type_t_diffu, &
                             type_vn_diffu, &
                             hdiff_efdt_ratio, &
                             hdiff_w_efdt_ratio, &
                             smagorinski_scaling_factor, &
                             smagorinski_scaling_factor2, &
                             smagorinski_scaling_factor3, &
                             smagorinski_scaling_factor4, &
                             smagorinski_scaling_height, &
                             smagorinski_scaling_height2, &
                             smagorinski_scaling_height3, &
                             smagorinski_scaling_height4, &
                             hdiff_temp, &
                             denom_diffu_v, &
                             nudge_max_coeff, &
                             itype_sher, &
                             iforcing, &
                             a_hshr, &
                             loutshs, &
                             backend, &
                             rc)
      use, intrinsic :: iso_c_binding

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: theta_ref_mc

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: wgtfac_c

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: e_bln_c_s

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: geofac_div

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: geofac_grg_x

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: geofac_grg_y

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: geofac_n2s

      real(c_double), dimension(:), contiguous, intent(inout), target :: nudgecoeff_e

      real(c_double), dimension(:, :, :), contiguous, intent(inout), target :: rbf_vec_coeff_v

      integer(c_int), dimension(:, :), contiguous, intent(inout), pointer :: zd_cellidx

      integer(c_int), dimension(:, :), contiguous, intent(inout), pointer :: zd_vertidx

      real(c_double), dimension(:, :), contiguous, intent(inout), pointer :: zd_intcoef

      real(c_double), dimension(:), contiguous, intent(inout), pointer :: zd_diffcoef

      integer(c_int), value, target :: ndyn_substeps

      integer(c_int), value, target :: diffusion_type

      logical(c_bool), value, target :: hdiff_w

      logical(c_bool), value, target :: hdiff_vn

      logical(c_bool), value, target :: hdiff_smag_w

      logical(c_bool), value, target :: zdiffu_t

      integer(c_int), value, target :: type_t_diffu

      integer(c_int), value, target :: type_vn_diffu

      real(c_double), value, target :: hdiff_efdt_ratio

      real(c_double), value, target :: hdiff_w_efdt_ratio

      real(c_double), value, target :: smagorinski_scaling_factor

      real(c_double), value, target :: smagorinski_scaling_factor2

      real(c_double), value, target :: smagorinski_scaling_factor3

      real(c_double), value, target :: smagorinski_scaling_factor4

      real(c_double), value, target :: smagorinski_scaling_height

      real(c_double), value, target :: smagorinski_scaling_height2

      real(c_double), value, target :: smagorinski_scaling_height3

      real(c_double), value, target :: smagorinski_scaling_height4

      logical(c_bool), value, target :: hdiff_temp

      real(c_double), value, target :: denom_diffu_v

      real(c_double), value, target :: nudge_max_coeff

      integer(c_int), value, target :: itype_sher

      integer(c_int), value, target :: iforcing

      real(c_double), value, target :: a_hshr

      logical(c_bool), value, target :: loutshs

      integer(c_int), value, target :: backend

      logical(c_bool) :: on_gpu

      integer(c_int) :: theta_ref_mc_size_0

      integer(c_int) :: theta_ref_mc_size_1

      integer(c_int) :: wgtfac_c_size_0

      integer(c_int) :: wgtfac_c_size_1

      integer(c_int) :: e_bln_c_s_size_0

      integer(c_int) :: e_bln_c_s_size_1

      integer(c_int) :: geofac_div_size_0

      integer(c_int) :: geofac_div_size_1

      integer(c_int) :: geofac_grg_x_size_0

      integer(c_int) :: geofac_grg_x_size_1

      integer(c_int) :: geofac_grg_y_size_0

      integer(c_int) :: geofac_grg_y_size_1

      integer(c_int) :: geofac_n2s_size_0

      integer(c_int) :: geofac_n2s_size_1

      integer(c_int) :: nudgecoeff_e_size_0

      integer(c_int) :: rbf_vec_coeff_v_size_0

      integer(c_int) :: rbf_vec_coeff_v_size_1

      integer(c_int) :: rbf_vec_coeff_v_size_2

      integer(c_int) :: zd_cellidx_size_0

      integer(c_int) :: zd_cellidx_size_1

      integer(c_int) :: zd_vertidx_size_0

      integer(c_int) :: zd_vertidx_size_1

      integer(c_int) :: zd_intcoef_size_0

      integer(c_int) :: zd_intcoef_size_1

      integer(c_int) :: zd_diffcoef_size_0

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      type(c_ptr) :: zd_cellidx_ptr

      type(c_ptr) :: zd_vertidx_ptr

      type(c_ptr) :: zd_intcoef_ptr

      type(c_ptr) :: zd_diffcoef_ptr

      zd_cellidx_ptr = c_null_ptr

      zd_vertidx_ptr = c_null_ptr

      zd_intcoef_ptr = c_null_ptr

      zd_diffcoef_ptr = c_null_ptr

      !$acc host_data use_device(theta_ref_mc)
      !$acc host_data use_device(wgtfac_c)
      !$acc host_data use_device(e_bln_c_s)
      !$acc host_data use_device(geofac_div)
      !$acc host_data use_device(geofac_grg_x)
      !$acc host_data use_device(geofac_grg_y)
      !$acc host_data use_device(geofac_n2s)
      !$acc host_data use_device(nudgecoeff_e)
      !$acc host_data use_device(rbf_vec_coeff_v)
      !$acc host_data use_device(zd_cellidx) if(associated(zd_cellidx))
      !$acc host_data use_device(zd_vertidx) if(associated(zd_vertidx))
      !$acc host_data use_device(zd_intcoef) if(associated(zd_intcoef))
      !$acc host_data use_device(zd_diffcoef) if(associated(zd_diffcoef))

#ifdef _OPENACC
      on_gpu = .True.
#else
      on_gpu = .False.
#endif

      theta_ref_mc_size_0 = SIZE(theta_ref_mc, 1)
      theta_ref_mc_size_1 = SIZE(theta_ref_mc, 2)

      wgtfac_c_size_0 = SIZE(wgtfac_c, 1)
      wgtfac_c_size_1 = SIZE(wgtfac_c, 2)

      e_bln_c_s_size_0 = SIZE(e_bln_c_s, 1)
      e_bln_c_s_size_1 = SIZE(e_bln_c_s, 2)

      geofac_div_size_0 = SIZE(geofac_div, 1)
      geofac_div_size_1 = SIZE(geofac_div, 2)

      geofac_grg_x_size_0 = SIZE(geofac_grg_x, 1)
      geofac_grg_x_size_1 = SIZE(geofac_grg_x, 2)

      geofac_grg_y_size_0 = SIZE(geofac_grg_y, 1)
      geofac_grg_y_size_1 = SIZE(geofac_grg_y, 2)

      geofac_n2s_size_0 = SIZE(geofac_n2s, 1)
      geofac_n2s_size_1 = SIZE(geofac_n2s, 2)

      nudgecoeff_e_size_0 = SIZE(nudgecoeff_e, 1)

      rbf_vec_coeff_v_size_0 = SIZE(rbf_vec_coeff_v, 1)
      rbf_vec_coeff_v_size_1 = SIZE(rbf_vec_coeff_v, 2)
      rbf_vec_coeff_v_size_2 = SIZE(rbf_vec_coeff_v, 3)

      if (associated(zd_cellidx)) then
         zd_cellidx_ptr = c_loc(zd_cellidx)
         zd_cellidx_size_0 = SIZE(zd_cellidx, 1)
         zd_cellidx_size_1 = SIZE(zd_cellidx, 2)
      end if

      if (associated(zd_vertidx)) then
         zd_vertidx_ptr = c_loc(zd_vertidx)
         zd_vertidx_size_0 = SIZE(zd_vertidx, 1)
         zd_vertidx_size_1 = SIZE(zd_vertidx, 2)
      end if

      if (associated(zd_intcoef)) then
         zd_intcoef_ptr = c_loc(zd_intcoef)
         zd_intcoef_size_0 = SIZE(zd_intcoef, 1)
         zd_intcoef_size_1 = SIZE(zd_intcoef, 2)
      end if

      if (associated(zd_diffcoef)) then
         zd_diffcoef_ptr = c_loc(zd_diffcoef)
         zd_diffcoef_size_0 = SIZE(zd_diffcoef, 1)
      end if

      rc = diffusion_init_wrapper(theta_ref_mc=c_loc(theta_ref_mc), &
                                  theta_ref_mc_size_0=theta_ref_mc_size_0, &
                                  theta_ref_mc_size_1=theta_ref_mc_size_1, &
                                  wgtfac_c=c_loc(wgtfac_c), &
                                  wgtfac_c_size_0=wgtfac_c_size_0, &
                                  wgtfac_c_size_1=wgtfac_c_size_1, &
                                  e_bln_c_s=c_loc(e_bln_c_s), &
                                  e_bln_c_s_size_0=e_bln_c_s_size_0, &
                                  e_bln_c_s_size_1=e_bln_c_s_size_1, &
                                  geofac_div=c_loc(geofac_div), &
                                  geofac_div_size_0=geofac_div_size_0, &
                                  geofac_div_size_1=geofac_div_size_1, &
                                  geofac_grg_x=c_loc(geofac_grg_x), &
                                  geofac_grg_x_size_0=geofac_grg_x_size_0, &
                                  geofac_grg_x_size_1=geofac_grg_x_size_1, &
                                  geofac_grg_y=c_loc(geofac_grg_y), &
                                  geofac_grg_y_size_0=geofac_grg_y_size_0, &
                                  geofac_grg_y_size_1=geofac_grg_y_size_1, &
                                  geofac_n2s=c_loc(geofac_n2s), &
                                  geofac_n2s_size_0=geofac_n2s_size_0, &
                                  geofac_n2s_size_1=geofac_n2s_size_1, &
                                  nudgecoeff_e=c_loc(nudgecoeff_e), &
                                  nudgecoeff_e_size_0=nudgecoeff_e_size_0, &
                                  rbf_vec_coeff_v=c_loc(rbf_vec_coeff_v), &
                                  rbf_vec_coeff_v_size_0=rbf_vec_coeff_v_size_0, &
                                  rbf_vec_coeff_v_size_1=rbf_vec_coeff_v_size_1, &
                                  rbf_vec_coeff_v_size_2=rbf_vec_coeff_v_size_2, &
                                  zd_cellidx=zd_cellidx_ptr, &
                                  zd_cellidx_size_0=zd_cellidx_size_0, &
                                  zd_cellidx_size_1=zd_cellidx_size_1, &
                                  zd_vertidx=zd_vertidx_ptr, &
                                  zd_vertidx_size_0=zd_vertidx_size_0, &
                                  zd_vertidx_size_1=zd_vertidx_size_1, &
                                  zd_intcoef=zd_intcoef_ptr, &
                                  zd_intcoef_size_0=zd_intcoef_size_0, &
                                  zd_intcoef_size_1=zd_intcoef_size_1, &
                                  zd_diffcoef=zd_diffcoef_ptr, &
                                  zd_diffcoef_size_0=zd_diffcoef_size_0, &
                                  ndyn_substeps=ndyn_substeps, &
                                  diffusion_type=diffusion_type, &
                                  hdiff_w=hdiff_w, &
                                  hdiff_vn=hdiff_vn, &
                                  hdiff_smag_w=hdiff_smag_w, &
                                  zdiffu_t=zdiffu_t, &
                                  type_t_diffu=type_t_diffu, &
                                  type_vn_diffu=type_vn_diffu, &
                                  hdiff_efdt_ratio=hdiff_efdt_ratio, &
                                  hdiff_w_efdt_ratio=hdiff_w_efdt_ratio, &
                                  smagorinski_scaling_factor=smagorinski_scaling_factor, &
                                  smagorinski_scaling_factor2=smagorinski_scaling_factor2, &
                                  smagorinski_scaling_factor3=smagorinski_scaling_factor3, &
                                  smagorinski_scaling_factor4=smagorinski_scaling_factor4, &
                                  smagorinski_scaling_height=smagorinski_scaling_height, &
                                  smagorinski_scaling_height2=smagorinski_scaling_height2, &
                                  smagorinski_scaling_height3=smagorinski_scaling_height3, &
                                  smagorinski_scaling_height4=smagorinski_scaling_height4, &
                                  hdiff_temp=hdiff_temp, &
                                  denom_diffu_v=denom_diffu_v, &
                                  nudge_max_coeff=nudge_max_coeff, &
                                  itype_sher=itype_sher, &
                                  iforcing=iforcing, &
                                  a_hshr=a_hshr, &
                                  loutshs=loutshs, &
                                  backend=backend, &
                                  on_gpu=on_gpu)
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
   end subroutine diffusion_init

   subroutine diffusion_run(w, &
                            vn, &
                            exner, &
                            theta_v, &
                            rho, &
                            hdef_ic, &
                            div_ic, &
                            dwdx, &
                            dwdy, &
                            dtime, &
                            linit, &
                            rc)
      use, intrinsic :: iso_c_binding

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: w

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: vn

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: exner

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: theta_v

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: rho

      real(c_double), dimension(:, :), contiguous, intent(inout), pointer :: hdef_ic

      real(c_double), dimension(:, :), contiguous, intent(inout), pointer :: div_ic

      real(c_double), dimension(:, :), contiguous, intent(inout), pointer :: dwdx

      real(c_double), dimension(:, :), contiguous, intent(inout), pointer :: dwdy

      real(c_double), value, target :: dtime

      logical(c_bool), value, target :: linit

      logical(c_bool) :: on_gpu

      integer(c_int) :: w_size_0

      integer(c_int) :: w_size_1

      integer(c_int) :: vn_size_0

      integer(c_int) :: vn_size_1

      integer(c_int) :: exner_size_0

      integer(c_int) :: exner_size_1

      integer(c_int) :: theta_v_size_0

      integer(c_int) :: theta_v_size_1

      integer(c_int) :: rho_size_0

      integer(c_int) :: rho_size_1

      integer(c_int) :: hdef_ic_size_0

      integer(c_int) :: hdef_ic_size_1

      integer(c_int) :: div_ic_size_0

      integer(c_int) :: div_ic_size_1

      integer(c_int) :: dwdx_size_0

      integer(c_int) :: dwdx_size_1

      integer(c_int) :: dwdy_size_0

      integer(c_int) :: dwdy_size_1

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      type(c_ptr) :: hdef_ic_ptr

      type(c_ptr) :: div_ic_ptr

      type(c_ptr) :: dwdx_ptr

      type(c_ptr) :: dwdy_ptr

      hdef_ic_ptr = c_null_ptr

      div_ic_ptr = c_null_ptr

      dwdx_ptr = c_null_ptr

      dwdy_ptr = c_null_ptr

      !$acc host_data use_device(w)
      !$acc host_data use_device(vn)
      !$acc host_data use_device(exner)
      !$acc host_data use_device(theta_v)
      !$acc host_data use_device(rho)
      !$acc host_data use_device(hdef_ic) if(associated(hdef_ic))
      !$acc host_data use_device(div_ic) if(associated(div_ic))
      !$acc host_data use_device(dwdx) if(associated(dwdx))
      !$acc host_data use_device(dwdy) if(associated(dwdy))

#ifdef _OPENACC
      on_gpu = .True.
#else
      on_gpu = .False.
#endif

      w_size_0 = SIZE(w, 1)
      w_size_1 = SIZE(w, 2)

      vn_size_0 = SIZE(vn, 1)
      vn_size_1 = SIZE(vn, 2)

      exner_size_0 = SIZE(exner, 1)
      exner_size_1 = SIZE(exner, 2)

      theta_v_size_0 = SIZE(theta_v, 1)
      theta_v_size_1 = SIZE(theta_v, 2)

      rho_size_0 = SIZE(rho, 1)
      rho_size_1 = SIZE(rho, 2)

      if (associated(hdef_ic)) then
         hdef_ic_ptr = c_loc(hdef_ic)
         hdef_ic_size_0 = SIZE(hdef_ic, 1)
         hdef_ic_size_1 = SIZE(hdef_ic, 2)
      end if

      if (associated(div_ic)) then
         div_ic_ptr = c_loc(div_ic)
         div_ic_size_0 = SIZE(div_ic, 1)
         div_ic_size_1 = SIZE(div_ic, 2)
      end if

      if (associated(dwdx)) then
         dwdx_ptr = c_loc(dwdx)
         dwdx_size_0 = SIZE(dwdx, 1)
         dwdx_size_1 = SIZE(dwdx, 2)
      end if

      if (associated(dwdy)) then
         dwdy_ptr = c_loc(dwdy)
         dwdy_size_0 = SIZE(dwdy, 1)
         dwdy_size_1 = SIZE(dwdy, 2)
      end if

      rc = diffusion_run_wrapper(w=c_loc(w), &
                                 w_size_0=w_size_0, &
                                 w_size_1=w_size_1, &
                                 vn=c_loc(vn), &
                                 vn_size_0=vn_size_0, &
                                 vn_size_1=vn_size_1, &
                                 exner=c_loc(exner), &
                                 exner_size_0=exner_size_0, &
                                 exner_size_1=exner_size_1, &
                                 theta_v=c_loc(theta_v), &
                                 theta_v_size_0=theta_v_size_0, &
                                 theta_v_size_1=theta_v_size_1, &
                                 rho=c_loc(rho), &
                                 rho_size_0=rho_size_0, &
                                 rho_size_1=rho_size_1, &
                                 hdef_ic=hdef_ic_ptr, &
                                 hdef_ic_size_0=hdef_ic_size_0, &
                                 hdef_ic_size_1=hdef_ic_size_1, &
                                 div_ic=div_ic_ptr, &
                                 div_ic_size_0=div_ic_size_0, &
                                 div_ic_size_1=div_ic_size_1, &
                                 dwdx=dwdx_ptr, &
                                 dwdx_size_0=dwdx_size_0, &
                                 dwdx_size_1=dwdx_size_1, &
                                 dwdy=dwdy_ptr, &
                                 dwdy_size_0=dwdy_size_0, &
                                 dwdy_size_1=dwdy_size_1, &
                                 dtime=dtime, &
                                 linit=linit, &
                                 on_gpu=on_gpu)
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
   end subroutine diffusion_run

   subroutine grid_init(cell_starts, &
                        cell_ends, &
                        vertex_starts, &
                        vertex_ends, &
                        edge_starts, &
                        edge_ends, &
                        c2e, &
                        e2c, &
                        c2e2c, &
                        e2c2e, &
                        e2v, &
                        v2e, &
                        v2c, &
                        e2c2v, &
                        c2v, &
                        c_owner_mask, &
                        e_owner_mask, &
                        v_owner_mask, &
                        c_glb_index, &
                        e_glb_index, &
                        v_glb_index, &
                        tangent_orientation, &
                        inverse_primal_edge_lengths, &
                        inv_dual_edge_length, &
                        inv_vert_vert_length, &
                        edge_areas, &
                        f_e, &
                        cell_center_lat, &
                        cell_center_lon, &
                        cell_areas, &
                        primal_normal_vert_x, &
                        primal_normal_vert_y, &
                        dual_normal_vert_x, &
                        dual_normal_vert_y, &
                        primal_normal_cell_x, &
                        primal_normal_cell_y, &
                        dual_normal_cell_x, &
                        dual_normal_cell_y, &
                        edge_center_lat, &
                        edge_center_lon, &
                        primal_normal_x, &
                        primal_normal_y, &
                        vct_a, &
                        lowest_layer_thickness, &
                        model_top_height, &
                        stretch_factor, &
                        flat_height, &
                        rayleigh_damping_height, &
                        mean_cell_area, &
                        comm_id, &
                        num_vertices, &
                        num_cells, &
                        num_edges, &
                        vertical_size, &
                        limited_area, &
                        backend, &
                        rc)
      use, intrinsic :: iso_c_binding

      integer(c_int), dimension(:), contiguous, intent(inout), target :: cell_starts

      integer(c_int), dimension(:), contiguous, intent(inout), target :: cell_ends

      integer(c_int), dimension(:), contiguous, intent(inout), target :: vertex_starts

      integer(c_int), dimension(:), contiguous, intent(inout), target :: vertex_ends

      integer(c_int), dimension(:), contiguous, intent(inout), target :: edge_starts

      integer(c_int), dimension(:), contiguous, intent(inout), target :: edge_ends

      integer(c_int), dimension(:, :), contiguous, intent(inout), target :: c2e

      integer(c_int), dimension(:, :), contiguous, intent(inout), target :: e2c

      integer(c_int), dimension(:, :), contiguous, intent(inout), target :: c2e2c

      integer(c_int), dimension(:, :), contiguous, intent(inout), target :: e2c2e

      integer(c_int), dimension(:, :), contiguous, intent(inout), target :: e2v

      integer(c_int), dimension(:, :), contiguous, intent(inout), target :: v2e

      integer(c_int), dimension(:, :), contiguous, intent(inout), target :: v2c

      integer(c_int), dimension(:, :), contiguous, intent(inout), target :: e2c2v

      integer(c_int), dimension(:, :), contiguous, intent(inout), target :: c2v

      logical(c_bool), dimension(:), contiguous, intent(inout), target :: c_owner_mask

      logical(c_bool), dimension(:), contiguous, intent(inout), target :: e_owner_mask

      logical(c_bool), dimension(:), contiguous, intent(inout), target :: v_owner_mask

      integer(c_int), dimension(:), contiguous, intent(inout), target :: c_glb_index

      integer(c_int), dimension(:), contiguous, intent(inout), target :: e_glb_index

      integer(c_int), dimension(:), contiguous, intent(inout), target :: v_glb_index

      real(c_double), dimension(:), contiguous, intent(inout), target :: tangent_orientation

      real(c_double), dimension(:), contiguous, intent(inout), target :: inverse_primal_edge_lengths

      real(c_double), dimension(:), contiguous, intent(inout), target :: inv_dual_edge_length

      real(c_double), dimension(:), contiguous, intent(inout), target :: inv_vert_vert_length

      real(c_double), dimension(:), contiguous, intent(inout), target :: edge_areas

      real(c_double), dimension(:), contiguous, intent(inout), target :: f_e

      real(c_double), dimension(:), contiguous, intent(inout), target :: cell_center_lat

      real(c_double), dimension(:), contiguous, intent(inout), target :: cell_center_lon

      real(c_double), dimension(:), contiguous, intent(inout), target :: cell_areas

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: primal_normal_vert_x

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: primal_normal_vert_y

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: dual_normal_vert_x

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: dual_normal_vert_y

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: primal_normal_cell_x

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: primal_normal_cell_y

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: dual_normal_cell_x

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: dual_normal_cell_y

      real(c_double), dimension(:), contiguous, intent(inout), target :: edge_center_lat

      real(c_double), dimension(:), contiguous, intent(inout), target :: edge_center_lon

      real(c_double), dimension(:), contiguous, intent(inout), target :: primal_normal_x

      real(c_double), dimension(:), contiguous, intent(inout), target :: primal_normal_y

      real(c_double), dimension(:), contiguous, intent(inout), target :: vct_a

      real(c_double), value, target :: lowest_layer_thickness

      real(c_double), value, target :: model_top_height

      real(c_double), value, target :: stretch_factor

      real(c_double), value, target :: flat_height

      real(c_double), value, target :: rayleigh_damping_height

      real(c_double), value, target :: mean_cell_area

      integer(c_int), value, target :: comm_id

      integer(c_int), value, target :: num_vertices

      integer(c_int), value, target :: num_cells

      integer(c_int), value, target :: num_edges

      integer(c_int), value, target :: vertical_size

      logical(c_bool), value, target :: limited_area

      integer(c_int), value, target :: backend

      logical(c_bool) :: on_gpu

      integer(c_int) :: cell_starts_size_0

      integer(c_int) :: cell_ends_size_0

      integer(c_int) :: vertex_starts_size_0

      integer(c_int) :: vertex_ends_size_0

      integer(c_int) :: edge_starts_size_0

      integer(c_int) :: edge_ends_size_0

      integer(c_int) :: c2e_size_0

      integer(c_int) :: c2e_size_1

      integer(c_int) :: e2c_size_0

      integer(c_int) :: e2c_size_1

      integer(c_int) :: c2e2c_size_0

      integer(c_int) :: c2e2c_size_1

      integer(c_int) :: e2c2e_size_0

      integer(c_int) :: e2c2e_size_1

      integer(c_int) :: e2v_size_0

      integer(c_int) :: e2v_size_1

      integer(c_int) :: v2e_size_0

      integer(c_int) :: v2e_size_1

      integer(c_int) :: v2c_size_0

      integer(c_int) :: v2c_size_1

      integer(c_int) :: e2c2v_size_0

      integer(c_int) :: e2c2v_size_1

      integer(c_int) :: c2v_size_0

      integer(c_int) :: c2v_size_1

      integer(c_int) :: c_owner_mask_size_0

      integer(c_int) :: e_owner_mask_size_0

      integer(c_int) :: v_owner_mask_size_0

      integer(c_int) :: c_glb_index_size_0

      integer(c_int) :: e_glb_index_size_0

      integer(c_int) :: v_glb_index_size_0

      integer(c_int) :: tangent_orientation_size_0

      integer(c_int) :: inverse_primal_edge_lengths_size_0

      integer(c_int) :: inv_dual_edge_length_size_0

      integer(c_int) :: inv_vert_vert_length_size_0

      integer(c_int) :: edge_areas_size_0

      integer(c_int) :: f_e_size_0

      integer(c_int) :: cell_center_lat_size_0

      integer(c_int) :: cell_center_lon_size_0

      integer(c_int) :: cell_areas_size_0

      integer(c_int) :: primal_normal_vert_x_size_0

      integer(c_int) :: primal_normal_vert_x_size_1

      integer(c_int) :: primal_normal_vert_y_size_0

      integer(c_int) :: primal_normal_vert_y_size_1

      integer(c_int) :: dual_normal_vert_x_size_0

      integer(c_int) :: dual_normal_vert_x_size_1

      integer(c_int) :: dual_normal_vert_y_size_0

      integer(c_int) :: dual_normal_vert_y_size_1

      integer(c_int) :: primal_normal_cell_x_size_0

      integer(c_int) :: primal_normal_cell_x_size_1

      integer(c_int) :: primal_normal_cell_y_size_0

      integer(c_int) :: primal_normal_cell_y_size_1

      integer(c_int) :: dual_normal_cell_x_size_0

      integer(c_int) :: dual_normal_cell_x_size_1

      integer(c_int) :: dual_normal_cell_y_size_0

      integer(c_int) :: dual_normal_cell_y_size_1

      integer(c_int) :: edge_center_lat_size_0

      integer(c_int) :: edge_center_lon_size_0

      integer(c_int) :: primal_normal_x_size_0

      integer(c_int) :: primal_normal_y_size_0

      integer(c_int) :: vct_a_size_0

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      !$acc host_data use_device(c2e)
      !$acc host_data use_device(e2c)
      !$acc host_data use_device(c2e2c)
      !$acc host_data use_device(e2c2e)
      !$acc host_data use_device(e2v)
      !$acc host_data use_device(v2e)
      !$acc host_data use_device(v2c)
      !$acc host_data use_device(e2c2v)
      !$acc host_data use_device(c2v)
      !$acc host_data use_device(tangent_orientation)
      !$acc host_data use_device(inverse_primal_edge_lengths)
      !$acc host_data use_device(inv_dual_edge_length)
      !$acc host_data use_device(inv_vert_vert_length)
      !$acc host_data use_device(edge_areas)
      !$acc host_data use_device(f_e)
      !$acc host_data use_device(cell_center_lat)
      !$acc host_data use_device(cell_center_lon)
      !$acc host_data use_device(cell_areas)
      !$acc host_data use_device(primal_normal_vert_x)
      !$acc host_data use_device(primal_normal_vert_y)
      !$acc host_data use_device(dual_normal_vert_x)
      !$acc host_data use_device(dual_normal_vert_y)
      !$acc host_data use_device(primal_normal_cell_x)
      !$acc host_data use_device(primal_normal_cell_y)
      !$acc host_data use_device(dual_normal_cell_x)
      !$acc host_data use_device(dual_normal_cell_y)
      !$acc host_data use_device(edge_center_lat)
      !$acc host_data use_device(edge_center_lon)
      !$acc host_data use_device(primal_normal_x)
      !$acc host_data use_device(primal_normal_y)
      !$acc host_data use_device(vct_a)

#ifdef _OPENACC
      on_gpu = .True.
#else
      on_gpu = .False.
#endif

      cell_starts_size_0 = SIZE(cell_starts, 1)

      cell_ends_size_0 = SIZE(cell_ends, 1)

      vertex_starts_size_0 = SIZE(vertex_starts, 1)

      vertex_ends_size_0 = SIZE(vertex_ends, 1)

      edge_starts_size_0 = SIZE(edge_starts, 1)

      edge_ends_size_0 = SIZE(edge_ends, 1)

      c2e_size_0 = SIZE(c2e, 1)
      c2e_size_1 = SIZE(c2e, 2)

      e2c_size_0 = SIZE(e2c, 1)
      e2c_size_1 = SIZE(e2c, 2)

      c2e2c_size_0 = SIZE(c2e2c, 1)
      c2e2c_size_1 = SIZE(c2e2c, 2)

      e2c2e_size_0 = SIZE(e2c2e, 1)
      e2c2e_size_1 = SIZE(e2c2e, 2)

      e2v_size_0 = SIZE(e2v, 1)
      e2v_size_1 = SIZE(e2v, 2)

      v2e_size_0 = SIZE(v2e, 1)
      v2e_size_1 = SIZE(v2e, 2)

      v2c_size_0 = SIZE(v2c, 1)
      v2c_size_1 = SIZE(v2c, 2)

      e2c2v_size_0 = SIZE(e2c2v, 1)
      e2c2v_size_1 = SIZE(e2c2v, 2)

      c2v_size_0 = SIZE(c2v, 1)
      c2v_size_1 = SIZE(c2v, 2)

      c_owner_mask_size_0 = SIZE(c_owner_mask, 1)

      e_owner_mask_size_0 = SIZE(e_owner_mask, 1)

      v_owner_mask_size_0 = SIZE(v_owner_mask, 1)

      c_glb_index_size_0 = SIZE(c_glb_index, 1)

      e_glb_index_size_0 = SIZE(e_glb_index, 1)

      v_glb_index_size_0 = SIZE(v_glb_index, 1)

      tangent_orientation_size_0 = SIZE(tangent_orientation, 1)

      inverse_primal_edge_lengths_size_0 = SIZE(inverse_primal_edge_lengths, 1)

      inv_dual_edge_length_size_0 = SIZE(inv_dual_edge_length, 1)

      inv_vert_vert_length_size_0 = SIZE(inv_vert_vert_length, 1)

      edge_areas_size_0 = SIZE(edge_areas, 1)

      f_e_size_0 = SIZE(f_e, 1)

      cell_center_lat_size_0 = SIZE(cell_center_lat, 1)

      cell_center_lon_size_0 = SIZE(cell_center_lon, 1)

      cell_areas_size_0 = SIZE(cell_areas, 1)

      primal_normal_vert_x_size_0 = SIZE(primal_normal_vert_x, 1)
      primal_normal_vert_x_size_1 = SIZE(primal_normal_vert_x, 2)

      primal_normal_vert_y_size_0 = SIZE(primal_normal_vert_y, 1)
      primal_normal_vert_y_size_1 = SIZE(primal_normal_vert_y, 2)

      dual_normal_vert_x_size_0 = SIZE(dual_normal_vert_x, 1)
      dual_normal_vert_x_size_1 = SIZE(dual_normal_vert_x, 2)

      dual_normal_vert_y_size_0 = SIZE(dual_normal_vert_y, 1)
      dual_normal_vert_y_size_1 = SIZE(dual_normal_vert_y, 2)

      primal_normal_cell_x_size_0 = SIZE(primal_normal_cell_x, 1)
      primal_normal_cell_x_size_1 = SIZE(primal_normal_cell_x, 2)

      primal_normal_cell_y_size_0 = SIZE(primal_normal_cell_y, 1)
      primal_normal_cell_y_size_1 = SIZE(primal_normal_cell_y, 2)

      dual_normal_cell_x_size_0 = SIZE(dual_normal_cell_x, 1)
      dual_normal_cell_x_size_1 = SIZE(dual_normal_cell_x, 2)

      dual_normal_cell_y_size_0 = SIZE(dual_normal_cell_y, 1)
      dual_normal_cell_y_size_1 = SIZE(dual_normal_cell_y, 2)

      edge_center_lat_size_0 = SIZE(edge_center_lat, 1)

      edge_center_lon_size_0 = SIZE(edge_center_lon, 1)

      primal_normal_x_size_0 = SIZE(primal_normal_x, 1)

      primal_normal_y_size_0 = SIZE(primal_normal_y, 1)

      vct_a_size_0 = SIZE(vct_a, 1)

      rc = grid_init_wrapper(cell_starts=c_loc(cell_starts), &
                             cell_starts_size_0=cell_starts_size_0, &
                             cell_ends=c_loc(cell_ends), &
                             cell_ends_size_0=cell_ends_size_0, &
                             vertex_starts=c_loc(vertex_starts), &
                             vertex_starts_size_0=vertex_starts_size_0, &
                             vertex_ends=c_loc(vertex_ends), &
                             vertex_ends_size_0=vertex_ends_size_0, &
                             edge_starts=c_loc(edge_starts), &
                             edge_starts_size_0=edge_starts_size_0, &
                             edge_ends=c_loc(edge_ends), &
                             edge_ends_size_0=edge_ends_size_0, &
                             c2e=c_loc(c2e), &
                             c2e_size_0=c2e_size_0, &
                             c2e_size_1=c2e_size_1, &
                             e2c=c_loc(e2c), &
                             e2c_size_0=e2c_size_0, &
                             e2c_size_1=e2c_size_1, &
                             c2e2c=c_loc(c2e2c), &
                             c2e2c_size_0=c2e2c_size_0, &
                             c2e2c_size_1=c2e2c_size_1, &
                             e2c2e=c_loc(e2c2e), &
                             e2c2e_size_0=e2c2e_size_0, &
                             e2c2e_size_1=e2c2e_size_1, &
                             e2v=c_loc(e2v), &
                             e2v_size_0=e2v_size_0, &
                             e2v_size_1=e2v_size_1, &
                             v2e=c_loc(v2e), &
                             v2e_size_0=v2e_size_0, &
                             v2e_size_1=v2e_size_1, &
                             v2c=c_loc(v2c), &
                             v2c_size_0=v2c_size_0, &
                             v2c_size_1=v2c_size_1, &
                             e2c2v=c_loc(e2c2v), &
                             e2c2v_size_0=e2c2v_size_0, &
                             e2c2v_size_1=e2c2v_size_1, &
                             c2v=c_loc(c2v), &
                             c2v_size_0=c2v_size_0, &
                             c2v_size_1=c2v_size_1, &
                             c_owner_mask=c_loc(c_owner_mask), &
                             c_owner_mask_size_0=c_owner_mask_size_0, &
                             e_owner_mask=c_loc(e_owner_mask), &
                             e_owner_mask_size_0=e_owner_mask_size_0, &
                             v_owner_mask=c_loc(v_owner_mask), &
                             v_owner_mask_size_0=v_owner_mask_size_0, &
                             c_glb_index=c_loc(c_glb_index), &
                             c_glb_index_size_0=c_glb_index_size_0, &
                             e_glb_index=c_loc(e_glb_index), &
                             e_glb_index_size_0=e_glb_index_size_0, &
                             v_glb_index=c_loc(v_glb_index), &
                             v_glb_index_size_0=v_glb_index_size_0, &
                             tangent_orientation=c_loc(tangent_orientation), &
                             tangent_orientation_size_0=tangent_orientation_size_0, &
                             inverse_primal_edge_lengths=c_loc(inverse_primal_edge_lengths), &
                             inverse_primal_edge_lengths_size_0=inverse_primal_edge_lengths_size_0, &
                             inv_dual_edge_length=c_loc(inv_dual_edge_length), &
                             inv_dual_edge_length_size_0=inv_dual_edge_length_size_0, &
                             inv_vert_vert_length=c_loc(inv_vert_vert_length), &
                             inv_vert_vert_length_size_0=inv_vert_vert_length_size_0, &
                             edge_areas=c_loc(edge_areas), &
                             edge_areas_size_0=edge_areas_size_0, &
                             f_e=c_loc(f_e), &
                             f_e_size_0=f_e_size_0, &
                             cell_center_lat=c_loc(cell_center_lat), &
                             cell_center_lat_size_0=cell_center_lat_size_0, &
                             cell_center_lon=c_loc(cell_center_lon), &
                             cell_center_lon_size_0=cell_center_lon_size_0, &
                             cell_areas=c_loc(cell_areas), &
                             cell_areas_size_0=cell_areas_size_0, &
                             primal_normal_vert_x=c_loc(primal_normal_vert_x), &
                             primal_normal_vert_x_size_0=primal_normal_vert_x_size_0, &
                             primal_normal_vert_x_size_1=primal_normal_vert_x_size_1, &
                             primal_normal_vert_y=c_loc(primal_normal_vert_y), &
                             primal_normal_vert_y_size_0=primal_normal_vert_y_size_0, &
                             primal_normal_vert_y_size_1=primal_normal_vert_y_size_1, &
                             dual_normal_vert_x=c_loc(dual_normal_vert_x), &
                             dual_normal_vert_x_size_0=dual_normal_vert_x_size_0, &
                             dual_normal_vert_x_size_1=dual_normal_vert_x_size_1, &
                             dual_normal_vert_y=c_loc(dual_normal_vert_y), &
                             dual_normal_vert_y_size_0=dual_normal_vert_y_size_0, &
                             dual_normal_vert_y_size_1=dual_normal_vert_y_size_1, &
                             primal_normal_cell_x=c_loc(primal_normal_cell_x), &
                             primal_normal_cell_x_size_0=primal_normal_cell_x_size_0, &
                             primal_normal_cell_x_size_1=primal_normal_cell_x_size_1, &
                             primal_normal_cell_y=c_loc(primal_normal_cell_y), &
                             primal_normal_cell_y_size_0=primal_normal_cell_y_size_0, &
                             primal_normal_cell_y_size_1=primal_normal_cell_y_size_1, &
                             dual_normal_cell_x=c_loc(dual_normal_cell_x), &
                             dual_normal_cell_x_size_0=dual_normal_cell_x_size_0, &
                             dual_normal_cell_x_size_1=dual_normal_cell_x_size_1, &
                             dual_normal_cell_y=c_loc(dual_normal_cell_y), &
                             dual_normal_cell_y_size_0=dual_normal_cell_y_size_0, &
                             dual_normal_cell_y_size_1=dual_normal_cell_y_size_1, &
                             edge_center_lat=c_loc(edge_center_lat), &
                             edge_center_lat_size_0=edge_center_lat_size_0, &
                             edge_center_lon=c_loc(edge_center_lon), &
                             edge_center_lon_size_0=edge_center_lon_size_0, &
                             primal_normal_x=c_loc(primal_normal_x), &
                             primal_normal_x_size_0=primal_normal_x_size_0, &
                             primal_normal_y=c_loc(primal_normal_y), &
                             primal_normal_y_size_0=primal_normal_y_size_0, &
                             vct_a=c_loc(vct_a), &
                             vct_a_size_0=vct_a_size_0, &
                             lowest_layer_thickness=lowest_layer_thickness, &
                             model_top_height=model_top_height, &
                             stretch_factor=stretch_factor, &
                             flat_height=flat_height, &
                             rayleigh_damping_height=rayleigh_damping_height, &
                             mean_cell_area=mean_cell_area, &
                             comm_id=comm_id, &
                             num_vertices=num_vertices, &
                             num_cells=num_cells, &
                             num_edges=num_edges, &
                             vertical_size=vertical_size, &
                             limited_area=limited_area, &
                             backend=backend, &
                             on_gpu=on_gpu)
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
   end subroutine grid_init

   subroutine solve_nh_init(c_lin_e, &
                            c_intp, &
                            e_flx_avg, &
                            geofac_grdiv, &
                            geofac_rot, &
                            pos_on_tplane_e_1, &
                            pos_on_tplane_e_2, &
                            rbf_vec_coeff_e, &
                            e_bln_c_s, &
                            rbf_vec_coeff_v, &
                            geofac_div, &
                            geofac_n2s, &
                            geofac_grg_x, &
                            geofac_grg_y, &
                            nudgecoeff_e, &
                            mask_prog_halo_c, &
                            rayleigh_w, &
                            exner_exfac, &
                            exner_ref_mc, &
                            wgtfac_c, &
                            wgtfacq_c, &
                            inv_ddqz_z_full, &
                            rho_ref_mc, &
                            theta_ref_mc, &
                            vwind_expl_wgt, &
                            d_exner_dz_ref_ic, &
                            ddqz_z_half, &
                            theta_ref_ic, &
                            d2dexdz2_fac1_mc, &
                            d2dexdz2_fac2_mc, &
                            rho_ref_me, &
                            theta_ref_me, &
                            ddxn_z_full, &
                            zdiff_gradp, &
                            vertidx_gradp, &
                            pg_edgeidx, &
                            pg_vertidx, &
                            pg_exdist, &
                            ddqz_z_full_e, &
                            ddxt_z_full, &
                            wgtfac_e, &
                            wgtfacq_e, &
                            vwind_impl_wgt, &
                            hmask_dd3d, &
                            scalfac_dd3d, &
                            coeff1_dwdz, &
                            coeff2_dwdz, &
                            coeff_gradekin, &
                            c_owner_mask, &
                            itime_scheme, &
                            iadv_rhotheta, &
                            igradp_method, &
                            rayleigh_type, &
                            divdamp_order, &
                            divdamp_type, &
                            l_vert_nested, &
                            ldeepatmo, &
                            iau_init, &
                            extra_diffu, &
                            rhotheta_offctr, &
                            veladv_offctr, &
                            nudge_max_coeff, &
                            divdamp_fac, &
                            divdamp_fac2, &
                            divdamp_fac3, &
                            divdamp_fac4, &
                            divdamp_z, &
                            divdamp_z2, &
                            divdamp_z3, &
                            divdamp_z4, &
                            nflat_gradp, &
                            backend, &
                            rc)
      use, intrinsic :: iso_c_binding

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: c_lin_e

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: c_intp

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: e_flx_avg

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: geofac_grdiv

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: geofac_rot

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: pos_on_tplane_e_1

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: pos_on_tplane_e_2

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: rbf_vec_coeff_e

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: e_bln_c_s

      real(c_double), dimension(:, :, :), contiguous, intent(inout), target :: rbf_vec_coeff_v

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: geofac_div

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: geofac_n2s

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: geofac_grg_x

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: geofac_grg_y

      real(c_double), dimension(:), contiguous, intent(inout), target :: nudgecoeff_e

      logical(c_bool), dimension(:), contiguous, intent(inout), target :: mask_prog_halo_c

      real(c_double), dimension(:), contiguous, intent(inout), target :: rayleigh_w

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: exner_exfac

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: exner_ref_mc

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: wgtfac_c

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: wgtfacq_c

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: inv_ddqz_z_full

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: rho_ref_mc

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: theta_ref_mc

      real(c_double), dimension(:), contiguous, intent(inout), target :: vwind_expl_wgt

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: d_exner_dz_ref_ic

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: ddqz_z_half

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: theta_ref_ic

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: d2dexdz2_fac1_mc

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: d2dexdz2_fac2_mc

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: rho_ref_me

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: theta_ref_me

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: ddxn_z_full

      real(c_double), dimension(:, :, :), contiguous, intent(inout), target :: zdiff_gradp

      integer(c_int), dimension(:, :, :), contiguous, intent(inout), target :: vertidx_gradp

      integer(c_int), dimension(:), contiguous, intent(inout), pointer :: pg_edgeidx

      integer(c_int), dimension(:), contiguous, intent(inout), pointer :: pg_vertidx

      real(c_double), dimension(:), contiguous, intent(inout), pointer :: pg_exdist

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: ddqz_z_full_e

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: ddxt_z_full

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: wgtfac_e

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: wgtfacq_e

      real(c_double), dimension(:), contiguous, intent(inout), target :: vwind_impl_wgt

      real(c_double), dimension(:), contiguous, intent(inout), target :: hmask_dd3d

      real(c_double), dimension(:), contiguous, intent(inout), target :: scalfac_dd3d

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: coeff1_dwdz

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: coeff2_dwdz

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: coeff_gradekin

      logical(c_bool), dimension(:), contiguous, intent(inout), target :: c_owner_mask

      integer(c_int), value, target :: itime_scheme

      integer(c_int), value, target :: iadv_rhotheta

      integer(c_int), value, target :: igradp_method

      integer(c_int), value, target :: rayleigh_type

      integer(c_int), value, target :: divdamp_order

      integer(c_int), value, target :: divdamp_type

      logical(c_bool), value, target :: l_vert_nested

      logical(c_bool), value, target :: ldeepatmo

      logical(c_bool), value, target :: iau_init

      logical(c_bool), value, target :: extra_diffu

      real(c_double), value, target :: rhotheta_offctr

      real(c_double), value, target :: veladv_offctr

      real(c_double), value, target :: nudge_max_coeff

      real(c_double), value, target :: divdamp_fac

      real(c_double), value, target :: divdamp_fac2

      real(c_double), value, target :: divdamp_fac3

      real(c_double), value, target :: divdamp_fac4

      real(c_double), value, target :: divdamp_z

      real(c_double), value, target :: divdamp_z2

      real(c_double), value, target :: divdamp_z3

      real(c_double), value, target :: divdamp_z4

      integer(c_int), value, target :: nflat_gradp

      integer(c_int), value, target :: backend

      logical(c_bool) :: on_gpu

      integer(c_int) :: c_lin_e_size_0

      integer(c_int) :: c_lin_e_size_1

      integer(c_int) :: c_intp_size_0

      integer(c_int) :: c_intp_size_1

      integer(c_int) :: e_flx_avg_size_0

      integer(c_int) :: e_flx_avg_size_1

      integer(c_int) :: geofac_grdiv_size_0

      integer(c_int) :: geofac_grdiv_size_1

      integer(c_int) :: geofac_rot_size_0

      integer(c_int) :: geofac_rot_size_1

      integer(c_int) :: pos_on_tplane_e_1_size_0

      integer(c_int) :: pos_on_tplane_e_1_size_1

      integer(c_int) :: pos_on_tplane_e_2_size_0

      integer(c_int) :: pos_on_tplane_e_2_size_1

      integer(c_int) :: rbf_vec_coeff_e_size_0

      integer(c_int) :: rbf_vec_coeff_e_size_1

      integer(c_int) :: e_bln_c_s_size_0

      integer(c_int) :: e_bln_c_s_size_1

      integer(c_int) :: rbf_vec_coeff_v_size_0

      integer(c_int) :: rbf_vec_coeff_v_size_1

      integer(c_int) :: rbf_vec_coeff_v_size_2

      integer(c_int) :: geofac_div_size_0

      integer(c_int) :: geofac_div_size_1

      integer(c_int) :: geofac_n2s_size_0

      integer(c_int) :: geofac_n2s_size_1

      integer(c_int) :: geofac_grg_x_size_0

      integer(c_int) :: geofac_grg_x_size_1

      integer(c_int) :: geofac_grg_y_size_0

      integer(c_int) :: geofac_grg_y_size_1

      integer(c_int) :: nudgecoeff_e_size_0

      integer(c_int) :: mask_prog_halo_c_size_0

      integer(c_int) :: rayleigh_w_size_0

      integer(c_int) :: exner_exfac_size_0

      integer(c_int) :: exner_exfac_size_1

      integer(c_int) :: exner_ref_mc_size_0

      integer(c_int) :: exner_ref_mc_size_1

      integer(c_int) :: wgtfac_c_size_0

      integer(c_int) :: wgtfac_c_size_1

      integer(c_int) :: wgtfacq_c_size_0

      integer(c_int) :: wgtfacq_c_size_1

      integer(c_int) :: inv_ddqz_z_full_size_0

      integer(c_int) :: inv_ddqz_z_full_size_1

      integer(c_int) :: rho_ref_mc_size_0

      integer(c_int) :: rho_ref_mc_size_1

      integer(c_int) :: theta_ref_mc_size_0

      integer(c_int) :: theta_ref_mc_size_1

      integer(c_int) :: vwind_expl_wgt_size_0

      integer(c_int) :: d_exner_dz_ref_ic_size_0

      integer(c_int) :: d_exner_dz_ref_ic_size_1

      integer(c_int) :: ddqz_z_half_size_0

      integer(c_int) :: ddqz_z_half_size_1

      integer(c_int) :: theta_ref_ic_size_0

      integer(c_int) :: theta_ref_ic_size_1

      integer(c_int) :: d2dexdz2_fac1_mc_size_0

      integer(c_int) :: d2dexdz2_fac1_mc_size_1

      integer(c_int) :: d2dexdz2_fac2_mc_size_0

      integer(c_int) :: d2dexdz2_fac2_mc_size_1

      integer(c_int) :: rho_ref_me_size_0

      integer(c_int) :: rho_ref_me_size_1

      integer(c_int) :: theta_ref_me_size_0

      integer(c_int) :: theta_ref_me_size_1

      integer(c_int) :: ddxn_z_full_size_0

      integer(c_int) :: ddxn_z_full_size_1

      integer(c_int) :: zdiff_gradp_size_0

      integer(c_int) :: zdiff_gradp_size_1

      integer(c_int) :: zdiff_gradp_size_2

      integer(c_int) :: vertidx_gradp_size_0

      integer(c_int) :: vertidx_gradp_size_1

      integer(c_int) :: vertidx_gradp_size_2

      integer(c_int) :: pg_edgeidx_size_0

      integer(c_int) :: pg_vertidx_size_0

      integer(c_int) :: pg_exdist_size_0

      integer(c_int) :: ddqz_z_full_e_size_0

      integer(c_int) :: ddqz_z_full_e_size_1

      integer(c_int) :: ddxt_z_full_size_0

      integer(c_int) :: ddxt_z_full_size_1

      integer(c_int) :: wgtfac_e_size_0

      integer(c_int) :: wgtfac_e_size_1

      integer(c_int) :: wgtfacq_e_size_0

      integer(c_int) :: wgtfacq_e_size_1

      integer(c_int) :: vwind_impl_wgt_size_0

      integer(c_int) :: hmask_dd3d_size_0

      integer(c_int) :: scalfac_dd3d_size_0

      integer(c_int) :: coeff1_dwdz_size_0

      integer(c_int) :: coeff1_dwdz_size_1

      integer(c_int) :: coeff2_dwdz_size_0

      integer(c_int) :: coeff2_dwdz_size_1

      integer(c_int) :: coeff_gradekin_size_0

      integer(c_int) :: coeff_gradekin_size_1

      integer(c_int) :: c_owner_mask_size_0

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      type(c_ptr) :: pg_edgeidx_ptr

      type(c_ptr) :: pg_vertidx_ptr

      type(c_ptr) :: pg_exdist_ptr

      pg_edgeidx_ptr = c_null_ptr

      pg_vertidx_ptr = c_null_ptr

      pg_exdist_ptr = c_null_ptr

      !$acc host_data use_device(c_lin_e)
      !$acc host_data use_device(c_intp)
      !$acc host_data use_device(e_flx_avg)
      !$acc host_data use_device(geofac_grdiv)
      !$acc host_data use_device(geofac_rot)
      !$acc host_data use_device(pos_on_tplane_e_1)
      !$acc host_data use_device(pos_on_tplane_e_2)
      !$acc host_data use_device(rbf_vec_coeff_e)
      !$acc host_data use_device(e_bln_c_s)
      !$acc host_data use_device(rbf_vec_coeff_v)
      !$acc host_data use_device(geofac_div)
      !$acc host_data use_device(geofac_n2s)
      !$acc host_data use_device(geofac_grg_x)
      !$acc host_data use_device(geofac_grg_y)
      !$acc host_data use_device(nudgecoeff_e)
      !$acc host_data use_device(mask_prog_halo_c)
      !$acc host_data use_device(rayleigh_w)
      !$acc host_data use_device(exner_exfac)
      !$acc host_data use_device(exner_ref_mc)
      !$acc host_data use_device(wgtfac_c)
      !$acc host_data use_device(wgtfacq_c)
      !$acc host_data use_device(inv_ddqz_z_full)
      !$acc host_data use_device(rho_ref_mc)
      !$acc host_data use_device(theta_ref_mc)
      !$acc host_data use_device(vwind_expl_wgt)
      !$acc host_data use_device(d_exner_dz_ref_ic)
      !$acc host_data use_device(ddqz_z_half)
      !$acc host_data use_device(theta_ref_ic)
      !$acc host_data use_device(d2dexdz2_fac1_mc)
      !$acc host_data use_device(d2dexdz2_fac2_mc)
      !$acc host_data use_device(rho_ref_me)
      !$acc host_data use_device(theta_ref_me)
      !$acc host_data use_device(ddxn_z_full)
      !$acc host_data use_device(zdiff_gradp)
      !$acc host_data use_device(vertidx_gradp)
      !$acc host_data use_device(ddqz_z_full_e)
      !$acc host_data use_device(ddxt_z_full)
      !$acc host_data use_device(wgtfac_e)
      !$acc host_data use_device(wgtfacq_e)
      !$acc host_data use_device(vwind_impl_wgt)
      !$acc host_data use_device(hmask_dd3d)
      !$acc host_data use_device(scalfac_dd3d)
      !$acc host_data use_device(coeff1_dwdz)
      !$acc host_data use_device(coeff2_dwdz)
      !$acc host_data use_device(coeff_gradekin)
      !$acc host_data use_device(c_owner_mask)
      !$acc host_data use_device(pg_edgeidx) if(associated(pg_edgeidx))
      !$acc host_data use_device(pg_vertidx) if(associated(pg_vertidx))
      !$acc host_data use_device(pg_exdist) if(associated(pg_exdist))

#ifdef _OPENACC
      on_gpu = .True.
#else
      on_gpu = .False.
#endif

      c_lin_e_size_0 = SIZE(c_lin_e, 1)
      c_lin_e_size_1 = SIZE(c_lin_e, 2)

      c_intp_size_0 = SIZE(c_intp, 1)
      c_intp_size_1 = SIZE(c_intp, 2)

      e_flx_avg_size_0 = SIZE(e_flx_avg, 1)
      e_flx_avg_size_1 = SIZE(e_flx_avg, 2)

      geofac_grdiv_size_0 = SIZE(geofac_grdiv, 1)
      geofac_grdiv_size_1 = SIZE(geofac_grdiv, 2)

      geofac_rot_size_0 = SIZE(geofac_rot, 1)
      geofac_rot_size_1 = SIZE(geofac_rot, 2)

      pos_on_tplane_e_1_size_0 = SIZE(pos_on_tplane_e_1, 1)
      pos_on_tplane_e_1_size_1 = SIZE(pos_on_tplane_e_1, 2)

      pos_on_tplane_e_2_size_0 = SIZE(pos_on_tplane_e_2, 1)
      pos_on_tplane_e_2_size_1 = SIZE(pos_on_tplane_e_2, 2)

      rbf_vec_coeff_e_size_0 = SIZE(rbf_vec_coeff_e, 1)
      rbf_vec_coeff_e_size_1 = SIZE(rbf_vec_coeff_e, 2)

      e_bln_c_s_size_0 = SIZE(e_bln_c_s, 1)
      e_bln_c_s_size_1 = SIZE(e_bln_c_s, 2)

      rbf_vec_coeff_v_size_0 = SIZE(rbf_vec_coeff_v, 1)
      rbf_vec_coeff_v_size_1 = SIZE(rbf_vec_coeff_v, 2)
      rbf_vec_coeff_v_size_2 = SIZE(rbf_vec_coeff_v, 3)

      geofac_div_size_0 = SIZE(geofac_div, 1)
      geofac_div_size_1 = SIZE(geofac_div, 2)

      geofac_n2s_size_0 = SIZE(geofac_n2s, 1)
      geofac_n2s_size_1 = SIZE(geofac_n2s, 2)

      geofac_grg_x_size_0 = SIZE(geofac_grg_x, 1)
      geofac_grg_x_size_1 = SIZE(geofac_grg_x, 2)

      geofac_grg_y_size_0 = SIZE(geofac_grg_y, 1)
      geofac_grg_y_size_1 = SIZE(geofac_grg_y, 2)

      nudgecoeff_e_size_0 = SIZE(nudgecoeff_e, 1)

      mask_prog_halo_c_size_0 = SIZE(mask_prog_halo_c, 1)

      rayleigh_w_size_0 = SIZE(rayleigh_w, 1)

      exner_exfac_size_0 = SIZE(exner_exfac, 1)
      exner_exfac_size_1 = SIZE(exner_exfac, 2)

      exner_ref_mc_size_0 = SIZE(exner_ref_mc, 1)
      exner_ref_mc_size_1 = SIZE(exner_ref_mc, 2)

      wgtfac_c_size_0 = SIZE(wgtfac_c, 1)
      wgtfac_c_size_1 = SIZE(wgtfac_c, 2)

      wgtfacq_c_size_0 = SIZE(wgtfacq_c, 1)
      wgtfacq_c_size_1 = SIZE(wgtfacq_c, 2)

      inv_ddqz_z_full_size_0 = SIZE(inv_ddqz_z_full, 1)
      inv_ddqz_z_full_size_1 = SIZE(inv_ddqz_z_full, 2)

      rho_ref_mc_size_0 = SIZE(rho_ref_mc, 1)
      rho_ref_mc_size_1 = SIZE(rho_ref_mc, 2)

      theta_ref_mc_size_0 = SIZE(theta_ref_mc, 1)
      theta_ref_mc_size_1 = SIZE(theta_ref_mc, 2)

      vwind_expl_wgt_size_0 = SIZE(vwind_expl_wgt, 1)

      d_exner_dz_ref_ic_size_0 = SIZE(d_exner_dz_ref_ic, 1)
      d_exner_dz_ref_ic_size_1 = SIZE(d_exner_dz_ref_ic, 2)

      ddqz_z_half_size_0 = SIZE(ddqz_z_half, 1)
      ddqz_z_half_size_1 = SIZE(ddqz_z_half, 2)

      theta_ref_ic_size_0 = SIZE(theta_ref_ic, 1)
      theta_ref_ic_size_1 = SIZE(theta_ref_ic, 2)

      d2dexdz2_fac1_mc_size_0 = SIZE(d2dexdz2_fac1_mc, 1)
      d2dexdz2_fac1_mc_size_1 = SIZE(d2dexdz2_fac1_mc, 2)

      d2dexdz2_fac2_mc_size_0 = SIZE(d2dexdz2_fac2_mc, 1)
      d2dexdz2_fac2_mc_size_1 = SIZE(d2dexdz2_fac2_mc, 2)

      rho_ref_me_size_0 = SIZE(rho_ref_me, 1)
      rho_ref_me_size_1 = SIZE(rho_ref_me, 2)

      theta_ref_me_size_0 = SIZE(theta_ref_me, 1)
      theta_ref_me_size_1 = SIZE(theta_ref_me, 2)

      ddxn_z_full_size_0 = SIZE(ddxn_z_full, 1)
      ddxn_z_full_size_1 = SIZE(ddxn_z_full, 2)

      zdiff_gradp_size_0 = SIZE(zdiff_gradp, 1)
      zdiff_gradp_size_1 = SIZE(zdiff_gradp, 2)
      zdiff_gradp_size_2 = SIZE(zdiff_gradp, 3)

      vertidx_gradp_size_0 = SIZE(vertidx_gradp, 1)
      vertidx_gradp_size_1 = SIZE(vertidx_gradp, 2)
      vertidx_gradp_size_2 = SIZE(vertidx_gradp, 3)

      ddqz_z_full_e_size_0 = SIZE(ddqz_z_full_e, 1)
      ddqz_z_full_e_size_1 = SIZE(ddqz_z_full_e, 2)

      ddxt_z_full_size_0 = SIZE(ddxt_z_full, 1)
      ddxt_z_full_size_1 = SIZE(ddxt_z_full, 2)

      wgtfac_e_size_0 = SIZE(wgtfac_e, 1)
      wgtfac_e_size_1 = SIZE(wgtfac_e, 2)

      wgtfacq_e_size_0 = SIZE(wgtfacq_e, 1)
      wgtfacq_e_size_1 = SIZE(wgtfacq_e, 2)

      vwind_impl_wgt_size_0 = SIZE(vwind_impl_wgt, 1)

      hmask_dd3d_size_0 = SIZE(hmask_dd3d, 1)

      scalfac_dd3d_size_0 = SIZE(scalfac_dd3d, 1)

      coeff1_dwdz_size_0 = SIZE(coeff1_dwdz, 1)
      coeff1_dwdz_size_1 = SIZE(coeff1_dwdz, 2)

      coeff2_dwdz_size_0 = SIZE(coeff2_dwdz, 1)
      coeff2_dwdz_size_1 = SIZE(coeff2_dwdz, 2)

      coeff_gradekin_size_0 = SIZE(coeff_gradekin, 1)
      coeff_gradekin_size_1 = SIZE(coeff_gradekin, 2)

      c_owner_mask_size_0 = SIZE(c_owner_mask, 1)

      if (associated(pg_edgeidx)) then
         pg_edgeidx_ptr = c_loc(pg_edgeidx)
         pg_edgeidx_size_0 = SIZE(pg_edgeidx, 1)
      end if

      if (associated(pg_vertidx)) then
         pg_vertidx_ptr = c_loc(pg_vertidx)
         pg_vertidx_size_0 = SIZE(pg_vertidx, 1)
      end if

      if (associated(pg_exdist)) then
         pg_exdist_ptr = c_loc(pg_exdist)
         pg_exdist_size_0 = SIZE(pg_exdist, 1)
      end if

      rc = solve_nh_init_wrapper(c_lin_e=c_loc(c_lin_e), &
                                 c_lin_e_size_0=c_lin_e_size_0, &
                                 c_lin_e_size_1=c_lin_e_size_1, &
                                 c_intp=c_loc(c_intp), &
                                 c_intp_size_0=c_intp_size_0, &
                                 c_intp_size_1=c_intp_size_1, &
                                 e_flx_avg=c_loc(e_flx_avg), &
                                 e_flx_avg_size_0=e_flx_avg_size_0, &
                                 e_flx_avg_size_1=e_flx_avg_size_1, &
                                 geofac_grdiv=c_loc(geofac_grdiv), &
                                 geofac_grdiv_size_0=geofac_grdiv_size_0, &
                                 geofac_grdiv_size_1=geofac_grdiv_size_1, &
                                 geofac_rot=c_loc(geofac_rot), &
                                 geofac_rot_size_0=geofac_rot_size_0, &
                                 geofac_rot_size_1=geofac_rot_size_1, &
                                 pos_on_tplane_e_1=c_loc(pos_on_tplane_e_1), &
                                 pos_on_tplane_e_1_size_0=pos_on_tplane_e_1_size_0, &
                                 pos_on_tplane_e_1_size_1=pos_on_tplane_e_1_size_1, &
                                 pos_on_tplane_e_2=c_loc(pos_on_tplane_e_2), &
                                 pos_on_tplane_e_2_size_0=pos_on_tplane_e_2_size_0, &
                                 pos_on_tplane_e_2_size_1=pos_on_tplane_e_2_size_1, &
                                 rbf_vec_coeff_e=c_loc(rbf_vec_coeff_e), &
                                 rbf_vec_coeff_e_size_0=rbf_vec_coeff_e_size_0, &
                                 rbf_vec_coeff_e_size_1=rbf_vec_coeff_e_size_1, &
                                 e_bln_c_s=c_loc(e_bln_c_s), &
                                 e_bln_c_s_size_0=e_bln_c_s_size_0, &
                                 e_bln_c_s_size_1=e_bln_c_s_size_1, &
                                 rbf_vec_coeff_v=c_loc(rbf_vec_coeff_v), &
                                 rbf_vec_coeff_v_size_0=rbf_vec_coeff_v_size_0, &
                                 rbf_vec_coeff_v_size_1=rbf_vec_coeff_v_size_1, &
                                 rbf_vec_coeff_v_size_2=rbf_vec_coeff_v_size_2, &
                                 geofac_div=c_loc(geofac_div), &
                                 geofac_div_size_0=geofac_div_size_0, &
                                 geofac_div_size_1=geofac_div_size_1, &
                                 geofac_n2s=c_loc(geofac_n2s), &
                                 geofac_n2s_size_0=geofac_n2s_size_0, &
                                 geofac_n2s_size_1=geofac_n2s_size_1, &
                                 geofac_grg_x=c_loc(geofac_grg_x), &
                                 geofac_grg_x_size_0=geofac_grg_x_size_0, &
                                 geofac_grg_x_size_1=geofac_grg_x_size_1, &
                                 geofac_grg_y=c_loc(geofac_grg_y), &
                                 geofac_grg_y_size_0=geofac_grg_y_size_0, &
                                 geofac_grg_y_size_1=geofac_grg_y_size_1, &
                                 nudgecoeff_e=c_loc(nudgecoeff_e), &
                                 nudgecoeff_e_size_0=nudgecoeff_e_size_0, &
                                 mask_prog_halo_c=c_loc(mask_prog_halo_c), &
                                 mask_prog_halo_c_size_0=mask_prog_halo_c_size_0, &
                                 rayleigh_w=c_loc(rayleigh_w), &
                                 rayleigh_w_size_0=rayleigh_w_size_0, &
                                 exner_exfac=c_loc(exner_exfac), &
                                 exner_exfac_size_0=exner_exfac_size_0, &
                                 exner_exfac_size_1=exner_exfac_size_1, &
                                 exner_ref_mc=c_loc(exner_ref_mc), &
                                 exner_ref_mc_size_0=exner_ref_mc_size_0, &
                                 exner_ref_mc_size_1=exner_ref_mc_size_1, &
                                 wgtfac_c=c_loc(wgtfac_c), &
                                 wgtfac_c_size_0=wgtfac_c_size_0, &
                                 wgtfac_c_size_1=wgtfac_c_size_1, &
                                 wgtfacq_c=c_loc(wgtfacq_c), &
                                 wgtfacq_c_size_0=wgtfacq_c_size_0, &
                                 wgtfacq_c_size_1=wgtfacq_c_size_1, &
                                 inv_ddqz_z_full=c_loc(inv_ddqz_z_full), &
                                 inv_ddqz_z_full_size_0=inv_ddqz_z_full_size_0, &
                                 inv_ddqz_z_full_size_1=inv_ddqz_z_full_size_1, &
                                 rho_ref_mc=c_loc(rho_ref_mc), &
                                 rho_ref_mc_size_0=rho_ref_mc_size_0, &
                                 rho_ref_mc_size_1=rho_ref_mc_size_1, &
                                 theta_ref_mc=c_loc(theta_ref_mc), &
                                 theta_ref_mc_size_0=theta_ref_mc_size_0, &
                                 theta_ref_mc_size_1=theta_ref_mc_size_1, &
                                 vwind_expl_wgt=c_loc(vwind_expl_wgt), &
                                 vwind_expl_wgt_size_0=vwind_expl_wgt_size_0, &
                                 d_exner_dz_ref_ic=c_loc(d_exner_dz_ref_ic), &
                                 d_exner_dz_ref_ic_size_0=d_exner_dz_ref_ic_size_0, &
                                 d_exner_dz_ref_ic_size_1=d_exner_dz_ref_ic_size_1, &
                                 ddqz_z_half=c_loc(ddqz_z_half), &
                                 ddqz_z_half_size_0=ddqz_z_half_size_0, &
                                 ddqz_z_half_size_1=ddqz_z_half_size_1, &
                                 theta_ref_ic=c_loc(theta_ref_ic), &
                                 theta_ref_ic_size_0=theta_ref_ic_size_0, &
                                 theta_ref_ic_size_1=theta_ref_ic_size_1, &
                                 d2dexdz2_fac1_mc=c_loc(d2dexdz2_fac1_mc), &
                                 d2dexdz2_fac1_mc_size_0=d2dexdz2_fac1_mc_size_0, &
                                 d2dexdz2_fac1_mc_size_1=d2dexdz2_fac1_mc_size_1, &
                                 d2dexdz2_fac2_mc=c_loc(d2dexdz2_fac2_mc), &
                                 d2dexdz2_fac2_mc_size_0=d2dexdz2_fac2_mc_size_0, &
                                 d2dexdz2_fac2_mc_size_1=d2dexdz2_fac2_mc_size_1, &
                                 rho_ref_me=c_loc(rho_ref_me), &
                                 rho_ref_me_size_0=rho_ref_me_size_0, &
                                 rho_ref_me_size_1=rho_ref_me_size_1, &
                                 theta_ref_me=c_loc(theta_ref_me), &
                                 theta_ref_me_size_0=theta_ref_me_size_0, &
                                 theta_ref_me_size_1=theta_ref_me_size_1, &
                                 ddxn_z_full=c_loc(ddxn_z_full), &
                                 ddxn_z_full_size_0=ddxn_z_full_size_0, &
                                 ddxn_z_full_size_1=ddxn_z_full_size_1, &
                                 zdiff_gradp=c_loc(zdiff_gradp), &
                                 zdiff_gradp_size_0=zdiff_gradp_size_0, &
                                 zdiff_gradp_size_1=zdiff_gradp_size_1, &
                                 zdiff_gradp_size_2=zdiff_gradp_size_2, &
                                 vertidx_gradp=c_loc(vertidx_gradp), &
                                 vertidx_gradp_size_0=vertidx_gradp_size_0, &
                                 vertidx_gradp_size_1=vertidx_gradp_size_1, &
                                 vertidx_gradp_size_2=vertidx_gradp_size_2, &
                                 pg_edgeidx=pg_edgeidx_ptr, &
                                 pg_edgeidx_size_0=pg_edgeidx_size_0, &
                                 pg_vertidx=pg_vertidx_ptr, &
                                 pg_vertidx_size_0=pg_vertidx_size_0, &
                                 pg_exdist=pg_exdist_ptr, &
                                 pg_exdist_size_0=pg_exdist_size_0, &
                                 ddqz_z_full_e=c_loc(ddqz_z_full_e), &
                                 ddqz_z_full_e_size_0=ddqz_z_full_e_size_0, &
                                 ddqz_z_full_e_size_1=ddqz_z_full_e_size_1, &
                                 ddxt_z_full=c_loc(ddxt_z_full), &
                                 ddxt_z_full_size_0=ddxt_z_full_size_0, &
                                 ddxt_z_full_size_1=ddxt_z_full_size_1, &
                                 wgtfac_e=c_loc(wgtfac_e), &
                                 wgtfac_e_size_0=wgtfac_e_size_0, &
                                 wgtfac_e_size_1=wgtfac_e_size_1, &
                                 wgtfacq_e=c_loc(wgtfacq_e), &
                                 wgtfacq_e_size_0=wgtfacq_e_size_0, &
                                 wgtfacq_e_size_1=wgtfacq_e_size_1, &
                                 vwind_impl_wgt=c_loc(vwind_impl_wgt), &
                                 vwind_impl_wgt_size_0=vwind_impl_wgt_size_0, &
                                 hmask_dd3d=c_loc(hmask_dd3d), &
                                 hmask_dd3d_size_0=hmask_dd3d_size_0, &
                                 scalfac_dd3d=c_loc(scalfac_dd3d), &
                                 scalfac_dd3d_size_0=scalfac_dd3d_size_0, &
                                 coeff1_dwdz=c_loc(coeff1_dwdz), &
                                 coeff1_dwdz_size_0=coeff1_dwdz_size_0, &
                                 coeff1_dwdz_size_1=coeff1_dwdz_size_1, &
                                 coeff2_dwdz=c_loc(coeff2_dwdz), &
                                 coeff2_dwdz_size_0=coeff2_dwdz_size_0, &
                                 coeff2_dwdz_size_1=coeff2_dwdz_size_1, &
                                 coeff_gradekin=c_loc(coeff_gradekin), &
                                 coeff_gradekin_size_0=coeff_gradekin_size_0, &
                                 coeff_gradekin_size_1=coeff_gradekin_size_1, &
                                 c_owner_mask=c_loc(c_owner_mask), &
                                 c_owner_mask_size_0=c_owner_mask_size_0, &
                                 itime_scheme=itime_scheme, &
                                 iadv_rhotheta=iadv_rhotheta, &
                                 igradp_method=igradp_method, &
                                 rayleigh_type=rayleigh_type, &
                                 divdamp_order=divdamp_order, &
                                 divdamp_type=divdamp_type, &
                                 l_vert_nested=l_vert_nested, &
                                 ldeepatmo=ldeepatmo, &
                                 iau_init=iau_init, &
                                 extra_diffu=extra_diffu, &
                                 rhotheta_offctr=rhotheta_offctr, &
                                 veladv_offctr=veladv_offctr, &
                                 nudge_max_coeff=nudge_max_coeff, &
                                 divdamp_fac=divdamp_fac, &
                                 divdamp_fac2=divdamp_fac2, &
                                 divdamp_fac3=divdamp_fac3, &
                                 divdamp_fac4=divdamp_fac4, &
                                 divdamp_z=divdamp_z, &
                                 divdamp_z2=divdamp_z2, &
                                 divdamp_z3=divdamp_z3, &
                                 divdamp_z4=divdamp_z4, &
                                 nflat_gradp=nflat_gradp, &
                                 backend=backend, &
                                 on_gpu=on_gpu)
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
   end subroutine solve_nh_init

   subroutine solve_nh_run(rho_now, &
                           rho_new, &
                           exner_now, &
                           exner_new, &
                           w_now, &
                           w_new, &
                           theta_v_now, &
                           theta_v_new, &
                           vn_now, &
                           vn_new, &
                           w_concorr_c, &
                           ddt_vn_apc_ntl1, &
                           ddt_vn_apc_ntl2, &
                           ddt_w_adv_ntl1, &
                           ddt_w_adv_ntl2, &
                           theta_v_ic, &
                           rho_ic, &
                           exner_pr, &
                           exner_dyn_incr, &
                           ddt_exner_phy, &
                           grf_tend_rho, &
                           grf_tend_thv, &
                           grf_tend_w, &
                           mass_fl_e, &
                           ddt_vn_phy, &
                           grf_tend_vn, &
                           vn_ie, &
                           vt, &
                           vn_incr, &
                           rho_incr, &
                           exner_incr, &
                           mass_flx_me, &
                           mass_flx_ic, &
                           vol_flx_ic, &
                           vn_traj, &
                           dtime, &
                           max_vcfl_size1_array, &
                           lprep_adv, &
                           at_initial_timestep, &
                           divdamp_fac_o2, &
                           ndyn_substeps_var, &
                           idyn_timestep, &
                           is_iau_active, &
                           iau_wgt_dyn, &
                           rc)
      use, intrinsic :: iso_c_binding

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: rho_now

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: rho_new

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: exner_now

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: exner_new

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: w_now

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: w_new

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: theta_v_now

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: theta_v_new

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: vn_now

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: vn_new

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: w_concorr_c

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: ddt_vn_apc_ntl1

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: ddt_vn_apc_ntl2

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: ddt_w_adv_ntl1

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: ddt_w_adv_ntl2

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: theta_v_ic

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: rho_ic

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: exner_pr

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: exner_dyn_incr

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: ddt_exner_phy

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: grf_tend_rho

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: grf_tend_thv

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: grf_tend_w

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: mass_fl_e

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: ddt_vn_phy

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: grf_tend_vn

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: vn_ie

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: vt

      real(c_double), dimension(:, :), contiguous, intent(inout), pointer :: vn_incr

      real(c_double), dimension(:, :), contiguous, intent(inout), pointer :: rho_incr

      real(c_double), dimension(:, :), contiguous, intent(inout), pointer :: exner_incr

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: mass_flx_me

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: mass_flx_ic

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: vol_flx_ic

      real(c_double), dimension(:, :), contiguous, intent(inout), target :: vn_traj

      real(c_double), value, target :: dtime

      real(c_double), dimension(:), contiguous, intent(inout), target :: max_vcfl_size1_array

      logical(c_bool), value, target :: lprep_adv

      logical(c_bool), value, target :: at_initial_timestep

      real(c_double), value, target :: divdamp_fac_o2

      integer(c_int), value, target :: ndyn_substeps_var

      integer(c_int), value, target :: idyn_timestep

      logical(c_bool), value, target :: is_iau_active

      real(c_double), value, target :: iau_wgt_dyn

      logical(c_bool) :: on_gpu

      integer(c_int) :: rho_now_size_0

      integer(c_int) :: rho_now_size_1

      integer(c_int) :: rho_new_size_0

      integer(c_int) :: rho_new_size_1

      integer(c_int) :: exner_now_size_0

      integer(c_int) :: exner_now_size_1

      integer(c_int) :: exner_new_size_0

      integer(c_int) :: exner_new_size_1

      integer(c_int) :: w_now_size_0

      integer(c_int) :: w_now_size_1

      integer(c_int) :: w_new_size_0

      integer(c_int) :: w_new_size_1

      integer(c_int) :: theta_v_now_size_0

      integer(c_int) :: theta_v_now_size_1

      integer(c_int) :: theta_v_new_size_0

      integer(c_int) :: theta_v_new_size_1

      integer(c_int) :: vn_now_size_0

      integer(c_int) :: vn_now_size_1

      integer(c_int) :: vn_new_size_0

      integer(c_int) :: vn_new_size_1

      integer(c_int) :: w_concorr_c_size_0

      integer(c_int) :: w_concorr_c_size_1

      integer(c_int) :: ddt_vn_apc_ntl1_size_0

      integer(c_int) :: ddt_vn_apc_ntl1_size_1

      integer(c_int) :: ddt_vn_apc_ntl2_size_0

      integer(c_int) :: ddt_vn_apc_ntl2_size_1

      integer(c_int) :: ddt_w_adv_ntl1_size_0

      integer(c_int) :: ddt_w_adv_ntl1_size_1

      integer(c_int) :: ddt_w_adv_ntl2_size_0

      integer(c_int) :: ddt_w_adv_ntl2_size_1

      integer(c_int) :: theta_v_ic_size_0

      integer(c_int) :: theta_v_ic_size_1

      integer(c_int) :: rho_ic_size_0

      integer(c_int) :: rho_ic_size_1

      integer(c_int) :: exner_pr_size_0

      integer(c_int) :: exner_pr_size_1

      integer(c_int) :: exner_dyn_incr_size_0

      integer(c_int) :: exner_dyn_incr_size_1

      integer(c_int) :: ddt_exner_phy_size_0

      integer(c_int) :: ddt_exner_phy_size_1

      integer(c_int) :: grf_tend_rho_size_0

      integer(c_int) :: grf_tend_rho_size_1

      integer(c_int) :: grf_tend_thv_size_0

      integer(c_int) :: grf_tend_thv_size_1

      integer(c_int) :: grf_tend_w_size_0

      integer(c_int) :: grf_tend_w_size_1

      integer(c_int) :: mass_fl_e_size_0

      integer(c_int) :: mass_fl_e_size_1

      integer(c_int) :: ddt_vn_phy_size_0

      integer(c_int) :: ddt_vn_phy_size_1

      integer(c_int) :: grf_tend_vn_size_0

      integer(c_int) :: grf_tend_vn_size_1

      integer(c_int) :: vn_ie_size_0

      integer(c_int) :: vn_ie_size_1

      integer(c_int) :: vt_size_0

      integer(c_int) :: vt_size_1

      integer(c_int) :: vn_incr_size_0

      integer(c_int) :: vn_incr_size_1

      integer(c_int) :: rho_incr_size_0

      integer(c_int) :: rho_incr_size_1

      integer(c_int) :: exner_incr_size_0

      integer(c_int) :: exner_incr_size_1

      integer(c_int) :: mass_flx_me_size_0

      integer(c_int) :: mass_flx_me_size_1

      integer(c_int) :: mass_flx_ic_size_0

      integer(c_int) :: mass_flx_ic_size_1

      integer(c_int) :: vol_flx_ic_size_0

      integer(c_int) :: vol_flx_ic_size_1

      integer(c_int) :: vn_traj_size_0

      integer(c_int) :: vn_traj_size_1

      integer(c_int) :: max_vcfl_size1_array_size_0

      integer(c_int) :: rc  ! Stores the return code
      ! ptrs

      type(c_ptr) :: vn_incr_ptr

      type(c_ptr) :: rho_incr_ptr

      type(c_ptr) :: exner_incr_ptr

      vn_incr_ptr = c_null_ptr

      rho_incr_ptr = c_null_ptr

      exner_incr_ptr = c_null_ptr

      !$acc host_data use_device(rho_now)
      !$acc host_data use_device(rho_new)
      !$acc host_data use_device(exner_now)
      !$acc host_data use_device(exner_new)
      !$acc host_data use_device(w_now)
      !$acc host_data use_device(w_new)
      !$acc host_data use_device(theta_v_now)
      !$acc host_data use_device(theta_v_new)
      !$acc host_data use_device(vn_now)
      !$acc host_data use_device(vn_new)
      !$acc host_data use_device(w_concorr_c)
      !$acc host_data use_device(ddt_vn_apc_ntl1)
      !$acc host_data use_device(ddt_vn_apc_ntl2)
      !$acc host_data use_device(ddt_w_adv_ntl1)
      !$acc host_data use_device(ddt_w_adv_ntl2)
      !$acc host_data use_device(theta_v_ic)
      !$acc host_data use_device(rho_ic)
      !$acc host_data use_device(exner_pr)
      !$acc host_data use_device(exner_dyn_incr)
      !$acc host_data use_device(ddt_exner_phy)
      !$acc host_data use_device(grf_tend_rho)
      !$acc host_data use_device(grf_tend_thv)
      !$acc host_data use_device(grf_tend_w)
      !$acc host_data use_device(mass_fl_e)
      !$acc host_data use_device(ddt_vn_phy)
      !$acc host_data use_device(grf_tend_vn)
      !$acc host_data use_device(vn_ie)
      !$acc host_data use_device(vt)
      !$acc host_data use_device(mass_flx_me)
      !$acc host_data use_device(mass_flx_ic)
      !$acc host_data use_device(vol_flx_ic)
      !$acc host_data use_device(vn_traj)
      !$acc host_data use_device(vn_incr) if(associated(vn_incr))
      !$acc host_data use_device(rho_incr) if(associated(rho_incr))
      !$acc host_data use_device(exner_incr) if(associated(exner_incr))

#ifdef _OPENACC
      on_gpu = .True.
#else
      on_gpu = .False.
#endif

      rho_now_size_0 = SIZE(rho_now, 1)
      rho_now_size_1 = SIZE(rho_now, 2)

      rho_new_size_0 = SIZE(rho_new, 1)
      rho_new_size_1 = SIZE(rho_new, 2)

      exner_now_size_0 = SIZE(exner_now, 1)
      exner_now_size_1 = SIZE(exner_now, 2)

      exner_new_size_0 = SIZE(exner_new, 1)
      exner_new_size_1 = SIZE(exner_new, 2)

      w_now_size_0 = SIZE(w_now, 1)
      w_now_size_1 = SIZE(w_now, 2)

      w_new_size_0 = SIZE(w_new, 1)
      w_new_size_1 = SIZE(w_new, 2)

      theta_v_now_size_0 = SIZE(theta_v_now, 1)
      theta_v_now_size_1 = SIZE(theta_v_now, 2)

      theta_v_new_size_0 = SIZE(theta_v_new, 1)
      theta_v_new_size_1 = SIZE(theta_v_new, 2)

      vn_now_size_0 = SIZE(vn_now, 1)
      vn_now_size_1 = SIZE(vn_now, 2)

      vn_new_size_0 = SIZE(vn_new, 1)
      vn_new_size_1 = SIZE(vn_new, 2)

      w_concorr_c_size_0 = SIZE(w_concorr_c, 1)
      w_concorr_c_size_1 = SIZE(w_concorr_c, 2)

      ddt_vn_apc_ntl1_size_0 = SIZE(ddt_vn_apc_ntl1, 1)
      ddt_vn_apc_ntl1_size_1 = SIZE(ddt_vn_apc_ntl1, 2)

      ddt_vn_apc_ntl2_size_0 = SIZE(ddt_vn_apc_ntl2, 1)
      ddt_vn_apc_ntl2_size_1 = SIZE(ddt_vn_apc_ntl2, 2)

      ddt_w_adv_ntl1_size_0 = SIZE(ddt_w_adv_ntl1, 1)
      ddt_w_adv_ntl1_size_1 = SIZE(ddt_w_adv_ntl1, 2)

      ddt_w_adv_ntl2_size_0 = SIZE(ddt_w_adv_ntl2, 1)
      ddt_w_adv_ntl2_size_1 = SIZE(ddt_w_adv_ntl2, 2)

      theta_v_ic_size_0 = SIZE(theta_v_ic, 1)
      theta_v_ic_size_1 = SIZE(theta_v_ic, 2)

      rho_ic_size_0 = SIZE(rho_ic, 1)
      rho_ic_size_1 = SIZE(rho_ic, 2)

      exner_pr_size_0 = SIZE(exner_pr, 1)
      exner_pr_size_1 = SIZE(exner_pr, 2)

      exner_dyn_incr_size_0 = SIZE(exner_dyn_incr, 1)
      exner_dyn_incr_size_1 = SIZE(exner_dyn_incr, 2)

      ddt_exner_phy_size_0 = SIZE(ddt_exner_phy, 1)
      ddt_exner_phy_size_1 = SIZE(ddt_exner_phy, 2)

      grf_tend_rho_size_0 = SIZE(grf_tend_rho, 1)
      grf_tend_rho_size_1 = SIZE(grf_tend_rho, 2)

      grf_tend_thv_size_0 = SIZE(grf_tend_thv, 1)
      grf_tend_thv_size_1 = SIZE(grf_tend_thv, 2)

      grf_tend_w_size_0 = SIZE(grf_tend_w, 1)
      grf_tend_w_size_1 = SIZE(grf_tend_w, 2)

      mass_fl_e_size_0 = SIZE(mass_fl_e, 1)
      mass_fl_e_size_1 = SIZE(mass_fl_e, 2)

      ddt_vn_phy_size_0 = SIZE(ddt_vn_phy, 1)
      ddt_vn_phy_size_1 = SIZE(ddt_vn_phy, 2)

      grf_tend_vn_size_0 = SIZE(grf_tend_vn, 1)
      grf_tend_vn_size_1 = SIZE(grf_tend_vn, 2)

      vn_ie_size_0 = SIZE(vn_ie, 1)
      vn_ie_size_1 = SIZE(vn_ie, 2)

      vt_size_0 = SIZE(vt, 1)
      vt_size_1 = SIZE(vt, 2)

      mass_flx_me_size_0 = SIZE(mass_flx_me, 1)
      mass_flx_me_size_1 = SIZE(mass_flx_me, 2)

      mass_flx_ic_size_0 = SIZE(mass_flx_ic, 1)
      mass_flx_ic_size_1 = SIZE(mass_flx_ic, 2)

      vol_flx_ic_size_0 = SIZE(vol_flx_ic, 1)
      vol_flx_ic_size_1 = SIZE(vol_flx_ic, 2)

      vn_traj_size_0 = SIZE(vn_traj, 1)
      vn_traj_size_1 = SIZE(vn_traj, 2)

      max_vcfl_size1_array_size_0 = SIZE(max_vcfl_size1_array, 1)

      if (associated(vn_incr)) then
         vn_incr_ptr = c_loc(vn_incr)
         vn_incr_size_0 = SIZE(vn_incr, 1)
         vn_incr_size_1 = SIZE(vn_incr, 2)
      end if

      if (associated(rho_incr)) then
         rho_incr_ptr = c_loc(rho_incr)
         rho_incr_size_0 = SIZE(rho_incr, 1)
         rho_incr_size_1 = SIZE(rho_incr, 2)
      end if

      if (associated(exner_incr)) then
         exner_incr_ptr = c_loc(exner_incr)
         exner_incr_size_0 = SIZE(exner_incr, 1)
         exner_incr_size_1 = SIZE(exner_incr, 2)
      end if

      rc = solve_nh_run_wrapper(rho_now=c_loc(rho_now), &
                                rho_now_size_0=rho_now_size_0, &
                                rho_now_size_1=rho_now_size_1, &
                                rho_new=c_loc(rho_new), &
                                rho_new_size_0=rho_new_size_0, &
                                rho_new_size_1=rho_new_size_1, &
                                exner_now=c_loc(exner_now), &
                                exner_now_size_0=exner_now_size_0, &
                                exner_now_size_1=exner_now_size_1, &
                                exner_new=c_loc(exner_new), &
                                exner_new_size_0=exner_new_size_0, &
                                exner_new_size_1=exner_new_size_1, &
                                w_now=c_loc(w_now), &
                                w_now_size_0=w_now_size_0, &
                                w_now_size_1=w_now_size_1, &
                                w_new=c_loc(w_new), &
                                w_new_size_0=w_new_size_0, &
                                w_new_size_1=w_new_size_1, &
                                theta_v_now=c_loc(theta_v_now), &
                                theta_v_now_size_0=theta_v_now_size_0, &
                                theta_v_now_size_1=theta_v_now_size_1, &
                                theta_v_new=c_loc(theta_v_new), &
                                theta_v_new_size_0=theta_v_new_size_0, &
                                theta_v_new_size_1=theta_v_new_size_1, &
                                vn_now=c_loc(vn_now), &
                                vn_now_size_0=vn_now_size_0, &
                                vn_now_size_1=vn_now_size_1, &
                                vn_new=c_loc(vn_new), &
                                vn_new_size_0=vn_new_size_0, &
                                vn_new_size_1=vn_new_size_1, &
                                w_concorr_c=c_loc(w_concorr_c), &
                                w_concorr_c_size_0=w_concorr_c_size_0, &
                                w_concorr_c_size_1=w_concorr_c_size_1, &
                                ddt_vn_apc_ntl1=c_loc(ddt_vn_apc_ntl1), &
                                ddt_vn_apc_ntl1_size_0=ddt_vn_apc_ntl1_size_0, &
                                ddt_vn_apc_ntl1_size_1=ddt_vn_apc_ntl1_size_1, &
                                ddt_vn_apc_ntl2=c_loc(ddt_vn_apc_ntl2), &
                                ddt_vn_apc_ntl2_size_0=ddt_vn_apc_ntl2_size_0, &
                                ddt_vn_apc_ntl2_size_1=ddt_vn_apc_ntl2_size_1, &
                                ddt_w_adv_ntl1=c_loc(ddt_w_adv_ntl1), &
                                ddt_w_adv_ntl1_size_0=ddt_w_adv_ntl1_size_0, &
                                ddt_w_adv_ntl1_size_1=ddt_w_adv_ntl1_size_1, &
                                ddt_w_adv_ntl2=c_loc(ddt_w_adv_ntl2), &
                                ddt_w_adv_ntl2_size_0=ddt_w_adv_ntl2_size_0, &
                                ddt_w_adv_ntl2_size_1=ddt_w_adv_ntl2_size_1, &
                                theta_v_ic=c_loc(theta_v_ic), &
                                theta_v_ic_size_0=theta_v_ic_size_0, &
                                theta_v_ic_size_1=theta_v_ic_size_1, &
                                rho_ic=c_loc(rho_ic), &
                                rho_ic_size_0=rho_ic_size_0, &
                                rho_ic_size_1=rho_ic_size_1, &
                                exner_pr=c_loc(exner_pr), &
                                exner_pr_size_0=exner_pr_size_0, &
                                exner_pr_size_1=exner_pr_size_1, &
                                exner_dyn_incr=c_loc(exner_dyn_incr), &
                                exner_dyn_incr_size_0=exner_dyn_incr_size_0, &
                                exner_dyn_incr_size_1=exner_dyn_incr_size_1, &
                                ddt_exner_phy=c_loc(ddt_exner_phy), &
                                ddt_exner_phy_size_0=ddt_exner_phy_size_0, &
                                ddt_exner_phy_size_1=ddt_exner_phy_size_1, &
                                grf_tend_rho=c_loc(grf_tend_rho), &
                                grf_tend_rho_size_0=grf_tend_rho_size_0, &
                                grf_tend_rho_size_1=grf_tend_rho_size_1, &
                                grf_tend_thv=c_loc(grf_tend_thv), &
                                grf_tend_thv_size_0=grf_tend_thv_size_0, &
                                grf_tend_thv_size_1=grf_tend_thv_size_1, &
                                grf_tend_w=c_loc(grf_tend_w), &
                                grf_tend_w_size_0=grf_tend_w_size_0, &
                                grf_tend_w_size_1=grf_tend_w_size_1, &
                                mass_fl_e=c_loc(mass_fl_e), &
                                mass_fl_e_size_0=mass_fl_e_size_0, &
                                mass_fl_e_size_1=mass_fl_e_size_1, &
                                ddt_vn_phy=c_loc(ddt_vn_phy), &
                                ddt_vn_phy_size_0=ddt_vn_phy_size_0, &
                                ddt_vn_phy_size_1=ddt_vn_phy_size_1, &
                                grf_tend_vn=c_loc(grf_tend_vn), &
                                grf_tend_vn_size_0=grf_tend_vn_size_0, &
                                grf_tend_vn_size_1=grf_tend_vn_size_1, &
                                vn_ie=c_loc(vn_ie), &
                                vn_ie_size_0=vn_ie_size_0, &
                                vn_ie_size_1=vn_ie_size_1, &
                                vt=c_loc(vt), &
                                vt_size_0=vt_size_0, &
                                vt_size_1=vt_size_1, &
                                vn_incr=vn_incr_ptr, &
                                vn_incr_size_0=vn_incr_size_0, &
                                vn_incr_size_1=vn_incr_size_1, &
                                rho_incr=rho_incr_ptr, &
                                rho_incr_size_0=rho_incr_size_0, &
                                rho_incr_size_1=rho_incr_size_1, &
                                exner_incr=exner_incr_ptr, &
                                exner_incr_size_0=exner_incr_size_0, &
                                exner_incr_size_1=exner_incr_size_1, &
                                mass_flx_me=c_loc(mass_flx_me), &
                                mass_flx_me_size_0=mass_flx_me_size_0, &
                                mass_flx_me_size_1=mass_flx_me_size_1, &
                                mass_flx_ic=c_loc(mass_flx_ic), &
                                mass_flx_ic_size_0=mass_flx_ic_size_0, &
                                mass_flx_ic_size_1=mass_flx_ic_size_1, &
                                vol_flx_ic=c_loc(vol_flx_ic), &
                                vol_flx_ic_size_0=vol_flx_ic_size_0, &
                                vol_flx_ic_size_1=vol_flx_ic_size_1, &
                                vn_traj=c_loc(vn_traj), &
                                vn_traj_size_0=vn_traj_size_0, &
                                vn_traj_size_1=vn_traj_size_1, &
                                dtime=dtime, &
                                max_vcfl_size1_array=c_loc(max_vcfl_size1_array), &
                                max_vcfl_size1_array_size_0=max_vcfl_size1_array_size_0, &
                                lprep_adv=lprep_adv, &
                                at_initial_timestep=at_initial_timestep, &
                                divdamp_fac_o2=divdamp_fac_o2, &
                                ndyn_substeps_var=ndyn_substeps_var, &
                                idyn_timestep=idyn_timestep, &
                                is_iau_active=is_iau_active, &
                                iau_wgt_dyn=iau_wgt_dyn, &
                                on_gpu=on_gpu)
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
      !$acc end host_data
   end subroutine solve_nh_run

end module