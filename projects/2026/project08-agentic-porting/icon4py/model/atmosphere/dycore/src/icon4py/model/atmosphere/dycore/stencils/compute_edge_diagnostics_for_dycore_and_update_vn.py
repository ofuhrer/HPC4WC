# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import gt4py.next as gtx
from gt4py.next import broadcast, where
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.dycore.stencils.add_analysis_increments_to_vn import (
    _add_analysis_increments_to_vn,
)
from icon4py.model.atmosphere.dycore.stencils.add_temporal_tendencies_to_vn import (
    _add_temporal_tendencies_to_vn,
)
from icon4py.model.atmosphere.dycore.stencils.add_temporal_tendencies_to_vn_by_interpolating_between_time_levels import (
    _add_temporal_tendencies_to_vn_by_interpolating_between_time_levels,
)
from icon4py.model.atmosphere.dycore.stencils.add_vertical_wind_derivative_to_divergence_damping import (
    _add_vertical_wind_derivative_to_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.apply_2nd_order_divergence_damping import (
    _apply_2nd_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.apply_4th_order_divergence_damping import (
    _apply_4th_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.apply_weighted_2nd_and_4th_order_divergence_damping import (
    _apply_weighted_2nd_and_4th_order_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.compute_graddiv2_of_vn import _compute_graddiv2_of_vn
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_of_rho_and_theta import (
    _compute_horizontal_advection_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates import (
    _compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_multiple_levels import (
    _compute_horizontal_gradient_of_exner_pressure_for_multiple_levels,
)
from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates import (
    _compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates,
)
from icon4py.model.atmosphere.dycore.stencils.compute_vn_on_lateral_boundary import (
    _compute_vn_on_lateral_boundary,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def apply_on_vertical_level(
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    on_flatlevels: fa.EdgeKField[ta.wpfloat],
    between_flat_and_flatgradp: fa.EdgeKField[ta.wpfloat],
    below_flatgradp: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    return concat_where(
        dims.KDim < nflatlev,
        on_flatlevels,
        concat_where(nflat_gradp + 1 <= dims.KDim, below_flatgradp, between_flat_and_flatgradp),
    )


@gtx.field_operator
def apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
    pg_exdist: fa.EdgeKField[ta.vpfloat],
    z_hydro_corr: fa.EdgeField[ta.wpfloat],
    z_gradh_exner: fa.EdgeKField[ta.vpfloat],
) -> fa.EdgeKField[ta.vpfloat]:
    # Note: In the original Fortran code `pg_exdist` is implemented as a list,
    # in ICON4Py it's a full field intialized with zeros for points that are not in the list.
    z_gradh_exner_vp = where(
        pg_exdist != 0.0, z_gradh_exner + z_hydro_corr * pg_exdist, z_gradh_exner
    )
    return z_gradh_exner_vp


@gtx.field_operator
def _compute_horizontal_pressure_gradient(
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[ta.vpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    hydrostatic_correction_on_lowest_level: fa.EdgeField[ta.wpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    c_lin_e: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ikoffset: gtx.Field[[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[[dims.EdgeDim, dims.E2CDim, dims.KDim], ta.vpfloat],
    pg_exdist: fa.EdgeKField[ta.vpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
) -> fa.EdgeKField[ta.wpfloat]:
    # Note: we only support `TAYLOR_HYDRO`
    horizontal_pressure_gradient = apply_on_vertical_level(
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
        on_flatlevels=_compute_horizontal_gradient_of_exner_pressure_for_flat_coordinates(
            inv_dual_edge_length=inv_dual_edge_length,
            z_exner_ex_pr=temporal_extrapolation_of_perturbed_exner,
        ),
        between_flat_and_flatgradp=_compute_horizontal_gradient_of_exner_pressure_for_nonflat_coordinates(
            inv_dual_edge_length=inv_dual_edge_length,
            z_exner_ex_pr=temporal_extrapolation_of_perturbed_exner,
            ddxn_z_full=ddxn_z_full,
            c_lin_e=c_lin_e,
            z_dexner_dz_c_1=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        ),
        below_flatgradp=_compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
            inv_dual_edge_length=inv_dual_edge_length,
            z_exner_ex_pr=temporal_extrapolation_of_perturbed_exner,
            zdiff_gradp=zdiff_gradp,
            ikoffset=ikoffset,
            z_dexner_dz_c_1=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            z_dexner_dz_c_2=d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        ),
    )

    return apply_hydrostatic_correction_to_horizontal_gradient_of_exner_pressure(
        pg_exdist=pg_exdist,
        z_hydro_corr=hydrostatic_correction_on_lowest_level,
        z_gradh_exner=horizontal_pressure_gradient,
    )


@gtx.field_operator
def _compute_rho_theta_pgrad_and_update_vn(
    next_vn: fa.EdgeKField[ta.wpfloat],
    current_vn: fa.EdgeKField[ta.wpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    reference_rho_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    reference_theta_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    perturbed_rho_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    perturbed_theta_v_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[ta.vpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    hydrostatic_correction_on_lowest_level: fa.EdgeField[ta.wpfloat],
    predictor_normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    normal_wind_tendency_due_to_slow_physics_process: fa.EdgeKField[ta.vpfloat],
    normal_wind_iau_increment: fa.EdgeKField[ta.vpfloat],
    grf_tend_vn: fa.EdgeKField[ta.wpfloat],
    geofac_grg_x: gtx.Field[[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    geofac_grg_y: gtx.Field[[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    pos_on_tplane_e_x: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    pos_on_tplane_e_y: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_x: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_x: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_y: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_y: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    c_lin_e: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ikoffset: gtx.Field[[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[[dims.EdgeDim, dims.E2CDim, dims.KDim], ta.vpfloat],
    pg_exdist: fa.EdgeKField[ta.vpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    dtime: ta.wpfloat,
    is_iau_active: bool,
    iau_wgt_dyn: ta.wpfloat,
    limited_area: bool,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    start_edge_lateral_boundary: gtx.int32,
    start_edge_lateral_boundary_level_7: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_nudging: gtx.int32,
    end_edge_local: gtx.int32,
    end_edge_halo: gtx.int32,
) -> tuple[
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
]:
    # TODO(havogt): it would be nice if we could shrink the start of the compute domain to `start_edge_lateral_boundary_level_7 <= dims.EdgeDim`,
    # but that would require to put the correct lateral boundary condition where this is consumed.
    # TODO(havogt): most likely it is possible to remove the `end_edge_halo` bound here (and shrink the compute domain), the corresponding
    # Fortran code states "Initialize halo edges with zero in order to avoid access of uninitialized array elements".
    (rho_at_edges_on_model_levels, theta_v_at_edges_on_model_levels) = concat_where(
        ((start_edge_lateral_boundary_level_7 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_halo)),
        _compute_horizontal_advection_of_rho_and_theta(
            p_vn=current_vn,
            p_vt=tangential_wind,
            pos_on_tplane_e_1=pos_on_tplane_e_x,
            pos_on_tplane_e_2=pos_on_tplane_e_y,
            primal_normal_cell_1=primal_normal_cell_x,
            dual_normal_cell_1=dual_normal_cell_x,
            primal_normal_cell_2=primal_normal_cell_y,
            dual_normal_cell_2=dual_normal_cell_y,
            p_dthalf=wpfloat("0.5") * dtime,
            rho_ref_me=reference_rho_at_edges_on_model_levels,
            theta_ref_me=reference_theta_at_edges_on_model_levels,
            perturbed_rho_at_cells_on_model_levels=perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
        ),
        (
            broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim)),
            broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim)),
        ),
    )

    # The pressure gradient is computed on [start_edge_nudging_level_2, end_edge_local),
    # matching the Fortran compute domain rl = [grf_bdywidth_e+1, min_rledge_int]. Outside
    # this interval the E2C neighbors and the ikoffset/zdiff_gradp/pg_exdist metric fields
    # contain skip values (-1) or are uninitialized (outer lateral-boundary rows and halo
    # edges), and the result is never consumed there; write 0.0 instead.
    horizontal_pressure_gradient = concat_where(
        (start_edge_nudging_level_2 <= dims.EdgeDim) & (dims.EdgeDim < end_edge_local),
        _compute_horizontal_pressure_gradient(
            temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            hydrostatic_correction_on_lowest_level=hydrostatic_correction_on_lowest_level,
            ddxn_z_full=ddxn_z_full,
            c_lin_e=c_lin_e,
            ikoffset=ikoffset,
            zdiff_gradp=zdiff_gradp,
            pg_exdist=pg_exdist,
            inv_dual_edge_length=inv_dual_edge_length,
            nflatlev=nflatlev,
            nflat_gradp=nflat_gradp,
        ),
        broadcast(wpfloat("0.0"), (dims.EdgeDim, dims.KDim)),
    )

    # Note: we overcompute `next_vn`, which is only needed
    # up to dims.EdgeDim < end_edge_local
    # TODO(havogt): with multiple output domains this should be fixed.
    next_vn = concat_where(
        start_edge_nudging_level_2 <= dims.EdgeDim,
        _add_temporal_tendencies_to_vn(
            vn_nnow=current_vn,
            ddt_vn_apc_ntl1=predictor_normal_wind_advective_tendency,
            ddt_vn_phy=normal_wind_tendency_due_to_slow_physics_process,
            z_theta_v_e=theta_v_at_edges_on_model_levels,
            z_gradh_exner=horizontal_pressure_gradient,
            dtime=dtime,
        ),
        next_vn,
    )

    if is_iau_active:
        # Note: we overcompute `next_vn`, see above.
        next_vn = concat_where(
            start_edge_nudging_level_2 <= dims.EdgeDim,
            _add_analysis_increments_to_vn(
                vn_incr=normal_wind_iau_increment, vn=next_vn, iau_wgt_dyn=iau_wgt_dyn
            ),
            next_vn,
        )

    if limited_area:
        next_vn = concat_where(
            (start_edge_lateral_boundary <= dims.EdgeDim) & (dims.EdgeDim < end_edge_nudging),
            _compute_vn_on_lateral_boundary(
                vn_now=current_vn,
                grf_tend_vn=grf_tend_vn,
                dtime=dtime,
            ),
            next_vn,
        )

    return (
        rho_at_edges_on_model_levels,
        theta_v_at_edges_on_model_levels,
        horizontal_pressure_gradient,
        next_vn,
    )


@gtx.field_operator
def _apply_divergence_damping_and_update_vn(
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[ta.vpfloat],
    next_vn: fa.EdgeKField[ta.wpfloat],
    current_vn: fa.EdgeKField[ta.wpfloat],
    dwdz_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    predictor_normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    corrector_normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    normal_wind_tendency_due_to_slow_physics_process: fa.EdgeKField[ta.vpfloat],
    normal_wind_iau_increment: fa.EdgeKField[ta.vpfloat],
    second_order_divdamp_scaling_coeff: ta.wpfloat,
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    horizontal_pressure_gradient: fa.EdgeKField[ta.vpfloat],
    horizontal_mask_for_3d_divdamp: fa.EdgeField[ta.wpfloat],
    scaling_factor_for_3d_divdamp: fa.KField[ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    nudgecoeff_e: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    interpolated_fourth_order_divdamp_factor: fa.KField[ta.wpfloat],
    advection_explicit_weight_parameter: ta.wpfloat,
    advection_implicit_weight_parameter: ta.wpfloat,
    dtime: ta.wpfloat,
    is_iau_active: bool,
    iau_wgt_dyn: ta.wpfloat,
    limited_area: bool,
    apply_2nd_order_divergence_damping: bool,
    apply_4th_order_divergence_damping: bool,
    divdamp_order: gtx.int32,
    mean_cell_area: float,
    second_order_divdamp_factor: float,
    max_nudging_coefficient: float,
    dbl_eps: float,
) -> fa.EdgeKField[ta.wpfloat]:
    # add dw/dz for divergence damping term. In ICON, this stencil starts from k = kstart_dd3d until k = nlev - 1.
    # Since scaling_factor_for_3d_divdamp is zero when k < kstart_dd3d, it is meaningless to execute computation
    # above level kstart_dd3d. But we have decided to remove this manual optimization in icon4py.
    # See discussion in this PR https://github.com/C2SM/icon4py/pull/793
    horizontal_gradient_of_total_divergence = _add_vertical_wind_derivative_to_divergence_damping(
        hmask_dd3d=horizontal_mask_for_3d_divdamp,
        scalfac_dd3d=scaling_factor_for_3d_divdamp,
        inv_dual_edge_length=inv_dual_edge_length,
        z_dwdz_dd=dwdz_at_cells_on_model_levels,
        z_graddiv_vn=horizontal_gradient_of_normal_wind_divergence,
    )

    next_vn = _add_temporal_tendencies_to_vn_by_interpolating_between_time_levels(
        vn_nnow=current_vn,
        ddt_vn_apc_ntl1=predictor_normal_wind_advective_tendency,
        ddt_vn_apc_ntl2=corrector_normal_wind_advective_tendency,
        ddt_vn_phy=normal_wind_tendency_due_to_slow_physics_process,
        z_theta_v_e=theta_v_at_edges_on_model_levels,
        z_gradh_exner=horizontal_pressure_gradient,
        dtime=dtime,
        wgt_nnow_vel=advection_explicit_weight_parameter,
        wgt_nnew_vel=advection_implicit_weight_parameter,
        cpd=PhysicsConstants.cpd,
    )

    if apply_2nd_order_divergence_damping:
        next_vn = _apply_2nd_order_divergence_damping(
            z_graddiv_vn=horizontal_gradient_of_total_divergence,
            vn=next_vn,
            scal_divdamp_o2=second_order_divdamp_scaling_coeff,
        )

    if apply_4th_order_divergence_damping:
        squared_horizontal_gradient_of_total_divergence = _compute_graddiv2_of_vn(
            geofac_grdiv=geofac_grdiv, z_graddiv_vn=horizontal_gradient_of_total_divergence
        )
        if limited_area:
            next_vn = _apply_weighted_2nd_and_4th_order_divergence_damping(
                interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
                nudgecoeff_e=nudgecoeff_e,
                z_graddiv2_vn=squared_horizontal_gradient_of_total_divergence,
                vn=next_vn,
                divdamp_order=divdamp_order,
                mean_cell_area=mean_cell_area,
                second_order_divdamp_factor=second_order_divdamp_factor,
                max_nudging_coefficient=max_nudging_coefficient,
                dbl_eps=dbl_eps,
            )
        else:
            next_vn = _apply_4th_order_divergence_damping(
                interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
                z_graddiv2_vn=squared_horizontal_gradient_of_total_divergence,
                vn=next_vn,
                divdamp_order=divdamp_order,
                mean_cell_area=mean_cell_area,
                second_order_divdamp_factor=second_order_divdamp_factor,
            )

    if is_iau_active:
        next_vn = _add_analysis_increments_to_vn(
            vn_incr=normal_wind_iau_increment, vn=next_vn, iau_wgt_dyn=iau_wgt_dyn
        )

    return next_vn


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_rho_theta_pgrad_and_update_vn(
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    horizontal_pressure_gradient: fa.EdgeKField[ta.vpfloat],
    next_vn: fa.EdgeKField[ta.wpfloat],
    current_vn: fa.EdgeKField[ta.wpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    reference_rho_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    reference_theta_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    perturbed_rho_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    perturbed_theta_v_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[ta.vpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    hydrostatic_correction_on_lowest_level: fa.EdgeField[ta.wpfloat],
    predictor_normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    normal_wind_tendency_due_to_slow_physics_process: fa.EdgeKField[ta.vpfloat],
    normal_wind_iau_increment: fa.EdgeKField[ta.vpfloat],
    grf_tend_vn: fa.EdgeKField[ta.wpfloat],
    geofac_grg_x: gtx.Field[[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    geofac_grg_y: gtx.Field[[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
    pos_on_tplane_e_x: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    pos_on_tplane_e_y: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_x: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_x: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    primal_normal_cell_y: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    dual_normal_cell_y: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    c_lin_e: gtx.Field[[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ikoffset: gtx.Field[[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[[dims.EdgeDim, dims.E2CDim, dims.KDim], ta.vpfloat],
    pg_exdist: fa.EdgeKField[ta.vpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    dtime: ta.wpfloat,
    is_iau_active: bool,
    iau_wgt_dyn: ta.wpfloat,
    limited_area: bool,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    start_edge_lateral_boundary: gtx.int32,
    start_edge_lateral_boundary_level_7: gtx.int32,
    start_edge_nudging_level_2: gtx.int32,
    end_edge_nudging: gtx.int32,
    end_edge_local: gtx.int32,
    end_edge_halo: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Formerly known as fused_solve_nonhydro_stencil_15_to_28_predictor.

    This program computes the air density and virtual potential temperature at edges on model levels.
    It also computes horizontal pressure gradient and updates normal wind by adding all the tendency terms
    in the Navier-Stokes equation. If data assimilation is considered, an increment is added to
    normal wind.

    Args:
        - rho_at_edges_on_model_levels: air density on edges on model levels [kg m-3]
        - theta_v_at_edges_on_model_levels: virtual potential temperature at edges on model levels [K]
        - horizontal_pressure_gradient: horizontal pressure gradient at edges on model levels [Pa m-1]
        - next_vn: normal wind to be updated [m s-1]
        - current_vn: normal wind at previous substep [m s-1]
        - tangential_wind: tangential wind at edges on model levels [m s-1]
        - reference_rho_at_edges_on_model_levels: reference air density at cells on model levels [kg m-3]
        - reference_theta_at_edges_on_model_levels: reference virtual potential temperature at edges on model levels [K]
        - perturbed_rho_at_cells_on_model_levels: perturbed air density (actual density minus reference density) at cells on model levels [kg m-3]
        - perturbed_theta_v_at_cells_on_model_levels: reference virtual potential temperature (actual potential temperature minus reference potential temperature) at cells on model levels [K]
        - temporal_extrapolation_of_perturbed_exner: temporal extrapolation of perturbed exner function (actual exner function minus reference exner function)
        - ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: vertical gradient of temporal extrapolation of perturbed exner function [m-1]
        - d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: second vertical gradient of temporal extrapolation of perturbed exner function [m-2]
        - hydrostatic_correction_on_lowest_level: hydrostatic correction for steep slope (see https://doi.org/10.1175/MWR-D-12-00049.1) [m-1]
        - predictor_normal_wind_advective_tendency: horizontal advection tendency of the normal wind at predictor step [m s-2]
        - normal_wind_tendency_due_to_slow_physics_process: normal wind tendeny due to slow physics [m s-2]
        - normal_wind_iau_increment: iau increment to normal wind (data assimilation) [m s-1]
        - geofac_grg_x: interpolation coefficient for computation of x-derivative of a cell-based variable at cell center using Green-Gauss theorem [m-1]
        - geofac_grg_y: interpolation coefficient for computation of y-derivative of a cell-based variable at cell center using Green-Gauss theorem [m-1]
        - pos_on_tplane_e_x: x-position of the edge on the tangential plane centered at cell center [m]
        - pos_on_tplane_e_y: y-position of the edge on the tangential plane centered at cell center [m]
        - primal_normal_cell_x: x component of normal vector of edges of triangles
        - primal_normal_cell_y: y component of normal vector of edges of triangles
        - dual_normal_cell_x: x component of normal vector of edges of dual hexagons
        - dual_normal_cell_y: y component of normal vector of edges of dual hexagons
        - ddxn_z_full: metric coefficient for computation of vertical derivative at model levels
        - c_lin_e: interpolation coefficient for computation of interpolating a cell-based variables to an edge-based variable
        - ikoffset: k offset index (offset from the lowest k index where the neighboring cell centers lie within the thickness of the layer) for hyrostatic correction
        - zdiff_gradp: vertical distance between current cell height and neighboring cell height for pressure gradient over multiple levels [m]
        - pg_exdist: vertical distance between current cell height and neighboring cell height for hydrostatic correction [m]
        - inv_dual_edge_length: inverse dual edge length [m]
        - dtime: time step [s]
        - is_iau_active: option for iau increment analysis
        - iau_wgt_dyn: a scaling factor for iau increment
        - limited_area: option indicating the grid is limited area or not
        - nflatlev: starting vertical index of flat levels
        - nflat_gradp: starting vertical index when neighboring cell centers lie within the thickness of the layer
        - start_edge_halo_level_2: start index of second halo level zone for edges
        - end_edge_halo_level_2: end index of second halo level zone for edges
        - start_edge_lateral_boundary: start index of first lateral boundary level (counting from outermost) zone for edges
        - end_edge_halo: end index of first halo level zone for edges
        - start_edge_lateral_boundary_level_7: start index of 7th lateral boundary level (counting from outermost) zone for edges
        - start_edge_nudging_level_2: start index of second nudging level zone for edges
        - end_edge_local: end index of local zone for edges

    Returns:
        - next_vn: normal wind to be updated [m s-1]
    """
    _compute_rho_theta_pgrad_and_update_vn(
        next_vn=next_vn,
        current_vn=current_vn,
        tangential_wind=tangential_wind,
        reference_rho_at_edges_on_model_levels=reference_rho_at_edges_on_model_levels,
        reference_theta_at_edges_on_model_levels=reference_theta_at_edges_on_model_levels,
        perturbed_rho_at_cells_on_model_levels=perturbed_rho_at_cells_on_model_levels,
        perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
        temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        hydrostatic_correction_on_lowest_level=hydrostatic_correction_on_lowest_level,
        predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
        normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
        normal_wind_iau_increment=normal_wind_iau_increment,
        grf_tend_vn=grf_tend_vn,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        pos_on_tplane_e_x=pos_on_tplane_e_x,
        pos_on_tplane_e_y=pos_on_tplane_e_y,
        primal_normal_cell_x=primal_normal_cell_x,
        dual_normal_cell_x=dual_normal_cell_x,
        primal_normal_cell_y=primal_normal_cell_y,
        dual_normal_cell_y=dual_normal_cell_y,
        ddxn_z_full=ddxn_z_full,
        c_lin_e=c_lin_e,
        ikoffset=ikoffset,
        zdiff_gradp=zdiff_gradp,
        pg_exdist=pg_exdist,
        inv_dual_edge_length=inv_dual_edge_length,
        dtime=dtime,
        is_iau_active=is_iau_active,
        iau_wgt_dyn=iau_wgt_dyn,
        limited_area=limited_area,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
        start_edge_lateral_boundary=start_edge_lateral_boundary,
        start_edge_lateral_boundary_level_7=start_edge_lateral_boundary_level_7,
        start_edge_nudging_level_2=start_edge_nudging_level_2,
        end_edge_nudging=end_edge_nudging,
        end_edge_local=end_edge_local,
        end_edge_halo=end_edge_halo,
        out=(
            rho_at_edges_on_model_levels,
            theta_v_at_edges_on_model_levels,
            horizontal_pressure_gradient,
            next_vn,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_divergence_damping_and_update_vn(
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[ta.vpfloat],
    next_vn: fa.EdgeKField[ta.wpfloat],
    current_vn: fa.EdgeKField[ta.wpfloat],
    dwdz_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    predictor_normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    corrector_normal_wind_advective_tendency: fa.EdgeKField[ta.vpfloat],
    normal_wind_tendency_due_to_slow_physics_process: fa.EdgeKField[ta.vpfloat],
    normal_wind_iau_increment: fa.EdgeKField[ta.vpfloat],
    second_order_divdamp_scaling_coeff: ta.wpfloat,
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    horizontal_pressure_gradient: fa.EdgeKField[ta.vpfloat],
    horizontal_mask_for_3d_divdamp: fa.EdgeField[ta.wpfloat],
    scaling_factor_for_3d_divdamp: fa.KField[ta.wpfloat],
    inv_dual_edge_length: fa.EdgeField[ta.wpfloat],
    nudgecoeff_e: fa.EdgeField[ta.wpfloat],
    geofac_grdiv: gtx.Field[[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    interpolated_fourth_order_divdamp_factor: fa.KField[ta.wpfloat],
    advection_explicit_weight_parameter: ta.wpfloat,
    advection_implicit_weight_parameter: ta.wpfloat,
    dtime: ta.wpfloat,
    is_iau_active: bool,
    iau_wgt_dyn: ta.wpfloat,
    limited_area: bool,
    apply_2nd_order_divergence_damping: bool,
    apply_4th_order_divergence_damping: bool,
    divdamp_order: gtx.int32,
    mean_cell_area: float,
    second_order_divdamp_factor: float,
    max_nudging_coefficient: float,
    dbl_eps: float,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Formerly known as fused_solve_nonhydro_stencil_15_to_28_corrector.

    This program updates normal wind by adding all tendency terms in the Navier-Stokes equation with
    interpolated advective tendency computed in the predictor and corrector steps, and then computes
    the divergence damping and add it to normal wind. If data assimilation is considered, an
    increment is added to normal wind.

    Args:
        - horizontal_gradient_of_normal_wind_divergence: horizontal gradient of divergence of normal wind at edges [m-1 s-1]
        - next_vn: normal wind to be updated [m s-1]
        - current_vn: normal wind at previous substep [m s-1]
        - dwdz_at_cells_on_model_levels: vertical derivative of vertical wind [s-1]
        - predictor_normal_wind_advective_tendency: horizontal advection tendency of the normal wind at predictor step [m s-2]
        - corrector_normal_wind_advective_tendency: horizontal advection tendency of the normal wind at corrector step [m s-2]
        - normal_wind_tendency_due_to_slow_physics_process: normal wind tendeny due to slow physics [m s-2]
        - normal_wind_iau_increment: iau increment to normal wind (data assimilation) [m s-1]
        - second_order_divdamp_scaling_coeff: second order divergence damping coefficient [m s]
        - theta_v_at_edges_on_model_levels: virtual potential temperature at edges on model levels [K]
        - horizontal_pressure_gradient: horizontal pressure gradient at edges on model levels [Pa m-1]
        - horizontal_mask_for_3d_divdamp: horizontal mask for 3D divergence damping (including dw/dz) at edges on model levels
        - scaling_factor_for_3d_divdamp: scaling factor in vertical dimension for 3D divergence damping (including dw/dz) on model levels
        - inv_dual_edge_length: inverse dual edge length
        - nudgecoeff_e: nudging coefficient for fourth order divergence damping at nest boundary
        - geofac_grdiv: metric coefficient for computation of horizontal gradient of divergence
        - interpolated_fourth_order_divdamp_factor
        - advection_explicit_weight_parameter: explicitness weight of normal_wind_advective_tendency
        - advection_implicit_weight_parameter: implicitness weight of normal_wind_advective_tendency
        - dtime: time step [s]
        - is_iau_active: option for iau increment analysis
        - iau_wgt_dyn: a scaling factor for iau increment
        - limited_area: option indicating the grid is limited area or not
        - apply_2nd_order_divergence_damping: scaling factor for second order divergence damping
        - apply_4th_order_divergence_damping: scaling factor for fourth order divergence damping
        - divdamp_order: divergence damping order (see the class DivergenceDampingOrder)
        - mean_cell_area
        - second_order_divdamp_factor
        - max_nudging_coefficient
        - dbl_eps

    Returns:
        - next_vn: normal wind to be updated [m s-1]
    """
    _apply_divergence_damping_and_update_vn(
        horizontal_gradient_of_normal_wind_divergence=horizontal_gradient_of_normal_wind_divergence,
        next_vn=next_vn,
        current_vn=current_vn,
        dwdz_at_cells_on_model_levels=dwdz_at_cells_on_model_levels,
        predictor_normal_wind_advective_tendency=predictor_normal_wind_advective_tendency,
        corrector_normal_wind_advective_tendency=corrector_normal_wind_advective_tendency,
        normal_wind_tendency_due_to_slow_physics_process=normal_wind_tendency_due_to_slow_physics_process,
        normal_wind_iau_increment=normal_wind_iau_increment,
        theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
        horizontal_pressure_gradient=horizontal_pressure_gradient,
        second_order_divdamp_scaling_coeff=second_order_divdamp_scaling_coeff,
        horizontal_mask_for_3d_divdamp=horizontal_mask_for_3d_divdamp,
        scaling_factor_for_3d_divdamp=scaling_factor_for_3d_divdamp,
        inv_dual_edge_length=inv_dual_edge_length,
        nudgecoeff_e=nudgecoeff_e,
        geofac_grdiv=geofac_grdiv,
        interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
        advection_explicit_weight_parameter=advection_explicit_weight_parameter,
        advection_implicit_weight_parameter=advection_implicit_weight_parameter,
        dtime=dtime,
        is_iau_active=is_iau_active,
        iau_wgt_dyn=iau_wgt_dyn,
        limited_area=limited_area,
        apply_2nd_order_divergence_damping=apply_2nd_order_divergence_damping,
        apply_4th_order_divergence_damping=apply_4th_order_divergence_damping,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        second_order_divdamp_factor=second_order_divdamp_factor,
        max_nudging_coefficient=max_nudging_coefficient,
        dbl_eps=dbl_eps,
        out=next_vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
