# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import gt4py.next as gtx
from gt4py.next import astype, broadcast
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.dycore.stencils.add_analysis_increments_from_data_assimilation import (
    _add_analysis_increments_from_data_assimilation,
)
from icon4py.model.atmosphere.dycore.stencils.apply_rayleigh_damping_mechanism import (
    _apply_rayleigh_damping_mechanism,
)
from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction_of_w import (
    _compute_contravariant_correction_of_w,
)
from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction_of_w_for_lower_boundary import (
    _compute_contravariant_correction_of_w_for_lower_boundary,
)
from icon4py.model.atmosphere.dycore.stencils.compute_divergence_of_fluxes_of_rho_and_theta import (
    _compute_divergence_of_fluxes_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.compute_dwdz_for_divergence_damping import (
    _compute_dwdz_for_divergence_damping,
)
from icon4py.model.atmosphere.dycore.stencils.compute_explicit_part_for_rho_and_exner import (
    _compute_explicit_part_for_rho_and_exner,
)
from icon4py.model.atmosphere.dycore.stencils.compute_results_for_thermodynamic_variables import (
    _compute_results_for_thermodynamic_variables,
)
from icon4py.model.atmosphere.dycore.stencils.solve_tridiagonal_matrix_for_w_back_substitution import (
    _solve_tridiagonal_matrix_for_w_back_substitution_scan,
)
from icon4py.model.atmosphere.dycore.stencils.solve_tridiagonal_matrix_for_w_forward_sweep import (
    _solve_tridiagonal_matrix_for_w_forward_sweep,
)
from icon4py.model.atmosphere.dycore.stencils.update_dynamical_exner_time_increment import (
    _update_dynamical_exner_time_increment,
)
from icon4py.model.atmosphere.dycore.stencils.update_mass_volume_flux import (
    _update_mass_volume_flux,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants, RayleighType
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels(
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[vpfloat]:
    contravariant_correction_at_cells_on_half_levels = concat_where(
        dims.KDim < nlev,
        _compute_contravariant_correction_of_w(
            e_bln_c_s, contravariant_correction_at_edges_on_model_levels, wgtfac_c
        ),
        _compute_contravariant_correction_of_w_for_lower_boundary(
            e_bln_c_s, contravariant_correction_at_edges_on_model_levels, wgtfacq_c
        ),
    )
    return contravariant_correction_at_cells_on_half_levels


@gtx.field_operator
def _set_surface_boundary_condition_for_computation_of_w(
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
) -> fa.CellKField[ta.wpfloat]:
    return astype(contravariant_correction_at_cells_on_half_levels, wpfloat)


@gtx.field_operator
def _compute_w_explicit_term_with_predictor_advective_tendency(
    current_w: fa.CellKField[wpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKField[vpfloat],
    nonhydro_buoy_at_cells_on_half_levels: fa.CellKField[vpfloat],
    dtime: wpfloat,
) -> fa.CellKField[wpfloat]:
    (
        predictor_vertical_wind_advective_tendency_wp,
        nonhydro_buoy_at_cells_on_half_levels_wp,
    ) = astype(
        (
            predictor_vertical_wind_advective_tendency,
            nonhydro_buoy_at_cells_on_half_levels,
        ),
        wpfloat,
    )

    w_explicit_term_wp = current_w + dtime * (
        predictor_vertical_wind_advective_tendency_wp
        - PhysicsConstants.cpd * nonhydro_buoy_at_cells_on_half_levels_wp
    )
    return w_explicit_term_wp


@gtx.field_operator
def _compute_w_explicit_term_with_interpolated_predictor_corrector_advective_tendency(
    current_w: fa.CellKField[wpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKField[vpfloat],
    corrector_vertical_wind_advective_tendency: fa.CellKField[vpfloat],
    nonhydro_buoy_at_cells_on_half_levels: fa.CellKField[vpfloat],
    dtime: wpfloat,
    advection_explicit_weight_parameter: wpfloat,
    advection_implicit_weight_parameter: wpfloat,
) -> fa.CellKField[wpfloat]:
    (
        predictor_vertical_wind_advective_tendency_wp,
        corrector_vertical_wind_advective_tendency_wp,
        nonhydro_buoy_at_cells_on_half_levels_wp,
    ) = astype(
        (
            predictor_vertical_wind_advective_tendency,
            corrector_vertical_wind_advective_tendency,
            nonhydro_buoy_at_cells_on_half_levels,
        ),
        wpfloat,
    )

    w_explicit_term_wp = current_w + dtime * (
        advection_explicit_weight_parameter * predictor_vertical_wind_advective_tendency_wp
        + advection_implicit_weight_parameter * corrector_vertical_wind_advective_tendency_wp
        - PhysicsConstants.cpd * nonhydro_buoy_at_cells_on_half_levels_wp
    )
    return w_explicit_term_wp


@gtx.field_operator
def _compute_solver_coefficients_matrix(
    current_exner: fa.CellKField[wpfloat],
    current_rho: fa.CellKField[wpfloat],
    current_theta_v: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[wpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    inv_ddqz_z_full_wp = astype(inv_ddqz_z_full, wpfloat)

    z_beta_wp = (
        dtime
        * PhysicsConstants.rd
        * current_exner
        / (PhysicsConstants.cvd * current_rho * current_theta_v)
        * inv_ddqz_z_full_wp
    )
    z_alpha_wp = (
        exner_w_implicit_weight_parameter
        * theta_v_at_cells_on_half_levels
        * rho_at_cells_on_half_levels
    )
    return astype((z_beta_wp, z_alpha_wp), vpfloat)


@gtx.field_operator
def solve_w(
    last_inner_level: gtx.int32,
    next_w: fa.CellKField[wpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    ddqz_z_half: fa.CellKField[vpfloat],
    z_alpha: fa.CellKField[vpfloat],
    z_beta: fa.CellKField[vpfloat],
    z_w_expl: fa.CellKField[wpfloat],
    z_exner_expl: fa.CellKField[wpfloat],
    dtime: wpfloat,
    cpd: wpfloat,
) -> fa.CellKField[wpfloat]:
    (
        tridiagonal_intermediate_result,
        next_w_intermediate_result,
    ) = concat_where(
        dims.KDim > 0,
        _solve_tridiagonal_matrix_for_w_forward_sweep(
            vwind_impl_wgt=vwind_impl_wgt,
            theta_v_ic=theta_v_ic,
            ddqz_z_half=ddqz_z_half,
            z_alpha=z_alpha,
            z_beta=z_beta,
            z_w_expl=z_w_expl,
            z_exner_expl=z_exner_expl,
            dtime=dtime,
            cpd=cpd,
        ),
        (broadcast(vpfloat("0.0"), (dims.CellDim,)), broadcast(wpfloat("0.0"), (dims.CellDim,))),
    )
    next_w = concat_where(
        dims.KDim < last_inner_level,
        _solve_tridiagonal_matrix_for_w_back_substitution_scan(
            z_q=tridiagonal_intermediate_result,
            w=next_w_intermediate_result,
        ),
        next_w,
    )
    return next_w


@gtx.field_operator
def _vertically_implicit_solver_at_predictor_step(
    next_w: fa.CellKField[
        ta.wpfloat
    ],  # necessary input because the last vertical level is set outside this field operator
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    nonhydro_buoy_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    current_w: fa.CellKField[ta.wpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.vpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[ta.wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    exner_tendency_due_to_slow_physics: fa.CellKField[ta.vpfloat],
    rho_iau_increment: fa.CellKField[ta.vpfloat],
    exner_iau_increment: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    exner_dynamical_increment: fa.CellKField[ta.wpfloat],
    dwdz_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    rayleigh_damping_factor: fa.KField[ta.wpfloat],
    reference_exner_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    iau_wgt_dyn: ta.wpfloat,
    dtime: ta.wpfloat,
    rayleigh_type: gtx.int32,
    divdamp_type: gtx.int32,
    is_iau_active: bool,
    at_first_substep: bool,
    end_index_of_damping_layer: gtx.int32,
    kstart_moist: gtx.int32,
    n_lev: gtx.int32,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    divergence_of_mass, divergence_of_theta_v = _compute_divergence_of_fluxes_of_rho_and_theta(
        geofac_div=geofac_div,
        mass_fl_e=mass_flux_at_edges_on_model_levels,
        z_theta_v_fl_e=theta_v_flux_at_edges_on_model_levels,
    )

    w_explicit_term = concat_where(
        1 <= dims.KDim,
        _compute_w_explicit_term_with_predictor_advective_tendency(
            current_w=current_w,
            predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
            nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy_at_cells_on_half_levels,
            dtime=dtime,
        ),
        broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
    )

    vertical_mass_flux_at_cells_on_half_levels = concat_where(
        (1 <= dims.KDim) & (dims.KDim < n_lev),
        rho_at_cells_on_half_levels
        * (
            -astype(contravariant_correction_at_cells_on_half_levels, wpfloat)
            + exner_w_explicit_weight_parameter * current_w
        ),
        broadcast(wpfloat("0.0"), (dims.CellDim,)),
    )

    (
        tridiagonal_beta_coeff_at_cells_on_model_levels,
        tridiagonal_alpha_coeff_at_cells_on_half_levels,
    ) = _compute_solver_coefficients_matrix(
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        inv_ddqz_z_full=inv_ddqz_z_full,
        exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
        dtime=dtime,
    )
    tridiagonal_alpha_coeff_at_cells_on_half_levels = concat_where(
        dims.KDim < n_lev,
        tridiagonal_alpha_coeff_at_cells_on_half_levels,
        broadcast(vpfloat("0.0"), (dims.CellDim,)),
    )

    (rho_explicit_term, exner_explicit_term) = _compute_explicit_part_for_rho_and_exner(
        rho_nnow=current_rho,
        inv_ddqz_z_full=inv_ddqz_z_full,
        z_flxdiv_mass=divergence_of_mass,
        z_contr_w_fl_l=vertical_mass_flux_at_cells_on_half_levels,
        exner_pr=perturbed_exner_at_cells_on_model_levels,
        z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
        z_flxdiv_theta=divergence_of_theta_v,
        theta_v_ic=theta_v_at_cells_on_half_levels,
        ddt_exner_phy=exner_tendency_due_to_slow_physics,
        dtime=dtime,
    )

    if is_iau_active:
        rho_explicit_term, exner_explicit_term = _add_analysis_increments_from_data_assimilation(
            z_rho_expl=rho_explicit_term,
            z_exner_expl=exner_explicit_term,
            rho_incr=rho_iau_increment,
            exner_incr=exner_iau_increment,
            iau_wgt_dyn=iau_wgt_dyn,
        )

    next_w = solve_w(
        last_inner_level=n_lev,
        next_w=next_w,  # n_lev value is set by _set_surface_boundary_condtion_for_computation_of_w
        vwind_impl_wgt=exner_w_implicit_weight_parameter,
        theta_v_ic=theta_v_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        z_alpha=tridiagonal_alpha_coeff_at_cells_on_half_levels,
        z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
        z_w_expl=w_explicit_term,
        z_exner_expl=exner_explicit_term,
        dtime=dtime,
        cpd=PhysicsConstants.cpd,
    )

    if rayleigh_type == RayleighType.KLEMP:
        next_w = concat_where(
            (dims.KDim > 0) & (dims.KDim < end_index_of_damping_layer + 1),
            _apply_rayleigh_damping_mechanism(
                z_raylfac=rayleigh_damping_factor,
                w=next_w,
            ),
            next_w,
        )

    next_rho, next_exner, next_theta_v = _compute_results_for_thermodynamic_variables(
        z_rho_expl=rho_explicit_term,
        vwind_impl_wgt=exner_w_implicit_weight_parameter,
        inv_ddqz_z_full=inv_ddqz_z_full,
        rho_ic=rho_at_cells_on_half_levels,
        w=next_w,
        z_exner_expl=exner_explicit_term,
        exner_ref_mc=reference_exner_at_cells_on_model_levels,
        z_alpha=tridiagonal_alpha_coeff_at_cells_on_half_levels,
        z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
        rho_now=current_rho,
        theta_v_now=current_theta_v,
        exner_now=current_exner,
        dtime=dtime,
    )

    # compute dw/dz for divergence damping term. In ICON, dwdz_at_cells_on_model_levels is
    # computed from k >= kstart_dd3d. We have decided to remove this manual optimization in icon4py.
    # See discussion in this PR https://github.com/C2SM/icon4py/pull/793
    if divdamp_type >= 3:
        dwdz_at_cells_on_model_levels = _compute_dwdz_for_divergence_damping(
            inv_ddqz_z_full=inv_ddqz_z_full,
            w=next_w,
            w_concorr_c=contravariant_correction_at_cells_on_half_levels,
        )

    exner_dynamical_increment = (
        concat_where(
            kstart_moist <= dims.KDim,
            astype(current_exner, vpfloat),
            exner_dynamical_increment,
        )
        if at_first_substep
        else exner_dynamical_increment
    )

    return (
        next_w,
        next_rho,
        next_exner,
        next_theta_v,
        dwdz_at_cells_on_model_levels,
        exner_dynamical_increment,
    )


@gtx.program
def vertically_implicit_solver_at_predictor_step(
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    next_w: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    next_exner: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    dwdz_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    exner_dynamical_increment: fa.CellKField[ta.vpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    nonhydro_buoy_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    current_w: fa.CellKField[ta.wpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.vpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[ta.wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    exner_tendency_due_to_slow_physics: fa.CellKField[ta.vpfloat],
    rho_iau_increment: fa.CellKField[ta.vpfloat],
    exner_iau_increment: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    rayleigh_damping_factor: fa.KField[ta.wpfloat],
    reference_exner_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    e_bln_c_s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
    iau_wgt_dyn: ta.wpfloat,
    dtime: ta.wpfloat,
    is_iau_active: bool,
    rayleigh_type: gtx.int32,
    divdamp_type: gtx.int32,
    at_first_substep: bool,
    end_index_of_damping_layer: gtx.int32,
    kstart_moist: gtx.int32,
    flat_level_index_plus1: gtx.int32,
    start_cell_index_nudging: gtx.int32,
    end_cell_index_local: gtx.int32,
    start_cell_index_lateral_lvl3: gtx.int32,
    end_cell_index_halo_lvl1: gtx.int32,
    vertical_start_index_model_top: gtx.int32,
    vertical_end_index_model_surface: gtx.int32,
) -> None:
    _interpolate_contravariant_correction_from_edges_on_model_levels_to_cells_on_half_levels(
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        e_bln_c_s=e_bln_c_s,
        wgtfac_c=wgtfac_c,
        wgtfacq_c=wgtfacq_c,
        nlev=vertical_end_index_model_surface - 1,
        out=contravariant_correction_at_cells_on_half_levels,
        domain={
            dims.CellDim: (
                start_cell_index_lateral_lvl3,
                end_cell_index_halo_lvl1,
            ),
            dims.KDim: (flat_level_index_plus1, vertical_end_index_model_surface),
        },
    )
    _set_surface_boundary_condition_for_computation_of_w(
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        out=next_w,
        domain={
            dims.CellDim: (start_cell_index_nudging, end_cell_index_local),
            dims.KDim: (vertical_end_index_model_surface - 1, vertical_end_index_model_surface),
        },
    )
    _vertically_implicit_solver_at_predictor_step(
        next_w=next_w,
        geofac_div=geofac_div,
        mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
        theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
        predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
        nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy_at_cells_on_half_levels,
        rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        current_w=current_w,
        inv_ddqz_z_full=inv_ddqz_z_full,
        exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
        rho_iau_increment=rho_iau_increment,
        exner_iau_increment=exner_iau_increment,
        ddqz_z_half=ddqz_z_half,
        dwdz_at_cells_on_model_levels=dwdz_at_cells_on_model_levels,
        exner_dynamical_increment=exner_dynamical_increment,
        rayleigh_damping_factor=rayleigh_damping_factor,
        reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
        iau_wgt_dyn=iau_wgt_dyn,
        dtime=dtime,
        rayleigh_type=rayleigh_type,
        divdamp_type=divdamp_type,
        is_iau_active=is_iau_active,
        at_first_substep=at_first_substep,
        end_index_of_damping_layer=end_index_of_damping_layer,
        kstart_moist=kstart_moist,
        n_lev=vertical_end_index_model_surface - 1,
        out=(
            next_w,
            next_rho,
            next_exner,
            next_theta_v,
            dwdz_at_cells_on_model_levels,
            exner_dynamical_increment,
        ),
        domain={
            dims.CellDim: (start_cell_index_nudging, end_cell_index_local),
            dims.KDim: (vertical_start_index_model_top, vertical_end_index_model_surface - 1),
        },
    )


@gtx.field_operator
def _vertically_implicit_solver_at_corrector_step(
    next_w: fa.CellKField[
        ta.wpfloat
    ],  # necessary input because the last vertical level is set outside this field operator
    dynamical_vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    exner_dynamical_increment: fa.CellKField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    corrector_vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    nonhydro_buoy_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    current_w: fa.CellKField[ta.wpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.vpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[ta.wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    exner_tendency_due_to_slow_physics: fa.CellKField[ta.vpfloat],
    rho_iau_increment: fa.CellKField[ta.vpfloat],
    exner_iau_increment: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    rayleigh_damping_factor: fa.KField[ta.wpfloat],
    reference_exner_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    advection_explicit_weight_parameter: ta.wpfloat,
    advection_implicit_weight_parameter: ta.wpfloat,
    lprep_adv: bool,
    r_nsubsteps: ta.wpfloat,
    ndyn_substeps_var: ta.wpfloat,
    iau_wgt_dyn: ta.wpfloat,
    dtime: ta.wpfloat,
    is_iau_active: bool,
    rayleigh_type: gtx.int32,
    at_first_substep: bool,
    at_last_substep: bool,
    end_index_of_damping_layer: gtx.int32,
    kstart_moist: gtx.int32,
    n_lev: gtx.int32,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    divergence_of_mass, divergence_of_theta_v = _compute_divergence_of_fluxes_of_rho_and_theta(
        geofac_div=geofac_div,
        mass_fl_e=mass_flux_at_edges_on_model_levels,
        z_theta_v_fl_e=theta_v_flux_at_edges_on_model_levels,
    )
    w_explicit_term = concat_where(
        1 <= dims.KDim,
        _compute_w_explicit_term_with_interpolated_predictor_corrector_advective_tendency(
            current_w=current_w,
            predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
            corrector_vertical_wind_advective_tendency=corrector_vertical_wind_advective_tendency,
            nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy_at_cells_on_half_levels,
            dtime=dtime,
            advection_explicit_weight_parameter=advection_explicit_weight_parameter,
            advection_implicit_weight_parameter=advection_implicit_weight_parameter,
        ),
        broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
    )
    vertical_mass_flux_at_cells_on_half_levels = concat_where(
        (1 <= dims.KDim) & (dims.KDim < n_lev),
        rho_at_cells_on_half_levels
        * (
            -astype(contravariant_correction_at_cells_on_half_levels, wpfloat)
            + exner_w_explicit_weight_parameter * current_w
        ),
        broadcast(wpfloat("0.0"), (dims.CellDim,)),
    )
    (
        tridiagonal_beta_coeff_at_cells_on_model_levels,
        tridiagonal_alpha_coeff_at_cells_on_half_levels,
    ) = _compute_solver_coefficients_matrix(
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        inv_ddqz_z_full=inv_ddqz_z_full,
        exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
        dtime=dtime,
    )
    tridiagonal_alpha_coeff_at_cells_on_half_levels = concat_where(
        dims.KDim < n_lev,
        tridiagonal_alpha_coeff_at_cells_on_half_levels,
        broadcast(vpfloat("0.0"), (dims.CellDim,)),
    )
    (rho_explicit_term, exner_explicit_term) = _compute_explicit_part_for_rho_and_exner(
        rho_nnow=current_rho,
        inv_ddqz_z_full=inv_ddqz_z_full,
        z_flxdiv_mass=divergence_of_mass,
        z_contr_w_fl_l=vertical_mass_flux_at_cells_on_half_levels,
        exner_pr=perturbed_exner_at_cells_on_model_levels,
        z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
        z_flxdiv_theta=divergence_of_theta_v,
        theta_v_ic=theta_v_at_cells_on_half_levels,
        ddt_exner_phy=exner_tendency_due_to_slow_physics,
        dtime=dtime,
    )
    if is_iau_active:
        rho_explicit_term, exner_explicit_term = _add_analysis_increments_from_data_assimilation(
            z_rho_expl=rho_explicit_term,
            z_exner_expl=exner_explicit_term,
            rho_incr=rho_iau_increment,
            exner_incr=exner_iau_increment,
            iau_wgt_dyn=iau_wgt_dyn,
        )

    next_w = solve_w(
        last_inner_level=n_lev,
        next_w=next_w,  # n_lev value is set by _set_surface_boundary_condtion_for_computation_of_w
        vwind_impl_wgt=exner_w_implicit_weight_parameter,
        theta_v_ic=theta_v_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        z_alpha=tridiagonal_alpha_coeff_at_cells_on_half_levels,
        z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
        z_w_expl=w_explicit_term,
        z_exner_expl=exner_explicit_term,
        dtime=dtime,
        cpd=PhysicsConstants.cpd,
    )

    if rayleigh_type == RayleighType.KLEMP:
        next_w = concat_where(
            (dims.KDim > 0) & (dims.KDim < end_index_of_damping_layer + 1),
            _apply_rayleigh_damping_mechanism(
                z_raylfac=rayleigh_damping_factor,
                w=next_w,
            ),
            next_w,
        )

    next_rho, next_exner, next_theta_v = _compute_results_for_thermodynamic_variables(
        z_rho_expl=rho_explicit_term,
        vwind_impl_wgt=exner_w_implicit_weight_parameter,
        inv_ddqz_z_full=inv_ddqz_z_full,
        rho_ic=rho_at_cells_on_half_levels,
        w=next_w,
        z_exner_expl=exner_explicit_term,
        exner_ref_mc=reference_exner_at_cells_on_model_levels,
        z_alpha=tridiagonal_alpha_coeff_at_cells_on_half_levels,
        z_beta=tridiagonal_beta_coeff_at_cells_on_model_levels,
        rho_now=current_rho,
        theta_v_now=current_theta_v,
        exner_now=current_exner,
        dtime=dtime,
    )

    if lprep_adv:
        if at_first_substep:
            (
                dynamical_vertical_mass_flux_at_cells_on_half_levels,
                dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            ) = (
                broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
                broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim)),
            )

        (
            dynamical_vertical_mass_flux_at_cells_on_half_levels,
            dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
        ) = concat_where(
            1 <= dims.KDim,
            _update_mass_volume_flux(
                z_contr_w_fl_l=vertical_mass_flux_at_cells_on_half_levels,
                rho_ic=rho_at_cells_on_half_levels,
                vwind_impl_wgt=exner_w_implicit_weight_parameter,
                w=next_w,
                mass_flx_ic=dynamical_vertical_mass_flux_at_cells_on_half_levels,
                vol_flx_ic=dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
                r_nsubsteps=r_nsubsteps,
            ),
            (
                dynamical_vertical_mass_flux_at_cells_on_half_levels,
                dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            ),
        )

    if at_last_substep:
        exner_dynamical_increment = concat_where(
            dims.KDim >= kstart_moist,
            _update_dynamical_exner_time_increment(
                exner=next_exner,
                ddt_exner_phy=exner_tendency_due_to_slow_physics,
                exner_dyn_incr=exner_dynamical_increment,
                ndyn_substeps_var=ndyn_substeps_var,
                dtime=dtime,
            ),
            exner_dynamical_increment,
        )

    return (
        next_w,
        next_rho,
        next_exner,
        next_theta_v,
        dynamical_vertical_mass_flux_at_cells_on_half_levels,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
        exner_dynamical_increment,
    )


@gtx.program
def vertically_implicit_solver_at_corrector_step(
    next_w: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    next_exner: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    dynamical_vertical_mass_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    dynamical_vertical_volumetric_flux_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    exner_dynamical_increment: fa.CellKField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    predictor_vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    corrector_vertical_wind_advective_tendency: fa.CellKField[ta.vpfloat],
    nonhydro_buoy_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    current_w: fa.CellKField[ta.wpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.vpfloat],
    exner_w_implicit_weight_parameter: fa.CellField[ta.wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    exner_tendency_due_to_slow_physics: fa.CellKField[ta.vpfloat],
    rho_iau_increment: fa.CellKField[ta.vpfloat],
    exner_iau_increment: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    rayleigh_damping_factor: fa.KField[ta.wpfloat],
    reference_exner_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    advection_explicit_weight_parameter: ta.wpfloat,
    advection_implicit_weight_parameter: ta.wpfloat,
    lprep_adv: bool,
    r_nsubsteps: ta.wpfloat,
    ndyn_substeps_var: ta.wpfloat,
    iau_wgt_dyn: ta.wpfloat,
    dtime: ta.wpfloat,
    is_iau_active: bool,
    rayleigh_type: gtx.int32,
    at_first_substep: bool,
    at_last_substep: bool,
    end_index_of_damping_layer: gtx.int32,
    kstart_moist: gtx.int32,
    start_cell_index_nudging: gtx.int32,
    end_cell_index_local: gtx.int32,
    vertical_start_index_model_top: gtx.int32,
    vertical_end_index_model_surface: gtx.int32,
) -> None:
    _set_surface_boundary_condition_for_computation_of_w(
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        out=next_w,
        domain={
            dims.CellDim: (start_cell_index_nudging, end_cell_index_local),
            dims.KDim: (vertical_end_index_model_surface - 1, vertical_end_index_model_surface),
        },
    )
    _vertically_implicit_solver_at_corrector_step(
        next_w=next_w,
        dynamical_vertical_mass_flux_at_cells_on_half_levels=dynamical_vertical_mass_flux_at_cells_on_half_levels,
        dynamical_vertical_volumetric_flux_at_cells_on_half_levels=dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
        exner_dynamical_increment=exner_dynamical_increment,
        geofac_div=geofac_div,
        mass_flux_at_edges_on_model_levels=mass_flux_at_edges_on_model_levels,
        theta_v_flux_at_edges_on_model_levels=theta_v_flux_at_edges_on_model_levels,
        predictor_vertical_wind_advective_tendency=predictor_vertical_wind_advective_tendency,
        corrector_vertical_wind_advective_tendency=corrector_vertical_wind_advective_tendency,
        nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy_at_cells_on_half_levels,
        rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
        current_exner=current_exner,
        current_rho=current_rho,
        current_theta_v=current_theta_v,
        current_w=current_w,
        inv_ddqz_z_full=inv_ddqz_z_full,
        exner_w_implicit_weight_parameter=exner_w_implicit_weight_parameter,
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        exner_tendency_due_to_slow_physics=exner_tendency_due_to_slow_physics,
        rho_iau_increment=rho_iau_increment,
        exner_iau_increment=exner_iau_increment,
        ddqz_z_half=ddqz_z_half,
        rayleigh_damping_factor=rayleigh_damping_factor,
        reference_exner_at_cells_on_model_levels=reference_exner_at_cells_on_model_levels,
        advection_explicit_weight_parameter=advection_explicit_weight_parameter,
        advection_implicit_weight_parameter=advection_implicit_weight_parameter,
        lprep_adv=lprep_adv,
        r_nsubsteps=r_nsubsteps,
        ndyn_substeps_var=ndyn_substeps_var,
        iau_wgt_dyn=iau_wgt_dyn,
        dtime=dtime,
        is_iau_active=is_iau_active,
        rayleigh_type=rayleigh_type,
        at_first_substep=at_first_substep,
        at_last_substep=at_last_substep,
        end_index_of_damping_layer=end_index_of_damping_layer,
        kstart_moist=kstart_moist,
        n_lev=vertical_end_index_model_surface - 1,
        out=(
            next_w,
            next_rho,
            next_exner,
            next_theta_v,
            dynamical_vertical_mass_flux_at_cells_on_half_levels,
            dynamical_vertical_volumetric_flux_at_cells_on_half_levels,
            exner_dynamical_increment,
        ),
        domain={
            dims.CellDim: (start_cell_index_nudging, end_cell_index_local),
            dims.KDim: (vertical_start_index_model_top, vertical_end_index_model_surface - 1),
        },
    )
