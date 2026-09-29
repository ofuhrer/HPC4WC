# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2CDim
from icon4py.model.common.math.stencils.cell_horizontal_gradients_by_green_gauss_method import (
    cell_horizontal_gradients_by_green_gauss_method,
)
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_backward_trajectory_from_edge_center(
    p_vn: fa.EdgeKField[wpfloat],
    p_vt: fa.EdgeKField[vpfloat],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    dual_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    dual_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    p_dthalf: wpfloat,
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    lvn_pos = where(p_vn >= wpfloat("0.0"), True, False)

    z_ntdistv_bary_1 = -(
        p_vn * p_dthalf + where(lvn_pos, pos_on_tplane_e_1[E2CDim(0)], pos_on_tplane_e_1[E2CDim(1)])
    )

    z_ntdistv_bary_2 = -(
        astype(p_vt, wpfloat) * p_dthalf
        + where(lvn_pos, pos_on_tplane_e_2[E2CDim(0)], pos_on_tplane_e_2[E2CDim(1)])
    )

    p_distv_bary_1 = where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_1[E2CDim(0)]
        + z_ntdistv_bary_2 * dual_normal_cell_1[E2CDim(0)],
        z_ntdistv_bary_1 * primal_normal_cell_1[E2CDim(1)]
        + z_ntdistv_bary_2 * dual_normal_cell_1[E2CDim(1)],
    )

    p_distv_bary_2 = where(
        lvn_pos,
        z_ntdistv_bary_1 * primal_normal_cell_2[E2CDim(0)]
        + z_ntdistv_bary_2 * dual_normal_cell_2[E2CDim(0)],
        z_ntdistv_bary_1 * primal_normal_cell_2[E2CDim(1)]
        + z_ntdistv_bary_2 * dual_normal_cell_2[E2CDim(1)],
    )

    return p_distv_bary_1, p_distv_bary_2


@gtx.field_operator
def _compute_upwind_values_of_rho_and_theta_v_at_edges(
    p_vn: fa.EdgeKField[wpfloat],
    rho_ref_me: fa.EdgeKField[vpfloat],
    theta_ref_me: fa.EdgeKField[vpfloat],
    p_distv_bary_1: fa.EdgeKField[wpfloat],
    p_distv_bary_2: fa.EdgeKField[wpfloat],
    z_grad_rth_1: fa.CellKField[vpfloat],
    z_grad_rth_2: fa.CellKField[vpfloat],
    z_grad_rth_3: fa.CellKField[vpfloat],
    z_grad_rth_4: fa.CellKField[vpfloat],
    z_rth_pr_1: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    (
        theta_ref_me_wp,
        rho_ref_me_wp,
        z_grad_rth_1_wp,
        z_grad_rth_2_wp,
        z_grad_rth_3_wp,
        z_grad_rth_4_wp,
        z_rth_pr_1_wp,
        z_rth_pr_2_wp,
    ) = astype(
        (
            theta_ref_me,
            rho_ref_me,
            z_grad_rth_1,
            z_grad_rth_2,
            z_grad_rth_3,
            z_grad_rth_4,
            z_rth_pr_1,
            z_rth_pr_2,
        ),
        wpfloat,
    )

    z_rho_e_wp = where(
        p_vn >= wpfloat("0.0"),
        rho_ref_me_wp
        + z_rth_pr_1_wp(E2C[0])
        + p_distv_bary_1 * z_grad_rth_1_wp(E2C[0])
        + p_distv_bary_2 * z_grad_rth_2_wp(E2C[0]),
        rho_ref_me_wp
        + z_rth_pr_1_wp(E2C[1])
        + p_distv_bary_1 * z_grad_rth_1_wp(E2C[1])
        + p_distv_bary_2 * z_grad_rth_2_wp(E2C[1]),
    )

    z_theta_v_e_wp = where(
        p_vn >= wpfloat("0.0"),
        theta_ref_me_wp
        + z_rth_pr_2_wp(E2C[0])
        + p_distv_bary_1 * z_grad_rth_3_wp(E2C[0])
        + p_distv_bary_2 * z_grad_rth_4_wp(E2C[0]),
        theta_ref_me_wp
        + z_rth_pr_2_wp(E2C[1])
        + p_distv_bary_1 * z_grad_rth_3_wp(E2C[1])
        + p_distv_bary_2 * z_grad_rth_4_wp(E2C[1]),
    )

    return z_rho_e_wp, z_theta_v_e_wp


@gtx.field_operator
def _compute_horizontal_advection_of_rho_and_theta(
    p_vn: fa.EdgeKField[wpfloat],
    p_vt: fa.EdgeKField[vpfloat],
    pos_on_tplane_e_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    pos_on_tplane_e_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    dual_normal_cell_1: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    primal_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    dual_normal_cell_2: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], wpfloat],
    p_dthalf: wpfloat,
    rho_ref_me: fa.EdgeKField[vpfloat],
    theta_ref_me: fa.EdgeKField[vpfloat],
    perturbed_rho_at_cells_on_model_levels: fa.CellKField[vpfloat],
    perturbed_theta_v_at_cells_on_model_levels: fa.CellKField[vpfloat],
    geofac_grg_x: gtx.Field[[dims.CellDim, dims.C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[[dims.CellDim, dims.C2E2CODim], wpfloat],
) -> tuple[fa.EdgeKField[wpfloat], fa.EdgeKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_16_fused_btraj_traj_o1."""
    (
        ddx_perturbed_theta_v,
        ddy_perturbed_theta_v,
    ) = cell_horizontal_gradients_by_green_gauss_method(
        scalar_field=perturbed_theta_v_at_cells_on_model_levels,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
    )

    (ddx_perturbed_rho, ddy_perturbed_rho) = cell_horizontal_gradients_by_green_gauss_method(
        scalar_field=perturbed_rho_at_cells_on_model_levels,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
    )

    (p_distv_bary_1, p_distv_bary_2) = _compute_backward_trajectory_from_edge_center(
        p_vn=p_vn,
        p_vt=p_vt,
        pos_on_tplane_e_1=pos_on_tplane_e_1,
        pos_on_tplane_e_2=pos_on_tplane_e_2,
        primal_normal_cell_1=primal_normal_cell_1,
        dual_normal_cell_1=dual_normal_cell_1,
        primal_normal_cell_2=primal_normal_cell_2,
        dual_normal_cell_2=dual_normal_cell_2,
        p_dthalf=p_dthalf,
    )

    (
        rho_at_edges_on_model_levels,
        theta_at_edges_on_model_levels,
    ) = _compute_upwind_values_of_rho_and_theta_v_at_edges(
        p_vn=p_vn,
        rho_ref_me=rho_ref_me,
        theta_ref_me=theta_ref_me,
        p_distv_bary_1=p_distv_bary_1,
        p_distv_bary_2=p_distv_bary_2,
        z_grad_rth_1=ddx_perturbed_rho,
        z_grad_rth_2=ddy_perturbed_rho,
        z_grad_rth_3=ddx_perturbed_theta_v,
        z_grad_rth_4=ddy_perturbed_theta_v,
        z_rth_pr_1=perturbed_rho_at_cells_on_model_levels,
        z_rth_pr_2=perturbed_theta_v_at_cells_on_model_levels,
    )

    return rho_at_edges_on_model_levels, theta_at_edges_on_model_levels
