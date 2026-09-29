# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_perturbation_of_rho_and_theta(
    rho: fa.CellKField[wpfloat],
    rho_ref_mc: fa.CellKField[vpfloat],
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_07 or _mo_solve_nonhydro_stencil_13."""
    rho_ref_mc_wp, theta_ref_mc_wp = astype((rho_ref_mc, theta_ref_mc), wpfloat)

    z_rth_pr_1_wp = rho - rho_ref_mc_wp
    z_rth_pr_2_wp = theta_v - theta_ref_mc_wp
    return astype((z_rth_pr_1_wp, z_rth_pr_2_wp), vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_perturbation_of_rho_and_theta(
    rho: fa.CellKField[wpfloat],
    rho_ref_mc: fa.CellKField[vpfloat],
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    z_rth_pr_1: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_perturbation_of_rho_and_theta(
        rho=rho,
        rho_ref_mc=rho_ref_mc,
        theta_v=theta_v,
        theta_ref_mc=theta_ref_mc,
        out=(z_rth_pr_1, z_rth_pr_2),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
