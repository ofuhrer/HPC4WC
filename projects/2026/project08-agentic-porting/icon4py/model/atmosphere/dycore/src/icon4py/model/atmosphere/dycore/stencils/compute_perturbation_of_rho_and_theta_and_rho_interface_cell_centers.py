# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.atmosphere.dycore.stencils.compute_perturbation_of_rho_and_theta import (
    _compute_perturbation_of_rho_and_theta,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers(
    wgtfac_c: fa.CellKField[vpfloat],
    rho: fa.CellKField[wpfloat],
    rho_ref_mc: fa.CellKField[vpfloat],
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    """Formerly known as _mo_solve_nonhydro_stencil_08."""
    wgtfac_c_wp = astype(wgtfac_c, wpfloat)

    rho_ic = wgtfac_c_wp * rho + (wpfloat("1.0") - wgtfac_c_wp) * rho(KDim - 1)
    z_rth_pr_1, z_rth_pr_2 = _compute_perturbation_of_rho_and_theta(
        rho=rho, rho_ref_mc=rho_ref_mc, theta_v=theta_v, theta_ref_mc=theta_ref_mc
    )
    return rho_ic, z_rth_pr_1, z_rth_pr_2


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers(
    wgtfac_c: fa.CellKField[vpfloat],
    rho: fa.CellKField[wpfloat],
    rho_ref_mc: fa.CellKField[vpfloat],
    theta_v: fa.CellKField[wpfloat],
    theta_ref_mc: fa.CellKField[vpfloat],
    rho_ic: fa.CellKField[wpfloat],
    z_rth_pr_1: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_perturbation_of_rho_and_theta_and_rho_interface_cell_centers(
        wgtfac_c=wgtfac_c,
        rho=rho,
        rho_ref_mc=rho_ref_mc,
        theta_v=theta_v,
        theta_ref_mc=theta_ref_mc,
        out=(rho_ic, z_rth_pr_1, z_rth_pr_2),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
