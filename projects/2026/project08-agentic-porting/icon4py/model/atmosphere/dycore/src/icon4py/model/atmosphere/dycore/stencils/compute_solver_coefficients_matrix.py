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
def _compute_solver_coefficients_matrix(
    exner_nnow: fa.CellKField[wpfloat],
    rho_nnow: fa.CellKField[wpfloat],
    theta_v_nnow: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    dtime: wpfloat,
    rd: wpfloat,
    cvd: wpfloat,
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_44."""
    inv_ddqz_z_full_wp = astype(inv_ddqz_z_full, wpfloat)

    z_beta_wp = dtime * rd * exner_nnow / (cvd * rho_nnow * theta_v_nnow) * inv_ddqz_z_full_wp
    z_alpha_wp = vwind_impl_wgt * theta_v_ic * rho_ic
    return astype((z_beta_wp, z_alpha_wp), vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_solver_coefficients_matrix(
    z_beta: fa.CellKField[vpfloat],
    exner_nnow: fa.CellKField[wpfloat],
    rho_nnow: fa.CellKField[wpfloat],
    theta_v_nnow: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    z_alpha: fa.CellKField[vpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    dtime: wpfloat,
    rd: wpfloat,
    cvd: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_solver_coefficients_matrix(
        exner_nnow=exner_nnow,
        rho_nnow=rho_nnow,
        theta_v_nnow=theta_v_nnow,
        inv_ddqz_z_full=inv_ddqz_z_full,
        vwind_impl_wgt=vwind_impl_wgt,
        theta_v_ic=theta_v_ic,
        rho_ic=rho_ic,
        dtime=dtime,
        rd=rd,
        cvd=cvd,
        out=(z_beta, z_alpha),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
