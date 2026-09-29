# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _compute_approx_of_2nd_vertical_derivative_of_exner(
    z_theta_v_pr_ic: fa.CellKField[vpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_12."""
    z_dexner_dz_c_2_vp = -vpfloat("0.5") * (
        (z_theta_v_pr_ic - z_theta_v_pr_ic(KDim + 1)) * d2dexdz2_fac1_mc
        + z_rth_pr_2 * d2dexdz2_fac2_mc
    )
    return z_dexner_dz_c_2_vp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_approx_of_2nd_vertical_derivative_of_exner(
    z_theta_v_pr_ic: fa.CellKField[vpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[vpfloat],
    z_rth_pr_2: fa.CellKField[vpfloat],
    z_dexner_dz_c_2: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_approx_of_2nd_vertical_derivative_of_exner(
        z_theta_v_pr_ic=z_theta_v_pr_ic,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        z_rth_pr_2=z_rth_pr_2,
        out=z_dexner_dz_c_2,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
