# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.atmosphere.dycore.stencils.init_cell_kdim_field_with_zero_wp import (
    _init_cell_kdim_field_with_zero_wp,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _set_lower_boundary_condition_for_w_and_contravariant_correction(
    w_concorr_c: fa.CellKField[vpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_47."""
    w_concorr_c_wp = astype(w_concorr_c, wpfloat)

    w_nnew_wp = w_concorr_c_wp
    z_contr_w_fl_l_wp = _init_cell_kdim_field_with_zero_wp()
    return w_nnew_wp, z_contr_w_fl_l_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def set_lower_boundary_condition_for_w_and_contravariant_correction(
    w_nnew: fa.CellKField[wpfloat],
    z_contr_w_fl_l: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _set_lower_boundary_condition_for_w_and_contravariant_correction(
        w_concorr_c=w_concorr_c,
        out=(w_nnew, z_contr_w_fl_l),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
