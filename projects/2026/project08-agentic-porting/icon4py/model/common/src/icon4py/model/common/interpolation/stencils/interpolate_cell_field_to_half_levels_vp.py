# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _interpolate_cell_field_to_half_levels_vp(
    wgtfac_c: fa.CellKField[ta.vpfloat],
    interpolant: fa.CellKField[ta.vpfloat],
) -> fa.CellKField[ta.vpfloat]:
    """
    Interpolate a CellDim variable of floating precision from full levels to half levels.
    The return variable also has floating precision.
        var_half_k-1/2 = wgt_fac_c_k-1 var_half_k-1 + wgt_fac_c_k var_half_k

    Args:
        wgtfac_c: weight factor
        interpolant: CellDim variables at full levels
    Returns:
        CellDim variables at half levels
    """
    interpolation_to_half_levels_vp = wgtfac_c * interpolant + (
        vpfloat("1.0") - wgtfac_c
    ) * interpolant(KDim - 1)
    return interpolation_to_half_levels_vp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_cell_field_to_half_levels_vp(
    wgtfac_c: fa.CellKField[ta.vpfloat],
    interpolant: fa.CellKField[ta.vpfloat],
    interpolation_to_half_levels_vp: fa.CellKField[ta.vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_cell_field_to_half_levels_vp(
        wgtfac_c=wgtfac_c,
        interpolant=interpolant,
        out=interpolation_to_half_levels_vp,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
