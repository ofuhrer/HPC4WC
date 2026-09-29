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
def _interpolate_edge_field_to_half_levels_vp(
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    interpolant: fa.EdgeKField[ta.vpfloat],
) -> fa.EdgeKField[ta.vpfloat]:
    """
    Interpolate a EdgeDim variable of floating precision from full levels to half levels.
    The return variable also has floating precision.
        var_half_k-1/2 = wgt_fac_c_k-1 var_half_k-1 + wgt_fac_c_k var_half_k

    Args:
        wgtfac_e: weight factor
        interpolant: EdgeDim variables at full levels
    Returns:
        EdgeDim variables at half levels
    """
    interpolation_to_half_levels_vp = wgtfac_e * interpolant + (
        vpfloat("1.0") - wgtfac_e
    ) * interpolant(KDim - 1)
    return interpolation_to_half_levels_vp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_edge_field_to_half_levels_vp(
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    interpolant: fa.EdgeKField[ta.vpfloat],
    interpolation_to_half_levels_vp: fa.EdgeKField[ta.vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_edge_field_to_half_levels_vp(
        wgtfac_e=wgtfac_e,
        interpolant=interpolant,
        out=interpolation_to_half_levels_vp,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
