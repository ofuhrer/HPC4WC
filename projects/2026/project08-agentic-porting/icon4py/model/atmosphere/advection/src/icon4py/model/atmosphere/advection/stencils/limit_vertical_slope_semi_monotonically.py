# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import abs, minimum, where  # noqa: A004

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim


@gtx.field_operator
def _limit_vertical_slope_semi_monotonically(
    p_cc: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    elev: gtx.int32,
) -> fa.CellKField[ta.wpfloat]:
    p_cc_min_last = minimum(p_cc(KDim - 1), p_cc)
    p_cc_min = where(k == elev, p_cc_min_last, minimum(p_cc_min_last, p_cc(KDim + 1)))
    slope_l = minimum(abs(z_slope), 2.0 * (p_cc - p_cc_min))
    slope = where(z_slope >= 0.0, slope_l, -slope_l)
    return slope


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def limit_vertical_slope_semi_monotonically(
    p_cc: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    elev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _limit_vertical_slope_semi_monotonically(
        p_cc=p_cc,
        z_slope=z_slope,
        k=k,
        elev=elev,
        out=z_slope,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
