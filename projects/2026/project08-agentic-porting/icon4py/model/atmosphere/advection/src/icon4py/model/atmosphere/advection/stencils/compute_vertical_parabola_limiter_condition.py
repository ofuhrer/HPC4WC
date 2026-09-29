# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import abs, where  # noqa: A004

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim


@gtx.field_operator
def _compute_vertical_parabola_limiter_condition(
    p_face: fa.CellKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[gtx.int32]:
    z_delta = p_face - p_face(KDim + 1)
    z_a6i = 6.0 * (p_cc - 0.5 * (p_face + p_face(KDim + 1)))

    l_limit = where(abs(z_delta) < -1.0 * z_a6i, 1, 0)

    return l_limit


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vertical_parabola_limiter_condition(
    p_face: fa.CellKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    l_limit: fa.CellKField[gtx.int32],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_vertical_parabola_limiter_condition(
        p_face=p_face,
        p_cc=p_cc,
        out=l_limit,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
