# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import minimum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim


@gtx.field_operator
def _limit_vertical_parabola_semi_monotonically(
    l_limit: fa.CellKField[gtx.int32],
    p_face: fa.CellKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    q_face_up, q_face_low = where(
        l_limit != 0,
        where(
            (p_cc < minimum(p_face, p_face(KDim + 1))),
            (p_cc, p_cc),
            where(
                p_face > p_face(KDim + 1),
                (3.0 * p_cc - 2.0 * p_face(KDim + 1), p_face(KDim + 1)),
                (p_face, 3.0 * p_cc - 2.0 * p_face),
            ),
        ),
        (p_face, p_face(KDim + 1)),
    )

    return q_face_up, q_face_low


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def limit_vertical_parabola_semi_monotonically(
    l_limit: fa.CellKField[gtx.int32],
    p_face: fa.CellKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    p_face_up: fa.CellKField[ta.wpfloat],
    p_face_low: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _limit_vertical_parabola_semi_monotonically(
        l_limit=l_limit,
        p_face=p_face,
        p_cc=p_cc,
        out=(p_face_up, p_face_low),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
