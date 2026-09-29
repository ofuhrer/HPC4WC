# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def _compute_ppm4gpu_parabola_coefficients(
    z_face_up: fa.CellKField[ta.wpfloat],
    z_face_low: fa.CellKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    z_delta_q = 0.5 * (z_face_up - z_face_low)
    z_a1 = p_cc - 0.5 * (z_face_up + z_face_low)

    return z_delta_q, z_a1


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm4gpu_parabola_coefficients(
    z_face_up: fa.CellKField[ta.wpfloat],
    z_face_low: fa.CellKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    z_delta_q: fa.CellKField[ta.wpfloat],
    z_a1: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm4gpu_parabola_coefficients(
        z_face_up=z_face_up,
        z_face_low=z_face_low,
        p_cc=p_cc,
        out=(z_delta_q, z_a1),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
