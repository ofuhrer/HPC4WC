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


@gtx.field_operator
def _compute_ppm_quartic_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    zgeo1 = p_cellhgt_mc_now(KDim - 1) / (p_cellhgt_mc_now(KDim - 1) + p_cellhgt_mc_now)
    zgeo2 = 1.0 / (
        p_cellhgt_mc_now(KDim - 2)
        + p_cellhgt_mc_now(KDim - 1)
        + p_cellhgt_mc_now
        + p_cellhgt_mc_now(KDim + 1)
    )
    zgeo3 = (p_cellhgt_mc_now(KDim - 2) + p_cellhgt_mc_now(KDim - 1)) / (
        2.0 * p_cellhgt_mc_now(KDim - 1) + p_cellhgt_mc_now
    )
    zgeo4 = (p_cellhgt_mc_now(KDim + 1) + p_cellhgt_mc_now) / (
        2.0 * p_cellhgt_mc_now + p_cellhgt_mc_now(KDim - 1)
    )

    p_face = (
        p_cc(KDim - 1)
        + zgeo1 * (p_cc - p_cc(KDim - 1))
        + zgeo2
        * (
            (2.0 * p_cellhgt_mc_now * zgeo1) * (zgeo3 - zgeo4) * (p_cc - p_cc(KDim - 1))
            - zgeo3 * p_cellhgt_mc_now(KDim - 1) * z_slope
            + zgeo4 * p_cellhgt_mc_now * z_slope(KDim - 1)
        )
    )

    return p_face


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_quartic_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
    p_face: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_quartic_face_values(
        p_cc=p_cc,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        z_slope=z_slope,
        out=p_face,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
