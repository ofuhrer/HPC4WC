# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def _compute_ffsl_backtrajectory_counterclockwise_indicator(
    p_vn: fa.EdgeKField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    lcounterclock: bool,
) -> fa.EdgeKField[bool]:
    return where(p_vn * tangent_orientation >= 0.0, lcounterclock, False)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ffsl_backtrajectory_counterclockwise_indicator(
    p_vn: fa.EdgeKField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    lvn_sys_pos: fa.EdgeKField[bool],
    lcounterclock: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ffsl_backtrajectory_counterclockwise_indicator(
        p_vn=p_vn,
        tangent_orientation=tangent_orientation,
        lcounterclock=lcounterclock,
        out=lvn_sys_pos,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
