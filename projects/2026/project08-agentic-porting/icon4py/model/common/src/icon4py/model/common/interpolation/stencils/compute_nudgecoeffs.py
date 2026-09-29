# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, exp, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_nudgecoeffs(
    refin_ctrl: fa.EdgeField[gtx.int32],
    grf_nudge_start_e: gtx.int32,
    max_nudging_coefficient: wpfloat,
    nudge_efold_width: wpfloat,
    nudge_zone_width: gtx.int32,
) -> fa.EdgeField[wpfloat]:
    return where(
        ((refin_ctrl > 0) & (refin_ctrl <= (2 * nudge_zone_width + (grf_nudge_start_e - 3)))),
        max_nudging_coefficient
        * exp((-(astype(refin_ctrl - grf_nudge_start_e, wpfloat))) / (2.0 * nudge_efold_width)),
        0.0,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_nudgecoeffs(
    refin_ctrl: fa.EdgeField[gtx.int32],
    nudging_coefficients_for_edges: fa.EdgeField[wpfloat],
    grf_nudge_start_e: gtx.int32,
    max_nudging_coefficient: wpfloat,
    nudge_efold_width: wpfloat,
    nudge_zone_width: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    """Compute nudging coefficient for edges based the grid refinement level of an edge."""
    _compute_nudgecoeffs(
        refin_ctrl=refin_ctrl,
        grf_nudge_start_e=grf_nudge_start_e,
        max_nudging_coefficient=max_nudging_coefficient,
        nudge_efold_width=nudge_efold_width,
        nudge_zone_width=nudge_zone_width,
        out=nudging_coefficients_for_edges,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )
