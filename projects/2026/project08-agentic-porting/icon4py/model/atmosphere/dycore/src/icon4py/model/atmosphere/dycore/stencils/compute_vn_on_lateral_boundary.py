# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_vn_on_lateral_boundary(
    grf_tend_vn: fa.EdgeKField[wpfloat],
    vn_now: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
) -> fa.EdgeKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_29."""
    vn_new_wp = vn_now + dtime * grf_tend_vn
    return vn_new_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_vn_on_lateral_boundary(
    grf_tend_vn: fa.EdgeKField[wpfloat],
    vn_now: fa.EdgeKField[wpfloat],
    vn_new: fa.EdgeKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_vn_on_lateral_boundary(
        grf_tend_vn=grf_tend_vn,
        vn_now=vn_now,
        dtime=dtime,
        out=vn_new,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
