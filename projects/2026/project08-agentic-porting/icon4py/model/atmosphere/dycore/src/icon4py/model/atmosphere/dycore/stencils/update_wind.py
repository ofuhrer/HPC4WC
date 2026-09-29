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
def _update_wind(
    w_now: fa.CellKField[wpfloat],
    grf_tend_w: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> fa.CellKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_62."""
    w_new_wp = w_now + dtime * grf_tend_w
    return w_new_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_wind(
    w_now: fa.CellKField[wpfloat],
    grf_tend_w: fa.CellKField[wpfloat],
    w_new: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_wind(
        w_now=w_now,
        grf_tend_w=grf_tend_w,
        dtime=dtime,
        out=w_new,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
