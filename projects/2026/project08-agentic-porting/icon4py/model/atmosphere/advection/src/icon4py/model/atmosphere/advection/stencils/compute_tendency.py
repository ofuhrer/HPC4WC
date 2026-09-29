# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


# TODO(dastrm): move to common and rename arguments


@gtx.field_operator
def _compute_tendency(
    p_tracer_now: fa.CellKField[ta.wpfloat],
    p_tracer_new: fa.CellKField[ta.wpfloat],
    p_dtime: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    opt_ddt_tracer_adv = (p_tracer_new - p_tracer_now) / p_dtime
    return opt_ddt_tracer_adv


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tendency(
    p_tracer_now: fa.CellKField[ta.wpfloat],
    p_tracer_new: fa.CellKField[ta.wpfloat],
    opt_ddt_tracer_adv: fa.CellKField[ta.wpfloat],
    p_dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_tendency(
        p_tracer_now=p_tracer_now,
        p_tracer_new=p_tracer_new,
        p_dtime=p_dtime,
        out=opt_ddt_tracer_adv,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
