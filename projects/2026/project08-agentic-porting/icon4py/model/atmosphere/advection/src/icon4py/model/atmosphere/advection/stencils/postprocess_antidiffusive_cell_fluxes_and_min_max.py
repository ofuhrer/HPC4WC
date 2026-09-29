# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype, maximum, minimum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _postprocess_antidiffusive_cell_fluxes_and_min_max(
    refin_ctrl: fa.CellField[gtx.int32],
    p_cc: fa.CellKField[ta.wpfloat],
    z_tracer_new_low: fa.CellKField[ta.wpfloat],
    z_tracer_max: fa.CellKField[ta.vpfloat],
    z_tracer_min: fa.CellKField[ta.vpfloat],
    lo_bound: gtx.int32,
    hi_bound: gtx.int32,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
]:
    condition = (refin_ctrl == lo_bound) | (refin_ctrl == hi_bound)
    z_tracer_new_out = where(
        condition,
        minimum(1.1 * p_cc, maximum(0.9 * p_cc, z_tracer_new_low)),
        z_tracer_new_low,
    )

    z_tracer_max_out = where(
        condition, astype(maximum(p_cc, z_tracer_new_out), vpfloat), z_tracer_max
    )
    z_tracer_min_out = where(
        condition, astype(minimum(p_cc, z_tracer_new_out), vpfloat), z_tracer_min
    )

    return (z_tracer_new_out, z_tracer_max_out, z_tracer_min_out)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def postprocess_antidiffusive_cell_fluxes_and_min_max(
    refin_ctrl: fa.CellField[gtx.int32],
    p_cc: fa.CellKField[ta.wpfloat],
    z_tracer_new_low: fa.CellKField[ta.wpfloat],
    z_tracer_max: fa.CellKField[ta.vpfloat],
    z_tracer_min: fa.CellKField[ta.vpfloat],
    z_tracer_new_low_out: fa.CellKField[ta.wpfloat],
    z_tracer_max_out: fa.CellKField[ta.vpfloat],
    z_tracer_min_out: fa.CellKField[ta.vpfloat],
    lo_bound: gtx.int32,
    hi_bound: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _postprocess_antidiffusive_cell_fluxes_and_min_max(
        refin_ctrl=refin_ctrl,
        p_cc=p_cc,
        z_tracer_new_low=z_tracer_new_low,
        z_tracer_max=z_tracer_max,
        z_tracer_min=z_tracer_min,
        lo_bound=lo_bound,
        hi_bound=hi_bound,
        out=(z_tracer_new_low_out, z_tracer_max_out, z_tracer_min_out),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
