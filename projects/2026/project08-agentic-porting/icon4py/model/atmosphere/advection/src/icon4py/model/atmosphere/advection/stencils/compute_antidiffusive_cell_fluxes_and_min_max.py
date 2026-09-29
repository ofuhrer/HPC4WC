# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype, maximum, minimum, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E, C2EDim
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _compute_antidiffusive_cell_fluxes_and_min_max(
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    p_rhodz_now: fa.CellKField[ta.wpfloat],
    p_rhodz_new: fa.CellKField[ta.wpfloat],
    z_mflx_low: fa.EdgeKField[ta.wpfloat],
    z_anti: fa.EdgeKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    p_dtime: ta.wpfloat,
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
]:
    z_mflx_anti_1 = astype(p_dtime * geofac_div[C2EDim(0)] / p_rhodz_new * z_anti(C2E[0]), vpfloat)
    z_mflx_anti_2 = astype(p_dtime * geofac_div[C2EDim(1)] / p_rhodz_new * z_anti(C2E[1]), vpfloat)
    z_mflx_anti_3 = astype(p_dtime * geofac_div[C2EDim(2)] / p_rhodz_new * z_anti(C2E[2]), vpfloat)

    z_mflx_anti_in = -1.0 * (
        minimum(0.0, z_mflx_anti_1) + minimum(0.0, z_mflx_anti_2) + minimum(0.0, z_mflx_anti_3)
    )

    z_mflx_anti_out = (
        maximum(0.0, z_mflx_anti_1) + maximum(0.0, z_mflx_anti_2) + maximum(0.0, z_mflx_anti_3)
    )

    z_fluxdiv_c = neighbor_sum(z_mflx_low(C2E) * geofac_div, axis=dims.C2EDim)

    z_tracer_new_low = (p_cc * p_rhodz_now - p_dtime * z_fluxdiv_c) / p_rhodz_new
    z_tracer_max = astype(maximum(p_cc, z_tracer_new_low), vpfloat)
    z_tracer_min = astype(minimum(p_cc, z_tracer_new_low), vpfloat)

    return (
        z_mflx_anti_in,
        z_mflx_anti_out,
        z_tracer_new_low,
        z_tracer_max,
        z_tracer_min,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_antidiffusive_cell_fluxes_and_min_max(
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    p_rhodz_now: fa.CellKField[ta.wpfloat],
    p_rhodz_new: fa.CellKField[ta.wpfloat],
    z_mflx_low: fa.EdgeKField[ta.wpfloat],
    z_anti: fa.EdgeKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    z_mflx_anti_in: fa.CellKField[ta.vpfloat],
    z_mflx_anti_out: fa.CellKField[ta.vpfloat],
    z_tracer_new_low: fa.CellKField[ta.wpfloat],
    z_tracer_max: fa.CellKField[ta.vpfloat],
    z_tracer_min: fa.CellKField[ta.vpfloat],
    p_dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_antidiffusive_cell_fluxes_and_min_max(
        geofac_div=geofac_div,
        p_rhodz_now=p_rhodz_now,
        p_rhodz_new=p_rhodz_new,
        z_mflx_low=z_mflx_low,
        z_anti=z_anti,
        p_cc=p_cc,
        p_dtime=p_dtime,
        out=(
            z_mflx_anti_in,
            z_mflx_anti_out,
            z_tracer_new_low,
            z_tracer_max,
            z_tracer_min,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
