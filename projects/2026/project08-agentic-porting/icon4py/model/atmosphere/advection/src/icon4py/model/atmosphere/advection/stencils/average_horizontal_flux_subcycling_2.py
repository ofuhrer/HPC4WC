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
def _average_horizontal_flux_subcycling_2(
    z_tracer_mflx_1_dsl: fa.EdgeKField[ta.wpfloat],
    z_tracer_mflx_2_dsl: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    p_out_e = (z_tracer_mflx_1_dsl + z_tracer_mflx_2_dsl) / 2.0
    return p_out_e


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def average_horizontal_flux_subcycling_2(
    z_tracer_mflx_1_dsl: fa.EdgeKField[ta.wpfloat],
    z_tracer_mflx_2_dsl: fa.EdgeKField[ta.wpfloat],
    p_out_e: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _average_horizontal_flux_subcycling_2(
        z_tracer_mflx_1_dsl=z_tracer_mflx_1_dsl,
        z_tracer_mflx_2_dsl=z_tracer_mflx_2_dsl,
        out=p_out_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
