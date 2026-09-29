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
from icon4py.model.common.dimension import E2C


# TODO(dastrm): this stencil has no test


@gtx.field_operator
def _apply_positive_definite_horizontal_multiplicative_flux_factor(
    r_m: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    p_mflx_tracer_h_out = where(
        p_mflx_tracer_h >= 0.0,
        p_mflx_tracer_h * r_m(E2C[0]),
        p_mflx_tracer_h * r_m(E2C[1]),
    )
    return p_mflx_tracer_h_out


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_positive_definite_horizontal_multiplicative_flux_factor(
    r_m: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_positive_definite_horizontal_multiplicative_flux_factor(
        r_m,
        p_mflx_tracer_h,
        out=p_mflx_tracer_h,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
