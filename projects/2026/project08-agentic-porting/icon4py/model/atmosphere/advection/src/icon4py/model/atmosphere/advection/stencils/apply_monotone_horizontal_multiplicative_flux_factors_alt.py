# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import minimum, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C


# TODO(dastrm): this stencil is unused
# TODO(dastrm): this stencil has no test


@gtx.field_operator
def _apply_monotone_horizontal_multiplicative_flux_factors_alt(
    z_anti: fa.EdgeKField[ta.wpfloat],
    z_mflx_low: fa.EdgeKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    r_p: fa.CellKField[ta.wpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    z_signum = where((z_anti > 0.0), 1.0, -1.0)

    r_frac = 0.5 * (
        (1.0 + z_signum) * minimum(r_m(E2C[0]), r_p(E2C[1]))
        + (1.0 - z_signum) * minimum(r_m(E2C[1]), r_p(E2C[0]))
    )

    p_mflx_tracer_h = z_mflx_low + minimum(1.0, r_frac) * z_anti

    return p_mflx_tracer_h


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_monotone_horizontal_multiplicative_flux_factors_alt(
    z_anti: fa.EdgeKField[ta.wpfloat],
    z_mflx_low: fa.EdgeKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    r_p: fa.CellKField[ta.wpfloat],
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_monotone_horizontal_multiplicative_flux_factors_alt(
        z_anti=z_anti,
        z_mflx_low=z_mflx_low,
        r_m=r_m,
        r_p=r_p,
        out=p_mflx_tracer_h,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
