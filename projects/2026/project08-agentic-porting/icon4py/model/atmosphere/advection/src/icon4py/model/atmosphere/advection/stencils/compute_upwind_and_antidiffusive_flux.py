# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import abs  # noqa: A004

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C


# TODO(dastrm): this stencil has no test


@gtx.field_operator
def _compute_upwind_and_antidiffusive_flux(
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
) -> tuple[fa.EdgeKField[ta.wpfloat], fa.EdgeKField[ta.wpfloat]]:
    z_mflx_low = 0.5 * (
        p_mass_flx_e * (p_cc(E2C[0]) + p_cc(E2C[1]))
        - abs(p_mass_flx_e) * (p_cc(E2C[1]) - p_cc(E2C[0]))
    )

    z_anti = p_mflx_tracer_h - z_mflx_low

    return (z_mflx_low, z_anti)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_upwind_and_antidiffusive_flux(
    p_mflx_tracer_h: fa.EdgeKField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    p_cc: fa.CellKField[ta.wpfloat],
    z_mflx_low: fa.EdgeKField[ta.wpfloat],
    z_anti: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_upwind_and_antidiffusive_flux(
        p_mflx_tracer_h=p_mflx_tracer_h,
        p_mass_flx_e=p_mass_flx_e,
        p_cc=p_cc,
        out=(z_mflx_low, z_anti),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
