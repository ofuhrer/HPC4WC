# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_horizontal_tracer_flux_from_cubic_coefficients(
    p_out_e_hybrid_2: fa.EdgeKField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    z_dreg_area: fa.EdgeKField[ta.vpfloat],
) -> fa.EdgeKField[ta.wpfloat]:
    p_out_e_hybrid_2 = p_mass_flx_e * p_out_e_hybrid_2 / astype(z_dreg_area, wpfloat)

    return p_out_e_hybrid_2


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_tracer_flux_from_cubic_coefficients(
    p_out_e_hybrid_2: fa.EdgeKField[ta.wpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    z_dreg_area: fa.EdgeKField[ta.vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_horizontal_tracer_flux_from_cubic_coefficients(
        p_out_e_hybrid_2=p_out_e_hybrid_2,
        p_mass_flx_e=p_mass_flx_e,
        z_dreg_area=z_dreg_area,
        out=p_out_e_hybrid_2,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
