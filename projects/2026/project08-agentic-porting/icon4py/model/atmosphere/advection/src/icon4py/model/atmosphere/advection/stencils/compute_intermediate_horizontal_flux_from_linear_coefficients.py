# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C
from icon4py.model.common.type_alias import wpfloat


# TODO(dastrm): this stencil has no test


@gtx.field_operator
def _compute_intermediate_horizontal_flux_from_linear_coefficients(
    z_lsq_coeff_1_dsl: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_2_dsl: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_3_dsl: fa.CellKField[ta.wpfloat],
    distv_bary_1: fa.EdgeKField[ta.vpfloat],
    distv_bary_2: fa.EdgeKField[ta.vpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
) -> fa.EdgeKField[ta.wpfloat]:
    z_tracer_mflx_dsl = (
        where(
            cell_rel_idx_dsl == 1,
            z_lsq_coeff_1_dsl(E2C[1]),
            z_lsq_coeff_1_dsl(E2C[0]),
        )
        + astype(distv_bary_1, wpfloat)
        * where(
            cell_rel_idx_dsl == 1,
            z_lsq_coeff_2_dsl(E2C[1]),
            z_lsq_coeff_2_dsl(E2C[0]),
        )
        + astype(distv_bary_2, wpfloat)
        * where(
            cell_rel_idx_dsl == 1,
            z_lsq_coeff_3_dsl(E2C[1]),
            z_lsq_coeff_3_dsl(E2C[0]),
        )
    ) * p_mass_flx_e

    return z_tracer_mflx_dsl


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_intermediate_horizontal_flux_from_linear_coefficients(
    z_lsq_coeff_1_dsl: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_2_dsl: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_3_dsl: fa.CellKField[ta.wpfloat],
    distv_bary_1: fa.EdgeKField[ta.vpfloat],
    distv_bary_2: fa.EdgeKField[ta.vpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
    z_tracer_mflx_dsl: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_intermediate_horizontal_flux_from_linear_coefficients(
        z_lsq_coeff_1_dsl=z_lsq_coeff_1_dsl,
        z_lsq_coeff_2_dsl=z_lsq_coeff_2_dsl,
        z_lsq_coeff_3_dsl=z_lsq_coeff_3_dsl,
        distv_bary_1=distv_bary_1,
        distv_bary_2=distv_bary_2,
        p_mass_flx_e=p_mass_flx_e,
        cell_rel_idx_dsl=cell_rel_idx_dsl,
        out=z_tracer_mflx_dsl,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
