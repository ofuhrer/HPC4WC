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
def _compute_intermediate_horizontal_tracer_flux_from_cubic_coefficients(
    z_lsq_coeff_1: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_2: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_3: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_4: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_5: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_6: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_7: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_8: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_9: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_10: fa.CellKField[ta.wpfloat],
    z_quad_vector_sum0_1: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_2: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_3: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_4: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_5: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_6: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_7: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_8: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_9: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_10: fa.EdgeKField[ta.vpfloat],
    patch0_cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
) -> fa.EdgeKField[ta.wpfloat]:
    p_out_e_hybrid_1a = (
        where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_1(E2C[1]),
            z_lsq_coeff_1(E2C[0]),
        )
        * astype(z_quad_vector_sum0_1, wpfloat)
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_2(E2C[1]),
            z_lsq_coeff_2(E2C[0]),
        )
        * astype(z_quad_vector_sum0_2, wpfloat)
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_3(E2C[1]),
            z_lsq_coeff_3(E2C[0]),
        )
        * astype(z_quad_vector_sum0_3, wpfloat)
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_4(E2C[1]),
            z_lsq_coeff_4(E2C[0]),
        )
        * astype(z_quad_vector_sum0_4, wpfloat)
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_5(E2C[1]),
            z_lsq_coeff_5(E2C[0]),
        )
        * astype(z_quad_vector_sum0_5, wpfloat)
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_6(E2C[1]),
            z_lsq_coeff_6(E2C[0]),
        )
        * astype(z_quad_vector_sum0_6, wpfloat)
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_7(E2C[1]),
            z_lsq_coeff_7(E2C[0]),
        )
        * astype(z_quad_vector_sum0_7, wpfloat)
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_8(E2C[1]),
            z_lsq_coeff_8(E2C[0]),
        )
        * astype(z_quad_vector_sum0_8, wpfloat)
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_9(E2C[1]),
            z_lsq_coeff_9(E2C[0]),
        )
        * astype(z_quad_vector_sum0_9, wpfloat)
        + where(
            patch0_cell_rel_idx_dsl == 1,
            z_lsq_coeff_10(E2C[1]),
            z_lsq_coeff_10(E2C[0]),
        )
        * astype(z_quad_vector_sum0_10, wpfloat)
    )

    return p_out_e_hybrid_1a


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_intermediate_horizontal_tracer_flux_from_cubic_coefficients(
    z_lsq_coeff_1: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_2: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_3: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_4: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_5: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_6: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_7: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_8: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_9: fa.CellKField[ta.wpfloat],
    z_lsq_coeff_10: fa.CellKField[ta.wpfloat],
    z_quad_vector_sum0_1: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_2: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_3: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_4: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_5: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_6: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_7: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_8: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_9: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum0_10: fa.EdgeKField[ta.vpfloat],
    patch0_cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
    p_out_e_hybrid_1a: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_intermediate_horizontal_tracer_flux_from_cubic_coefficients(
        z_lsq_coeff_1=z_lsq_coeff_1,
        z_lsq_coeff_2=z_lsq_coeff_2,
        z_lsq_coeff_3=z_lsq_coeff_3,
        z_lsq_coeff_4=z_lsq_coeff_4,
        z_lsq_coeff_5=z_lsq_coeff_5,
        z_lsq_coeff_6=z_lsq_coeff_6,
        z_lsq_coeff_7=z_lsq_coeff_7,
        z_lsq_coeff_8=z_lsq_coeff_8,
        z_lsq_coeff_9=z_lsq_coeff_9,
        z_lsq_coeff_10=z_lsq_coeff_10,
        z_quad_vector_sum0_1=z_quad_vector_sum0_1,
        z_quad_vector_sum0_2=z_quad_vector_sum0_2,
        z_quad_vector_sum0_3=z_quad_vector_sum0_3,
        z_quad_vector_sum0_4=z_quad_vector_sum0_4,
        z_quad_vector_sum0_5=z_quad_vector_sum0_5,
        z_quad_vector_sum0_6=z_quad_vector_sum0_6,
        z_quad_vector_sum0_7=z_quad_vector_sum0_7,
        z_quad_vector_sum0_8=z_quad_vector_sum0_8,
        z_quad_vector_sum0_9=z_quad_vector_sum0_9,
        z_quad_vector_sum0_10=z_quad_vector_sum0_10,
        patch0_cell_rel_idx_dsl=patch0_cell_rel_idx_dsl,
        out=p_out_e_hybrid_1a,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
