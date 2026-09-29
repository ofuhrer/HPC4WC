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


# TODO(dastrm): this stencil is unused
# TODO(dastrm): this stencil has no test


@gtx.field_operator
def _compute_intermediate_horizontal_tracer_flux_from_cubic_coeffs_alt(
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
    z_quad_vector_sum_1: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_2: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_3: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_4: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_5: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_6: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_7: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_8: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_9: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_10: fa.EdgeKField[ta.vpfloat],
    z_dreg_area: fa.EdgeKField[ta.vpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
) -> fa.EdgeKField[ta.wpfloat]:
    p_out_e_miura3 = (
        (
            where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_1(E2C[1]),
                z_lsq_coeff_1(E2C[0]),
            )
            * astype(z_quad_vector_sum_1, wpfloat)
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_2(E2C[1]),
                z_lsq_coeff_2(E2C[0]),
            )
            * astype(z_quad_vector_sum_2, wpfloat)
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_3(E2C[1]),
                z_lsq_coeff_3(E2C[0]),
            )
            * astype(z_quad_vector_sum_3, wpfloat)
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_4(E2C[1]),
                z_lsq_coeff_4(E2C[0]),
            )
            * astype(z_quad_vector_sum_4, wpfloat)
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_5(E2C[1]),
                z_lsq_coeff_5(E2C[0]),
            )
            * astype(z_quad_vector_sum_5, wpfloat)
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_6(E2C[1]),
                z_lsq_coeff_6(E2C[0]),
            )
            * astype(z_quad_vector_sum_6, wpfloat)
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_7(E2C[1]),
                z_lsq_coeff_7(E2C[0]),
            )
            * astype(z_quad_vector_sum_7, wpfloat)
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_8(E2C[1]),
                z_lsq_coeff_8(E2C[0]),
            )
            * astype(z_quad_vector_sum_8, wpfloat)
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_9(E2C[1]),
                z_lsq_coeff_9(E2C[0]),
            )
            * astype(z_quad_vector_sum_9, wpfloat)
            + where(
                cell_rel_idx_dsl == 1,
                z_lsq_coeff_10(E2C[1]),
                z_lsq_coeff_10(E2C[0]),
            )
            * astype(z_quad_vector_sum_10, wpfloat)
        )
        / astype(z_dreg_area, wpfloat)
        * p_mass_flx_e
    )

    return p_out_e_miura3


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_intermediate_horizontal_tracer_flux_from_cubic_coeffs_alt(
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
    z_quad_vector_sum_1: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_2: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_3: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_4: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_5: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_6: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_7: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_8: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_9: fa.EdgeKField[ta.vpfloat],
    z_quad_vector_sum_10: fa.EdgeKField[ta.vpfloat],
    z_dreg_area: fa.EdgeKField[ta.vpfloat],
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    cell_rel_idx_dsl: fa.EdgeKField[gtx.int32],
    p_out_e_miura3: fa.EdgeKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_intermediate_horizontal_tracer_flux_from_cubic_coeffs_alt(
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
        z_quad_vector_sum_1=z_quad_vector_sum_1,
        z_quad_vector_sum_2=z_quad_vector_sum_2,
        z_quad_vector_sum_3=z_quad_vector_sum_3,
        z_quad_vector_sum_4=z_quad_vector_sum_4,
        z_quad_vector_sum_5=z_quad_vector_sum_5,
        z_quad_vector_sum_6=z_quad_vector_sum_6,
        z_quad_vector_sum_7=z_quad_vector_sum_7,
        z_quad_vector_sum_8=z_quad_vector_sum_8,
        z_quad_vector_sum_9=z_quad_vector_sum_9,
        z_quad_vector_sum_10=z_quad_vector_sum_10,
        z_dreg_area=z_dreg_area,
        p_mass_flx_e=p_mass_flx_e,
        cell_rel_idx_dsl=cell_rel_idx_dsl,
        out=p_out_e_miura3,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
