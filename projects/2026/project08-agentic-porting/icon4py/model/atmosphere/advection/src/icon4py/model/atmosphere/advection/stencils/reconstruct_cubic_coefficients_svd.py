# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E2C2E2C, C2E2C2E2CDim


@gtx.field_operator
def _reconstruct_cubic_coefficients_svd(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_4: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_5: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_6: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_7: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_8: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_9: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_moments_1: fa.CellField[ta.wpfloat],
    lsq_moments_2: fa.CellField[ta.wpfloat],
    lsq_moments_3: fa.CellField[ta.wpfloat],
    lsq_moments_4: fa.CellField[ta.wpfloat],
    lsq_moments_5: fa.CellField[ta.wpfloat],
    lsq_moments_6: fa.CellField[ta.wpfloat],
    lsq_moments_7: fa.CellField[ta.wpfloat],
    lsq_moments_8: fa.CellField[ta.wpfloat],
    lsq_moments_9: fa.CellField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    p_coeff_10_dsl = neighbor_sum(lsq_pseudoinv_9 * (p_cc(C2E2C2E2C) - p_cc), axis=C2E2C2E2CDim)
    p_coeff_9_dsl = neighbor_sum(lsq_pseudoinv_8 * (p_cc(C2E2C2E2C) - p_cc), axis=C2E2C2E2CDim)
    p_coeff_8_dsl = neighbor_sum(lsq_pseudoinv_7 * (p_cc(C2E2C2E2C) - p_cc), axis=C2E2C2E2CDim)
    p_coeff_7_dsl = neighbor_sum(lsq_pseudoinv_6 * (p_cc(C2E2C2E2C) - p_cc), axis=C2E2C2E2CDim)
    p_coeff_6_dsl = neighbor_sum(lsq_pseudoinv_5 * (p_cc(C2E2C2E2C) - p_cc), axis=C2E2C2E2CDim)
    p_coeff_5_dsl = neighbor_sum(lsq_pseudoinv_4 * (p_cc(C2E2C2E2C) - p_cc), axis=C2E2C2E2CDim)
    p_coeff_4_dsl = neighbor_sum(lsq_pseudoinv_3 * (p_cc(C2E2C2E2C) - p_cc), axis=C2E2C2E2CDim)
    p_coeff_3_dsl = neighbor_sum(lsq_pseudoinv_2 * (p_cc(C2E2C2E2C) - p_cc), axis=C2E2C2E2CDim)
    p_coeff_2_dsl = neighbor_sum(lsq_pseudoinv_1 * (p_cc(C2E2C2E2C) - p_cc), axis=C2E2C2E2CDim)

    p_coeff_1_dsl = p_cc - (
        p_coeff_2_dsl * lsq_moments_1
        + p_coeff_3_dsl * lsq_moments_2
        + p_coeff_4_dsl * lsq_moments_3
        + p_coeff_5_dsl * lsq_moments_4
        + p_coeff_6_dsl * lsq_moments_5
        + p_coeff_7_dsl * lsq_moments_6
        + p_coeff_8_dsl * lsq_moments_7
        + p_coeff_9_dsl * lsq_moments_8
        + p_coeff_10_dsl * lsq_moments_9
    )

    return (
        p_coeff_1_dsl,
        p_coeff_2_dsl,
        p_coeff_3_dsl,
        p_coeff_4_dsl,
        p_coeff_5_dsl,
        p_coeff_6_dsl,
        p_coeff_7_dsl,
        p_coeff_8_dsl,
        p_coeff_9_dsl,
        p_coeff_10_dsl,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def reconstruct_cubic_coefficients_svd(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_3: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_4: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_5: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_6: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_7: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_8: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_9: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2E2CDim], ta.wpfloat],
    lsq_moments_1: fa.CellField[ta.wpfloat],
    lsq_moments_2: fa.CellField[ta.wpfloat],
    lsq_moments_3: fa.CellField[ta.wpfloat],
    lsq_moments_4: fa.CellField[ta.wpfloat],
    lsq_moments_5: fa.CellField[ta.wpfloat],
    lsq_moments_6: fa.CellField[ta.wpfloat],
    lsq_moments_7: fa.CellField[ta.wpfloat],
    lsq_moments_8: fa.CellField[ta.wpfloat],
    lsq_moments_9: fa.CellField[ta.wpfloat],
    p_coeff_1_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_2_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_3_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_4_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_5_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_6_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_7_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_8_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_9_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_10_dsl: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _reconstruct_cubic_coefficients_svd(
        p_cc=p_cc,
        lsq_pseudoinv_1=lsq_pseudoinv_1,
        lsq_pseudoinv_2=lsq_pseudoinv_2,
        lsq_pseudoinv_3=lsq_pseudoinv_3,
        lsq_pseudoinv_4=lsq_pseudoinv_4,
        lsq_pseudoinv_5=lsq_pseudoinv_5,
        lsq_pseudoinv_6=lsq_pseudoinv_6,
        lsq_pseudoinv_7=lsq_pseudoinv_7,
        lsq_pseudoinv_8=lsq_pseudoinv_8,
        lsq_pseudoinv_9=lsq_pseudoinv_9,
        lsq_moments_1=lsq_moments_1,
        lsq_moments_2=lsq_moments_2,
        lsq_moments_3=lsq_moments_3,
        lsq_moments_4=lsq_moments_4,
        lsq_moments_5=lsq_moments_5,
        lsq_moments_6=lsq_moments_6,
        lsq_moments_7=lsq_moments_7,
        lsq_moments_8=lsq_moments_8,
        lsq_moments_9=lsq_moments_9,
        out=(
            p_coeff_1_dsl,
            p_coeff_2_dsl,
            p_coeff_3_dsl,
            p_coeff_4_dsl,
            p_coeff_5_dsl,
            p_coeff_6_dsl,
            p_coeff_7_dsl,
            p_coeff_8_dsl,
            p_coeff_9_dsl,
            p_coeff_10_dsl,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
