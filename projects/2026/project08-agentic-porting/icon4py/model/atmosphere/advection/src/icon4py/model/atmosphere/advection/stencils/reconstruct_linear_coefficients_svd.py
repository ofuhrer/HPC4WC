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
from icon4py.model.common.dimension import C2E2C, C2E2CDim


# TODO(dastrm): this stencil has no test
# TODO(dastrm): move to common


@gtx.field_operator
def _reconstruct_linear_coefficients_svd(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    p_coeff_1_dsl = p_cc
    p_coeff_2_dsl = neighbor_sum(lsq_pseudoinv_1 * (p_cc(C2E2C) - p_cc), axis=C2E2CDim)
    p_coeff_3_dsl = neighbor_sum(lsq_pseudoinv_2 * (p_cc(C2E2C) - p_cc), axis=C2E2CDim)
    return p_coeff_1_dsl, p_coeff_2_dsl, p_coeff_3_dsl


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def reconstruct_linear_coefficients_svd(
    p_cc: fa.CellKField[ta.wpfloat],
    lsq_pseudoinv_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    lsq_pseudoinv_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], ta.wpfloat],
    p_coeff_1_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_2_dsl: fa.CellKField[ta.wpfloat],
    p_coeff_3_dsl: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _reconstruct_linear_coefficients_svd(
        p_cc=p_cc,
        lsq_pseudoinv_1=lsq_pseudoinv_1,
        lsq_pseudoinv_2=lsq_pseudoinv_2,
        out=(p_coeff_1_dsl, p_coeff_2_dsl, p_coeff_3_dsl),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
