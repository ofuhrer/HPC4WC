# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E, C2EDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _mo_intp_rbf_rbf_vec_interpol_cell(
    p_vn_in: fa.EdgeKField[wpfloat],
    ptr_coeff_1: gtx.Field[gtx.Dims[dims.CellDim, C2EDim], wpfloat],
    ptr_coeff_2: gtx.Field[gtx.Dims[dims.CellDim, C2EDim], wpfloat],
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    p_u_out = neighbor_sum(ptr_coeff_1 * p_vn_in(C2E), axis=C2EDim)
    p_v_out = neighbor_sum(ptr_coeff_2 * p_vn_in(C2E), axis=C2EDim)
    return p_u_out, p_v_out


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def mo_intp_rbf_rbf_vec_interpol_cell(
    p_vn_in: fa.EdgeKField[wpfloat],
    ptr_coeff_1: gtx.Field[gtx.Dims[dims.CellDim, C2EDim], wpfloat],
    ptr_coeff_2: gtx.Field[gtx.Dims[dims.CellDim, C2EDim], wpfloat],
    p_u_out: fa.CellKField[wpfloat],
    p_v_out: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _mo_intp_rbf_rbf_vec_interpol_cell(
        p_vn_in=p_vn_in,
        ptr_coeff_1=ptr_coeff_1,
        ptr_coeff_2=ptr_coeff_2,
        out=(p_u_out, p_v_out),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
