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
from icon4py.model.common.dimension import C2E2C2E


@gtx.field_operator
def _edge_2_cell_vector_rbf_interpolation(
    p_e_in: fa.EdgeKField[ta.wpfloat],
    ptr_coeff_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat],
    ptr_coeff_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """
    Performs vector RBF reconstruction at cell center from edge center.
    It is ported from subroutine rbf_vec_interpol_cell in mo_intp_rbf.f90 in ICON.

    The theory is described in Narcowich and Ward (Math Comp. 1994) and Bonaventura and Baudisch (Mox Report n. 75).
    It takes edge based variables as input and combines them into three dimensional cartesian vectors at each cell center.
    TODO(OngChia): This stencil actually just use the c2e2c2e connectivity and the corresponding coefficients to compute cell-center value without knowledge of how the coefficients are computed. A better name is perferred.

    Args:
        p_e_in: Input values at edge center.
        ptr_coeff_1: RBF coefficient in zonal direction.
        ptr_coeff_2: RBF coefficient in meridional direction.
    Returns:
        RBF reconstructed vector at cell center.
    """
    p_u_out = neighbor_sum(ptr_coeff_1 * p_e_in(C2E2C2E), axis=dims.C2E2C2EDim)
    p_v_out = neighbor_sum(ptr_coeff_2 * p_e_in(C2E2C2E), axis=dims.C2E2C2EDim)
    return p_u_out, p_v_out


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def edge_2_cell_vector_rbf_interpolation(
    p_e_in: fa.EdgeKField[ta.wpfloat],
    ptr_coeff_1: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat],
    ptr_coeff_2: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2C2EDim], ta.wpfloat],
    p_u_out: fa.CellKField[ta.wpfloat],
    p_v_out: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _edge_2_cell_vector_rbf_interpolation(
        p_e_in=p_e_in,
        ptr_coeff_1=ptr_coeff_1,
        ptr_coeff_2=ptr_coeff_2,
        out=(p_u_out, p_v_out),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
