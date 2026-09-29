# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E2CO, C2E2CODim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _compute_nabla2_on_cell(
    psi_c: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
) -> fa.CellField[ta.wpfloat]:
    """
    Computes the Laplacian (nabla squared) of a scalar field defined on cell
    centres.
    """
    nabla2_psi_c = neighbor_sum(psi_c(C2E2CO) * geofac_n2s, axis=C2E2CODim)

    return nabla2_psi_c


@gtx.field_operator
def _compute_nabla2_on_cell_k(
    psi_c: fa.CellKField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CODim], ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Computes the Laplacian (nabla squared) of a scalar field defined on cell
    centres for all vertical levels.
    """
    nabla2_psi_c = neighbor_sum(psi_c(C2E2CO) * geofac_n2s, axis=C2E2CODim)

    return nabla2_psi_c


@gtx.field_operator
def _compute_difference_on_cell_k(
    field_a: fa.CellKField[ta.wpfloat],
    field_b: fa.CellKField[ta.vpfloat],
) -> fa.CellKField[ta.wpfloat]:
    return field_a - astype(field_b, wpfloat)
