# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_nabla2_for_z(
    kh_smag_e: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    kh_smag_e_wp = astype(kh_smag_e, wpfloat)

    z_nabla2_e_wp = kh_smag_e_wp * inv_dual_edge_length * (theta_v(E2C[1]) - theta_v(E2C[0]))
    return z_nabla2_e_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_nabla2_for_z(
    kh_smag_e: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _calculate_nabla2_for_z(
        kh_smag_e=kh_smag_e,
        inv_dual_edge_length=inv_dual_edge_length,
        theta_v=theta_v,
        out=z_nabla2_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
