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
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _apply_nabla2_and_nabla4_global_to_vn(
    area_edge: fa.EdgeField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    z_nabla4_e2: fa.EdgeKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    kh_smag_e_wp, z_nabla4_e2_wp = astype((kh_smag_e, z_nabla4_e2), wpfloat)

    vn_wp = vn + area_edge * (
        kh_smag_e_wp * z_nabla2_e - diff_multfac_vn * z_nabla4_e2_wp * area_edge
    )
    return vn_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_nabla2_and_nabla4_global_to_vn(
    area_edge: fa.EdgeField[wpfloat],
    kh_smag_e: fa.EdgeKField[vpfloat],
    z_nabla2_e: fa.EdgeKField[wpfloat],
    z_nabla4_e2: fa.EdgeKField[vpfloat],
    diff_multfac_vn: fa.KField[wpfloat],
    vn: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_nabla2_and_nabla4_global_to_vn(
        area_edge=area_edge,
        kh_smag_e=kh_smag_e,
        z_nabla2_e=z_nabla2_e,
        z_nabla4_e2=z_nabla4_e2,
        diff_multfac_vn=diff_multfac_vn,
        vn=vn,
        out=vn,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
