# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import max_over, maximum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2CDim
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _enhance_diffusion_coefficient_for_grid_point_cold_pools(
    kh_smag_e: fa.EdgeKField[vpfloat],
    enh_diffu_3d: fa.CellKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    kh_smag_e_vp = maximum(kh_smag_e, max_over(enh_diffu_3d(E2C), axis=E2CDim))
    return kh_smag_e_vp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def enhance_diffusion_coefficient_for_grid_point_cold_pools(
    kh_smag_e: fa.EdgeKField[vpfloat],
    enh_diffu_3d: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _enhance_diffusion_coefficient_for_grid_point_cold_pools(
        kh_smag_e=kh_smag_e,
        enh_diffu_3d=enh_diffu_3d,
        out=kh_smag_e,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
