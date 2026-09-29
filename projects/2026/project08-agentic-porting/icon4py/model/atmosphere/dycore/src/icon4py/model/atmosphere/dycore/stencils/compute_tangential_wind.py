# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C2E, E2C2EDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_tangential_wind(
    vn: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly knowan as _mo_velocity_advection_stencil_01."""
    vt_wp = neighbor_sum(rbf_vec_coeff_e * vn(E2C2E), axis=E2C2EDim)
    return astype(vt_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_tangential_wind(
    vn: fa.EdgeKField[wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], wpfloat],
    vt: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_tangential_wind(
        vn=vn,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        out=vt,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
