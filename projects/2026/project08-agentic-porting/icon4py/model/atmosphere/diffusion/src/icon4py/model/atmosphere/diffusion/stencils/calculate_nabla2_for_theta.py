# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_z import (
    _calculate_nabla2_for_z,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_of_theta import (
    _calculate_nabla2_of_theta,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_nabla2_for_theta(
    kh_smag_e: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
) -> fa.CellKField[vpfloat]:
    z_nabla2_e = _calculate_nabla2_for_z(kh_smag_e, inv_dual_edge_length, theta_v)
    z_temp = _calculate_nabla2_of_theta(z_nabla2_e, geofac_div)
    return z_temp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_nabla2_for_theta(
    kh_smag_e: fa.EdgeKField[float],
    inv_dual_edge_length: fa.EdgeField[float],
    theta_v: fa.CellKField[float],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], float],
    z_temp: fa.CellKField[float],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _calculate_nabla2_for_theta(
        kh_smag_e=kh_smag_e,
        inv_dual_edge_length=inv_dual_edge_length,
        theta_v=theta_v,
        geofac_div=geofac_div,
        out=z_temp,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
