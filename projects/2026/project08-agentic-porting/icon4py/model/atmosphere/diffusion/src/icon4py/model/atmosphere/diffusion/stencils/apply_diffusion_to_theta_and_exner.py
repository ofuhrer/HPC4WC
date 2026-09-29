# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import where

from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_z import (
    _calculate_nabla2_for_z,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_of_theta import (
    _calculate_nabla2_of_theta,
)
from icon4py.model.atmosphere.diffusion.stencils.truly_horizontal_diffusion_nabla_of_theta_over_steep_points import (
    _truly_horizontal_diffusion_nabla_of_theta_over_steep_points,
)
from icon4py.model.atmosphere.diffusion.stencils.update_theta_and_exner import (
    _update_theta_and_exner,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _apply_diffusion_to_theta_and_exner(
    kh_smag_e: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
    vcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    area: fa.CellField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: vpfloat,
    apply_zdiffusion_t: bool,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    z_nabla2_e = _calculate_nabla2_for_z(kh_smag_e, inv_dual_edge_length, theta_v)
    z_temp = _calculate_nabla2_of_theta(z_nabla2_e, geofac_div)

    if apply_zdiffusion_t:
        # Only where `zd_diffcoef` is not 0 to avoid loading the other fields.
        # Eventually the toolchain could consider extracting a mask `zd_diffcoef != 0` if we tell gt4py that `zd_diffcoef` is static.
        z_temp = where(
            zd_diffcoef != 0.0,
            _truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
                zd_vertoffset,
                zd_diffcoef,
                geofac_n2s_c,
                geofac_n2s_nbh,
                vcoef,
                theta_v,
                z_temp,
            ),
            z_temp,
        )

    theta_v, exner = _update_theta_and_exner(z_temp, area, theta_v, exner, rd_o_cvd)

    return theta_v, exner


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_diffusion_to_theta_and_exner(
    kh_smag_e: fa.EdgeKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    theta_v_in: fa.CellKField[wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], wpfloat],
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
    vcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    area: fa.CellField[wpfloat],
    theta_v: fa.CellKField[wpfloat],
    exner: fa.CellKField[wpfloat],
    rd_o_cvd: vpfloat,
    apply_zdiffusion_t: bool,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_diffusion_to_theta_and_exner(
        kh_smag_e=kh_smag_e,
        inv_dual_edge_length=inv_dual_edge_length,
        theta_v=theta_v_in,
        geofac_div=geofac_div,
        zd_vertoffset=zd_vertoffset,
        zd_diffcoef=zd_diffcoef,
        geofac_n2s_c=geofac_n2s_c,
        geofac_n2s_nbh=geofac_n2s_nbh,
        vcoef=vcoef,
        area=area,
        exner=exner,
        rd_o_cvd=rd_o_cvd,
        apply_zdiffusion_t=apply_zdiffusion_t,
        out=(theta_v, exner),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
