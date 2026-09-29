# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype
from gt4py.next.experimental import as_offset

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import C2E2C, C2E2CDim, Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
    vcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    theta_v: fa.CellKField[wpfloat],
    z_temp: fa.CellKField[vpfloat],
) -> fa.CellKField[vpfloat]:
    z_temp_wp = astype(z_temp, wpfloat)

    theta_v_0 = theta_v(C2E2C[0])(as_offset(Koff, zd_vertoffset[C2E2CDim(0)]))
    theta_v_1 = theta_v(C2E2C[1])(as_offset(Koff, zd_vertoffset[C2E2CDim(1)]))
    theta_v_2 = theta_v(C2E2C[2])(as_offset(Koff, zd_vertoffset[C2E2CDim(2)]))

    theta_v_0_m1 = theta_v(C2E2C[0])(as_offset(Koff, zd_vertoffset[C2E2CDim(0)] + 1))
    theta_v_1_m1 = theta_v(C2E2C[1])(as_offset(Koff, zd_vertoffset[C2E2CDim(1)] + 1))
    theta_v_2_m1 = theta_v(C2E2C[2])(as_offset(Koff, zd_vertoffset[C2E2CDim(2)] + 1))

    sum_tmp = (
        theta_v * geofac_n2s_c
        + geofac_n2s_nbh[C2E2CDim(0)]
        * (vcoef[C2E2CDim(0)] * theta_v_0 + (wpfloat("1.0") - vcoef[C2E2CDim(0)]) * theta_v_0_m1)
        + geofac_n2s_nbh[C2E2CDim(1)]
        * (vcoef[C2E2CDim(1)] * theta_v_1 + (wpfloat("1.0") - vcoef[C2E2CDim(1)]) * theta_v_1_m1)
        + geofac_n2s_nbh[C2E2CDim(2)]
        * (vcoef[C2E2CDim(2)] * theta_v_2 + (wpfloat("1.0") - vcoef[C2E2CDim(2)]) * theta_v_2_m1)
    )

    # Note: In the original Fortran code `zd_diffcoef` is implemented as a list,
    # in ICON4Py it's a full field intialized with zeros for points that are not in the list.
    z_temp_wp = z_temp_wp + zd_diffcoef * sum_tmp

    return astype(z_temp_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
    zd_vertoffset: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], gtx.int32],
    zd_diffcoef: fa.CellKField[wpfloat],
    geofac_n2s_c: fa.CellField[wpfloat],
    geofac_n2s_nbh: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim], wpfloat],
    vcoef: gtx.Field[gtx.Dims[dims.CellDim, dims.C2E2CDim, dims.KDim], wpfloat],
    theta_v: fa.CellKField[wpfloat],
    z_temp: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _truly_horizontal_diffusion_nabla_of_theta_over_steep_points(
        zd_vertoffset=zd_vertoffset,
        zd_diffcoef=zd_diffcoef,
        geofac_n2s_c=geofac_n2s_c,
        geofac_n2s_nbh=geofac_n2s_nbh,
        vcoef=vcoef,
        theta_v=theta_v,
        z_temp=z_temp,
        out=z_temp,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
