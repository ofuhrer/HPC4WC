# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import abs, astype, minimum, neighbor_sum, where  # noqa: A004

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E2CO, C2E2CODim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _add_extra_diffusion_for_w_con_approaching_cfl(
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    z_w_con_c: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], ta.wpfloat],
    w: fa.CellKField[ta.wpfloat],
    ddt_w_adv: fa.CellKField[ta.vpfloat],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
) -> fa.CellKField[ta.vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_18."""
    z_w_con_c_wp, ddqz_z_half_wp, ddt_w_adv_wp, cfl_w_limit_wp = astype(
        (z_w_con_c, ddqz_z_half, ddt_w_adv, cfl_w_limit), wpfloat
    )

    difcoef = where(
        cfl_clipping & owner_mask,
        scalfac_exdiff
        * minimum(
            wpfloat("0.85") - cfl_w_limit_wp * dtime,
            abs(z_w_con_c_wp) * dtime / ddqz_z_half_wp - cfl_w_limit_wp * dtime,
        ),
        wpfloat("0.0"),
    )

    ddt_w_adv_wp = where(
        cfl_clipping & owner_mask,
        ddt_w_adv_wp + difcoef * area * neighbor_sum(w(C2E2CO) * geofac_n2s, axis=C2E2CODim),
        ddt_w_adv_wp,
    )

    return astype(ddt_w_adv_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def add_extra_diffusion_for_w_con_approaching_cfl(
    cfl_clipping: fa.CellKField[bool],
    owner_mask: fa.CellField[bool],
    z_w_con_c: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    area: fa.CellField[ta.wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[dims.CellDim, C2E2CODim], ta.wpfloat],
    w: fa.CellKField[ta.wpfloat],
    ddt_w_adv: fa.CellKField[ta.vpfloat],
    scalfac_exdiff: ta.wpfloat,
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _add_extra_diffusion_for_w_con_approaching_cfl(
        cfl_clipping=cfl_clipping,
        owner_mask=owner_mask,
        z_w_con_c=z_w_con_c,
        ddqz_z_half=ddqz_z_half,
        area=area,
        geofac_n2s=geofac_n2s,
        w=w,
        ddt_w_adv=ddt_w_adv,
        scalfac_exdiff=scalfac_exdiff,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        out=ddt_w_adv,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
