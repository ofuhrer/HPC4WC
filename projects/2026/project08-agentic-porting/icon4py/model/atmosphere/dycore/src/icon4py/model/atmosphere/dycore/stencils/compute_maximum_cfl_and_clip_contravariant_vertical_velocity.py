# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import abs, astype, broadcast, where  # noqa: A004

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    z_w_con_c: fa.CellKField[ta.vpfloat],
    cfl_w_limit: ta.vpfloat,
    dtime: ta.wpfloat,
) -> tuple[
    fa.CellKField[bool],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
]:
    """Formerly know as _mo_velocity_advection_stencil_14."""
    z_w_con_c_wp, ddqz_z_half_wp = astype((z_w_con_c, ddqz_z_half), wpfloat)

    cfl_clipping = where(
        abs(z_w_con_c) > cfl_w_limit * ddqz_z_half,
        broadcast(True, (dims.CellDim, dims.KDim)),
        False,
    )

    vcfl = where(cfl_clipping, z_w_con_c_wp * dtime / ddqz_z_half_wp, wpfloat("0.0"))
    vcfl_vp = astype(vcfl, vpfloat)

    z_w_con_c_wp = where(
        (cfl_clipping) & (vcfl_vp < -vpfloat("0.85")),
        astype(-vpfloat("0.85") * ddqz_z_half, wpfloat) / dtime,
        z_w_con_c_wp,
    )

    z_w_con_c_wp = where(
        (cfl_clipping) & (vcfl_vp > vpfloat("0.85")),
        astype(vpfloat("0.85") * ddqz_z_half, wpfloat) / dtime,
        z_w_con_c_wp,
    )

    return cfl_clipping, vcfl_vp, astype(z_w_con_c_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
    ddqz_z_half: fa.CellKField[vpfloat],
    z_w_con_c: fa.CellKField[vpfloat],
    cfl_clipping: fa.CellKField[bool],
    vcfl: fa.CellKField[vpfloat],
    cfl_w_limit: vpfloat,
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_maximum_cfl_and_clip_contravariant_vertical_velocity(
        ddqz_z_half=ddqz_z_half,
        z_w_con_c=z_w_con_c,
        cfl_w_limit=cfl_w_limit,
        dtime=dtime,
        out=(cfl_clipping, vcfl, z_w_con_c),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
