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
from icon4py.model.common.dimension import E2C, E2C2EO, E2V, E2C2EODim, E2CDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
    levelmask: fa.KField[bool],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, E2CDim], ta.wpfloat],
    z_w_con_c_full: fa.CellKField[ta.vpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    zeta: fa.VertexKField[ta.vpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2EODim], ta.wpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    ddt_vn_apc: fa.EdgeKField[ta.vpfloat],
    cfl_w_limit: ta.vpfloat,
    scalfac_exdiff: ta.wpfloat,
    dtime: ta.wpfloat,
) -> fa.EdgeKField[ta.vpfloat]:
    """Formerly known as _mo_velocity_advection_stencil_20."""
    z_w_con_c_full_wp, ddqz_z_full_e_wp, ddt_vn_apc_wp, cfl_w_limit_wp = astype(
        (z_w_con_c_full, ddqz_z_full_e, ddt_vn_apc, cfl_w_limit), wpfloat
    )

    w_con_e = neighbor_sum(c_lin_e * z_w_con_c_full_wp(E2C), axis=E2CDim)
    difcoef = scalfac_exdiff * minimum(
        wpfloat("0.85") - cfl_w_limit_wp * dtime,
        abs(w_con_e) * dtime / ddqz_z_full_e_wp - cfl_w_limit_wp * dtime,
    )
    ddt_vn_apc_wp = where(
        # TODO(havogt): my guess is if the second condition is `True`, then
        # `(levelmask | levelmask(KDim + 1))` is also `True`
        (levelmask | levelmask(KDim + 1))
        & (abs(w_con_e) > astype(cfl_w_limit * ddqz_z_full_e, wpfloat)),
        ddt_vn_apc_wp
        + difcoef
        * area_edge
        * (
            neighbor_sum(geofac_grdiv * vn(E2C2EO), axis=E2C2EODim)
            + tangent_orientation
            * inv_primal_edge_length
            * astype(zeta(E2V[1]) - zeta(E2V[0]), wpfloat)
        ),
        ddt_vn_apc_wp,
    )
    return astype(ddt_vn_apc_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
    levelmask: gtx.Field[gtx.Dims[dims.KDim], bool],
    c_lin_e: gtx.Field[gtx.Dims[dims.EdgeDim, E2CDim], ta.wpfloat],
    z_w_con_c_full: fa.CellKField[ta.vpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    area_edge: fa.EdgeField[ta.wpfloat],
    tangent_orientation: fa.EdgeField[ta.wpfloat],
    inv_primal_edge_length: fa.EdgeField[ta.wpfloat],
    zeta: fa.VertexKField[ta.vpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, E2C2EODim], ta.wpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    ddt_vn_apc: fa.EdgeKField[ta.vpfloat],
    cfl_w_limit: ta.vpfloat,
    scalfac_exdiff: ta.wpfloat,
    dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _add_extra_diffusion_for_normal_wind_tendency_approaching_cfl(
        levelmask=levelmask,
        c_lin_e=c_lin_e,
        z_w_con_c_full=z_w_con_c_full,
        ddqz_z_full_e=ddqz_z_full_e,
        area_edge=area_edge,
        tangent_orientation=tangent_orientation,
        inv_primal_edge_length=inv_primal_edge_length,
        zeta=zeta,
        geofac_grdiv=geofac_grdiv,
        vn=vn,
        ddt_vn_apc=ddt_vn_apc,
        cfl_w_limit=cfl_w_limit,
        scalfac_exdiff=scalfac_exdiff,
        dtime=dtime,
        out=ddt_vn_apc,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
