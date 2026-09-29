# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _integrate_tracer_density_horizontally(
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    z_rhofluxdiv_c: fa.CellKField[ta.vpfloat],
    z_tracer_mflx: fa.EdgeKField[ta.wpfloat],
    z_rho_now: fa.CellKField[ta.wpfloat],
    z_tracer_now: fa.CellKField[ta.wpfloat],
    z_dtsub: ta.wpfloat,
    nsub: gtx.int32,
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    z_rhofluxdiv_c_out = (
        astype(neighbor_sum(p_mass_flx_e(C2E) * geofac_div, axis=dims.C2EDim), vpfloat)
        if nsub == 1
        else z_rhofluxdiv_c
    )

    z_fluxdiv_c_dsl = astype(
        neighbor_sum(z_tracer_mflx(C2E) * geofac_div, axis=dims.C2EDim), vpfloat
    )

    z_rho_new_dsl = z_rho_now - z_dtsub * astype(z_rhofluxdiv_c_out, wpfloat)

    z_tracer_new_dsl = (
        z_tracer_now * z_rho_now - z_dtsub * astype(z_fluxdiv_c_dsl, wpfloat)
    ) / z_rho_new_dsl

    return (
        z_rhofluxdiv_c_out,
        z_fluxdiv_c_dsl,
        z_rho_new_dsl,
        z_tracer_new_dsl,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def integrate_tracer_density_horizontally(
    p_mass_flx_e: fa.EdgeKField[ta.wpfloat],
    geofac_div: gtx.Field[gtx.Dims[dims.CellDim, dims.C2EDim], ta.wpfloat],
    z_rhofluxdiv_c: fa.CellKField[ta.vpfloat],
    z_tracer_mflx: fa.EdgeKField[ta.wpfloat],
    z_rho_now: fa.CellKField[ta.wpfloat],
    z_tracer_now: fa.CellKField[ta.wpfloat],
    z_rhofluxdiv_c_out: fa.CellKField[ta.vpfloat],
    z_fluxdiv_c_dsl: fa.CellKField[ta.vpfloat],
    z_rho_new_dsl: fa.CellKField[ta.wpfloat],
    z_tracer_new_dsl: fa.CellKField[ta.wpfloat],
    z_dtsub: ta.wpfloat,
    nsub: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _integrate_tracer_density_horizontally(
        p_mass_flx_e=p_mass_flx_e,
        geofac_div=geofac_div,
        z_rhofluxdiv_c=z_rhofluxdiv_c,
        z_tracer_mflx=z_tracer_mflx,
        z_rho_now=z_rho_now,
        z_tracer_now=z_tracer_now,
        z_dtsub=z_dtsub,
        nsub=nsub,
        out=(z_rhofluxdiv_c_out, z_fluxdiv_c_dsl, z_rho_new_dsl, z_tracer_new_dsl),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
