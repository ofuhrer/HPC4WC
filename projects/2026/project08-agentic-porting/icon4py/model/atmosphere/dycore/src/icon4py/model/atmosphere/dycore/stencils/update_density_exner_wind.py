# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.dycore.stencils.update_wind import _update_wind
from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _update_density_exner_wind(
    rho_now: fa.CellKField[wpfloat],
    grf_tend_rho: fa.CellKField[wpfloat],
    theta_v_now: fa.CellKField[wpfloat],
    grf_tend_thv: fa.CellKField[wpfloat],
    w_now: fa.CellKField[wpfloat],
    grf_tend_w: fa.CellKField[wpfloat],
    dtime: wpfloat,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
    fa.CellKField[wpfloat],
]:
    """Formerly known as _mo_solve_nonhydro_stencil_61."""
    rho_new_wp = rho_now + dtime * grf_tend_rho
    exner_new_wp = theta_v_now + dtime * grf_tend_thv
    w_new_wp = _update_wind(w_now=w_now, grf_tend_w=grf_tend_w, dtime=dtime)
    return rho_new_wp, exner_new_wp, w_new_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_density_exner_wind(
    rho_now: fa.CellKField[wpfloat],
    grf_tend_rho: fa.CellKField[wpfloat],
    theta_v_now: fa.CellKField[wpfloat],
    grf_tend_thv: fa.CellKField[wpfloat],
    w_now: fa.CellKField[wpfloat],
    grf_tend_w: fa.CellKField[wpfloat],
    rho_new: fa.CellKField[wpfloat],
    exner_new: fa.CellKField[wpfloat],
    w_new: fa.CellKField[wpfloat],
    dtime: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_density_exner_wind(
        rho_now=rho_now,
        grf_tend_rho=grf_tend_rho,
        theta_v_now=theta_v_now,
        grf_tend_thv=grf_tend_thv,
        w_now=w_now,
        grf_tend_w=grf_tend_w,
        dtime=dtime,
        out=(rho_new, exner_new, w_new),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
