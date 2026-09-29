# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _update_mass_flux_weighted(
    rho_ic: fa.CellKField[wpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
    w_now: fa.CellKField[wpfloat],
    w_new: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    mass_flx_ic: fa.CellKField[wpfloat],
    r_nsubsteps: wpfloat,
) -> fa.CellKField[wpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_65."""
    w_concorr_c_wp = astype(w_concorr_c, wpfloat)

    mass_flx_ic_wp = mass_flx_ic + (
        r_nsubsteps * rho_ic * (vwind_expl_wgt * w_now + vwind_impl_wgt * w_new - w_concorr_c_wp)
    )
    return mass_flx_ic_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_mass_flux_weighted(
    rho_ic: fa.CellKField[wpfloat],
    vwind_expl_wgt: fa.CellField[wpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
    w_now: fa.CellKField[wpfloat],
    w_new: fa.CellKField[wpfloat],
    w_concorr_c: fa.CellKField[vpfloat],
    mass_flx_ic: fa.CellKField[wpfloat],
    r_nsubsteps: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_mass_flux_weighted(
        rho_ic=rho_ic,
        vwind_expl_wgt=vwind_expl_wgt,
        vwind_impl_wgt=vwind_impl_wgt,
        w_now=w_now,
        w_new=w_new,
        w_concorr_c=w_concorr_c,
        mass_flx_ic=mass_flx_ic,
        r_nsubsteps=r_nsubsteps,
        out=mass_flx_ic,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
