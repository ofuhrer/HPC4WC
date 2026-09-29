# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _update_mass_volume_flux(
    z_contr_w_fl_l: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
    w: fa.CellKField[wpfloat],
    mass_flx_ic: fa.CellKField[wpfloat],
    vol_flx_ic: fa.CellKField[wpfloat],
    r_nsubsteps: wpfloat,
) -> tuple[fa.CellKField[wpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_58."""
    z_a = r_nsubsteps * (z_contr_w_fl_l + rho_ic * vwind_impl_wgt * w)
    mass_flx_ic_wp = mass_flx_ic + z_a
    vol_flx_ic_wp = vol_flx_ic + z_a / rho_ic
    return mass_flx_ic_wp, vol_flx_ic_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_mass_volume_flux(
    z_contr_w_fl_l: fa.CellKField[wpfloat],
    rho_ic: fa.CellKField[wpfloat],
    vwind_impl_wgt: fa.CellField[wpfloat],
    w: fa.CellKField[wpfloat],
    mass_flx_ic: fa.CellKField[wpfloat],
    vol_flx_ic: fa.CellKField[wpfloat],
    r_nsubsteps: wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _update_mass_volume_flux(
        z_contr_w_fl_l=z_contr_w_fl_l,
        rho_ic=rho_ic,
        vwind_impl_wgt=vwind_impl_wgt,
        w=w,
        mass_flx_ic=mass_flx_ic,
        vol_flx_ic=vol_flx_ic,
        r_nsubsteps=r_nsubsteps,
        out=(mass_flx_ic, vol_flx_ic),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
