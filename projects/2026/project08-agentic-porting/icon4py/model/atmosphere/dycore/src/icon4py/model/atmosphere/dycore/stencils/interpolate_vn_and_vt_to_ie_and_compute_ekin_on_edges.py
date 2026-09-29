# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx

from icon4py.model.atmosphere.dycore.stencils.interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges import (
    _interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_vt_to_interface_edges import (
    _interpolate_vt_to_interface_edges,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def _interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges(
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    vt: fa.EdgeKField[ta.vpfloat],
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    """Formerly known as _mo_solve_nonhydro_stencil_36."""
    z_vt_ie = _interpolate_vt_to_interface_edges(wgtfac_e=wgtfac_e, vt=vt)
    vn_ie, z_kin_hor_e = _interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges(
        wgtfac_e=wgtfac_e, vn=vn, vt=vt
    )
    return vn_ie, z_vt_ie, z_kin_hor_e


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges(
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    vt: fa.EdgeKField[ta.vpfloat],
    vn_ie: fa.EdgeKField[ta.vpfloat],
    z_vt_ie: fa.EdgeKField[ta.vpfloat],
    z_kin_hor_e: fa.EdgeKField[ta.vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_vn_and_vt_to_ie_and_compute_ekin_on_edges(
        wgtfac_e=wgtfac_e,
        vn=vn,
        vt=vt,
        out=(vn_ie, z_vt_ie, z_kin_hor_e),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
