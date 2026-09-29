# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_kinetic_energy import (
    _compute_horizontal_kinetic_energy,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges(
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    vt: fa.EdgeKField[ta.vpfloat],
) -> tuple[fa.EdgeKField[ta.vpfloat], fa.EdgeKField[ta.vpfloat]]:
    """Formerly known as _mo_velocity_advection_stencil_02."""
    # TODO(): This stencil fusion with the one below is not optimal:
    # _compute_horizontal_kinetic_energy is wasting computation by assigning
    # vn_ie and vt_ie which are thrown away.
    wgtfac_e_wp = astype(wgtfac_e, wpfloat)

    vn_ie_wp = wgtfac_e_wp * vn + (wpfloat("1.0") - wgtfac_e_wp) * vn(KDim - 1)
    _, _, z_kin_hor_e = _compute_horizontal_kinetic_energy(vn=vn, vt=vt)

    return astype(vn_ie_wp, vpfloat), z_kin_hor_e


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges(
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    vt: fa.EdgeKField[ta.vpfloat],
    vn_ie: fa.EdgeKField[ta.vpfloat],
    z_kin_hor_e: fa.EdgeKField[ta.vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _interpolate_vn_to_half_levels_and_compute_kinetic_energy_on_edges(
        wgtfac_e=wgtfac_e,
        vn=vn,
        vt=vt,
        out=(vn_ie, z_kin_hor_e),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
