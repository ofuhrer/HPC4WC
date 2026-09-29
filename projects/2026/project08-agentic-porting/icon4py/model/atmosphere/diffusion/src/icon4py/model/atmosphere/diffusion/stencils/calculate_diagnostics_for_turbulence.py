# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_diagnostics_for_turbulence(
    div: fa.CellKField[vpfloat],
    kh_c: fa.CellKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[vpfloat]]:
    wgtfac_c_wp, div_wp, kh_c_wp = astype((wgtfac_c, div, kh_c), wpfloat)

    div_ic_wp = astype(wgtfac_c * div, wpfloat) + (wpfloat("1.0") - wgtfac_c_wp) * div_wp(KDim - 1)
    hdef_ic_wp = astype(wgtfac_c * kh_c, wpfloat) + (wpfloat("1.0") - wgtfac_c_wp) * kh_c_wp(
        KDim - 1
    )
    hdef_ic_wp = hdef_ic_wp * hdef_ic_wp

    return astype((div_ic_wp, hdef_ic_wp), vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def calculate_diagnostics_for_turbulence(
    div: fa.CellKField[vpfloat],
    kh_c: fa.CellKField[vpfloat],
    wgtfac_c: fa.CellKField[vpfloat],
    div_ic: fa.CellKField[vpfloat],
    hdef_ic: fa.CellKField[vpfloat],
) -> None:
    _calculate_diagnostics_for_turbulence(div, kh_c, wgtfac_c, out=(div_ic[:, 1:], hdef_ic[:, 1:]))
