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
def _extrapolate_temporally_exner_pressure(
    exner_exfac: fa.CellKField[vpfloat],
    exner: fa.CellKField[wpfloat],
    exner_ref_mc: fa.CellKField[vpfloat],
    exner_pr: fa.CellKField[wpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[wpfloat]]:
    """Formerly known as _mo_solve_nonhydro_stencil_02."""
    exner_exfac_wp, exner_ref_mc_wp = astype((exner_exfac, exner_ref_mc), wpfloat)

    z_exner_ex_pr_wp = (wpfloat("1.0") + exner_exfac_wp) * (
        exner - exner_ref_mc_wp
    ) - exner_exfac_wp * exner_pr
    exner_pr_wp = exner - exner_ref_mc_wp
    return astype(z_exner_ex_pr_wp, vpfloat), exner_pr_wp


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def extrapolate_temporally_exner_pressure(
    exner_exfac: fa.CellKField[vpfloat],
    exner: fa.CellKField[wpfloat],
    exner_ref_mc: fa.CellKField[vpfloat],
    exner_pr: fa.CellKField[wpfloat],
    z_exner_ex_pr: fa.CellKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _extrapolate_temporally_exner_pressure(
        exner_exfac=exner_exfac,
        exner=exner,
        exner_ref_mc=exner_ref_mc,
        exner_pr=exner_pr,
        out=(z_exner_ex_pr, exner_pr),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
