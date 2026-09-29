# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype
from gt4py.next.experimental import as_offset

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import E2C, E2CDim, Koff
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    z_exner_ex_pr: fa.CellKField[vpfloat],
    zdiff_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], vpfloat],
    ikoffset: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
    z_dexner_dz_c_2: fa.CellKField[vpfloat],
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_20."""
    z_exner_ex_pr_0 = z_exner_ex_pr(E2C[0])(as_offset(Koff, ikoffset[E2CDim(0)]))
    z_exner_ex_pr_1 = z_exner_ex_pr(E2C[1])(as_offset(Koff, ikoffset[E2CDim(1)]))

    z_dexner_dz_c1_0 = z_dexner_dz_c_1(E2C[0])(as_offset(Koff, ikoffset[E2CDim(0)]))
    z_dexner_dz_c1_1 = z_dexner_dz_c_1(E2C[1])(as_offset(Koff, ikoffset[E2CDim(1)]))

    z_dexner_dz_c2_0 = z_dexner_dz_c_2(E2C[0])(as_offset(Koff, ikoffset[E2CDim(0)]))
    z_dexner_dz_c2_1 = z_dexner_dz_c_2(E2C[1])(as_offset(Koff, ikoffset[E2CDim(1)]))

    z_gradh_exner_wp = inv_dual_edge_length * (
        astype(
            (
                z_exner_ex_pr_1
                + zdiff_gradp[E2CDim(1)]
                * (z_dexner_dz_c1_1 + zdiff_gradp[E2CDim(1)] * z_dexner_dz_c2_1)
            )
            - (
                z_exner_ex_pr_0
                + zdiff_gradp[E2CDim(0)]
                * (z_dexner_dz_c1_0 + zdiff_gradp[E2CDim(0)] * z_dexner_dz_c2_0)
            ),
            wpfloat,
        )
    )

    return astype(z_gradh_exner_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    z_exner_ex_pr: fa.CellKField[vpfloat],
    zdiff_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], vpfloat],
    ikoffset: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32],
    z_dexner_dz_c_1: fa.CellKField[vpfloat],
    z_dexner_dz_c_2: fa.CellKField[vpfloat],
    z_gradh_exner: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_horizontal_gradient_of_exner_pressure_for_multiple_levels(
        inv_dual_edge_length=inv_dual_edge_length,
        z_exner_ex_pr=z_exner_ex_pr,
        zdiff_gradp=zdiff_gradp,
        ikoffset=ikoffset,
        z_dexner_dz_c_1=z_dexner_dz_c_1,
        z_dexner_dz_c_2=z_dexner_dz_c_2,
        out=z_gradh_exner,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
