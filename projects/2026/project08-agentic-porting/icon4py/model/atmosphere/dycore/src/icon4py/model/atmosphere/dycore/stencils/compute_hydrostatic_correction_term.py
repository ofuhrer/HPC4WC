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
def _compute_hydrostatic_correction_term(
    theta_v: fa.CellKField[wpfloat],
    ikoffset: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    grav_o_cpd: wpfloat,
) -> fa.EdgeKField[vpfloat]:
    """Formerly known as _mo_solve_nonhydro_stencil_21."""
    zdiff_gradp_wp = zdiff_gradp  # astype(zdiff_gradp, wpfloat) # TODO(): fix this cast

    theta_v_0 = theta_v(E2C[0])(as_offset(Koff, ikoffset[E2CDim(0)]))
    theta_v_1 = theta_v(E2C[1])(as_offset(Koff, ikoffset[E2CDim(1)]))

    theta_v_ic_0 = theta_v_ic(E2C[0])(as_offset(Koff, ikoffset[E2CDim(0)]))
    theta_v_ic_1 = theta_v_ic(E2C[1])(as_offset(Koff, ikoffset[E2CDim(1)]))

    theta_v_ic_p1_0 = theta_v_ic(E2C[0])(as_offset(Koff, ikoffset[E2CDim(0)] + 1))
    theta_v_ic_p1_1 = theta_v_ic(E2C[1])(as_offset(Koff, ikoffset[E2CDim(1)] + 1))

    inv_ddqz_z_full_0_wp = astype(
        inv_ddqz_z_full(E2C[0])(as_offset(Koff, ikoffset[E2CDim(0)])), wpfloat
    )
    inv_ddqz_z_full_1_wp = astype(
        inv_ddqz_z_full(E2C[1])(as_offset(Koff, ikoffset[E2CDim(1)])), wpfloat
    )

    z_theta_0 = (
        theta_v_0
        + zdiff_gradp_wp[E2CDim(0)] * (theta_v_ic_0 - theta_v_ic_p1_0) * inv_ddqz_z_full_0_wp
    )
    z_theta_1 = (
        theta_v_1
        + zdiff_gradp_wp[E2CDim(1)] * (theta_v_ic_1 - theta_v_ic_p1_1) * inv_ddqz_z_full_1_wp
    )
    z_hydro_corr_wp = (
        grav_o_cpd
        * inv_dual_edge_length
        * (z_theta_1 - z_theta_0)
        * wpfloat("4.0")
        / ((z_theta_0 + z_theta_1) * (z_theta_0 + z_theta_1))
    )

    return astype(z_hydro_corr_wp, vpfloat)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_hydrostatic_correction_term(
    theta_v: fa.CellKField[wpfloat],
    ikoffset: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], gtx.int32],
    zdiff_gradp: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim, dims.KDim], vpfloat],
    theta_v_ic: fa.CellKField[wpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    inv_dual_edge_length: fa.EdgeField[wpfloat],
    grav_o_cpd: wpfloat,
    z_hydro_corr: fa.EdgeKField[vpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_hydrostatic_correction_term(
        theta_v=theta_v,
        ikoffset=ikoffset,
        zdiff_gradp=zdiff_gradp,
        theta_v_ic=theta_v_ic,
        inv_ddqz_z_full=inv_ddqz_z_full,
        inv_dual_edge_length=inv_dual_edge_length,
        grav_o_cpd=grav_o_cpd,
        out=z_hydro_corr,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
