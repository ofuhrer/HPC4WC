# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import abs, where  # noqa: A004

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim


# TODO(dastrm): this stencil has no test
# TODO(dastrm): this stencil does not strictly match the fortran code


@gtx.field_operator
def _compute_courant_number_below(
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_mass: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    nlev: gtx.int32,
    dbl_eps: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    z_mass_pos = z_mass > 0.0

    in_bounds_p0 = k <= nlev - 1
    in_bounds_p1 = k <= nlev - 2
    in_bounds_p2 = k <= nlev - 3
    in_bounds_p3 = k <= nlev - 4

    mass_gt_cellmass_p0 = where(z_mass_pos & in_bounds_p0, z_mass >= p_cellmass_now, False)
    z_mass = z_mass - where(mass_gt_cellmass_p0, p_cellmass_now, 0.0)

    mass_gt_cellmass_p1 = mass_gt_cellmass_p0 & where(
        z_mass_pos & in_bounds_p1, z_mass >= p_cellmass_now(KDim + 1), False
    )
    z_mass = z_mass - where(mass_gt_cellmass_p1, p_cellmass_now(KDim + 1), 0.0)

    mass_gt_cellmass_p2 = mass_gt_cellmass_p1 & where(
        z_mass_pos & in_bounds_p2, z_mass >= p_cellmass_now(KDim + 2), False
    )
    z_mass = z_mass - where(mass_gt_cellmass_p2, p_cellmass_now(KDim + 2), 0.0)

    mass_gt_cellmass_p3 = mass_gt_cellmass_p2 & where(
        z_mass_pos & in_bounds_p3, z_mass >= p_cellmass_now(KDim + 3), False
    )
    z_mass = z_mass - where(mass_gt_cellmass_p3, p_cellmass_now(KDim + 3), 0.0)

    z_cfl = z_cfl + where(mass_gt_cellmass_p0, 1.0, 0.0)
    z_cfl = z_cfl + where(mass_gt_cellmass_p1, 1.0, 0.0)
    z_cfl = z_cfl + where(mass_gt_cellmass_p2, 1.0, 0.0)
    z_cfl = z_cfl + where(mass_gt_cellmass_p3, 1.0, 0.0)

    p_cellmass_now_jks = p_cellmass_now
    p_cellmass_now_jks = where(mass_gt_cellmass_p0, p_cellmass_now(KDim + 1), p_cellmass_now_jks)
    p_cellmass_now_jks = where(mass_gt_cellmass_p1, p_cellmass_now(KDim + 2), p_cellmass_now_jks)
    p_cellmass_now_jks = where(mass_gt_cellmass_p2, p_cellmass_now(KDim + 3), p_cellmass_now_jks)
    p_cellmass_now_jks = where(mass_gt_cellmass_p3, p_cellmass_now(KDim + 4), p_cellmass_now_jks)

    z_cflfrac = where(z_mass_pos, z_mass / p_cellmass_now_jks, 0.0)
    z_cfl = z_cfl + where(z_cflfrac < 1.0, z_cflfrac, 1.0 - dbl_eps)

    return z_cfl


@gtx.field_operator
def _compute_courant_number_above(
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_mass: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slevp1_ti: gtx.int32,
    dbl_eps: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    z_mass_neg = z_mass <= 0.0

    in_bounds_m0 = k >= slevp1_ti + 1
    in_bounds_m1 = k >= slevp1_ti + 2
    in_bounds_m2 = k >= slevp1_ti + 3
    in_bounds_m3 = k >= slevp1_ti + 4

    mass_gt_cellmass_m0 = where(
        z_mass_neg & in_bounds_m0, abs(z_mass) >= p_cellmass_now(KDim - 1), False
    )
    z_mass = z_mass + where(mass_gt_cellmass_m0, p_cellmass_now(KDim - 1), 0.0)

    mass_gt_cellmass_m1 = mass_gt_cellmass_m0 & where(
        z_mass_neg & in_bounds_m1, abs(z_mass) >= p_cellmass_now(KDim - 2), False
    )
    z_mass = z_mass + where(mass_gt_cellmass_m1, p_cellmass_now(KDim - 2), 0.0)

    mass_gt_cellmass_m2 = mass_gt_cellmass_m1 & where(
        z_mass_neg & in_bounds_m2, abs(z_mass) >= p_cellmass_now(KDim - 3), False
    )
    z_mass = z_mass + where(mass_gt_cellmass_m2, p_cellmass_now(KDim - 3), 0.0)

    mass_gt_cellmass_m3 = mass_gt_cellmass_m2 & where(
        z_mass_neg & in_bounds_m3, abs(z_mass) >= p_cellmass_now(KDim - 4), False
    )
    z_mass = z_mass + where(mass_gt_cellmass_m3, p_cellmass_now(KDim - 4), 0.0)

    p_cellmass_now_jks = p_cellmass_now(KDim - 1)
    p_cellmass_now_jks = where(mass_gt_cellmass_m0, p_cellmass_now(KDim - 2), p_cellmass_now_jks)
    p_cellmass_now_jks = where(mass_gt_cellmass_m1, p_cellmass_now(KDim - 3), p_cellmass_now_jks)
    p_cellmass_now_jks = where(mass_gt_cellmass_m2, p_cellmass_now(KDim - 4), p_cellmass_now_jks)
    p_cellmass_now_jks = where(mass_gt_cellmass_m3, p_cellmass_now(KDim - 5), p_cellmass_now_jks)

    z_cfl = z_cfl - where(mass_gt_cellmass_m0, 1.0, 0.0)
    z_cfl = z_cfl - where(mass_gt_cellmass_m1, 1.0, 0.0)
    z_cfl = z_cfl - where(mass_gt_cellmass_m2, 1.0, 0.0)
    z_cfl = z_cfl - where(mass_gt_cellmass_m3, 1.0, 0.0)

    z_cflfrac = where(z_mass_neg, z_mass / p_cellmass_now_jks, 0.0)
    z_cfl = z_cfl + where(abs(z_cflfrac) < 1.0, z_cflfrac, dbl_eps - 1.0)

    return z_cfl


@gtx.field_operator
def _compute_ppm4gpu_courant_number(
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slevp1_ti: gtx.int32,
    nlev: gtx.int32,
    dbl_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
) -> fa.CellKField[ta.wpfloat]:
    z_mass = p_dtime * p_mflx_contra_v

    cfl_below = _compute_courant_number_below(
        p_cellmass_now=p_cellmass_now,
        z_mass=z_mass,
        z_cfl=z_cfl,
        k=k,
        nlev=nlev,
        dbl_eps=dbl_eps,
    )
    cfl_above = _compute_courant_number_above(
        p_cellmass_now=p_cellmass_now,
        z_mass=z_mass,
        z_cfl=z_cfl,
        k=k,
        slevp1_ti=slevp1_ti,
        dbl_eps=dbl_eps,
    )

    z_cfl = cfl_below + cfl_above

    return z_cfl


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm4gpu_courant_number(
    p_mflx_contra_v: fa.CellKField[ta.wpfloat],
    p_cellmass_now: fa.CellKField[ta.wpfloat],
    z_cfl: fa.CellKField[ta.wpfloat],
    k: fa.KField[gtx.int32],
    slevp1_ti: gtx.int32,
    nlev: gtx.int32,
    dbl_eps: ta.wpfloat,
    p_dtime: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm4gpu_courant_number(
        p_mflx_contra_v=p_mflx_contra_v,
        p_cellmass_now=p_cellmass_now,
        z_cfl=z_cfl,
        k=k,
        slevp1_ti=slevp1_ti,
        nlev=nlev,
        dbl_eps=dbl_eps,
        p_dtime=p_dtime,
        out=z_cfl,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
