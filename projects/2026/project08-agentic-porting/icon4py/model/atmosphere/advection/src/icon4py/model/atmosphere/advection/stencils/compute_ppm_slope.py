# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim


@gtx.field_operator
def _compute_ppm_slope_a(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    zfac_m1 = (p_cc - p_cc(KDim - 1)) / (p_cellhgt_mc_now + p_cellhgt_mc_now(KDim - 1))
    zfac = (p_cc(KDim + 1) - p_cc) / (p_cellhgt_mc_now(KDim + 1) + p_cellhgt_mc_now)
    z_slope = (
        p_cellhgt_mc_now
        / (p_cellhgt_mc_now(KDim - 1) + p_cellhgt_mc_now + p_cellhgt_mc_now(KDim + 1))
    ) * (
        (2.0 * p_cellhgt_mc_now(KDim - 1) + p_cellhgt_mc_now) * zfac
        + (p_cellhgt_mc_now + 2.0 * p_cellhgt_mc_now(KDim + 1)) * zfac_m1
    )

    return z_slope


@gtx.field_operator
def _compute_ppm_slope_b(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    zfac_m1 = (p_cc - p_cc(KDim - 1)) / (p_cellhgt_mc_now + p_cellhgt_mc_now(KDim - 1))
    z_slope = (
        (p_cellhgt_mc_now / (p_cellhgt_mc_now(KDim - 1) + p_cellhgt_mc_now + p_cellhgt_mc_now))
        * (p_cellhgt_mc_now + 2.0 * p_cellhgt_mc_now)
        * zfac_m1
    )

    return z_slope


@gtx.field_operator
def _compute_ppm_slope(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    elev: gtx.int32,
) -> fa.CellKField[ta.wpfloat]:
    z_slope = concat_where(
        dims.KDim == elev,
        _compute_ppm_slope_b(p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now),
        _compute_ppm_slope_a(p_cc=p_cc, p_cellhgt_mc_now=p_cellhgt_mc_now),
    )

    return z_slope


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_slope(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    z_slope: fa.CellKField[ta.wpfloat],
    elev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_slope(
        p_cc=p_cc,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        elev=elev,
        out=z_slope,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
