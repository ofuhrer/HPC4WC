# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.advection.stencils.compute_ppm_quadratic_face_values import (
    _compute_ppm_quadratic_face_values,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim


# TODO(dastrm): this stencil is imported but never called
# TODO(dastrm): slev/elev and vertical_start/end are redundant


@gtx.field_operator
def _compute_ppm_all_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    p_face_in: fa.CellKField[ta.wpfloat],
    slev: gtx.int32,
    elev: gtx.int32,
    slevp1: gtx.int32,
    elevp1: gtx.int32,
) -> fa.CellKField[ta.wpfloat]:
    p_face = concat_where(
        (dims.KDim == slevp1) | (dims.KDim == elev),
        _compute_ppm_quadratic_face_values(p_cc, p_cellhgt_mc_now),
        p_face_in,
    )

    p_face = concat_where(dims.KDim == slev, p_cc, p_face)

    p_face = concat_where(dims.KDim == elevp1, p_cc(KDim - 1), p_face)

    return p_face


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ppm_all_face_values(
    p_cc: fa.CellKField[ta.wpfloat],
    p_cellhgt_mc_now: fa.CellKField[ta.wpfloat],
    p_face_in: fa.CellKField[ta.wpfloat],
    p_face: fa.CellKField[ta.wpfloat],
    slev: gtx.int32,
    elev: gtx.int32,
    slevp1: gtx.int32,
    elevp1: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ppm_all_face_values(
        p_cc=p_cc,
        p_cellhgt_mc_now=p_cellhgt_mc_now,
        p_face_in=p_face_in,
        slev=slev,
        elev=elev,
        slevp1=slevp1,
        elevp1=elevp1,
        out=p_face,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
