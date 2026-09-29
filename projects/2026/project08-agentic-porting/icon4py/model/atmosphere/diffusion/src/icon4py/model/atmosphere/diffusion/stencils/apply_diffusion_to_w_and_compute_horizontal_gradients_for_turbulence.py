# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w import _apply_nabla2_to_w
from icon4py.model.atmosphere.diffusion.stencils.apply_nabla2_to_w_in_upper_damping_layer import (
    _apply_nabla2_to_w_in_upper_damping_layer,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_horizontal_gradients_for_turbulence import (
    _calculate_horizontal_gradients_for_turbulence,
)
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla2_for_w import (
    _calculate_nabla2_for_w,
)
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import C2E2CODim, CellDim, KDim
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    w_old: fa.CellKField[wpfloat],
    type_shear: gtx.int32,
    dwdx: fa.CellKField[vpfloat],
    dwdy: fa.CellKField[vpfloat],
    diff_multfac_w: wpfloat,
    diff_multfac_n2w: fa.KField[wpfloat],
    nrdmax: gtx.int32,
    interior_idx: gtx.int32,
    halo_idx: gtx.int32,
) -> tuple[
    fa.CellKField[wpfloat],
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    dwdx, dwdy = (
        concat_where(
            0 < KDim,
            _calculate_horizontal_gradients_for_turbulence(w_old, geofac_grg_x, geofac_grg_y),
            (dwdx, dwdy),
        )
        if type_shear == 2
        else (dwdx, dwdy)
    )

    z_nabla2_c = _calculate_nabla2_for_w(w_old, geofac_n2s)

    w = concat_where(
        (interior_idx <= CellDim) & (CellDim < halo_idx),
        _apply_nabla2_to_w(area, z_nabla2_c, geofac_n2s, w_old, diff_multfac_w),
        w_old,
    )

    w = concat_where(
        (0 < KDim) & (KDim < nrdmax) & (interior_idx <= CellDim) & (CellDim < halo_idx),
        _apply_nabla2_to_w_in_upper_damping_layer(w, diff_multfac_n2w, area, z_nabla2_c),
        w,
    )

    return w, dwdx, dwdy


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
    area: fa.CellField[wpfloat],
    geofac_n2s: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    geofac_grg_x: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    geofac_grg_y: gtx.Field[gtx.Dims[CellDim, C2E2CODim], wpfloat],
    w_old: fa.CellKField[wpfloat],
    w: fa.CellKField[wpfloat],
    type_shear: gtx.int32,
    dwdx: fa.CellKField[vpfloat],
    dwdy: fa.CellKField[vpfloat],
    diff_multfac_w: wpfloat,
    diff_multfac_n2w: fa.KField[wpfloat],
    nrdmax: gtx.int32,
    interior_idx: gtx.int32,
    halo_idx: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _apply_diffusion_to_w_and_compute_horizontal_gradients_for_turbulence(
        area=area,
        geofac_n2s=geofac_n2s,
        geofac_grg_x=geofac_grg_x,
        geofac_grg_y=geofac_grg_y,
        w_old=w_old,
        type_shear=type_shear,
        dwdx=dwdx,
        dwdy=dwdy,
        diff_multfac_w=diff_multfac_w,
        diff_multfac_n2w=diff_multfac_n2w,
        nrdmax=nrdmax,
        interior_idx=interior_idx,
        halo_idx=halo_idx,
        out=(w, dwdx, dwdy),
        domain={
            CellDim: (horizontal_start, horizontal_end),
            KDim: (vertical_start, vertical_end),
        },
    )
