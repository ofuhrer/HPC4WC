# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import sys

import gt4py.next as gtx
from gt4py.next import astype, broadcast, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2CDim
from icon4py.model.common.type_alias import vpfloat


# TODO(dastrm): this stencil has no test


sys.setrecursionlimit(5500)


@gtx.field_operator
def _compute_ffsl_flux_area_list(
    famask_int: fa.EdgeKField[gtx.int32],
    p_vn: fa.EdgeKField[ta.wpfloat],
    bf_cc_patch1_lon: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    bf_cc_patch1_lat: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    bf_cc_patch2_lon: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    bf_cc_patch2_lat: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    butterfly_idx_patch1_vnpos: fa.EdgeField[gtx.int32],
    butterfly_idx_patch1_vnneg: fa.EdgeField[gtx.int32],
    butterfly_blk_patch1_vnpos: fa.EdgeField[gtx.int32],
    butterfly_blk_patch1_vnneg: fa.EdgeField[gtx.int32],
    butterfly_idx_patch2_vnpos: fa.EdgeField[gtx.int32],
    butterfly_idx_patch2_vnneg: fa.EdgeField[gtx.int32],
    butterfly_blk_patch2_vnpos: fa.EdgeField[gtx.int32],
    butterfly_blk_patch2_vnneg: fa.EdgeField[gtx.int32],
    dreg_patch1_1_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_1_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_2_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_2_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_3_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_3_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_4_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_4_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_1_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_1_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_2_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_2_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_3_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_3_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_4_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_4_lat_vmask: fa.EdgeKField[ta.vpfloat],
) -> tuple[
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[gtx.int32],
    fa.EdgeKField[gtx.int32],
    fa.EdgeKField[gtx.int32],
    fa.EdgeKField[gtx.int32],
]:
    famask_bool = famask_int == 1
    lvn_pos = p_vn >= 0.0
    # Translation of patch 1 and patch 2 in system relative to respective cell
    bf_cc_patch1_lon = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch1_lon[E2CDim(0)], bf_cc_patch1_lon[E2CDim(1)]),
        0.0,
    )
    bf_cc_patch1_lat = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch1_lat[E2CDim(0)], bf_cc_patch1_lat[E2CDim(1)]),
        0.0,
    )
    bf_cc_patch2_lon = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch2_lon[E2CDim(0)], bf_cc_patch2_lon[E2CDim(1)]),
        0.0,
    )
    bf_cc_patch2_lat = where(
        famask_bool,
        where(lvn_pos, bf_cc_patch2_lat[E2CDim(0)], bf_cc_patch2_lat[E2CDim(1)]),
        0.0,
    )

    # patch1 in translated system
    dreg_patch1_1_lon_vmask = dreg_patch1_1_lon_vmask - astype(bf_cc_patch1_lon, vpfloat)
    dreg_patch1_1_lat_vmask = dreg_patch1_1_lat_vmask - astype(bf_cc_patch1_lat, vpfloat)
    dreg_patch1_2_lon_vmask = dreg_patch1_2_lon_vmask - astype(bf_cc_patch1_lon, vpfloat)
    dreg_patch1_2_lat_vmask = dreg_patch1_2_lat_vmask - astype(bf_cc_patch1_lat, vpfloat)
    dreg_patch1_3_lon_vmask = dreg_patch1_3_lon_vmask - astype(bf_cc_patch1_lon, vpfloat)
    dreg_patch1_3_lat_vmask = dreg_patch1_3_lat_vmask - astype(bf_cc_patch1_lat, vpfloat)
    dreg_patch1_4_lon_vmask = dreg_patch1_4_lon_vmask - astype(bf_cc_patch1_lon, vpfloat)
    dreg_patch1_4_lat_vmask = dreg_patch1_4_lat_vmask - astype(bf_cc_patch1_lat, vpfloat)
    # patch2 in translated system
    dreg_patch2_1_lon_vmask = dreg_patch2_1_lon_vmask - astype(bf_cc_patch2_lon, vpfloat)
    dreg_patch2_1_lat_vmask = dreg_patch2_1_lat_vmask - astype(bf_cc_patch2_lat, vpfloat)
    dreg_patch2_2_lon_vmask = dreg_patch2_2_lon_vmask - astype(bf_cc_patch2_lon, vpfloat)
    dreg_patch2_2_lat_vmask = dreg_patch2_2_lat_vmask - astype(bf_cc_patch2_lat, vpfloat)
    dreg_patch2_3_lon_vmask = dreg_patch2_3_lon_vmask - astype(bf_cc_patch2_lon, vpfloat)
    dreg_patch2_3_lat_vmask = dreg_patch2_3_lat_vmask - astype(bf_cc_patch2_lat, vpfloat)
    dreg_patch2_4_lon_vmask = dreg_patch2_4_lon_vmask - astype(bf_cc_patch2_lon, vpfloat)
    dreg_patch2_4_lat_vmask = dreg_patch2_4_lat_vmask - astype(bf_cc_patch2_lat, vpfloat)

    # Store global index of the underlying grid cell
    # Adapt dimensions to fit ofr multiple levels
    butterfly_idx_patch1_vnpos_3d = broadcast(butterfly_idx_patch1_vnpos, (dims.EdgeDim, dims.KDim))
    butterfly_idx_patch1_vnneg_3d = broadcast(butterfly_idx_patch1_vnneg, (dims.EdgeDim, dims.KDim))
    butterfly_idx_patch2_vnpos_3d = broadcast(butterfly_idx_patch2_vnpos, (dims.EdgeDim, dims.KDim))
    butterfly_idx_patch2_vnneg_3d = broadcast(butterfly_idx_patch2_vnneg, (dims.EdgeDim, dims.KDim))
    butterfly_blk_patch1_vnpos_3d = broadcast(butterfly_blk_patch1_vnpos, (dims.EdgeDim, dims.KDim))
    butterfly_blk_patch1_vnneg_3d = broadcast(butterfly_blk_patch1_vnneg, (dims.EdgeDim, dims.KDim))
    butterfly_blk_patch2_vnpos_3d = broadcast(butterfly_blk_patch2_vnpos, (dims.EdgeDim, dims.KDim))
    butterfly_blk_patch2_vnneg_3d = broadcast(butterfly_blk_patch2_vnneg, (dims.EdgeDim, dims.KDim))
    patch1_cell_idx_vmask = where(
        famask_bool,
        where(lvn_pos, butterfly_idx_patch1_vnpos_3d, butterfly_idx_patch1_vnneg_3d),
        0,
    )
    patch2_cell_idx_vmask = where(
        famask_bool,
        where(lvn_pos, butterfly_idx_patch2_vnpos_3d, butterfly_idx_patch2_vnneg_3d),
        0,
    )
    patch1_cell_blk_vmask = where(
        famask_bool,
        where(lvn_pos, butterfly_blk_patch1_vnpos_3d, butterfly_blk_patch1_vnneg_3d),
        0,
    )
    patch2_cell_blk_vmask = where(
        famask_bool,
        where(lvn_pos, butterfly_blk_patch2_vnpos_3d, butterfly_blk_patch2_vnneg_3d),
        0,
    )

    return (
        dreg_patch1_1_lon_vmask,
        dreg_patch1_1_lat_vmask,
        dreg_patch1_2_lon_vmask,
        dreg_patch1_2_lat_vmask,
        dreg_patch1_3_lon_vmask,
        dreg_patch1_3_lat_vmask,
        dreg_patch1_4_lon_vmask,
        dreg_patch1_4_lat_vmask,
        dreg_patch2_1_lon_vmask,
        dreg_patch2_1_lat_vmask,
        dreg_patch2_2_lon_vmask,
        dreg_patch2_2_lat_vmask,
        dreg_patch2_3_lon_vmask,
        dreg_patch2_3_lat_vmask,
        dreg_patch2_4_lon_vmask,
        dreg_patch2_4_lat_vmask,
        patch1_cell_idx_vmask,
        patch1_cell_blk_vmask,
        patch2_cell_idx_vmask,
        patch2_cell_blk_vmask,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_ffsl_flux_area_list(
    famask_int: fa.EdgeKField[gtx.int32],
    p_vn: fa.EdgeKField[ta.wpfloat],
    bf_cc_patch1_lon: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    bf_cc_patch1_lat: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    bf_cc_patch2_lon: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    bf_cc_patch2_lat: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    butterfly_idx_patch1_vnpos: fa.EdgeField[gtx.int32],
    butterfly_idx_patch1_vnneg: fa.EdgeField[gtx.int32],
    butterfly_blk_patch1_vnpos: fa.EdgeField[gtx.int32],
    butterfly_blk_patch1_vnneg: fa.EdgeField[gtx.int32],
    butterfly_idx_patch2_vnpos: fa.EdgeField[gtx.int32],
    butterfly_idx_patch2_vnneg: fa.EdgeField[gtx.int32],
    butterfly_blk_patch2_vnpos: fa.EdgeField[gtx.int32],
    butterfly_blk_patch2_vnneg: fa.EdgeField[gtx.int32],
    dreg_patch1_1_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_1_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_2_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_2_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_3_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_3_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_4_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch1_4_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_1_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_1_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_2_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_2_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_3_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_3_lat_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_4_lon_vmask: fa.EdgeKField[ta.vpfloat],
    dreg_patch2_4_lat_vmask: fa.EdgeKField[ta.vpfloat],
    patch1_cell_idx_vmask: fa.EdgeKField[gtx.int32],
    patch1_cell_blk_vmask: fa.EdgeKField[gtx.int32],
    patch2_cell_idx_vmask: fa.EdgeKField[gtx.int32],
    patch2_cell_blk_vmask: fa.EdgeKField[gtx.int32],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_ffsl_flux_area_list(
        famask_int=famask_int,
        p_vn=p_vn,
        bf_cc_patch1_lon=bf_cc_patch1_lon,
        bf_cc_patch1_lat=bf_cc_patch1_lat,
        bf_cc_patch2_lon=bf_cc_patch2_lon,
        bf_cc_patch2_lat=bf_cc_patch2_lat,
        butterfly_idx_patch1_vnpos=butterfly_idx_patch1_vnpos,
        butterfly_idx_patch1_vnneg=butterfly_idx_patch1_vnneg,
        butterfly_blk_patch1_vnpos=butterfly_blk_patch1_vnpos,
        butterfly_blk_patch1_vnneg=butterfly_blk_patch1_vnneg,
        butterfly_idx_patch2_vnpos=butterfly_idx_patch2_vnpos,
        butterfly_idx_patch2_vnneg=butterfly_idx_patch2_vnneg,
        butterfly_blk_patch2_vnpos=butterfly_blk_patch2_vnpos,
        butterfly_blk_patch2_vnneg=butterfly_blk_patch2_vnneg,
        dreg_patch1_1_lon_vmask=dreg_patch1_1_lon_vmask,
        dreg_patch1_1_lat_vmask=dreg_patch1_1_lat_vmask,
        dreg_patch1_2_lon_vmask=dreg_patch1_2_lon_vmask,
        dreg_patch1_2_lat_vmask=dreg_patch1_2_lat_vmask,
        dreg_patch1_3_lon_vmask=dreg_patch1_3_lon_vmask,
        dreg_patch1_3_lat_vmask=dreg_patch1_3_lat_vmask,
        dreg_patch1_4_lon_vmask=dreg_patch1_4_lon_vmask,
        dreg_patch1_4_lat_vmask=dreg_patch1_4_lat_vmask,
        dreg_patch2_1_lon_vmask=dreg_patch2_1_lon_vmask,
        dreg_patch2_1_lat_vmask=dreg_patch2_1_lat_vmask,
        dreg_patch2_2_lon_vmask=dreg_patch2_2_lon_vmask,
        dreg_patch2_2_lat_vmask=dreg_patch2_2_lat_vmask,
        dreg_patch2_3_lon_vmask=dreg_patch2_3_lon_vmask,
        dreg_patch2_3_lat_vmask=dreg_patch2_3_lat_vmask,
        dreg_patch2_4_lon_vmask=dreg_patch2_4_lon_vmask,
        dreg_patch2_4_lat_vmask=dreg_patch2_4_lat_vmask,
        out=(
            dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask,
            dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask,
            dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask,
            dreg_patch2_1_lon_vmask,
            dreg_patch2_1_lat_vmask,
            dreg_patch2_2_lon_vmask,
            dreg_patch2_2_lat_vmask,
            dreg_patch2_3_lon_vmask,
            dreg_patch2_3_lat_vmask,
            dreg_patch2_4_lon_vmask,
            dreg_patch2_4_lat_vmask,
            patch1_cell_idx_vmask,
            patch1_cell_blk_vmask,
            patch2_cell_idx_vmask,
            patch2_cell_blk_vmask,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
