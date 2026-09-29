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
from icon4py.model.common.type_alias import vpfloat, wpfloat


sys.setrecursionlimit(5500)


# FUNCTIONS
@gtx.field_operator
def ccw(
    p0_lon: fa.EdgeKField[ta.wpfloat],
    p0_lat: fa.EdgeKField[ta.wpfloat],
    p1_lon: fa.EdgeKField[ta.wpfloat],
    p1_lat: fa.EdgeKField[ta.wpfloat],
    p2_lon: fa.EdgeKField[ta.wpfloat],
    p2_lat: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[gtx.int32]:
    """
    Counter-clockwise test.
    Given three points P0, P1, P2, it computes the sign of the cross product of vectors (P1-P0) and (P2-P0):
    cross = dx1*dy2 - dy1*dx2
    If cross > 0, the three points are in counter-clockwise order (returns 1), otherwise clockwise (returns -1).
    """
    dx1dy2 = (p1_lon - p0_lon) * (p2_lat - p0_lat)
    dy1dx2 = (p1_lat - p0_lat) * (p2_lon - p0_lon)
    lccw = dx1dy2 > dy1dx2
    ccw_out = where(lccw, 1, -1)  # 1: counterclockwise, -1: clockwise
    return ccw_out


@gtx.field_operator
def lintersect(
    line1_p1_lon: fa.EdgeKField[ta.wpfloat],
    line1_p1_lat: fa.EdgeKField[ta.wpfloat],
    line1_p2_lon: fa.EdgeKField[ta.wpfloat],
    line1_p2_lat: fa.EdgeKField[ta.wpfloat],
    line2_p1_lon: fa.EdgeKField[ta.wpfloat],
    line2_p1_lat: fa.EdgeKField[ta.wpfloat],
    line2_p2_lon: fa.EdgeKField[ta.wpfloat],
    line2_p2_lat: fa.EdgeKField[ta.wpfloat],
) -> fa.EdgeKField[bool]:
    """
    Line segment intersection test.
    Uses the CCW-based intersection test: two line segments AB and CD intersect if and only if:
    - A, B separate C and D (i.e. CCW(A,B,C) and CCW(A,B,D) have opposite signs → product = -1)
    - C, D separate A and B (i.e. CCW(C,D,A) and CCW(C,D,B) have opposite signs → product = -1)
    Both products must equal -1 simultaneously (sum = -2) for an intersection to occur.
    """
    intersect1 = ccw(
        p0_lon=line1_p1_lon,
        p0_lat=line1_p1_lat,
        p1_lon=line1_p2_lon,
        p1_lat=line1_p2_lat,
        p2_lon=line2_p1_lon,
        p2_lat=line2_p1_lat,
    ) * ccw(
        p0_lon=line1_p1_lon,
        p0_lat=line1_p1_lat,
        p1_lon=line1_p2_lon,
        p1_lat=line1_p2_lat,
        p2_lon=line2_p2_lon,
        p2_lat=line2_p2_lat,
    )
    intersect2 = ccw(
        p0_lon=line2_p1_lon,
        p0_lat=line2_p1_lat,
        p1_lon=line2_p2_lon,
        p1_lat=line2_p2_lat,
        p2_lon=line1_p1_lon,
        p2_lat=line1_p1_lat,
    ) * ccw(
        p0_lon=line2_p1_lon,
        p0_lat=line2_p1_lat,
        p1_lon=line2_p2_lon,
        p1_lat=line2_p2_lat,
        p2_lon=line1_p2_lon,
        p2_lat=line1_p2_lat,
    )
    lintersect_out = (intersect1 + intersect2) == -2

    return lintersect_out


# Compute intersection point of two lines in 2D
@gtx.field_operator
def line_intersect(
    line1_p1_lon: fa.EdgeKField[ta.wpfloat],
    line1_p1_lat: fa.EdgeKField[ta.wpfloat],
    line1_p2_lon: fa.EdgeKField[ta.wpfloat],
    line1_p2_lat: fa.EdgeKField[ta.wpfloat],
    line2_p1_lon: fa.EdgeKField[ta.wpfloat],
    line2_p1_lat: fa.EdgeKField[ta.wpfloat],
    line2_p2_lon: fa.EdgeKField[ta.wpfloat],
    line2_p2_lat: fa.EdgeKField[ta.wpfloat],
) -> tuple[fa.EdgeKField[ta.vpfloat], fa.EdgeKField[ta.vpfloat]]:
    # avoid division with zero
    d1 = line1_p2_lon - line1_p1_lon
    d1 = where(d1 != 0.0, d1, line1_p2_lon)

    d2 = line2_p2_lon - line2_p1_lon
    d2 = where(d2 != 0.0, d2, line2_p2_lon)

    m1 = (line1_p2_lat - line1_p1_lat) / d1
    m2 = (line2_p2_lat - line2_p1_lat) / d2

    intersect_1 = (line2_p1_lat - line1_p1_lat + m1 * line1_p1_lon - m2 * line2_p1_lon) / (m1 - m2)
    intersect_2 = line1_p1_lat + m1 * (intersect_1 - line1_p1_lon)

    return astype(intersect_1, vpfloat), astype(intersect_2, vpfloat)


@gtx.field_operator
def _prepare_ffsl_flux_area_patches_list(  # noqa: PLR0915 [too-many-statements]
    famask_int: fa.EdgeKField[gtx.int32],
    p_vn: fa.EdgeKField[ta.wpfloat],
    ptr_v3_lon: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ptr_v3_lat: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    tangent_orientation_dsl: fa.EdgeField[ta.wpfloat],
    dreg_patch0_1_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_1_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_2_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_2_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_3_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_3_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_4_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_4_lat_dsl: fa.EdgeKField[ta.vpfloat],
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
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    arrival_pts_1_lon_dsl = dreg_patch0_1_lon_dsl
    arrival_pts_1_lat_dsl = dreg_patch0_1_lat_dsl
    arrival_pts_2_lon_dsl = dreg_patch0_2_lon_dsl
    arrival_pts_2_lat_dsl = dreg_patch0_2_lat_dsl
    depart_pts_1_lon_dsl = dreg_patch0_4_lon_dsl  # indices have to be switched so that dep 1 belongs to arr 1 (and d2->a2)
    depart_pts_1_lat_dsl = dreg_patch0_4_lat_dsl
    depart_pts_2_lon_dsl = dreg_patch0_3_lon_dsl
    depart_pts_2_lat_dsl = dreg_patch0_3_lat_dsl

    lvn_pos = p_vn >= 0.0

    # get flux area departure-line segment
    fl_line_p1_lon = astype(depart_pts_1_lon_dsl, wpfloat)
    fl_line_p1_lat = astype(depart_pts_1_lat_dsl, wpfloat)
    fl_line_p2_lon = astype(depart_pts_2_lon_dsl, wpfloat)
    fl_line_p2_lat = astype(depart_pts_2_lat_dsl, wpfloat)

    # get triangle edge 1 (A1V3)
    tri_line1_p1_lon = astype(arrival_pts_1_lon_dsl, wpfloat)
    tri_line1_p1_lat = astype(arrival_pts_1_lat_dsl, wpfloat)
    tri_line1_p2_lon = where(lvn_pos, ptr_v3_lon[E2CDim(0)], ptr_v3_lon[E2CDim(1)])
    tri_line1_p2_lat = where(lvn_pos, ptr_v3_lat[E2CDim(0)], ptr_v3_lat[E2CDim(1)])

    # get triangle edge 2 (A2V3)
    tri_line2_p1_lon = astype(arrival_pts_2_lon_dsl, wpfloat)
    tri_line2_p1_lat = astype(arrival_pts_2_lat_dsl, wpfloat)
    tri_line2_p2_lon = where(lvn_pos, ptr_v3_lon[E2CDim(0)], ptr_v3_lon[E2CDim(1)])
    tri_line2_p2_lat = where(lvn_pos, ptr_v3_lat[E2CDim(0)], ptr_v3_lat[E2CDim(1)])

    # Create first mask does departure-line segment intersects with A1V3
    lintersect_line1 = lintersect(
        line1_p1_lon=fl_line_p1_lon,
        line1_p1_lat=fl_line_p1_lat,
        line1_p2_lon=fl_line_p2_lon,
        line1_p2_lat=fl_line_p2_lat,
        line2_p1_lon=tri_line1_p1_lon,
        line2_p1_lat=tri_line1_p1_lat,
        line2_p2_lon=tri_line1_p2_lon,
        line2_p2_lat=tri_line1_p2_lat,
    )
    # Create first mask does departure-line segment intersects with A2V3
    lintersect_line2 = lintersect(
        line1_p1_lon=fl_line_p1_lon,
        line1_p1_lat=fl_line_p1_lat,
        line1_p2_lon=fl_line_p2_lon,
        line1_p2_lat=fl_line_p2_lat,
        line2_p1_lon=tri_line2_p1_lon,
        line2_p1_lat=tri_line2_p1_lat,
        line2_p2_lon=tri_line2_p2_lon,
        line2_p2_lat=tri_line2_p2_lat,
    )

    lvn_sys_pos = (p_vn * broadcast(tangent_orientation_dsl, (dims.EdgeDim, dims.KDim))) >= 0.0
    famask_bool = famask_int == 1
    # ------------------------------------------------- Case 1
    mask_case1 = lintersect_line1 & lintersect_line2 & famask_bool
    ps1_x, ps1_y = line_intersect(
        line1_p1_lon=fl_line_p1_lon,
        line1_p1_lat=fl_line_p1_lat,
        line1_p2_lon=fl_line_p2_lon,
        line1_p2_lat=fl_line_p2_lat,
        line2_p1_lon=tri_line1_p1_lon,
        line2_p1_lat=tri_line1_p1_lat,
        line2_p2_lon=tri_line1_p2_lon,
        line2_p2_lat=tri_line1_p2_lat,
    )
    ps2_x, ps2_y = line_intersect(
        line1_p1_lon=fl_line_p1_lon,
        line1_p1_lat=fl_line_p1_lat,
        line1_p2_lon=fl_line_p2_lon,
        line1_p2_lat=fl_line_p2_lat,
        line2_p1_lon=tri_line2_p1_lon,
        line2_p1_lat=tri_line2_p1_lat,
        line2_p2_lon=tri_line2_p2_lon,
        line2_p2_lat=tri_line2_p2_lat,
    )

    # Case 1 - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case1, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case1, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case1, where(lvn_sys_pos, arrival_pts_2_lon_dsl, ps1_x), dreg_patch0_2_lon_dsl
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case1, where(lvn_sys_pos, arrival_pts_2_lat_dsl, ps1_y), dreg_patch0_2_lat_dsl
    )
    dreg_patch0_3_lon_dsl = where(mask_case1, ps2_x, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = where(mask_case1, ps2_y, dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = where(
        mask_case1, where(lvn_sys_pos, ps1_x, arrival_pts_2_lon_dsl), dreg_patch0_4_lon_dsl
    )
    dreg_patch0_4_lat_dsl = where(
        mask_case1, where(lvn_sys_pos, ps1_y, arrival_pts_2_lat_dsl), dreg_patch0_4_lat_dsl
    )
    # Case 1 - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case1, arrival_pts_1_lon_dsl, 0.0)
    dreg_patch1_1_lat_vmask = where(mask_case1, arrival_pts_1_lat_dsl, 0.0)
    dreg_patch1_4_lon_vmask = where(mask_case1, arrival_pts_1_lon_dsl, 0.0)
    dreg_patch1_4_lat_vmask = where(mask_case1, arrival_pts_1_lat_dsl, 0.0)
    dreg_patch1_2_lon_vmask = where(
        mask_case1, where(lvn_sys_pos, ps1_x, depart_pts_1_lon_dsl), 0.0
    )
    dreg_patch1_2_lat_vmask = where(
        mask_case1, where(lvn_sys_pos, ps1_y, depart_pts_1_lat_dsl), 0.0
    )
    dreg_patch1_3_lon_vmask = where(
        mask_case1, where(lvn_sys_pos, depart_pts_1_lon_dsl, ps1_x), 0.0
    )
    dreg_patch1_3_lat_vmask = where(
        mask_case1, where(lvn_sys_pos, depart_pts_1_lat_dsl, ps1_y), 0.0
    )
    # Case 1 - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case1, arrival_pts_2_lon_dsl, 0.0)
    dreg_patch2_1_lat_vmask = where(mask_case1, arrival_pts_2_lat_dsl, 0.0)
    dreg_patch2_4_lon_vmask = where(mask_case1, arrival_pts_2_lon_dsl, 0.0)
    dreg_patch2_4_lat_vmask = where(mask_case1, arrival_pts_2_lat_dsl, 0.0)
    dreg_patch2_2_lon_vmask = where(
        mask_case1, where(lvn_sys_pos, depart_pts_2_lon_dsl, ps2_x), 0.0
    )
    dreg_patch2_2_lat_vmask = where(
        mask_case1, where(lvn_sys_pos, depart_pts_2_lat_dsl, ps2_y), 0.0
    )
    dreg_patch2_3_lon_vmask = where(
        mask_case1, where(lvn_sys_pos, ps2_x, depart_pts_2_lon_dsl), 0.0
    )
    dreg_patch2_3_lat_vmask = where(
        mask_case1, where(lvn_sys_pos, ps2_y, depart_pts_2_lat_dsl), 0.0
    )

    # ------------------------------------------------- Case 2a
    mask_case2a = lintersect_line1 & (~lintersect_line2) & famask_bool
    # Case 2a - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case2a, where(lvn_sys_pos, arrival_pts_2_lon_dsl, ps1_x), dreg_patch0_2_lon_dsl
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case2a, where(lvn_sys_pos, arrival_pts_2_lat_dsl, ps1_y), dreg_patch0_2_lat_dsl
    )
    dreg_patch0_3_lon_dsl = where(mask_case2a, depart_pts_2_lon_dsl, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = where(mask_case2a, depart_pts_2_lat_dsl, dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = where(
        mask_case2a, where(lvn_sys_pos, ps1_x, arrival_pts_2_lon_dsl), dreg_patch0_4_lon_dsl
    )
    dreg_patch0_4_lat_dsl = where(
        mask_case2a, where(lvn_sys_pos, ps1_y, arrival_pts_2_lat_dsl), dreg_patch0_4_lat_dsl
    )
    # Case 2a - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch1_1_lat_vmask)
    dreg_patch1_4_lon_vmask = where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch1_4_lat_vmask)
    dreg_patch1_2_lon_vmask = where(
        mask_case2a, where(lvn_sys_pos, ps1_x, depart_pts_1_lon_dsl), dreg_patch1_2_lon_vmask
    )
    dreg_patch1_2_lat_vmask = where(
        mask_case2a, where(lvn_sys_pos, ps1_y, depart_pts_1_lat_dsl), dreg_patch1_2_lat_vmask
    )
    dreg_patch1_3_lon_vmask = where(
        mask_case2a, where(lvn_sys_pos, depart_pts_1_lon_dsl, ps1_x), dreg_patch1_3_lon_vmask
    )
    dreg_patch1_3_lat_vmask = where(
        mask_case2a, where(lvn_sys_pos, depart_pts_1_lat_dsl, ps1_y), dreg_patch1_3_lat_vmask
    )
    # Case 2a - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case2a, 0.0, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case2a, 0.0, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = where(mask_case2a, 0.0, dreg_patch2_2_lon_vmask)
    dreg_patch2_2_lat_vmask = where(mask_case2a, 0.0, dreg_patch2_2_lat_vmask)
    dreg_patch2_3_lon_vmask = where(mask_case2a, 0.0, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = where(mask_case2a, 0.0, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = where(mask_case2a, 0.0, dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = where(mask_case2a, 0.0, dreg_patch2_4_lat_vmask)

    # -------------------------------------------------- Case 2b
    mask_case2b = lintersect_line2 & (~lintersect_line1) & famask_bool
    # Case 2b - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case2b, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case2b, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case2b,
        where(lvn_sys_pos, arrival_pts_2_lon_dsl, depart_pts_1_lon_dsl),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case2b,
        where(lvn_sys_pos, arrival_pts_2_lat_dsl, depart_pts_1_lat_dsl),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = where(mask_case2b, ps2_x, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = where(mask_case2b, ps2_y, dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = where(
        mask_case2b,
        where(lvn_sys_pos, depart_pts_1_lon_dsl, arrival_pts_2_lon_dsl),
        dreg_patch0_4_lon_dsl,
    )
    dreg_patch0_4_lat_dsl = where(
        mask_case2b,
        where(lvn_sys_pos, depart_pts_1_lat_dsl, arrival_pts_2_lat_dsl),
        dreg_patch0_4_lat_dsl,
    )
    # Case 2b - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case2b, 0.0, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case2b, 0.0, dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = where(mask_case2b, 0.0, dreg_patch1_2_lon_vmask)
    dreg_patch1_2_lat_vmask = where(mask_case2b, 0.0, dreg_patch1_2_lat_vmask)
    dreg_patch1_3_lon_vmask = where(mask_case2b, 0.0, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = where(mask_case2b, 0.0, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = where(mask_case2b, 0.0, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = where(mask_case2b, 0.0, dreg_patch1_4_lat_vmask)
    # Case 2b - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case2b, arrival_pts_2_lon_dsl, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case2b, arrival_pts_2_lat_dsl, dreg_patch2_1_lat_vmask)
    dreg_patch2_4_lon_vmask = where(mask_case2b, arrival_pts_2_lon_dsl, dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = where(mask_case2b, arrival_pts_2_lat_dsl, dreg_patch2_4_lat_vmask)
    dreg_patch2_2_lon_vmask = where(
        mask_case2b, where(lvn_sys_pos, depart_pts_2_lon_dsl, ps2_x), dreg_patch2_2_lon_vmask
    )
    dreg_patch2_2_lat_vmask = where(
        mask_case2b, where(lvn_sys_pos, depart_pts_2_lat_dsl, ps2_y), dreg_patch2_2_lat_vmask
    )
    dreg_patch2_3_lon_vmask = where(
        mask_case2b, where(lvn_sys_pos, ps2_x, depart_pts_2_lon_dsl), dreg_patch2_3_lon_vmask
    )
    dreg_patch2_3_lat_vmask = where(
        mask_case2b, where(lvn_sys_pos, ps2_y, depart_pts_2_lat_dsl), dreg_patch2_3_lat_vmask
    )

    # flux area edge 1 and 2
    fl_e1_p1_lon = astype(arrival_pts_1_lon_dsl, wpfloat)
    fl_e1_p1_lat = astype(arrival_pts_1_lat_dsl, wpfloat)
    fl_e1_p2_lon = astype(depart_pts_1_lon_dsl, wpfloat)
    fl_e1_p2_lat = astype(depart_pts_1_lat_dsl, wpfloat)
    fl_e2_p1_lon = astype(arrival_pts_2_lon_dsl, wpfloat)
    fl_e2_p1_lat = astype(arrival_pts_2_lat_dsl, wpfloat)
    fl_e2_p2_lon = astype(depart_pts_2_lon_dsl, wpfloat)
    fl_e2_p2_lat = astype(depart_pts_2_lat_dsl, wpfloat)

    # ----------------------------------------------- Case 3a
    # Check whether flux area edge 2 intersects with triangle edge 1
    lintersect_e2_line1 = lintersect(
        line1_p1_lon=fl_e2_p1_lon,
        line1_p1_lat=fl_e2_p1_lat,
        line1_p2_lon=fl_e2_p2_lon,
        line1_p2_lat=fl_e2_p2_lat,
        line2_p1_lon=tri_line1_p1_lon,
        line2_p1_lat=tri_line1_p1_lat,
        line2_p2_lon=tri_line1_p2_lon,
        line2_p2_lat=tri_line1_p2_lat,
    )
    mask_case3a = lintersect_e2_line1 & famask_bool
    pi1_x, pi1_y = line_intersect(
        line1_p1_lon=fl_e2_p1_lon,
        line1_p1_lat=fl_e2_p1_lat,
        line1_p2_lon=fl_e2_p2_lon,
        line1_p2_lat=fl_e2_p2_lat,
        line2_p1_lon=tri_line1_p1_lon,
        line2_p1_lat=tri_line1_p1_lat,
        line2_p2_lon=tri_line1_p2_lon,
        line2_p2_lat=tri_line1_p2_lat,
    )
    # Case 3a - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case3a, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case3a, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case3a,
        where(lvn_sys_pos, arrival_pts_2_lon_dsl, depart_pts_1_lon_dsl),
        dreg_patch0_2_lon_dsl,
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case3a,
        where(lvn_sys_pos, arrival_pts_2_lat_dsl, depart_pts_1_lat_dsl),
        dreg_patch0_2_lat_dsl,
    )
    dreg_patch0_3_lon_dsl = where(mask_case3a, ps2_x, dreg_patch0_3_lon_dsl)
    dreg_patch0_3_lat_dsl = where(mask_case3a, ps2_y, dreg_patch0_3_lat_dsl)
    dreg_patch0_4_lon_dsl = where(
        mask_case3a,
        where(lvn_sys_pos, depart_pts_1_lon_dsl, arrival_pts_2_lon_dsl),
        dreg_patch0_4_lon_dsl,
    )
    dreg_patch0_4_lat_dsl = where(
        mask_case3a,
        where(lvn_sys_pos, depart_pts_1_lat_dsl, arrival_pts_2_lat_dsl),
        dreg_patch0_4_lat_dsl,
    )
    # Case 3a - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case3a, arrival_pts_1_lon_dsl, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case3a, arrival_pts_1_lat_dsl, dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = where(
        mask_case3a,
        where(lvn_sys_pos, pi1_x, depart_pts_2_lon_dsl),
        dreg_patch1_2_lon_vmask,
    )
    dreg_patch1_2_lat_vmask = where(
        mask_case3a,
        where(lvn_sys_pos, pi1_y, depart_pts_2_lat_dsl),
        dreg_patch1_2_lat_vmask,
    )
    dreg_patch1_3_lon_vmask = where(mask_case3a, depart_pts_1_lon_dsl, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = where(mask_case3a, depart_pts_1_lat_dsl, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = where(
        mask_case3a,
        where(lvn_sys_pos, depart_pts_1_lon_dsl, pi1_x),
        dreg_patch1_4_lon_vmask,
    )
    dreg_patch1_4_lat_vmask = where(
        mask_case3a,
        where(lvn_sys_pos, depart_pts_1_lat_dsl, pi1_y),
        dreg_patch1_4_lat_vmask,
    )
    # Case 3a - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case3a, 0.0, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case3a, 0.0, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = where(mask_case3a, 0.0, dreg_patch2_2_lon_vmask)
    dreg_patch2_2_lat_vmask = where(mask_case3a, 0.0, dreg_patch2_2_lat_vmask)
    dreg_patch2_3_lon_vmask = where(mask_case3a, 0.0, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = where(mask_case3a, 0.0, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = where(mask_case3a, 0.0, dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = where(mask_case3a, 0.0, dreg_patch2_4_lat_vmask)

    # ------------------------------------------------ Case 3b
    # Check whether flux area edge 1 intersects with triangle edge 2
    lintersect_e1_line2 = lintersect(
        line1_p1_lon=fl_e1_p1_lon,
        line1_p1_lat=fl_e1_p1_lat,
        line1_p2_lon=fl_e1_p2_lon,
        line1_p2_lat=fl_e1_p2_lat,
        line2_p1_lon=tri_line2_p1_lon,
        line2_p1_lat=tri_line2_p1_lat,
        line2_p2_lon=tri_line2_p2_lon,
        line2_p2_lat=tri_line2_p2_lat,
    )
    mask_case3b = lintersect_e1_line2 & famask_bool
    pi2_x, pi2_y = line_intersect(
        line1_p1_lon=fl_e1_p1_lon,
        line1_p1_lat=fl_e1_p1_lat,
        line1_p2_lon=fl_e1_p2_lon,
        line1_p2_lat=fl_e1_p2_lat,
        line2_p1_lon=tri_line2_p1_lon,
        line2_p1_lat=tri_line2_p1_lat,
        line2_p2_lon=tri_line2_p2_lon,
        line2_p2_lat=tri_line2_p2_lat,
    )
    # Case 3b - patch 0
    dreg_patch0_1_lon_dsl = where(mask_case3b, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
    dreg_patch0_1_lat_dsl = where(mask_case3b, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
    dreg_patch0_4_lon_dsl = where(mask_case3b, arrival_pts_1_lon_dsl, dreg_patch0_4_lon_dsl)
    dreg_patch0_4_lat_dsl = where(mask_case3b, arrival_pts_1_lat_dsl, dreg_patch0_4_lat_dsl)
    dreg_patch0_2_lon_dsl = where(
        mask_case3b, where(lvn_sys_pos, arrival_pts_2_lon_dsl, pi2_x), dreg_patch0_2_lon_dsl
    )
    dreg_patch0_2_lat_dsl = where(
        mask_case3b, where(lvn_sys_pos, arrival_pts_2_lat_dsl, pi2_y), dreg_patch0_2_lat_dsl
    )
    dreg_patch0_3_lon_dsl = where(
        mask_case3b, where(lvn_sys_pos, pi2_x, arrival_pts_2_lon_dsl), dreg_patch0_3_lon_dsl
    )
    dreg_patch0_3_lat_dsl = where(
        mask_case3b, where(lvn_sys_pos, pi2_y, arrival_pts_2_lat_dsl), dreg_patch0_3_lat_dsl
    )
    # Case 3b - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case3b, 0.0, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case3b, 0.0, dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = where(mask_case3b, 0.0, dreg_patch1_2_lon_vmask)
    dreg_patch1_2_lat_vmask = where(mask_case3b, 0.0, dreg_patch1_2_lat_vmask)
    dreg_patch1_3_lon_vmask = where(mask_case3b, 0.0, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = where(mask_case3b, 0.0, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = where(mask_case3b, 0.0, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = where(mask_case3b, 0.0, dreg_patch1_4_lat_vmask)
    # Case 3b - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case3b, arrival_pts_2_lon_dsl, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case3b, arrival_pts_2_lat_dsl, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = where(
        mask_case3b, where(lvn_sys_pos, depart_pts_2_lon_dsl, pi2_x), dreg_patch2_2_lon_vmask
    )
    dreg_patch2_2_lat_vmask = where(
        mask_case3b, where(lvn_sys_pos, depart_pts_2_lat_dsl, pi2_y), dreg_patch2_2_lat_vmask
    )
    dreg_patch2_3_lon_vmask = where(mask_case3b, depart_pts_1_lon_dsl, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = where(mask_case3b, depart_pts_1_lat_dsl, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = where(
        mask_case3b, where(lvn_sys_pos, pi2_x, depart_pts_2_lon_dsl), dreg_patch2_4_lon_vmask
    )
    dreg_patch2_4_lat_vmask = where(
        mask_case3b, where(lvn_sys_pos, pi2_y, depart_pts_2_lat_dsl), dreg_patch2_4_lat_vmask
    )

    # --------------------------------------------- Case 4
    # NB: Next line acts as the "ELSE IF", indices that already previously matched one of the above conditions
    # can't be overwritten by this new condition.
    indices_previously_matched = mask_case3b | mask_case3a | mask_case2b | mask_case2a | mask_case1
    #    mask_case4 = (abs(p_vn) < 0.1) & famask_bool & (not indices_previously_matched) we insert also the error indices
    mask_case4 = famask_bool & (~indices_previously_matched)
    # Case 4 - patch 0 - no change
    # Case 4 - patch 1
    dreg_patch1_1_lon_vmask = where(mask_case4, 0.0, dreg_patch1_1_lon_vmask)
    dreg_patch1_1_lat_vmask = where(mask_case4, 0.0, dreg_patch1_1_lat_vmask)
    dreg_patch1_2_lon_vmask = where(mask_case4, 0.0, dreg_patch1_2_lon_vmask)
    dreg_patch1_2_lat_vmask = where(mask_case4, 0.0, dreg_patch1_2_lat_vmask)
    dreg_patch1_3_lon_vmask = where(mask_case4, 0.0, dreg_patch1_3_lon_vmask)
    dreg_patch1_3_lat_vmask = where(mask_case4, 0.0, dreg_patch1_3_lat_vmask)
    dreg_patch1_4_lon_vmask = where(mask_case4, 0.0, dreg_patch1_4_lon_vmask)
    dreg_patch1_4_lat_vmask = where(mask_case4, 0.0, dreg_patch1_4_lat_vmask)
    # Case 4 - patch 2
    dreg_patch2_1_lon_vmask = where(mask_case4, 0.0, dreg_patch2_1_lon_vmask)
    dreg_patch2_1_lat_vmask = where(mask_case4, 0.0, dreg_patch2_1_lat_vmask)
    dreg_patch2_2_lon_vmask = where(mask_case4, 0.0, dreg_patch2_2_lon_vmask)
    dreg_patch2_2_lat_vmask = where(mask_case4, 0.0, dreg_patch2_2_lat_vmask)
    dreg_patch2_3_lon_vmask = where(mask_case4, 0.0, dreg_patch2_3_lon_vmask)
    dreg_patch2_3_lat_vmask = where(mask_case4, 0.0, dreg_patch2_3_lat_vmask)
    dreg_patch2_4_lon_vmask = where(mask_case4, 0.0, dreg_patch2_4_lon_vmask)
    dreg_patch2_4_lat_vmask = where(mask_case4, 0.0, dreg_patch2_4_lat_vmask)

    return (
        dreg_patch0_1_lon_dsl,
        dreg_patch0_1_lat_dsl,
        dreg_patch0_2_lon_dsl,
        dreg_patch0_2_lat_dsl,
        dreg_patch0_3_lon_dsl,
        dreg_patch0_3_lat_dsl,
        dreg_patch0_4_lon_dsl,
        dreg_patch0_4_lat_dsl,
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
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def prepare_ffsl_flux_area_patches_list(
    famask_int: fa.EdgeKField[gtx.int32],
    p_vn: fa.EdgeKField[ta.wpfloat],
    ptr_v3_lon: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    ptr_v3_lat: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2CDim], ta.wpfloat],
    tangent_orientation_dsl: fa.EdgeField[ta.wpfloat],
    dreg_patch0_1_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_1_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_2_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_2_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_3_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_3_lat_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_4_lon_dsl: fa.EdgeKField[ta.vpfloat],
    dreg_patch0_4_lat_dsl: fa.EdgeKField[ta.vpfloat],
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
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _prepare_ffsl_flux_area_patches_list(
        famask_int=famask_int,
        p_vn=p_vn,
        ptr_v3_lon=ptr_v3_lon,
        ptr_v3_lat=ptr_v3_lat,
        tangent_orientation_dsl=tangent_orientation_dsl,
        dreg_patch0_1_lon_dsl=dreg_patch0_1_lon_dsl,
        dreg_patch0_1_lat_dsl=dreg_patch0_1_lat_dsl,
        dreg_patch0_2_lon_dsl=dreg_patch0_2_lon_dsl,
        dreg_patch0_2_lat_dsl=dreg_patch0_2_lat_dsl,
        dreg_patch0_3_lon_dsl=dreg_patch0_3_lon_dsl,
        dreg_patch0_3_lat_dsl=dreg_patch0_3_lat_dsl,
        dreg_patch0_4_lon_dsl=dreg_patch0_4_lon_dsl,
        dreg_patch0_4_lat_dsl=dreg_patch0_4_lat_dsl,
        out=(
            dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl,
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
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
