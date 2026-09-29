# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.utils.data_allocation as data_alloc
from icon4py.model.atmosphere.advection.stencils.prepare_ffsl_flux_area_patches_list import (
    prepare_ffsl_flux_area_patches_list,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import stencil_tests


def _ccw_numpy(
    *,
    p0_lon,
    p0_lat,
    p1_lon,
    p1_lat,
    p2_lon,
    p2_lat,
):
    """
    Counter-clockwise test.
    Given three points P0, P1, P2, it computes the sign of the cross product of vectors (P1-P0) and (P2-P0):
    cross = dx1*dy2 - dy1*dx2
    If cross > 0, the three points are in counter-clockwise order (returns 1), otherwise clockwise (returns -1).
    """
    dx1 = p1_lon - p0_lon
    dy1 = p1_lat - p0_lat

    dx2 = p2_lon - p0_lon
    dy2 = p2_lat - p0_lat

    dx1dy2 = dx1 * dy2
    dy1dx2 = dy1 * dx2

    lccw = np.where(dx1dy2 > dy1dx2, True, False)
    ccw_out = np.where(lccw, 1, -1)  # 1: counterclockwise, -1: clockwise
    return ccw_out


def _lintersect_numpy(
    *,
    line1_p1_lon,
    line1_p1_lat,
    line1_p2_lon,
    line1_p2_lat,
    line2_p1_lon,
    line2_p1_lat,
    line2_p2_lon,
    line2_p2_lat,
):
    """
    Line segment intersection test.
    Uses the CCW-based intersection test: two line segments AB and CD intersect if and only if:
    - A, B separate C and D (i.e. CCW(A,B,C) and CCW(A,B,D) have opposite signs → product = -1)
    - C, D separate A and B (i.e. CCW(C,D,A) and CCW(C,D,B) have opposite signs → product = -1)
    Both products must equal -1 simultaneously (sum = -2) for an intersection to occur.
    """
    intersect1 = _ccw_numpy(
        p0_lon=line1_p1_lon,
        p0_lat=line1_p1_lat,
        p1_lon=line1_p2_lon,
        p1_lat=line1_p2_lat,
        p2_lon=line2_p1_lon,
        p2_lat=line2_p1_lat,
    ) * _ccw_numpy(
        p0_lon=line1_p1_lon,
        p0_lat=line1_p1_lat,
        p1_lon=line1_p2_lon,
        p1_lat=line1_p2_lat,
        p2_lon=line2_p2_lon,
        p2_lat=line2_p2_lat,
    )
    intersect2 = _ccw_numpy(
        p0_lon=line2_p1_lon,
        p0_lat=line2_p1_lat,
        p1_lon=line2_p2_lon,
        p1_lat=line2_p2_lat,
        p2_lon=line1_p1_lon,
        p2_lat=line1_p1_lat,
    ) * _ccw_numpy(
        p0_lon=line2_p1_lon,
        p0_lat=line2_p1_lat,
        p1_lon=line2_p2_lon,
        p1_lat=line2_p2_lat,
        p2_lon=line1_p2_lon,
        p2_lat=line1_p2_lat,
    )
    lintersect_out = np.where((intersect1 + intersect2) == -2, True, False)

    return lintersect_out


# Compute intersection point of two lines in 2D
def _line_intersect_numpy(
    *,
    line1_p1_lon,
    line1_p1_lat,
    line1_p2_lon,
    line1_p2_lat,
    line2_p1_lon,
    line2_p1_lat,
    line2_p2_lon,
    line2_p2_lat,
):
    d1 = line1_p2_lon - line1_p1_lon
    d1 = np.where(d1 != 0.0, d1, line1_p2_lon)

    d2 = line2_p2_lon - line2_p1_lon
    d2 = np.where(d2 != 0.0, d2, line2_p2_lon)

    m1 = (line1_p2_lat - line1_p1_lat) / d1
    m2 = (line2_p2_lat - line2_p1_lat) / d2

    intersect_1 = (line2_p1_lat - line1_p1_lat + m1 * line1_p1_lon - m2 * line2_p1_lon) / (m1 - m2)
    intersect_2 = line1_p1_lat + m1 * (intersect_1 - line1_p1_lon)

    return intersect_1, intersect_2


@pytest.mark.gtfn_too_slow
class TestPrepareFfslFluxAreaPatchesList(stencil_tests.StencilTest):
    PROGRAM = prepare_ffsl_flux_area_patches_list
    OUTPUTS = (
        "dreg_patch0_1_lon_dsl",
        "dreg_patch0_1_lat_dsl",
        "dreg_patch0_2_lon_dsl",
        "dreg_patch0_2_lat_dsl",
        "dreg_patch0_3_lon_dsl",
        "dreg_patch0_3_lat_dsl",
        "dreg_patch0_4_lon_dsl",
        "dreg_patch0_4_lat_dsl",
        "dreg_patch1_1_lon_vmask",
        "dreg_patch1_1_lat_vmask",
        "dreg_patch1_2_lon_vmask",
        "dreg_patch1_2_lat_vmask",
        "dreg_patch1_3_lon_vmask",
        "dreg_patch1_3_lat_vmask",
        "dreg_patch1_4_lon_vmask",
        "dreg_patch1_4_lat_vmask",
        "dreg_patch2_1_lon_vmask",
        "dreg_patch2_1_lat_vmask",
        "dreg_patch2_2_lon_vmask",
        "dreg_patch2_2_lat_vmask",
        "dreg_patch2_3_lon_vmask",
        "dreg_patch2_3_lat_vmask",
        "dreg_patch2_4_lon_vmask",
        "dreg_patch2_4_lat_vmask",
    )

    @staticmethod
    def _generate_flux_area_geometry(
        *,
        dreg_patch0_1_lon_dsl,
        dreg_patch0_1_lat_dsl,
        dreg_patch0_2_lon_dsl,
        dreg_patch0_2_lat_dsl,
        dreg_patch0_3_lon_dsl,
        dreg_patch0_3_lat_dsl,
        dreg_patch0_4_lon_dsl,
        dreg_patch0_4_lat_dsl,
        p_vn,
        ptr_v3_lon_e,
        ptr_v3_lat_e,
    ):
        arrival_pts_1_lon_dsl = dreg_patch0_1_lon_dsl
        arrival_pts_1_lat_dsl = dreg_patch0_1_lat_dsl
        arrival_pts_2_lon_dsl = dreg_patch0_2_lon_dsl
        arrival_pts_2_lat_dsl = dreg_patch0_2_lat_dsl
        depart_pts_1_lon_dsl = dreg_patch0_4_lon_dsl
        depart_pts_1_lat_dsl = dreg_patch0_4_lat_dsl
        depart_pts_2_lon_dsl = dreg_patch0_3_lon_dsl
        depart_pts_2_lat_dsl = dreg_patch0_3_lat_dsl

        lvn_pos = p_vn >= 0.0

        fl_line_p1_lon = depart_pts_1_lon_dsl
        fl_line_p1_lat = depart_pts_1_lat_dsl
        fl_line_p2_lon = depart_pts_2_lon_dsl
        fl_line_p2_lat = depart_pts_2_lat_dsl

        tri_line1_p1_lon = arrival_pts_1_lon_dsl
        tri_line1_p1_lat = arrival_pts_1_lat_dsl
        tri_line1_p2_lon = np.where(
            lvn_pos,
            np.broadcast_to(ptr_v3_lon_e[:, 0], p_vn.shape),
            np.broadcast_to(ptr_v3_lon_e[:, 1], p_vn.shape),
        )
        tri_line1_p2_lat = np.where(
            lvn_pos,
            np.broadcast_to(ptr_v3_lat_e[:, 0], p_vn.shape),
            np.broadcast_to(ptr_v3_lat_e[:, 1], p_vn.shape),
        )

        tri_line2_p1_lon = arrival_pts_2_lon_dsl
        tri_line2_p1_lat = arrival_pts_2_lat_dsl
        tri_line2_p2_lon = np.where(
            lvn_pos,
            np.broadcast_to(ptr_v3_lon_e[:, 0], p_vn.shape),
            np.broadcast_to(ptr_v3_lon_e[:, 1], p_vn.shape),
        )
        tri_line2_p2_lat = np.where(
            lvn_pos,
            np.broadcast_to(ptr_v3_lat_e[:, 0], p_vn.shape),
            np.broadcast_to(ptr_v3_lat_e[:, 1], p_vn.shape),
        )

        return (
            fl_line_p1_lon,
            fl_line_p1_lat,
            fl_line_p2_lon,
            fl_line_p2_lat,
            tri_line1_p1_lon,
            tri_line1_p1_lat,
            tri_line1_p2_lon,
            tri_line1_p2_lat,
            tri_line2_p1_lon,
            tri_line2_p1_lat,
            tri_line2_p2_lon,
            tri_line2_p2_lat,
            arrival_pts_1_lon_dsl,
            arrival_pts_1_lat_dsl,
            arrival_pts_2_lon_dsl,
            arrival_pts_2_lat_dsl,
            depart_pts_1_lon_dsl,
            depart_pts_1_lat_dsl,
            depart_pts_2_lon_dsl,
            depart_pts_2_lat_dsl,
        )

    @staticmethod
    def _apply_case1_patch0(
        *,
        mask_case1,
        lvn_sys_pos,
        arrival_pts_1_lon_dsl,
        arrival_pts_1_lat_dsl,
        arrival_pts_2_lon_dsl,
        arrival_pts_2_lat_dsl,
        ps1_x,
        ps1_y,
        ps2_x,
        ps2_y,
        depart_pts_1_lon_dsl,
        depart_pts_1_lat_dsl,
        depart_pts_2_lon_dsl,
        depart_pts_2_lat_dsl,
    ):
        dreg_patch0_1_lon_dsl = arrival_pts_1_lon_dsl
        dreg_patch0_1_lat_dsl = arrival_pts_1_lat_dsl
        dreg_patch0_2_lon_dsl = np.where(
            mask_case1,
            np.where(lvn_sys_pos, arrival_pts_2_lon_dsl, ps1_x),
            arrival_pts_2_lon_dsl,
        )
        dreg_patch0_2_lat_dsl = np.where(
            mask_case1,
            np.where(lvn_sys_pos, arrival_pts_2_lat_dsl, ps1_y),
            arrival_pts_2_lat_dsl,
        )
        dreg_patch0_3_lon_dsl = np.where(mask_case1, ps2_x, depart_pts_2_lon_dsl)
        dreg_patch0_3_lat_dsl = np.where(mask_case1, ps2_y, depart_pts_2_lat_dsl)
        dreg_patch0_4_lon_dsl = np.where(
            mask_case1,
            np.where(lvn_sys_pos, ps1_x, arrival_pts_2_lon_dsl),
            depart_pts_1_lon_dsl,
        )
        dreg_patch0_4_lat_dsl = np.where(
            mask_case1,
            np.where(lvn_sys_pos, ps1_y, arrival_pts_2_lat_dsl),
            depart_pts_1_lat_dsl,
        )

        return (
            dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl,
        )

    @staticmethod
    def _apply_case1_patch1(
        *,
        mask_case1,
        lvn_sys_pos,
        arrival_pts_1_lon_dsl,
        arrival_pts_1_lat_dsl,
        depart_pts_1_lon_dsl,
        depart_pts_1_lat_dsl,
        ps1_x,
        ps1_y,
    ):
        dreg_patch1_1_lon_vmask = np.where(mask_case1, arrival_pts_1_lon_dsl, 0.0)
        dreg_patch1_1_lat_vmask = np.where(mask_case1, arrival_pts_1_lat_dsl, 0.0)
        dreg_patch1_4_lon_vmask = np.where(mask_case1, arrival_pts_1_lon_dsl, 0.0)
        dreg_patch1_4_lat_vmask = np.where(mask_case1, arrival_pts_1_lat_dsl, 0.0)
        dreg_patch1_2_lon_vmask = np.where(
            mask_case1, np.where(lvn_sys_pos, ps1_x, depart_pts_1_lon_dsl), 0.0
        )
        dreg_patch1_2_lat_vmask = np.where(
            mask_case1, np.where(lvn_sys_pos, ps1_y, depart_pts_1_lat_dsl), 0.0
        )
        dreg_patch1_3_lon_vmask = np.where(
            mask_case1, np.where(lvn_sys_pos, depart_pts_1_lon_dsl, ps1_x), 0.0
        )
        dreg_patch1_3_lat_vmask = np.where(
            mask_case1, np.where(lvn_sys_pos, depart_pts_1_lat_dsl, ps1_y), 0.0
        )

        return (
            dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask,
            dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask,
            dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask,
        )

    @staticmethod
    def _apply_case1_patch2(
        *,
        mask_case1,
        lvn_sys_pos,
        arrival_pts_2_lon_dsl,
        arrival_pts_2_lat_dsl,
        depart_pts_2_lon_dsl,
        depart_pts_2_lat_dsl,
        ps2_x,
        ps2_y,
    ):
        # Case 1 - patch 2
        dreg_patch2_1_lon_vmask = np.where(mask_case1, arrival_pts_2_lon_dsl, 0.0)
        dreg_patch2_1_lat_vmask = np.where(mask_case1, arrival_pts_2_lat_dsl, 0.0)
        dreg_patch2_4_lon_vmask = np.where(mask_case1, arrival_pts_2_lon_dsl, 0.0)
        dreg_patch2_4_lat_vmask = np.where(mask_case1, arrival_pts_2_lat_dsl, 0.0)
        dreg_patch2_2_lon_vmask = np.where(
            mask_case1, np.where(lvn_sys_pos, depart_pts_2_lon_dsl, ps2_x), 0.0
        )
        dreg_patch2_2_lat_vmask = np.where(
            mask_case1, np.where(lvn_sys_pos, depart_pts_2_lat_dsl, ps2_y), 0.0
        )
        dreg_patch2_3_lon_vmask = np.where(
            mask_case1, np.where(lvn_sys_pos, ps2_x, depart_pts_2_lon_dsl), 0.0
        )
        dreg_patch2_3_lat_vmask = np.where(
            mask_case1, np.where(lvn_sys_pos, ps2_y, depart_pts_2_lat_dsl), 0.0
        )

        return (
            dreg_patch2_1_lon_vmask,
            dreg_patch2_1_lat_vmask,
            dreg_patch2_4_lon_vmask,
            dreg_patch2_4_lat_vmask,
            dreg_patch2_2_lon_vmask,
            dreg_patch2_2_lat_vmask,
            dreg_patch2_3_lon_vmask,
            dreg_patch2_3_lat_vmask,
        )

    @staticmethod
    def _apply_case2a_patch0(
        *,
        mask_case2a,
        lvn_sys_pos,
        arrival_pts_1_lon_dsl,
        arrival_pts_1_lat_dsl,
        arrival_pts_2_lon_dsl,
        arrival_pts_2_lat_dsl,
        ps1_x,
        ps1_y,
        depart_pts_2_lon_dsl,
        depart_pts_2_lat_dsl,
        dreg_patch0_1_lon_dsl,
        dreg_patch0_1_lat_dsl,
        dreg_patch0_2_lon_dsl,
        dreg_patch0_2_lat_dsl,
        dreg_patch0_3_lon_dsl,
        dreg_patch0_3_lat_dsl,
        dreg_patch0_4_lon_dsl,
        dreg_patch0_4_lat_dsl,
    ):
        dreg_patch0_1_lon_dsl = np.where(mask_case2a, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
        dreg_patch0_1_lat_dsl = np.where(mask_case2a, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
        dreg_patch0_2_lon_dsl = np.where(
            mask_case2a,
            np.where(lvn_sys_pos, arrival_pts_2_lon_dsl, ps1_x),
            dreg_patch0_2_lon_dsl,
        )
        dreg_patch0_2_lat_dsl = np.where(
            mask_case2a,
            np.where(lvn_sys_pos, arrival_pts_2_lat_dsl, ps1_y),
            dreg_patch0_2_lat_dsl,
        )
        dreg_patch0_3_lon_dsl = np.where(mask_case2a, depart_pts_2_lon_dsl, dreg_patch0_3_lon_dsl)
        dreg_patch0_3_lat_dsl = np.where(mask_case2a, depart_pts_2_lat_dsl, dreg_patch0_3_lat_dsl)
        dreg_patch0_4_lon_dsl = np.where(
            mask_case2a,
            np.where(lvn_sys_pos, ps1_x, arrival_pts_2_lon_dsl),
            dreg_patch0_4_lon_dsl,
        )
        dreg_patch0_4_lat_dsl = np.where(
            mask_case2a,
            np.where(lvn_sys_pos, ps1_y, arrival_pts_2_lat_dsl),
            dreg_patch0_4_lat_dsl,
        )

        return (
            dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl,
        )

    @staticmethod
    def _apply_case2a_patch1(
        *,
        mask_case2a,
        lvn_sys_pos,
        arrival_pts_1_lon_dsl,
        arrival_pts_1_lat_dsl,
        ps1_x,
        ps1_y,
        depart_pts_1_lon_dsl,
        depart_pts_1_lat_dsl,
        dreg_patch1_1_lon_vmask,
        dreg_patch1_1_lat_vmask,
        dreg_patch1_4_lon_vmask,
        dreg_patch1_4_lat_vmask,
        dreg_patch1_2_lon_vmask,
        dreg_patch1_2_lat_vmask,
        dreg_patch1_3_lon_vmask,
        dreg_patch1_3_lat_vmask,
    ):
        dreg_patch1_1_lon_vmask = np.where(
            mask_case2a, arrival_pts_1_lon_dsl, dreg_patch1_1_lon_vmask
        )
        dreg_patch1_1_lat_vmask = np.where(
            mask_case2a, arrival_pts_1_lat_dsl, dreg_patch1_1_lat_vmask
        )
        dreg_patch1_4_lon_vmask = np.where(
            mask_case2a, arrival_pts_1_lon_dsl, dreg_patch1_4_lon_vmask
        )
        dreg_patch1_4_lat_vmask = np.where(
            mask_case2a, arrival_pts_1_lat_dsl, dreg_patch1_4_lat_vmask
        )
        dreg_patch1_2_lon_vmask = np.where(
            mask_case2a,
            np.where(lvn_sys_pos, ps1_x, depart_pts_1_lon_dsl),
            dreg_patch1_2_lon_vmask,
        )
        dreg_patch1_2_lat_vmask = np.where(
            mask_case2a,
            np.where(lvn_sys_pos, ps1_y, depart_pts_1_lat_dsl),
            dreg_patch1_2_lat_vmask,
        )
        dreg_patch1_3_lon_vmask = np.where(
            mask_case2a,
            np.where(lvn_sys_pos, depart_pts_1_lon_dsl, ps1_x),
            dreg_patch1_3_lon_vmask,
        )
        dreg_patch1_3_lat_vmask = np.where(
            mask_case2a,
            np.where(lvn_sys_pos, depart_pts_1_lat_dsl, ps1_y),
            dreg_patch1_3_lat_vmask,
        )

        return (
            dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask,
            dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask,
            dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask,
        )

    @staticmethod
    def _apply_case2b_patch0(
        *,
        mask_case2b,
        lvn_sys_pos,
        arrival_pts_1_lon_dsl,
        arrival_pts_1_lat_dsl,
        arrival_pts_2_lon_dsl,
        arrival_pts_2_lat_dsl,
        depart_pts_1_lon_dsl,
        depart_pts_1_lat_dsl,
        ps2_x,
        ps2_y,
        dreg_patch0_1_lon_dsl,
        dreg_patch0_1_lat_dsl,
        dreg_patch0_2_lon_dsl,
        dreg_patch0_2_lat_dsl,
        dreg_patch0_3_lon_dsl,
        dreg_patch0_3_lat_dsl,
        dreg_patch0_4_lon_dsl,
        dreg_patch0_4_lat_dsl,
    ):
        dreg_patch0_1_lon_dsl = np.where(mask_case2b, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
        dreg_patch0_1_lat_dsl = np.where(mask_case2b, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
        dreg_patch0_2_lon_dsl = np.where(
            mask_case2b,
            np.where(lvn_sys_pos, arrival_pts_2_lon_dsl, depart_pts_1_lon_dsl),
            dreg_patch0_2_lon_dsl,
        )
        dreg_patch0_2_lat_dsl = np.where(
            mask_case2b,
            np.where(lvn_sys_pos, arrival_pts_2_lat_dsl, depart_pts_1_lat_dsl),
            dreg_patch0_2_lat_dsl,
        )
        dreg_patch0_3_lon_dsl = np.where(mask_case2b, ps2_x, dreg_patch0_3_lon_dsl)
        dreg_patch0_3_lat_dsl = np.where(mask_case2b, ps2_y, dreg_patch0_3_lat_dsl)
        dreg_patch0_4_lon_dsl = np.where(
            mask_case2b,
            np.where(lvn_sys_pos, depart_pts_1_lon_dsl, arrival_pts_2_lon_dsl),
            dreg_patch0_4_lon_dsl,
        )
        dreg_patch0_4_lat_dsl = np.where(
            mask_case2b,
            np.where(lvn_sys_pos, depart_pts_1_lat_dsl, arrival_pts_2_lat_dsl),
            dreg_patch0_4_lat_dsl,
        )

        return (
            dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl,
        )

    @staticmethod
    def _apply_case2b_patch1(
        *,
        mask_case2b,
        dreg_patch1_1_lon_vmask,
        dreg_patch1_1_lat_vmask,
        dreg_patch1_2_lon_vmask,
        dreg_patch1_2_lat_vmask,
        dreg_patch1_3_lon_vmask,
        dreg_patch1_3_lat_vmask,
        dreg_patch1_4_lon_vmask,
        dreg_patch1_4_lat_vmask,
    ):
        zeros_array = np.zeros_like(mask_case2b)

        dreg_patch1_1_lon_vmask = np.where(mask_case2b, zeros_array, dreg_patch1_1_lon_vmask)
        dreg_patch1_1_lat_vmask = np.where(mask_case2b, zeros_array, dreg_patch1_1_lat_vmask)
        dreg_patch1_2_lon_vmask = np.where(mask_case2b, zeros_array, dreg_patch1_2_lon_vmask)
        dreg_patch1_2_lat_vmask = np.where(mask_case2b, zeros_array, dreg_patch1_2_lat_vmask)
        dreg_patch1_3_lon_vmask = np.where(mask_case2b, zeros_array, dreg_patch1_3_lon_vmask)
        dreg_patch1_3_lat_vmask = np.where(mask_case2b, zeros_array, dreg_patch1_3_lat_vmask)
        dreg_patch1_4_lon_vmask = np.where(mask_case2b, zeros_array, dreg_patch1_4_lon_vmask)
        dreg_patch1_4_lat_vmask = np.where(mask_case2b, zeros_array, dreg_patch1_4_lat_vmask)

        return (
            dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask,
            dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask,
            dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask,
        )

    @staticmethod
    def _apply_case2b_patch2(
        *,
        mask_case2b,
        lvn_sys_pos,
        arrival_pts_2_lon_dsl,
        arrival_pts_2_lat_dsl,
        depart_pts_2_lon_dsl,
        depart_pts_2_lat_dsl,
        ps2_x,
        ps2_y,
        dreg_patch2_1_lon_vmask,
        dreg_patch2_1_lat_vmask,
        dreg_patch2_4_lon_vmask,
        dreg_patch2_4_lat_vmask,
        dreg_patch2_2_lon_vmask,
        dreg_patch2_2_lat_vmask,
        dreg_patch2_3_lon_vmask,
        dreg_patch2_3_lat_vmask,
    ):
        dreg_patch2_1_lon_vmask = np.where(
            mask_case2b, arrival_pts_2_lon_dsl, dreg_patch2_1_lon_vmask
        )
        dreg_patch2_1_lat_vmask = np.where(
            mask_case2b, arrival_pts_2_lat_dsl, dreg_patch2_1_lat_vmask
        )
        dreg_patch2_4_lon_vmask = np.where(
            mask_case2b, arrival_pts_2_lon_dsl, dreg_patch2_4_lon_vmask
        )
        dreg_patch2_4_lat_vmask = np.where(
            mask_case2b, arrival_pts_2_lat_dsl, dreg_patch2_4_lat_vmask
        )
        dreg_patch2_2_lon_vmask = np.where(
            mask_case2b,
            np.where(lvn_sys_pos, depart_pts_2_lon_dsl, ps2_x),
            dreg_patch2_2_lon_vmask,
        )
        dreg_patch2_2_lat_vmask = np.where(
            mask_case2b,
            np.where(lvn_sys_pos, depart_pts_2_lat_dsl, ps2_y),
            dreg_patch2_2_lat_vmask,
        )
        dreg_patch2_3_lon_vmask = np.where(
            mask_case2b,
            np.where(lvn_sys_pos, ps2_x, depart_pts_2_lon_dsl),
            dreg_patch2_3_lon_vmask,
        )
        dreg_patch2_3_lat_vmask = np.where(
            mask_case2b,
            np.where(lvn_sys_pos, ps2_y, depart_pts_2_lat_dsl),
            dreg_patch2_3_lat_vmask,
        )

        return (
            dreg_patch2_1_lon_vmask,
            dreg_patch2_1_lat_vmask,
            dreg_patch2_4_lon_vmask,
            dreg_patch2_4_lat_vmask,
            dreg_patch2_2_lon_vmask,
            dreg_patch2_2_lat_vmask,
            dreg_patch2_3_lon_vmask,
            dreg_patch2_3_lat_vmask,
        )

    @staticmethod
    def _apply_case3a_patch0(
        *,
        mask_case3a,
        arrival_pts_1_lon_dsl,
        arrival_pts_1_lat_dsl,
        arrival_pts_2_lon_dsl,
        arrival_pts_2_lat_dsl,
        depart_pts_1_lon_dsl,
        depart_pts_1_lat_dsl,
        lvn_sys_pos,
        ps2_x,
        ps2_y,
        dreg_patch0_1_lon_dsl,
        dreg_patch0_1_lat_dsl,
        dreg_patch0_2_lon_dsl,
        dreg_patch0_2_lat_dsl,
        dreg_patch0_3_lon_dsl,
        dreg_patch0_3_lat_dsl,
        dreg_patch0_4_lon_dsl,
        dreg_patch0_4_lat_dsl,
    ):
        dreg_patch0_1_lon_dsl = np.where(mask_case3a, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
        dreg_patch0_1_lat_dsl = np.where(mask_case3a, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
        dreg_patch0_2_lon_dsl = np.where(
            mask_case3a,
            np.where(lvn_sys_pos, arrival_pts_2_lon_dsl, depart_pts_1_lon_dsl),
            dreg_patch0_2_lon_dsl,
        )
        dreg_patch0_2_lat_dsl = np.where(
            mask_case3a,
            np.where(lvn_sys_pos, arrival_pts_2_lat_dsl, depart_pts_1_lat_dsl),
            dreg_patch0_2_lat_dsl,
        )
        dreg_patch0_3_lon_dsl = np.where(mask_case3a, ps2_x, dreg_patch0_3_lon_dsl)
        dreg_patch0_3_lat_dsl = np.where(mask_case3a, ps2_y, dreg_patch0_3_lat_dsl)
        dreg_patch0_4_lon_dsl = np.where(
            mask_case3a,
            np.where(lvn_sys_pos, depart_pts_1_lon_dsl, arrival_pts_2_lon_dsl),
            dreg_patch0_4_lon_dsl,
        )
        dreg_patch0_4_lat_dsl = np.where(
            mask_case3a,
            np.where(lvn_sys_pos, depart_pts_1_lat_dsl, arrival_pts_2_lat_dsl),
            dreg_patch0_4_lat_dsl,
        )

        return (
            dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl,
        )

    @staticmethod
    def _apply_case3a_patch1(
        *,
        mask_case3a,
        lvn_sys_pos,
        arrival_pts_1_lon_dsl,
        arrival_pts_1_lat_dsl,
        pi1_x,
        pi1_y,
        depart_pts_1_lon_dsl,
        depart_pts_1_lat_dsl,
        depart_pts_2_lon_dsl,
        depart_pts_2_lat_dsl,
        dreg_patch1_1_lon_vmask,
        dreg_patch1_1_lat_vmask,
        dreg_patch1_4_lon_vmask,
        dreg_patch1_4_lat_vmask,
        dreg_patch1_2_lon_vmask,
        dreg_patch1_2_lat_vmask,
        dreg_patch1_3_lon_vmask,
        dreg_patch1_3_lat_vmask,
    ):
        dreg_patch1_1_lon_vmask = np.where(
            mask_case3a, arrival_pts_1_lon_dsl, dreg_patch1_1_lon_vmask
        )
        dreg_patch1_1_lat_vmask = np.where(
            mask_case3a, arrival_pts_1_lat_dsl, dreg_patch1_1_lat_vmask
        )
        dreg_patch1_2_lon_vmask = np.where(
            mask_case3a,
            np.where(lvn_sys_pos, pi1_x, depart_pts_2_lon_dsl),
            dreg_patch1_2_lon_vmask,
        )
        dreg_patch1_2_lat_vmask = np.where(
            mask_case3a,
            np.where(lvn_sys_pos, pi1_y, depart_pts_2_lat_dsl),
            dreg_patch1_2_lat_vmask,
        )
        dreg_patch1_3_lon_vmask = np.where(
            mask_case3a, depart_pts_1_lon_dsl, dreg_patch1_3_lon_vmask
        )
        dreg_patch1_3_lat_vmask = np.where(
            mask_case3a, depart_pts_1_lat_dsl, dreg_patch1_3_lat_vmask
        )
        dreg_patch1_4_lon_vmask = np.where(
            mask_case3a,
            np.where(lvn_sys_pos, depart_pts_1_lon_dsl, pi1_x),
            dreg_patch1_4_lon_vmask,
        )
        dreg_patch1_4_lat_vmask = np.where(
            mask_case3a,
            np.where(lvn_sys_pos, depart_pts_1_lat_dsl, pi1_y),
            dreg_patch1_4_lat_vmask,
        )

        return (
            dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask,
            dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask,
            dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask,
        )

    @staticmethod
    def _apply_case3b_patch0(
        *,
        mask_case3b,
        arrival_pts_1_lon_dsl,
        arrival_pts_1_lat_dsl,
        arrival_pts_2_lon_dsl,
        arrival_pts_2_lat_dsl,
        pi2_x,
        pi2_y,
        lvn_sys_pos,
        dreg_patch0_1_lon_dsl,
        dreg_patch0_1_lat_dsl,
        dreg_patch0_4_lon_dsl,
        dreg_patch0_4_lat_dsl,
        dreg_patch0_2_lon_dsl,
        dreg_patch0_2_lat_dsl,
        dreg_patch0_3_lon_dsl,
        dreg_patch0_3_lat_dsl,
    ):
        dreg_patch0_1_lon_dsl = np.where(mask_case3b, arrival_pts_1_lon_dsl, dreg_patch0_1_lon_dsl)
        dreg_patch0_1_lat_dsl = np.where(mask_case3b, arrival_pts_1_lat_dsl, dreg_patch0_1_lat_dsl)
        dreg_patch0_4_lon_dsl = np.where(mask_case3b, arrival_pts_1_lon_dsl, dreg_patch0_4_lon_dsl)
        dreg_patch0_4_lat_dsl = np.where(mask_case3b, arrival_pts_1_lat_dsl, dreg_patch0_4_lat_dsl)
        dreg_patch0_2_lon_dsl = np.where(
            mask_case3b,
            np.where(lvn_sys_pos, arrival_pts_2_lon_dsl, pi2_x),
            dreg_patch0_2_lon_dsl,
        )
        dreg_patch0_2_lat_dsl = np.where(
            mask_case3b,
            np.where(lvn_sys_pos, arrival_pts_2_lat_dsl, pi2_y),
            dreg_patch0_2_lat_dsl,
        )
        dreg_patch0_3_lon_dsl = np.where(
            mask_case3b,
            np.where(lvn_sys_pos, pi2_x, arrival_pts_2_lon_dsl),
            dreg_patch0_3_lon_dsl,
        )
        dreg_patch0_3_lat_dsl = np.where(
            mask_case3b,
            np.where(lvn_sys_pos, pi2_y, arrival_pts_2_lat_dsl),
            dreg_patch0_3_lat_dsl,
        )

        return (
            dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl,
            dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl,
            dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl,
        )

    @staticmethod
    def _apply_case3b_patch2(
        *,
        mask_case3b,
        arrival_pts_2_lon_dsl,
        arrival_pts_2_lat_dsl,
        depart_pts_1_lon_dsl,
        depart_pts_1_lat_dsl,
        depart_pts_2_lon_dsl,
        depart_pts_2_lat_dsl,
        pi2_x,
        pi2_y,
        lvn_sys_pos,
        dreg_patch2_1_lon_vmask,
        dreg_patch2_1_lat_vmask,
        dreg_patch2_2_lon_vmask,
        dreg_patch2_2_lat_vmask,
        dreg_patch2_3_lon_vmask,
        dreg_patch2_3_lat_vmask,
        dreg_patch2_4_lon_vmask,
        dreg_patch2_4_lat_vmask,
    ):
        dreg_patch2_1_lon_vmask = np.where(
            mask_case3b, arrival_pts_2_lon_dsl, dreg_patch2_1_lon_vmask
        )
        dreg_patch2_1_lat_vmask = np.where(
            mask_case3b, arrival_pts_2_lat_dsl, dreg_patch2_1_lat_vmask
        )
        dreg_patch2_2_lon_vmask = np.where(
            mask_case3b,
            np.where(lvn_sys_pos, depart_pts_2_lon_dsl, pi2_x),
            dreg_patch2_2_lon_vmask,
        )
        dreg_patch2_2_lat_vmask = np.where(
            mask_case3b,
            np.where(lvn_sys_pos, depart_pts_2_lat_dsl, pi2_y),
            dreg_patch2_2_lat_vmask,
        )
        dreg_patch2_3_lon_vmask = np.where(
            mask_case3b, depart_pts_1_lon_dsl, dreg_patch2_3_lon_vmask
        )
        dreg_patch2_3_lat_vmask = np.where(
            mask_case3b, depart_pts_1_lat_dsl, dreg_patch2_3_lat_vmask
        )
        dreg_patch2_4_lon_vmask = np.where(
            mask_case3b,
            np.where(lvn_sys_pos, pi2_x, depart_pts_2_lon_dsl),
            dreg_patch2_4_lon_vmask,
        )
        dreg_patch2_4_lat_vmask = np.where(
            mask_case3b,
            np.where(lvn_sys_pos, pi2_y, depart_pts_2_lat_dsl),
            dreg_patch2_4_lat_vmask,
        )

        return (
            dreg_patch2_1_lon_vmask,
            dreg_patch2_1_lat_vmask,
            dreg_patch2_2_lon_vmask,
            dreg_patch2_2_lat_vmask,
            dreg_patch2_3_lon_vmask,
            dreg_patch2_3_lat_vmask,
            dreg_patch2_4_lon_vmask,
            dreg_patch2_4_lat_vmask,
        )

    @classmethod
    def reference(
        cls,
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        famask_int,
        p_vn,
        ptr_v3_lon,
        ptr_v3_lat,
        tangent_orientation_dsl,
        dreg_patch0_1_lon_dsl,
        dreg_patch0_1_lat_dsl,
        dreg_patch0_2_lon_dsl,
        dreg_patch0_2_lat_dsl,
        dreg_patch0_3_lon_dsl,
        dreg_patch0_3_lat_dsl,
        dreg_patch0_4_lon_dsl,
        dreg_patch0_4_lat_dsl,
        **kwargs,
    ) -> dict:
        ptr_v3_lon_e = np.expand_dims(ptr_v3_lon, axis=-1)
        ptr_v3_lat_e = np.expand_dims(ptr_v3_lat, axis=-1)
        tangent_orientation_dsl = np.expand_dims(tangent_orientation_dsl, axis=-1)

        result_tuple = cls._generate_flux_area_geometry(
            dreg_patch0_1_lon_dsl=dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl=dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl=dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl=dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl=dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl=dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl=dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl=dreg_patch0_4_lat_dsl,
            p_vn=p_vn,
            ptr_v3_lon_e=ptr_v3_lon_e,
            ptr_v3_lat_e=ptr_v3_lat_e,
        )

        (
            fl_line_p1_lon,
            fl_line_p1_lat,
            fl_line_p2_lon,
            fl_line_p2_lat,
            tri_line1_p1_lon,
            tri_line1_p1_lat,
            tri_line1_p2_lon,
            tri_line1_p2_lat,
            tri_line2_p1_lon,
            tri_line2_p1_lat,
            tri_line2_p2_lon,
            tri_line2_p2_lat,
            arrival_pts_1_lon_dsl,
            arrival_pts_1_lat_dsl,
            arrival_pts_2_lon_dsl,
            arrival_pts_2_lat_dsl,
            depart_pts_1_lon_dsl,
            depart_pts_1_lat_dsl,
            depart_pts_2_lon_dsl,
            depart_pts_2_lat_dsl,
        ) = result_tuple

        # Create first mask does departure-line segment intersects with A1V3
        lintersect_line1 = _lintersect_numpy(
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
        lintersect_line2 = _lintersect_numpy(
            line1_p1_lon=fl_line_p1_lon,
            line1_p1_lat=fl_line_p1_lat,
            line1_p2_lon=fl_line_p2_lon,
            line1_p2_lat=fl_line_p2_lat,
            line2_p1_lon=tri_line2_p1_lon,
            line2_p1_lat=tri_line2_p1_lat,
            line2_p2_lon=tri_line2_p2_lon,
            line2_p2_lat=tri_line2_p2_lat,
        )

        lvn_sys_pos = np.where(
            (p_vn * np.broadcast_to(tangent_orientation_dsl, p_vn.shape)) >= 0.0,
            True,
            False,
        )
        famask_bool = np.where(famask_int == 1, True, False)
        mask_case1 = np.logical_and.reduce([lintersect_line1, lintersect_line2, famask_bool])
        ps1_x, ps1_y = _line_intersect_numpy(
            line1_p1_lon=fl_line_p1_lon,
            line1_p1_lat=fl_line_p1_lat,
            line1_p2_lon=fl_line_p2_lon,
            line1_p2_lat=fl_line_p2_lat,
            line2_p1_lon=tri_line1_p1_lon,
            line2_p1_lat=tri_line1_p1_lat,
            line2_p2_lon=tri_line1_p2_lon,
            line2_p2_lat=tri_line1_p2_lat,
        )
        ps2_x, ps2_y = _line_intersect_numpy(
            line1_p1_lon=fl_line_p1_lon,
            line1_p1_lat=fl_line_p1_lat,
            line1_p2_lon=fl_line_p2_lon,
            line1_p2_lat=fl_line_p2_lat,
            line2_p1_lon=tri_line2_p1_lon,
            line2_p1_lat=tri_line2_p1_lat,
            line2_p2_lon=tri_line2_p2_lon,
            line2_p2_lat=tri_line2_p2_lat,
        )

        # ------------------------------------------------- Case 1
        # Case 1 - patch 0
        (
            dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl,
        ) = cls._apply_case1_patch0(
            mask_case1=mask_case1,
            lvn_sys_pos=lvn_sys_pos,
            arrival_pts_1_lon_dsl=arrival_pts_1_lon_dsl,
            arrival_pts_1_lat_dsl=arrival_pts_1_lat_dsl,
            arrival_pts_2_lon_dsl=arrival_pts_2_lon_dsl,
            arrival_pts_2_lat_dsl=arrival_pts_2_lat_dsl,
            ps1_x=ps1_x,
            ps1_y=ps1_y,
            ps2_x=ps2_x,
            ps2_y=ps2_y,
            depart_pts_1_lon_dsl=depart_pts_1_lon_dsl,
            depart_pts_1_lat_dsl=depart_pts_1_lat_dsl,
            depart_pts_2_lon_dsl=depart_pts_2_lon_dsl,
            depart_pts_2_lat_dsl=depart_pts_2_lat_dsl,
        )
        # Case 1 - patch 1
        (
            dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask,
            dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask,
            dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask,
        ) = cls._apply_case1_patch1(
            mask_case1=mask_case1,
            lvn_sys_pos=lvn_sys_pos,
            arrival_pts_1_lon_dsl=arrival_pts_1_lon_dsl,
            arrival_pts_1_lat_dsl=arrival_pts_1_lat_dsl,
            depart_pts_1_lon_dsl=depart_pts_1_lon_dsl,
            depart_pts_1_lat_dsl=depart_pts_1_lat_dsl,
            ps1_x=ps1_x,
            ps1_y=ps1_y,
        )
        # Case 1 - patch 2
        (
            dreg_patch2_1_lon_vmask,
            dreg_patch2_1_lat_vmask,
            dreg_patch2_4_lon_vmask,
            dreg_patch2_4_lat_vmask,
            dreg_patch2_2_lon_vmask,
            dreg_patch2_2_lat_vmask,
            dreg_patch2_3_lon_vmask,
            dreg_patch2_3_lat_vmask,
        ) = cls._apply_case1_patch2(
            mask_case1=mask_case1,
            lvn_sys_pos=lvn_sys_pos,
            arrival_pts_2_lon_dsl=arrival_pts_2_lon_dsl,
            arrival_pts_2_lat_dsl=arrival_pts_2_lat_dsl,
            depart_pts_2_lon_dsl=depart_pts_2_lon_dsl,
            depart_pts_2_lat_dsl=depart_pts_2_lat_dsl,
            ps2_x=ps2_x,
            ps2_y=ps2_y,
        )

        # ------------------------------------------------- Case 2a
        mask_case2a = np.logical_and.reduce(
            [lintersect_line1, np.logical_not(lintersect_line2), famask_bool]
        )
        # Case 2a - patch 0
        result_tuple_patch0 = cls._apply_case2a_patch0(
            mask_case2a=mask_case2a,
            lvn_sys_pos=lvn_sys_pos,
            arrival_pts_1_lon_dsl=arrival_pts_1_lon_dsl,
            arrival_pts_1_lat_dsl=arrival_pts_1_lat_dsl,
            arrival_pts_2_lon_dsl=arrival_pts_2_lon_dsl,
            arrival_pts_2_lat_dsl=arrival_pts_2_lat_dsl,
            ps1_x=ps1_x,
            ps1_y=ps1_y,
            depart_pts_2_lon_dsl=depart_pts_2_lon_dsl,
            depart_pts_2_lat_dsl=depart_pts_2_lat_dsl,
            dreg_patch0_1_lon_dsl=dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl=dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl=dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl=dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl=dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl=dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl=dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl=dreg_patch0_4_lat_dsl,
        )

        (
            dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl,
        ) = result_tuple_patch0
        # Case 2a - patch 1
        result_tuple_patch1 = cls._apply_case2a_patch1(
            mask_case2a=mask_case2a,
            lvn_sys_pos=lvn_sys_pos,
            arrival_pts_1_lon_dsl=arrival_pts_1_lon_dsl,
            arrival_pts_1_lat_dsl=arrival_pts_1_lat_dsl,
            ps1_x=ps1_x,
            ps1_y=ps1_y,
            depart_pts_1_lon_dsl=depart_pts_1_lon_dsl,
            depart_pts_1_lat_dsl=depart_pts_1_lat_dsl,
            dreg_patch1_1_lon_vmask=dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask=dreg_patch1_1_lat_vmask,
            dreg_patch1_4_lon_vmask=dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask=dreg_patch1_4_lat_vmask,
            dreg_patch1_2_lon_vmask=dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask=dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask=dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask=dreg_patch1_3_lat_vmask,
        )

        (
            dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask,
            dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask,
            dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask,
        ) = result_tuple_patch1
        # Case 2a - patch 2
        dreg_patch2_1_lon_vmask = np.where(mask_case2a, 0.0, dreg_patch2_1_lon_vmask)
        dreg_patch2_1_lat_vmask = np.where(mask_case2a, 0.0, dreg_patch2_1_lat_vmask)
        dreg_patch2_2_lon_vmask = np.where(mask_case2a, 0.0, dreg_patch2_2_lon_vmask)
        dreg_patch2_2_lat_vmask = np.where(mask_case2a, 0.0, dreg_patch2_2_lat_vmask)
        dreg_patch2_3_lon_vmask = np.where(mask_case2a, 0.0, dreg_patch2_3_lon_vmask)
        dreg_patch2_3_lat_vmask = np.where(mask_case2a, 0.0, dreg_patch2_3_lat_vmask)
        dreg_patch2_4_lon_vmask = np.where(mask_case2a, 0.0, dreg_patch2_4_lon_vmask)
        dreg_patch2_4_lat_vmask = np.where(mask_case2a, 0.0, dreg_patch2_4_lat_vmask)

        # -------------------------------------------------- Case 2b
        mask_case2b = np.logical_and.reduce(
            [lintersect_line2, np.logical_not(lintersect_line1), famask_bool]
        )
        # Case 2b - patch 0
        result_tuple_patch0_case2b = cls._apply_case2b_patch0(
            mask_case2b=mask_case2b,
            lvn_sys_pos=lvn_sys_pos,
            arrival_pts_1_lon_dsl=arrival_pts_1_lon_dsl,
            arrival_pts_1_lat_dsl=arrival_pts_1_lat_dsl,
            arrival_pts_2_lon_dsl=arrival_pts_2_lon_dsl,
            arrival_pts_2_lat_dsl=arrival_pts_2_lat_dsl,
            depart_pts_1_lon_dsl=depart_pts_1_lon_dsl,
            depart_pts_1_lat_dsl=depart_pts_1_lat_dsl,
            ps2_x=ps2_x,
            ps2_y=ps2_y,
            dreg_patch0_1_lon_dsl=dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl=dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl=dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl=dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl=dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl=dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl=dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl=dreg_patch0_4_lat_dsl,
        )

        (
            dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl,
        ) = result_tuple_patch0_case2b

        # Case 2b - patch 1
        result_tuple_patch1_case2b = cls._apply_case2b_patch1(
            mask_case2b=mask_case2b,
            dreg_patch1_1_lon_vmask=dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask=dreg_patch1_1_lat_vmask,
            dreg_patch1_2_lon_vmask=dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask=dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask=dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask=dreg_patch1_3_lat_vmask,
            dreg_patch1_4_lon_vmask=dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask=dreg_patch1_4_lat_vmask,
        )

        (
            dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask,
            dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask,
            dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask,
        ) = result_tuple_patch1_case2b

        # Case 2b - patch 2
        result_tuple_patch2_case2b = cls._apply_case2b_patch2(
            mask_case2b=mask_case2b,
            lvn_sys_pos=lvn_sys_pos,
            arrival_pts_2_lon_dsl=arrival_pts_2_lon_dsl,
            arrival_pts_2_lat_dsl=arrival_pts_2_lat_dsl,
            depart_pts_2_lon_dsl=depart_pts_2_lon_dsl,
            depart_pts_2_lat_dsl=depart_pts_2_lat_dsl,
            ps2_x=ps2_x,
            ps2_y=ps2_y,
            dreg_patch2_1_lon_vmask=dreg_patch2_1_lon_vmask,
            dreg_patch2_1_lat_vmask=dreg_patch2_1_lat_vmask,
            dreg_patch2_4_lon_vmask=dreg_patch2_4_lon_vmask,
            dreg_patch2_4_lat_vmask=dreg_patch2_4_lat_vmask,
            dreg_patch2_2_lon_vmask=dreg_patch2_2_lon_vmask,
            dreg_patch2_2_lat_vmask=dreg_patch2_2_lat_vmask,
            dreg_patch2_3_lon_vmask=dreg_patch2_3_lon_vmask,
            dreg_patch2_3_lat_vmask=dreg_patch2_3_lat_vmask,
        )

        (
            dreg_patch2_1_lon_vmask,
            dreg_patch2_1_lat_vmask,
            dreg_patch2_4_lon_vmask,
            dreg_patch2_4_lat_vmask,
            dreg_patch2_2_lon_vmask,
            dreg_patch2_2_lat_vmask,
            dreg_patch2_3_lon_vmask,
            dreg_patch2_3_lat_vmask,
        ) = result_tuple_patch2_case2b

        # flux area edge 1 and 2
        fl_e1_p1_lon = arrival_pts_1_lon_dsl
        fl_e1_p1_lat = arrival_pts_1_lat_dsl
        fl_e1_p2_lon = depart_pts_1_lon_dsl
        fl_e1_p2_lat = depart_pts_1_lat_dsl
        fl_e2_p1_lon = arrival_pts_2_lon_dsl
        fl_e2_p1_lat = arrival_pts_2_lat_dsl
        fl_e2_p2_lon = depart_pts_2_lon_dsl
        fl_e2_p2_lat = depart_pts_2_lat_dsl

        # ----------------------------------------------- Case 3a
        # Check whether flux area edge 2 intersects with triangle edge 1
        lintersect_e2_line1 = _lintersect_numpy(
            line1_p1_lon=fl_e2_p1_lon,
            line1_p1_lat=fl_e2_p1_lat,
            line1_p2_lon=fl_e2_p2_lon,
            line1_p2_lat=fl_e2_p2_lat,
            line2_p1_lon=tri_line1_p1_lon,
            line2_p1_lat=tri_line1_p1_lat,
            line2_p2_lon=tri_line1_p2_lon,
            line2_p2_lat=tri_line1_p2_lat,
        )
        mask_case3a = np.logical_and(lintersect_e2_line1, famask_bool)
        pi1_x, pi1_y = _line_intersect_numpy(
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
        result = cls._apply_case3a_patch0(
            mask_case3a=mask_case3a,
            arrival_pts_1_lon_dsl=arrival_pts_1_lon_dsl,
            arrival_pts_1_lat_dsl=arrival_pts_1_lat_dsl,
            arrival_pts_2_lon_dsl=arrival_pts_2_lon_dsl,
            arrival_pts_2_lat_dsl=arrival_pts_2_lat_dsl,
            depart_pts_1_lon_dsl=depart_pts_1_lon_dsl,
            depart_pts_1_lat_dsl=depart_pts_1_lat_dsl,
            lvn_sys_pos=lvn_sys_pos,
            ps2_x=ps2_x,
            ps2_y=ps2_y,
            dreg_patch0_1_lon_dsl=dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl=dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl=dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl=dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl=dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl=dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl=dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl=dreg_patch0_4_lat_dsl,
        )

        (
            dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl,
        ) = result

        # Case 3a - patch 1
        (
            dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask,
            dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask,
            dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask,
        ) = cls._apply_case3a_patch1(
            mask_case3a=mask_case3a,
            lvn_sys_pos=lvn_sys_pos,
            arrival_pts_1_lon_dsl=arrival_pts_1_lon_dsl,
            arrival_pts_1_lat_dsl=arrival_pts_1_lat_dsl,
            pi1_x=pi1_x,
            pi1_y=pi1_y,
            depart_pts_1_lon_dsl=depart_pts_1_lon_dsl,
            depart_pts_1_lat_dsl=depart_pts_1_lat_dsl,
            depart_pts_2_lon_dsl=depart_pts_2_lon_dsl,
            depart_pts_2_lat_dsl=depart_pts_2_lat_dsl,
            dreg_patch1_1_lon_vmask=dreg_patch1_1_lon_vmask,
            dreg_patch1_1_lat_vmask=dreg_patch1_1_lat_vmask,
            dreg_patch1_4_lon_vmask=dreg_patch1_4_lon_vmask,
            dreg_patch1_4_lat_vmask=dreg_patch1_4_lat_vmask,
            dreg_patch1_2_lon_vmask=dreg_patch1_2_lon_vmask,
            dreg_patch1_2_lat_vmask=dreg_patch1_2_lat_vmask,
            dreg_patch1_3_lon_vmask=dreg_patch1_3_lon_vmask,
            dreg_patch1_3_lat_vmask=dreg_patch1_3_lat_vmask,
        )
        # Case 3a - patch 2
        dreg_patch2_1_lon_vmask = np.where(mask_case3a, 0.0, dreg_patch2_1_lon_vmask)
        dreg_patch2_1_lat_vmask = np.where(mask_case3a, 0.0, dreg_patch2_1_lat_vmask)
        dreg_patch2_2_lon_vmask = np.where(mask_case3a, 0.0, dreg_patch2_2_lon_vmask)
        dreg_patch2_2_lat_vmask = np.where(mask_case3a, 0.0, dreg_patch2_2_lat_vmask)
        dreg_patch2_3_lon_vmask = np.where(mask_case3a, 0.0, dreg_patch2_3_lon_vmask)
        dreg_patch2_3_lat_vmask = np.where(mask_case3a, 0.0, dreg_patch2_3_lat_vmask)
        dreg_patch2_4_lon_vmask = np.where(mask_case3a, 0.0, dreg_patch2_4_lon_vmask)
        dreg_patch2_4_lat_vmask = np.where(mask_case3a, 0.0, dreg_patch2_4_lat_vmask)

        # ------------------------------------------------ Case 3b
        # Check whether flux area edge 1 intersects with triangle edge 2
        lintersect_e1_line2 = _lintersect_numpy(
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
        pi2_x, pi2_y = _line_intersect_numpy(
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
        (
            dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl,
            dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl,
            dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl,
        ) = cls._apply_case3b_patch0(
            mask_case3b=mask_case3b,
            arrival_pts_1_lon_dsl=arrival_pts_1_lon_dsl,
            arrival_pts_1_lat_dsl=arrival_pts_1_lat_dsl,
            arrival_pts_2_lon_dsl=arrival_pts_2_lon_dsl,
            arrival_pts_2_lat_dsl=arrival_pts_2_lat_dsl,
            pi2_x=pi2_x,
            pi2_y=pi2_y,
            lvn_sys_pos=lvn_sys_pos,
            dreg_patch0_1_lon_dsl=dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl=dreg_patch0_1_lat_dsl,
            dreg_patch0_4_lon_dsl=dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl=dreg_patch0_4_lat_dsl,
            dreg_patch0_2_lon_dsl=dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl=dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl=dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl=dreg_patch0_3_lat_dsl,
        )
        # Case 3b - patch 1
        dreg_patch1_1_lon_vmask = np.where(mask_case3b, 0.0, dreg_patch1_1_lon_vmask)
        dreg_patch1_1_lat_vmask = np.where(mask_case3b, 0.0, dreg_patch1_1_lat_vmask)
        dreg_patch1_2_lon_vmask = np.where(mask_case3b, 0.0, dreg_patch1_2_lon_vmask)
        dreg_patch1_2_lat_vmask = np.where(mask_case3b, 0.0, dreg_patch1_2_lat_vmask)
        dreg_patch1_3_lon_vmask = np.where(mask_case3b, 0.0, dreg_patch1_3_lon_vmask)
        dreg_patch1_3_lat_vmask = np.where(mask_case3b, 0.0, dreg_patch1_3_lat_vmask)
        dreg_patch1_4_lon_vmask = np.where(mask_case3b, 0.0, dreg_patch1_4_lon_vmask)
        dreg_patch1_4_lat_vmask = np.where(mask_case3b, 0.0, dreg_patch1_4_lat_vmask)

        # Case 3b - patch 2
        (
            dreg_patch2_1_lon_vmask,
            dreg_patch2_1_lat_vmask,
            dreg_patch2_2_lon_vmask,
            dreg_patch2_2_lat_vmask,
            dreg_patch2_3_lon_vmask,
            dreg_patch2_3_lat_vmask,
            dreg_patch2_4_lon_vmask,
            dreg_patch2_4_lat_vmask,
        ) = cls._apply_case3b_patch2(
            mask_case3b=mask_case3b,
            arrival_pts_2_lon_dsl=arrival_pts_2_lon_dsl,
            arrival_pts_2_lat_dsl=arrival_pts_2_lat_dsl,
            depart_pts_1_lon_dsl=depart_pts_1_lon_dsl,
            depart_pts_1_lat_dsl=depart_pts_1_lat_dsl,
            depart_pts_2_lon_dsl=depart_pts_2_lon_dsl,
            depart_pts_2_lat_dsl=depart_pts_2_lat_dsl,
            pi2_x=pi2_x,
            pi2_y=pi2_y,
            lvn_sys_pos=lvn_sys_pos,
            dreg_patch2_1_lon_vmask=dreg_patch2_1_lon_vmask,
            dreg_patch2_1_lat_vmask=dreg_patch2_1_lat_vmask,
            dreg_patch2_2_lon_vmask=dreg_patch2_2_lon_vmask,
            dreg_patch2_2_lat_vmask=dreg_patch2_2_lat_vmask,
            dreg_patch2_3_lon_vmask=dreg_patch2_3_lon_vmask,
            dreg_patch2_3_lat_vmask=dreg_patch2_3_lat_vmask,
            dreg_patch2_4_lon_vmask=dreg_patch2_4_lon_vmask,
            dreg_patch2_4_lat_vmask=dreg_patch2_4_lat_vmask,
        )
        # --------------------------------------------- Case 4
        # NB: Next line acts as the "ELSE IF", indices that already previously matched one of the above conditions
        # can't be overwritten by this new condition.
        indices_previously_matched = np.logical_or.reduce(
            [mask_case3b, mask_case3a, mask_case2b, mask_case2a, mask_case1]
        )
        #    mask_case4 = (abs(p_vn) < 0.1) & famask_bool & (not indices_previously_matched) we insert also the error indices
        mask_case4 = np.logical_and.reduce(
            [famask_bool, np.logical_not(indices_previously_matched)]
        )
        # Case 4 - patch 0 - no change
        # Case 4 - patch 1
        dreg_patch1_1_lon_vmask = np.where(mask_case4, 0.0, dreg_patch1_1_lon_vmask)
        dreg_patch1_1_lat_vmask = np.where(mask_case4, 0.0, dreg_patch1_1_lat_vmask)
        dreg_patch1_2_lon_vmask = np.where(mask_case4, 0.0, dreg_patch1_2_lon_vmask)
        dreg_patch1_2_lat_vmask = np.where(mask_case4, 0.0, dreg_patch1_2_lat_vmask)
        dreg_patch1_3_lon_vmask = np.where(mask_case4, 0.0, dreg_patch1_3_lon_vmask)
        dreg_patch1_3_lat_vmask = np.where(mask_case4, 0.0, dreg_patch1_3_lat_vmask)
        dreg_patch1_4_lon_vmask = np.where(mask_case4, 0.0, dreg_patch1_4_lon_vmask)
        dreg_patch1_4_lat_vmask = np.where(mask_case4, 0.0, dreg_patch1_4_lat_vmask)
        # Case 4 - patch 2
        dreg_patch2_1_lon_vmask = np.where(mask_case4, 0.0, dreg_patch2_1_lon_vmask)
        dreg_patch2_1_lat_vmask = np.where(mask_case4, 0.0, dreg_patch2_1_lat_vmask)
        dreg_patch2_2_lon_vmask = np.where(mask_case4, 0.0, dreg_patch2_2_lon_vmask)
        dreg_patch2_2_lat_vmask = np.where(mask_case4, 0.0, dreg_patch2_2_lat_vmask)
        dreg_patch2_3_lon_vmask = np.where(mask_case4, 0.0, dreg_patch2_3_lon_vmask)
        dreg_patch2_3_lat_vmask = np.where(mask_case4, 0.0, dreg_patch2_3_lat_vmask)
        dreg_patch2_4_lon_vmask = np.where(mask_case4, 0.0, dreg_patch2_4_lon_vmask)

        return dict(
            dreg_patch0_1_lon_dsl=dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl=dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl=dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl=dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl=dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl=dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl=dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl=dreg_patch0_4_lat_dsl,
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
        )

    @pytest.fixture
    def input_data(self, grid) -> dict:
        famask_int = data_alloc.random_mask(grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
        p_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        ptr_v3_lon_field = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, low=0.1, high=1.0
        )
        ptr_v3_lat_field = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2CDim, low=0.1, high=1.0
        )
        tangent_orientation_dsl = data_alloc.random_field(grid, dims.EdgeDim, low=0.1, high=1.0)
        dreg_patch0_1_lon_dsl = data_alloc.constant_field(grid, 1.0, dims.EdgeDim, dims.KDim)
        dreg_patch0_1_lat_dsl = data_alloc.constant_field(grid, 1.0, dims.EdgeDim, dims.KDim)
        dreg_patch0_2_lon_dsl = data_alloc.constant_field(grid, 2.0, dims.EdgeDim, dims.KDim)
        dreg_patch0_2_lat_dsl = data_alloc.constant_field(grid, 2.0, dims.EdgeDim, dims.KDim)
        dreg_patch0_3_lon_dsl = data_alloc.constant_field(grid, 3.0, dims.EdgeDim, dims.KDim)
        dreg_patch0_3_lat_dsl = data_alloc.constant_field(grid, 3.0, dims.EdgeDim, dims.KDim)
        dreg_patch0_4_lon_dsl = data_alloc.constant_field(grid, 4.0, dims.EdgeDim, dims.KDim)
        dreg_patch0_4_lat_dsl = data_alloc.constant_field(grid, 4.0, dims.EdgeDim, dims.KDim)
        dreg_patch1_1_lon_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch1_1_lat_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch1_2_lon_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch1_2_lat_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch1_3_lon_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch1_3_lat_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch1_4_lon_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch1_4_lat_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch2_1_lon_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch2_1_lat_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch2_2_lon_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch2_2_lat_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch2_3_lon_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch2_3_lat_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch2_4_lon_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        dreg_patch2_4_lat_vmask = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim)
        return dict(
            famask_int=famask_int,
            p_vn=p_vn,
            ptr_v3_lon=ptr_v3_lon_field,
            ptr_v3_lat=ptr_v3_lat_field,
            tangent_orientation_dsl=tangent_orientation_dsl,
            dreg_patch0_1_lon_dsl=dreg_patch0_1_lon_dsl,
            dreg_patch0_1_lat_dsl=dreg_patch0_1_lat_dsl,
            dreg_patch0_2_lon_dsl=dreg_patch0_2_lon_dsl,
            dreg_patch0_2_lat_dsl=dreg_patch0_2_lat_dsl,
            dreg_patch0_3_lon_dsl=dreg_patch0_3_lon_dsl,
            dreg_patch0_3_lat_dsl=dreg_patch0_3_lat_dsl,
            dreg_patch0_4_lon_dsl=dreg_patch0_4_lon_dsl,
            dreg_patch0_4_lat_dsl=dreg_patch0_4_lat_dsl,
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
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
