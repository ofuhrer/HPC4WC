# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.type_alias as ta
from icon4py.model.atmosphere.dycore.stencils.add_temporal_tendencies_to_vn_by_interpolating_between_time_levels import (
    add_temporal_tendencies_to_vn_by_interpolating_between_time_levels,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def add_temporal_tendencies_to_vn_by_interpolating_between_time_levels_numpy(
    *,
    vn_nnow: np.ndarray,
    ddt_vn_apc_ntl1: np.ndarray,
    ddt_vn_apc_ntl2: np.ndarray,
    ddt_vn_phy: np.ndarray,
    z_theta_v_e: np.ndarray,
    z_gradh_exner: np.ndarray,
    dtime: ta.wpfloat,
    wgt_nnow_vel: ta.wpfloat,
    wgt_nnew_vel: ta.wpfloat,
    cpd: ta.wpfloat,
) -> np.ndarray:
    vn_nnew = vn_nnow + dtime * (
        wgt_nnow_vel * ddt_vn_apc_ntl1
        + wgt_nnew_vel * ddt_vn_apc_ntl2
        + ddt_vn_phy
        - cpd * z_theta_v_e * z_gradh_exner
    )
    return vn_nnew


class TestAddTemporalTendenciesToVnByInterpolatingBetweenTimeLevels(StencilTest):
    PROGRAM = add_temporal_tendencies_to_vn_by_interpolating_between_time_levels
    OUTPUTS = ("vn_nnew",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        vn_nnow: np.ndarray,
        ddt_vn_apc_ntl1: np.ndarray,
        ddt_vn_apc_ntl2: np.ndarray,
        ddt_vn_phy: np.ndarray,
        z_theta_v_e: np.ndarray,
        z_gradh_exner: np.ndarray,
        dtime: ta.wpfloat,
        wgt_nnow_vel: ta.wpfloat,
        wgt_nnew_vel: ta.wpfloat,
        cpd: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        vn_nnew = add_temporal_tendencies_to_vn_by_interpolating_between_time_levels_numpy(
            vn_nnow=vn_nnow,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
            ddt_vn_phy=ddt_vn_phy,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            dtime=dtime,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            cpd=cpd,
        )
        return dict(vn_nnew=vn_nnew)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        vn_nnow = random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        ddt_vn_apc_ntl1 = random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        ddt_vn_apc_ntl2 = random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        ddt_vn_phy = random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        z_theta_v_e = random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        z_gradh_exner = random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)
        vn_nnew = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        dtime = ta.wpfloat("5.0")
        wgt_nnow_vel = ta.wpfloat("8.0")
        wgt_nnew_vel = ta.wpfloat("7.0")
        cpd = ta.wpfloat("2.0")

        return dict(
            vn_nnow=vn_nnow,
            ddt_vn_apc_ntl1=ddt_vn_apc_ntl1,
            ddt_vn_apc_ntl2=ddt_vn_apc_ntl2,
            ddt_vn_phy=ddt_vn_phy,
            z_theta_v_e=z_theta_v_e,
            z_gradh_exner=z_gradh_exner,
            vn_nnew=vn_nnew,
            dtime=dtime,
            wgt_nnow_vel=wgt_nnow_vel,
            wgt_nnew_vel=wgt_nnew_vel,
            cpd=cpd,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
