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

from icon4py.model.atmosphere.dycore.stencils.compute_maximum_cfl_and_clip_contravariant_vertical_velocity import (
    compute_maximum_cfl_and_clip_contravariant_vertical_velocity,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils.data_allocation import random_field, random_mask, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy(
    ddqz_z_half: np.ndarray, z_w_con_c: np.ndarray, cfl_w_limit: ta.wpfloat, dtime: ta.wpfloat
) -> tuple:
    num_rows, num_cols = z_w_con_c.shape
    cfl_clipping = np.where(
        np.abs(z_w_con_c) > cfl_w_limit * ddqz_z_half,
        np.ones([num_rows, num_cols]),
        np.zeros_like(z_w_con_c),
    )
    vcfl = np.where(cfl_clipping == 1.0, z_w_con_c * dtime / ddqz_z_half, 0.0)
    z_w_con_c = np.where(
        (cfl_clipping == 1.0) & (vcfl < -0.85),
        -0.85 * ddqz_z_half / dtime,
        z_w_con_c,
    )
    z_w_con_c = np.where(
        (cfl_clipping == 1.0) & (vcfl > 0.85), 0.85 * ddqz_z_half / dtime, z_w_con_c
    )

    return cfl_clipping, vcfl, z_w_con_c


class TestComputeMaximumCflAndClipContravariantVerticalVelocity(StencilTest):
    PROGRAM = compute_maximum_cfl_and_clip_contravariant_vertical_velocity
    OUTPUTS = ("cfl_clipping", "vcfl", "z_w_con_c")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        ddqz_z_half: np.ndarray,
        z_w_con_c: np.ndarray,
        cfl_w_limit: ta.wpfloat,
        dtime: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        (
            cfl_clipping,
            vcfl,
            z_w_con_c,
        ) = compute_maximum_cfl_and_clip_contravariant_vertical_velocity_numpy(
            ddqz_z_half, z_w_con_c, cfl_w_limit, dtime
        )

        return dict(
            cfl_clipping=cfl_clipping,
            vcfl=vcfl,
            z_w_con_c=z_w_con_c,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        ddqz_z_half = random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        z_w_con_c = random_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        cfl_clipping = random_mask(grid, dims.CellDim, dims.KDim, dtype=bool)
        vcfl = zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.vpfloat)
        cfl_w_limit = ta.vpfloat("5.0")
        dtime = ta.wpfloat("9.0")

        return dict(
            ddqz_z_half=ddqz_z_half,
            z_w_con_c=z_w_con_c,
            cfl_clipping=cfl_clipping,
            vcfl=vcfl,
            cfl_w_limit=cfl_w_limit,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
