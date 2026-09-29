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

from icon4py.model.atmosphere.dycore.stencils.update_density_exner_wind import (
    update_density_exner_wind,
)
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


class TestUpdateDensityExnerWind(StencilTest):
    PROGRAM = update_density_exner_wind
    OUTPUTS = ("rho_new", "exner_new", "w_new")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        rho_now: np.ndarray,
        grf_tend_rho: np.ndarray,
        theta_v_now: np.ndarray,
        grf_tend_thv: np.ndarray,
        w_now: np.ndarray,
        grf_tend_w: np.ndarray,
        dtime: ta.wpfloat,
        **kwargs: Any,
    ) -> dict:
        rho_new = rho_now + dtime * grf_tend_rho
        exner_new = theta_v_now + dtime * grf_tend_thv
        w_new = w_now + dtime * grf_tend_w
        return dict(rho_new=rho_new, exner_new=exner_new, w_new=w_new)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        rho_now = random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        grf_tend_rho = random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        theta_v_now = random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        grf_tend_thv = random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        w_now = random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        grf_tend_w = random_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        dtime = ta.wpfloat("5.0")
        rho_new = zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        exner_new = zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)
        w_new = zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

        return dict(
            rho_now=rho_now,
            grf_tend_rho=grf_tend_rho,
            theta_v_now=theta_v_now,
            grf_tend_thv=grf_tend_thv,
            w_now=w_now,
            grf_tend_w=grf_tend_w,
            rho_new=rho_new,
            exner_new=exner_new,
            w_new=w_new,
            dtime=dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
