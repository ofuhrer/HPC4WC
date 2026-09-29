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

from icon4py.model.atmosphere.advection.stencils.compute_ppm_slope import compute_ppm_slope
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


@pytest.mark.uses_concat_where
class TestComputePpmSlope(stencil_tests.StencilTest):
    PROGRAM = compute_ppm_slope
    OUTPUTS = (
        stencil_tests.Output(
            "z_slope", refslice=(slice(None), slice(None, -1)), gtslice=(slice(None), slice(1, -1))
        ),
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_cc: np.ndarray,
        p_cellhgt_mc_now: np.ndarray,
        elev: gtx.int32,
        **kwargs: Any,
    ) -> dict:
        zfac_m1 = (p_cc[:, 1:-1] - p_cc[:, :-2]) / (
            p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, :-2]
        )
        zfac = (p_cc[:, 2:] - p_cc[:, 1:-1]) / (p_cellhgt_mc_now[:, 2:] + p_cellhgt_mc_now[:, 1:-1])
        z_slope_a = (
            p_cellhgt_mc_now[:, 1:-1]
            / (p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, 2:])
        ) * (
            (2.0 * p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1]) * zfac
            + (p_cellhgt_mc_now[:, 1:-1] + 2.0 * p_cellhgt_mc_now[:, 2:]) * zfac_m1
        )

        zfac_m1 = (p_cc[:, 1:-1] - p_cc[:, :-2]) / (
            p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, :-2]
        )
        zfac = (p_cc[:, 1:-1] - p_cc[:, 1:-1]) / (
            p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, 1:-1]
        )
        z_slope_b = (
            p_cellhgt_mc_now[:, 1:-1]
            / (p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1] + p_cellhgt_mc_now[:, 1:-1])
        ) * (
            (2.0 * p_cellhgt_mc_now[:, :-2] + p_cellhgt_mc_now[:, 1:-1]) * zfac
            + (p_cellhgt_mc_now[:, 1:-1] + 2.0 * p_cellhgt_mc_now[:, 1:-1]) * zfac_m1
        )
        k = np.arange(p_cc.shape[1])
        z_slope = np.where(k[1:-1] < elev, z_slope_a, z_slope_b)
        return dict(z_slope=z_slope)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        z_slope = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        p_cellhgt_mc_now = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )

        elev = grid.num_levels - 2
        return dict(
            p_cc=p_cc,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            z_slope=z_slope,
            elev=elev,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels - 1),
        )
