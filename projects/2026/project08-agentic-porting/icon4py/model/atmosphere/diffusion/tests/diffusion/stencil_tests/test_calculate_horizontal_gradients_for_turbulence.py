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

from icon4py.model.atmosphere.diffusion.stencils.calculate_horizontal_gradients_for_turbulence import (
    calculate_horizontal_gradients_for_turbulence,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


def calculate_horizontal_gradients_for_turbulence_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    w: np.ndarray,
    geofac_grg_x: np.ndarray,
    geofac_grg_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    c2e2cO = connectivities[dims.C2E2CODim]
    geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
    dwdx = np.sum(np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_x * w[c2e2cO], 0.0), axis=1)

    geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
    dwdy = np.sum(np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_y * w[c2e2cO], 0.0), axis=1)
    return dwdx, dwdy


@pytest.mark.embedded_remap_error
class TestCalculateHorizontalGradientsForTurbulence(StencilTest):
    PROGRAM = calculate_horizontal_gradients_for_turbulence
    OUTPUTS = ("dwdx", "dwdy")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        w: np.ndarray,
        geofac_grg_x: np.ndarray,
        geofac_grg_y: np.ndarray,
        **kwargs,
    ) -> dict:
        dwdx, dwdy = calculate_horizontal_gradients_for_turbulence_numpy(
            connectivities, w, geofac_grg_x, geofac_grg_y
        )
        return dict(dwdx=dwdx, dwdy=dwdy)

    @pytest.fixture
    def input_data(self, grid):
        w = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        geofac_grg_x = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        geofac_grg_y = data_alloc.random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        dwdx = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        dwdy = data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            w=w,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            dwdx=dwdx,
            dwdy=dwdy,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
