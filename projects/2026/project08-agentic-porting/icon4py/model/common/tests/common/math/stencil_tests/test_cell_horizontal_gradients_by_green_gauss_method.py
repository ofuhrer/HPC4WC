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

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.math.stencils import (
    cell_horizontal_gradients_by_green_gauss_method as green_gauss,
)
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def cell_horizontal_gradients_by_green_gauss_method_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray],
    scalar_field: np.ndarray,
    geofac_grg_x: np.ndarray,
    geofac_grg_y: np.ndarray,
) -> tuple[np.ndarray, ...]:
    c2e2cO = connectivities[dims.C2E2CODim]
    geofac_grg_x = np.expand_dims(geofac_grg_x, axis=-1)
    p_grad_1_u = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_x * scalar_field[c2e2cO], 0), axis=1
    )
    geofac_grg_y = np.expand_dims(geofac_grg_y, axis=-1)
    p_grad_1_v = np.sum(
        np.where((c2e2cO != -1)[:, :, np.newaxis], geofac_grg_y * scalar_field[c2e2cO], 0), axis=1
    )
    return (
        p_grad_1_u,
        p_grad_1_v,
    )


@pytest.mark.embedded_remap_error
class TestMoMathGradientsGradGreenGaussCellDsl(StencilTest):
    PROGRAM = green_gauss.cell_horizontal_gradients_by_green_gauss_method
    OUTPUTS = ("out",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        scalar_field: np.ndarray,
        geofac_grg_x: np.ndarray,
        geofac_grg_y: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        (
            p_grad_1_u,
            p_grad_1_v,
        ) = cell_horizontal_gradients_by_green_gauss_method_numpy(
            connectivities, scalar_field, geofac_grg_x, geofac_grg_y
        )
        return dict(
            out=(p_grad_1_u, p_grad_1_v),
        )

    @pytest.fixture
    def input_data(
        self, grid: base.Grid
    ) -> dict[str, gtx.Field | state_utils.ScalarType | gtx.Domain | tuple[gtx.Field, ...]]:
        scalar_field = random_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        geofac_grg_x = random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        geofac_grg_y = random_field(grid, dims.CellDim, dims.C2E2CODim, dtype=wpfloat)
        p_grad_1_u = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)
        p_grad_1_v = zero_field(grid, dims.CellDim, dims.KDim, dtype=vpfloat)

        return dict(
            scalar_field=scalar_field,
            geofac_grg_x=geofac_grg_x,
            geofac_grg_y=geofac_grg_y,
            out=(p_grad_1_u, p_grad_1_v),
            domain=gtx.domain(
                {dims.CellDim: gtx.int32(grid.num_cells), dims.KDim: gtx.int32(grid.num_levels)}
            ),
        )
