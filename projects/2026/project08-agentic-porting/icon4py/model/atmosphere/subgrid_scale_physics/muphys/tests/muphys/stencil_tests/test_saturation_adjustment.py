# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.definitions import Q
from icon4py.model.atmosphere.subgrid_scale_physics.muphys.core.saturation_adjustment import (
    saturation_adjustment,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StencilTest


if TYPE_CHECKING:
    from icon4py.model.common.grid import base as base_grid


class TestSaturationAdjustment(StencilTest):
    PROGRAM = saturation_adjustment
    OUTPUTS = ("te_out", "qve_out", "qce_out")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        te: np.ndarray,
        **kwargs,
    ) -> dict:
        return dict(
            te_out=np.full(te.shape, 273.91226488486984),
            qve_out=np.full(te.shape, 4.4903852062454690e-003),
            qce_out=np.full(te.shape, 9.5724552280369163e-007),
        )

    @pytest.fixture
    def input_data(self, grid: base_grid.Grid) -> dict:
        return dict(
            te=data_alloc.constant_field(
                grid, 273.90911754406039, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            q_in=Q(
                v=data_alloc.constant_field(
                    grid, 4.4913424511676030e-003, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                c=data_alloc.constant_field(
                    grid, 6.0066941654987605e-013, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                r=data_alloc.constant_field(
                    grid, 2.5939378002267028e-004, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                s=data_alloc.constant_field(
                    grid, 3.582312533881839e-06, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                i=data_alloc.constant_field(
                    grid, 3.582312533881839e-06, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
                g=data_alloc.constant_field(
                    grid, 3.582312533881839e-06, dims.CellDim, dims.KDim, dtype=wpfloat
                ),
            ),
            rho=data_alloc.constant_field(
                grid, 1.1371657035251757, dims.CellDim, dims.KDim, dtype=wpfloat
            ),
            te_out=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            qve_out=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            qce_out=data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat),
            horizontal_start=0,
            horizontal_end=grid.num_cells,
            vertical_start=0,
            vertical_end=grid.num_levels,
        )
