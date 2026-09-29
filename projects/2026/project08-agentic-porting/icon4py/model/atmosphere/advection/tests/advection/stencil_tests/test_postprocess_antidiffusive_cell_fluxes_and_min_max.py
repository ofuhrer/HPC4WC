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

from icon4py.model.atmosphere.advection.stencils.postprocess_antidiffusive_cell_fluxes_and_min_max import (
    postprocess_antidiffusive_cell_fluxes_and_min_max,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestPostprocessAntidiffusiveCellFluxesAndMinMax(stencil_tests.StencilTest):
    PROGRAM = postprocess_antidiffusive_cell_fluxes_and_min_max
    OUTPUTS = ("z_tracer_new_low", "z_tracer_max", "z_tracer_min")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        refin_ctrl: np.ndarray,
        p_cc: np.ndarray,
        z_tracer_new_low: np.ndarray,
        z_tracer_max: np.ndarray,
        z_tracer_min: np.ndarray,
        lo_bound: float,
        hi_bound: float,
        **kwargs: Any,
    ) -> dict:
        refin_ctrl = np.expand_dims(refin_ctrl, axis=1)
        condition = np.logical_or(
            np.equal(refin_ctrl, np.full(refin_ctrl.shape, lo_bound, dtype=gtx.int32)),
            np.equal(refin_ctrl, np.full(refin_ctrl.shape, hi_bound, dtype=gtx.int32)),
        )
        z_tracer_new_out = np.where(
            condition,
            np.minimum(1.1 * p_cc, np.maximum(0.9 * p_cc, z_tracer_new_low)),
            z_tracer_new_low,
        )
        z_tracer_max_out = np.where(condition, np.maximum(p_cc, z_tracer_new_out), z_tracer_max)
        z_tracer_min_out = np.where(condition, np.minimum(p_cc, z_tracer_new_out), z_tracer_min)
        return dict(
            z_tracer_new_low=z_tracer_new_out,
            z_tracer_max=z_tracer_max_out,
            z_tracer_min=z_tracer_min_out,
        )

    @pytest.fixture()
    def input_data(self, grid: base.Grid) -> dict:
        hi_bound, lo_bound = 3, 1
        refin_ctrl = data_alloc.constant_field(grid, 2, dims.CellDim, dtype=gtx.int32)
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_tracer_new_low_in = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_tracer_max_in = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        z_tracer_min_in = data_alloc.random_field(grid, dims.CellDim, dims.KDim)

        z_tracer_new_low_out = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        z_tracer_max_out = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        z_tracer_min_out = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)

        return dict(
            refin_ctrl=refin_ctrl,
            p_cc=p_cc,
            z_tracer_new_low=z_tracer_new_low_in,
            z_tracer_max=z_tracer_max_in,
            z_tracer_min=z_tracer_min_in,
            z_tracer_new_low_out=z_tracer_new_low_out,
            z_tracer_max_out=z_tracer_max_out,
            z_tracer_min_out=z_tracer_min_out,
            lo_bound=lo_bound,
            hi_bound=hi_bound,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
