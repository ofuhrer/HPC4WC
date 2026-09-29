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

from icon4py.model.atmosphere.advection.stencils.compute_ffsl_backtrajectory_length_indicator import (
    compute_ffsl_backtrajectory_length_indicator,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestComputeFfslBacktrajectoryLengthIndicator(stencil_tests.StencilTest):
    PROGRAM = compute_ffsl_backtrajectory_length_indicator
    OUTPUTS = ("opt_famask_dsl",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_vn: np.ndarray,
        p_vt: np.ndarray,
        edge_cell_length: np.ndarray,
        p_dt: float,
        **kwargs,
    ) -> dict:
        lvn_pos = p_vn >= 0.0
        traj_length = np.sqrt(p_vn**2 + p_vt**2) * p_dt

        ec_length_0 = np.expand_dims(edge_cell_length[:, 0], axis=-1)
        ec_length_1 = np.expand_dims(edge_cell_length[:, 1], axis=-1)
        e2c_length = np.where(lvn_pos, ec_length_0, ec_length_1)

        opt_famask_dsl = np.where(
            traj_length > (1.25 * np.broadcast_to(e2c_length, p_vn.shape)),
            1,
            0,
        )

        return dict(opt_famask_dsl=opt_famask_dsl)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_vt = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        edge_cell_length = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        opt_famask_dsl = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=gtx.int32)
        p_dt = 1.0

        return dict(
            p_vn=p_vn,
            p_vt=p_vt,
            edge_cell_length=edge_cell_length,
            opt_famask_dsl=opt_famask_dsl,
            p_dt=p_dt,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
