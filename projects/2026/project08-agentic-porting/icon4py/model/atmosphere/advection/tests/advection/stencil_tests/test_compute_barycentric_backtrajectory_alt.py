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
from icon4py.model.atmosphere.advection.stencils.compute_barycentric_backtrajectory_alt import (
    compute_barycentric_backtrajectory_alt,
)
from icon4py.model.common import dimension as dims
from icon4py.model.testing import stencil_tests


class TestComputeBarycentricBacktrajectoryAlt(stencil_tests.StencilTest):
    PROGRAM = compute_barycentric_backtrajectory_alt
    OUTPUTS = ("p_distv_bary_1", "p_distv_bary_2")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_vn: np.ndarray,
        p_vt: np.ndarray,
        pos_on_tplane_e_1: np.ndarray,
        pos_on_tplane_e_2: np.ndarray,
        primal_normal_cell_1: np.ndarray,
        dual_normal_cell_1: np.ndarray,
        primal_normal_cell_2: np.ndarray,
        dual_normal_cell_2: np.ndarray,
        p_dthalf: float,
        **kwargs,
    ) -> dict:
        lvn_pos = p_vn >= 0.0
        pos_on_tplane_e_1 = np.expand_dims(pos_on_tplane_e_1, axis=-1)
        pos_on_tplane_e_2 = np.expand_dims(pos_on_tplane_e_2, axis=-1)
        primal_normal_cell_1 = np.expand_dims(primal_normal_cell_1, axis=-1)
        dual_normal_cell_1 = np.expand_dims(dual_normal_cell_1, axis=-1)
        primal_normal_cell_2 = np.expand_dims(primal_normal_cell_2, axis=-1)
        dual_normal_cell_2 = np.expand_dims(dual_normal_cell_2, axis=-1)

        z_ntdistv_bary_1 = -(
            p_vn * p_dthalf + np.where(lvn_pos, pos_on_tplane_e_1[:, 0], pos_on_tplane_e_1[:, 1])
        )
        z_ntdistv_bary_2 = -(
            p_vt * p_dthalf + np.where(lvn_pos, pos_on_tplane_e_2[:, 0], pos_on_tplane_e_2[:, 1])
        )

        p_distv_bary_1 = np.where(
            lvn_pos,
            z_ntdistv_bary_1 * primal_normal_cell_1[:, 0]
            + z_ntdistv_bary_2 * dual_normal_cell_1[:, 0],
            z_ntdistv_bary_1 * primal_normal_cell_1[:, 1]
            + z_ntdistv_bary_2 * dual_normal_cell_1[:, 1],
        )

        p_distv_bary_2 = np.where(
            lvn_pos,
            z_ntdistv_bary_1 * primal_normal_cell_2[:, 0]
            + z_ntdistv_bary_2 * dual_normal_cell_2[:, 0],
            z_ntdistv_bary_1 * primal_normal_cell_2[:, 1]
            + z_ntdistv_bary_2 * dual_normal_cell_2[:, 1],
        )

        return dict(
            p_distv_bary_1=p_distv_bary_1,
            p_distv_bary_2=p_distv_bary_2,
        )

    @pytest.fixture
    def input_data(self, grid) -> dict:
        p_vn = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_vt = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        pos_on_tplane_e_1 = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        pos_on_tplane_e_2 = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        primal_normal_cell_1 = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        dual_normal_cell_1 = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        primal_normal_cell_2 = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        dual_normal_cell_2 = data_alloc.random_field(grid, dims.EdgeDim, dims.E2CDim)
        p_distv_bary_1 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_distv_bary_2 = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
        p_dthalf = 2.0

        return dict(
            p_vn=p_vn,
            p_vt=p_vt,
            pos_on_tplane_e_1=pos_on_tplane_e_1,
            pos_on_tplane_e_2=pos_on_tplane_e_2,
            primal_normal_cell_1=primal_normal_cell_1,
            dual_normal_cell_1=dual_normal_cell_1,
            primal_normal_cell_2=primal_normal_cell_2,
            dual_normal_cell_2=dual_normal_cell_2,
            p_distv_bary_1=p_distv_bary_1,
            p_distv_bary_2=p_distv_bary_2,
            p_dthalf=p_dthalf,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
