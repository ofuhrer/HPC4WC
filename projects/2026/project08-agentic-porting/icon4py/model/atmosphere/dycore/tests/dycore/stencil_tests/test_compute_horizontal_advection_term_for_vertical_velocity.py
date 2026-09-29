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

from icon4py.model.atmosphere.dycore.stencils.compute_horizontal_advection_term_for_vertical_velocity import (
    compute_horizontal_advection_term_for_vertical_velocity,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def compute_horizontal_advection_term_for_vertical_velocity_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    vn_ie: np.ndarray,
    inv_dual_edge_length: np.ndarray,
    w: np.ndarray,
    z_vt_ie: np.ndarray,
    inv_primal_edge_length: np.ndarray,
    tangent_orientation: np.ndarray,
    z_w_v: np.ndarray,
) -> np.ndarray:
    inv_dual_edge_length = np.expand_dims(inv_dual_edge_length, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)
    tangent_orientation = np.expand_dims(tangent_orientation, axis=-1)

    w_e2c = w[connectivities[dims.E2CDim]]
    z_w_v_e2v = z_w_v[connectivities[dims.E2VDim]]

    red_w = w_e2c[:, 0] - w_e2c[:, 1]
    red_z_w_v = z_w_v_e2v[:, 0] - z_w_v_e2v[:, 1]

    z_v_grad_w = (
        vn_ie * inv_dual_edge_length * red_w
        + z_vt_ie * inv_primal_edge_length * tangent_orientation * red_z_w_v
    )
    return z_v_grad_w


class TestComputeHorizontalAdvectionTermForVerticalVelocity(StencilTest):
    PROGRAM = compute_horizontal_advection_term_for_vertical_velocity
    OUTPUTS = ("z_v_grad_w",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        vn_ie: np.ndarray,
        inv_dual_edge_length: np.ndarray,
        w: np.ndarray,
        z_vt_ie: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        tangent_orientation: np.ndarray,
        z_w_v: np.ndarray,
        z_v_grad_w: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict:
        z_v_grad_w[horizontal_start:horizontal_end, :] = (
            compute_horizontal_advection_term_for_vertical_velocity_numpy(
                connectivities=connectivities,
                vn_ie=vn_ie,
                inv_dual_edge_length=inv_dual_edge_length,
                w=w,
                z_vt_ie=z_vt_ie,
                inv_primal_edge_length=inv_primal_edge_length,
                tangent_orientation=tangent_orientation,
                z_w_v=z_w_v,
            )
        )[horizontal_start:horizontal_end, :]
        return dict(z_v_grad_w=z_v_grad_w)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        vn_ie = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        inv_dual_edge_length = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        w = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        z_vt_ie = random_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)
        inv_primal_edge_length = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        tangent_orientation = random_field(grid, dims.EdgeDim, dtype=wpfloat)
        z_w_v = random_field(grid, dims.VertexDim, dims.KDim, dtype=vpfloat)
        z_v_grad_w = zero_field(grid, dims.EdgeDim, dims.KDim, dtype=vpfloat)

        edge_domain = h_grid.domain(dims.EdgeDim)
        horizontal_start = grid.start_index(edge_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7))
        horizontal_end = grid.end_index(edge_domain(h_grid.Zone.HALO))

        return dict(
            vn_ie=vn_ie,
            inv_dual_edge_length=inv_dual_edge_length,
            w=w,
            z_vt_ie=z_vt_ie,
            inv_primal_edge_length=inv_primal_edge_length,
            tangent_orientation=tangent_orientation,
            z_w_v=z_w_v,
            z_v_grad_w=z_v_grad_w,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
