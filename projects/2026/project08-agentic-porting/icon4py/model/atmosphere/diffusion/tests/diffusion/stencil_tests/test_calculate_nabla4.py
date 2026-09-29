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
from icon4py.model.atmosphere.diffusion.stencils.calculate_nabla4 import calculate_nabla4
from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.testing.stencil_tests import StandardStaticVariants, StencilTest


def calculate_nabla4_numpy(
    *,
    connectivities: dict[gtx.Dimension, np.ndarray],
    u_vert: np.ndarray,
    v_vert: np.ndarray,
    primal_normal_vert_v1: np.ndarray,
    primal_normal_vert_v2: np.ndarray,
    z_nabla2_e: np.ndarray,
    inv_vert_vert_length: np.ndarray,
    inv_primal_edge_length: np.ndarray,
) -> np.ndarray:
    e2c2v = connectivities[dims.E2C2VDim]
    u_vert_e2c2v = u_vert[e2c2v]
    v_vert_e2c2v = v_vert[e2c2v]

    primal_normal_vert_v1 = np.expand_dims(primal_normal_vert_v1, axis=-1)
    primal_normal_vert_v2 = np.expand_dims(primal_normal_vert_v2, axis=-1)
    inv_vert_vert_length = np.expand_dims(inv_vert_vert_length, axis=-1)
    inv_primal_edge_length = np.expand_dims(inv_primal_edge_length, axis=-1)

    nabv_tang = (
        u_vert_e2c2v[:, 0] * primal_normal_vert_v1[:, 0]
        + v_vert_e2c2v[:, 0] * primal_normal_vert_v2[:, 0]
    ) + (
        u_vert_e2c2v[:, 1] * primal_normal_vert_v1[:, 1]
        + v_vert_e2c2v[:, 1] * primal_normal_vert_v2[:, 1]
    )
    nabv_norm = (
        u_vert_e2c2v[:, 2] * primal_normal_vert_v1[:, 2]
        + v_vert_e2c2v[:, 2] * primal_normal_vert_v2[:, 2]
    ) + (
        u_vert_e2c2v[:, 3] * primal_normal_vert_v1[:, 3]
        + v_vert_e2c2v[:, 3] * primal_normal_vert_v2[:, 3]
    )
    z_nabla4_e2 = 4.0 * (
        (nabv_norm - 2.0 * z_nabla2_e) * inv_vert_vert_length**2
        + (nabv_tang - 2.0 * z_nabla2_e) * inv_primal_edge_length**2
    )
    return z_nabla4_e2


@pytest.mark.continuous_benchmarking
class TestCalculateNabla4(StencilTest):
    PROGRAM = calculate_nabla4
    OUTPUTS = ("z_nabla4_e2",)
    STATIC_PARAMS = {
        StandardStaticVariants.NONE: (),
        StandardStaticVariants.COMPILE_TIME_DOMAIN: (
            "horizontal_start",
            "horizontal_end",
            "vertical_start",
            "vertical_end",
        ),
        StandardStaticVariants.COMPILE_TIME_VERTICAL: (
            "vertical_start",
            "vertical_end",
        ),
    }

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        u_vert: np.ndarray,
        v_vert: np.ndarray,
        primal_normal_vert_v1: np.ndarray,
        primal_normal_vert_v2: np.ndarray,
        z_nabla2_e: np.ndarray,
        inv_vert_vert_length: np.ndarray,
        inv_primal_edge_length: np.ndarray,
        **kwargs,
    ) -> dict:
        z_nabla4_e2 = calculate_nabla4_numpy(
            connectivities=connectivities,
            u_vert=u_vert,
            v_vert=v_vert,
            primal_normal_vert_v1=primal_normal_vert_v1,
            primal_normal_vert_v2=primal_normal_vert_v2,
            z_nabla2_e=z_nabla2_e,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_primal_edge_length=inv_primal_edge_length,
        )
        return dict(z_nabla4_e2=z_nabla4_e2)

    @pytest.fixture
    def input_data(self, grid) -> dict:
        u_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=ta.vpfloat)
        v_vert = data_alloc.random_field(grid, dims.VertexDim, dims.KDim, dtype=ta.vpfloat)

        primal_normal_vert_v1 = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )
        primal_normal_vert_v2 = data_alloc.random_field(
            grid, dims.EdgeDim, dims.E2C2VDim, dtype=ta.wpfloat
        )

        z_nabla2_e = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat)
        inv_vert_vert_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)
        inv_primal_edge_length = data_alloc.random_field(grid, dims.EdgeDim, dtype=ta.wpfloat)

        z_nabla4_e2 = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, dtype=ta.vpfloat)

        return dict(
            u_vert=u_vert,
            v_vert=v_vert,
            primal_normal_vert_v1=primal_normal_vert_v1,
            primal_normal_vert_v2=primal_normal_vert_v2,
            z_nabla2_e=z_nabla2_e,
            inv_vert_vert_length=inv_vert_vert_length,
            inv_primal_edge_length=inv_primal_edge_length,
            z_nabla4_e2=z_nabla4_e2,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
