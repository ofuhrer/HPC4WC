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
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.interpolation.stencils.mo_intp_rbf_rbf_vec_interpol_vertex import (
    mo_intp_rbf_rbf_vec_interpol_vertex,
)
from icon4py.model.common.type_alias import wpfloat
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing.stencil_tests import StandardStaticVariants, StencilTest


@pytest.mark.continuous_benchmarking
class TestMoIntpRbfRbfVecInterpolVertex(StencilTest):
    PROGRAM = mo_intp_rbf_rbf_vec_interpol_vertex
    OUTPUTS = ("p_u_out", "p_v_out")
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
        p_e_in: np.ndarray,
        ptr_coeff_1: np.ndarray,
        ptr_coeff_2: np.ndarray,
        horizontal_start: int,
        horizontal_end: int,
        **kwargs: Any,
    ) -> dict[str, np.ndarray]:
        v2e = connectivities[dims.V2EDim]
        ptr_coeff_1 = np.expand_dims(ptr_coeff_1, axis=-1)
        p_u_out = np.sum(
            np.where(np.expand_dims(v2e, axis=-1) >= 0, p_e_in[v2e] * ptr_coeff_1, 0.0), axis=1
        )

        ptr_coeff_2 = np.expand_dims(ptr_coeff_2, axis=-1)
        p_v_out = np.sum(
            np.where(np.expand_dims(v2e, axis=-1) >= 0, p_e_in[v2e] * ptr_coeff_2, 0.0), axis=1
        )
        p_u_final_out = np.zeros_like(p_u_out)  # Same as initial values of p_u_out
        p_v_final_out = np.zeros_like(p_v_out)  # Same as initial values of p_v_out
        p_u_final_out[horizontal_start:horizontal_end, :] = p_u_out[
            horizontal_start:horizontal_end, :
        ]
        p_v_final_out[horizontal_start:horizontal_end, :] = p_v_out[
            horizontal_start:horizontal_end, :
        ]

        return dict(p_v_out=p_v_final_out, p_u_out=p_u_final_out)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_e_in = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, dtype=wpfloat)
        ptr_coeff_1 = data_alloc.random_field(grid, dims.VertexDim, dims.V2EDim, dtype=wpfloat)
        ptr_coeff_2 = data_alloc.random_field(grid, dims.VertexDim, dims.V2EDim, dtype=wpfloat)
        p_v_out = data_alloc.zero_field(grid, dims.VertexDim, dims.KDim, dtype=wpfloat)
        p_u_out = data_alloc.zero_field(grid, dims.VertexDim, dims.KDim, dtype=wpfloat)

        vertex_domain = h_grid.domain(dims.VertexDim)
        horizontal_start = grid.start_index(vertex_domain(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2))
        horizontal_end = grid.end_index(vertex_domain(h_grid.Zone.LOCAL))

        return dict(
            p_e_in=p_e_in,
            ptr_coeff_1=ptr_coeff_1,
            ptr_coeff_2=ptr_coeff_2,
            p_v_out=p_v_out,
            p_u_out=p_u_out,
            horizontal_start=horizontal_start,
            horizontal_end=horizontal_end,
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
