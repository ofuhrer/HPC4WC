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

from icon4py.model.atmosphere.dycore.stencils.mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl import (
    mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.states import utils as state_utils
from icon4py.model.common.type_alias import vpfloat, wpfloat
from icon4py.model.common.utils.data_allocation import random_field, zero_field
from icon4py.model.testing.stencil_tests import StencilTest


def mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
    connectivities: dict[gtx.Dimension, np.ndarray], p_cell_in: np.ndarray, c_intp: np.ndarray
) -> np.ndarray:
    v2c = connectivities[dims.V2CDim]
    c_intp = np.expand_dims(c_intp, axis=-1)
    p_vert_out = np.sum(np.where((v2c != -1)[:, :, np.newaxis], p_cell_in[v2c] * c_intp, 0), axis=1)
    return p_vert_out


class TestMoIconInterpolationScalarCells2vertsScalarRiDsl(StencilTest):
    PROGRAM = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl
    OUTPUTS = ("p_vert_out",)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_cell_in: np.ndarray,
        c_intp: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        p_vert_out = mo_icon_interpolation_scalar_cells2verts_scalar_ri_dsl_numpy(
            connectivities, p_cell_in, c_intp
        )
        return dict(
            p_vert_out=p_vert_out,
        )

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict[str, gtx.Field | state_utils.ScalarType]:
        p_cell_in = random_field(grid, dims.CellDim, dims.KDim, dtype=wpfloat)
        c_intp = random_field(grid, dims.VertexDim, dims.V2CDim, dtype=wpfloat)
        p_vert_out = zero_field(grid, dims.VertexDim, dims.KDim, dtype=vpfloat)

        return dict(
            p_cell_in=p_cell_in,
            c_intp=c_intp,
            p_vert_out=p_vert_out,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_vertices),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
