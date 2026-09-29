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

from icon4py.model.atmosphere.advection.stencils.compute_ppm_quadratic_face_values import (
    compute_ppm_quadratic_face_values,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


outslice = (slice(None), slice(1, None))


class TestComputePpmQuadraticFaceValues(stencil_tests.StencilTest):
    PROGRAM = compute_ppm_quadratic_face_values
    OUTPUTS = (stencil_tests.Output("p_face", refslice=outslice, gtslice=outslice),)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_cc: np.ndarray,
        p_cellhgt_mc_now: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        p_face = p_cc.copy()
        p_face[:, 1:] = p_cc[:, 1:] * (
            1.0 - (p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1])
        ) + (p_cellhgt_mc_now[:, 1:] / (p_cellhgt_mc_now[:, :-1] + p_cellhgt_mc_now[:, 1:])) * (
            (p_cellhgt_mc_now[:, 1:] / p_cellhgt_mc_now[:, :-1]) * p_cc[:, 1:] + p_cc[:, :-1]
        )
        return dict(p_face=p_face)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_face = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_cellhgt_mc_now = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        return dict(
            p_cc=p_cc,
            p_cellhgt_mc_now=p_cellhgt_mc_now,
            p_face=p_face,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
