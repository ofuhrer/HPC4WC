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

from icon4py.model.atmosphere.advection.stencils.limit_vertical_parabola_semi_monotonically import (
    limit_vertical_parabola_semi_monotonically,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestLimitVerticalParabolaSemiMonotonically(stencil_tests.StencilTest):
    PROGRAM = limit_vertical_parabola_semi_monotonically
    OUTPUTS = ("p_face_up", "p_face_low")

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        l_limit: np.ndarray,
        p_face: np.ndarray,
        p_cc: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        q_face_up, q_face_low = np.where(
            l_limit != 0,
            np.where(
                (p_cc < np.minimum(p_face[:, :-1], p_face[:, 1:])),
                (p_cc, p_cc),
                np.where(
                    p_face[:, :-1] > p_face[:, 1:],
                    (3.0 * p_cc - 2.0 * p_face[:, 1:], p_face[:, 1:]),
                    (p_face[:, :-1], 3.0 * p_cc - 2.0 * p_face[:, :-1]),
                ),
            ),
            (p_face[:, :-1], p_face[:, 1:]),
        )
        return dict(p_face_up=q_face_up, p_face_low=q_face_low)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        l_limit = data_alloc.random_mask(grid, dims.CellDim, dims.KDim, dtype=gtx.int32)
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_face = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        p_face_up = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_face_low = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            l_limit=l_limit,
            p_cc=p_cc,
            p_face=p_face,
            p_face_up=p_face_up,
            p_face_low=p_face_low,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
