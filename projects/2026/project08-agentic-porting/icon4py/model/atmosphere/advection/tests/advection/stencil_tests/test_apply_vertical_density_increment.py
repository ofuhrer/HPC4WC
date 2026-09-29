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

from icon4py.model.atmosphere.advection.stencils.apply_vertical_density_increment import (
    apply_vertical_density_increment,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


class TestApplyVerticalDensityIncrement(stencil_tests.StencilTest):
    PROGRAM = apply_vertical_density_increment
    OUTPUTS = (
        stencil_tests.Output(
            "rhodz_ast2",
            refslice=(slice(None), slice(None, -1)),
            gtslice=(slice(None), slice(None, -1)),
        ),
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        rhodz_ast: np.ndarray,
        p_mflx_contra_v: np.ndarray,
        deepatmo_divzl: np.ndarray,
        deepatmo_divzu: np.ndarray,
        p_dtime: float,
        **kwargs: Any,
    ) -> dict:
        tmp = p_dtime * (
            p_mflx_contra_v[:, 1:] * deepatmo_divzl - p_mflx_contra_v[:, :-1] * deepatmo_divzu
        )
        rhodz_ast2 = rhodz_ast + tmp

        return dict(rhodz_ast2=rhodz_ast2)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        rhodz_ast = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_mflx_contra_v = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1}
        )
        deepatmo_divzl = data_alloc.random_field(grid, dims.KDim)
        deepatmo_divzu = data_alloc.random_field(grid, dims.KDim)
        rhodz_ast2 = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        p_dtime = 0.1
        return dict(
            rhodz_ast=rhodz_ast,
            p_mflx_contra_v=p_mflx_contra_v,
            deepatmo_divzl=deepatmo_divzl,
            deepatmo_divzu=deepatmo_divzu,
            rhodz_ast2=rhodz_ast2,
            p_dtime=p_dtime,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=0,
            vertical_end=gtx.int32(grid.num_levels),
        )
