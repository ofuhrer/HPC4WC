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

from icon4py.model.atmosphere.advection.stencils.compute_vertical_tracer_flux_upwind import (
    compute_vertical_tracer_flux_upwind,
)
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests


outslice = (slice(None), slice(1, None))


class TestComputeVerticalTracerFluxUpwind(stencil_tests.StencilTest):
    PROGRAM = compute_vertical_tracer_flux_upwind
    OUTPUTS = (stencil_tests.Output("p_upflux", refslice=outslice, gtslice=outslice),)

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        p_cc: np.ndarray,
        p_mflx_contra_v: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        p_upflux = p_cc.copy()
        p_upflux[:, 1:] = (
            np.where(p_mflx_contra_v[:, 1:] >= 0.0, p_cc[:, 1:], p_cc[:, :-1])
            * p_mflx_contra_v[:, 1:]
        )
        return dict(p_upflux=p_upflux)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        p_cc = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
        p_mflx_contra_v = data_alloc.random_field(
            grid, dims.CellDim, dims.KDim
        )  # TODO(dastrm): should be KHalfDim
        p_upflux = data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim
        )  # TODO(dastrm): should be KHalfDim
        return dict(
            p_cc=p_cc,
            p_mflx_contra_v=p_mflx_contra_v,
            p_upflux=p_upflux,
            horizontal_start=0,
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=1,
            vertical_end=gtx.int32(grid.num_levels),
        )
