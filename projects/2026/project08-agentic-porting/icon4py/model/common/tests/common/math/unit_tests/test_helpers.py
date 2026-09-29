# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Any

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

import icon4py.model.testing.test_utils
from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base, simple
from icon4py.model.common.math import (
    vector_operations as vector_ops,
    vertical_operations as vertical_ops,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import stencil_tests
from icon4py.model.testing.fixtures.datatest import backend, backend_like
from icon4py.model.testing.fixtures.stencil_tests import grid, grid_manager


def test_cross_product(backend: gtx_typing.Backend) -> None:
    mesh = simple.simple_grid(allocator=backend)
    x1 = data_alloc.random_field(mesh, dims.EdgeDim, allocator=backend)
    y1 = data_alloc.random_field(mesh, dims.EdgeDim, allocator=backend)
    z1 = data_alloc.random_field(mesh, dims.EdgeDim, allocator=backend)
    x2 = data_alloc.random_field(mesh, dims.EdgeDim, allocator=backend)
    y2 = data_alloc.random_field(mesh, dims.EdgeDim, allocator=backend)
    z2 = data_alloc.random_field(mesh, dims.EdgeDim, allocator=backend)
    x = data_alloc.zero_field(mesh, dims.EdgeDim, allocator=backend)
    y = data_alloc.zero_field(mesh, dims.EdgeDim, allocator=backend)
    z = data_alloc.zero_field(mesh, dims.EdgeDim, allocator=backend)

    vector_ops.cross_product_on_edges.with_backend(backend)(
        x1, x2, y1, y2, z1, z2, out=(x, y, z), offset_provider={}
    )
    a = np.column_stack((x1.asnumpy(), y1.asnumpy(), z1.asnumpy()))
    b = np.column_stack((x2.asnumpy(), y2.asnumpy(), z2.asnumpy()))
    c = np.cross(a, b)

    assert icon4py.model.testing.test_utils.dallclose(c[:, 0], x.asnumpy())
    assert icon4py.model.testing.test_utils.dallclose(c[:, 1], y.asnumpy())
    assert icon4py.model.testing.test_utils.dallclose(c[:, 2], z.asnumpy())


class TestAverageTwoVerticalLevelsDownwardsOnEdges(stencil_tests.StencilTest):
    PROGRAM = vertical_ops.average_two_vertical_levels_downwards_on_edges
    OUTPUTS = (
        stencil_tests.Output(
            "average",
            refslice=(slice(None), slice(None, -1)),
            gtslice=(slice(None), slice(None, -1)),
        ),
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        input_field: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        offset = np.roll(input_field, shift=1, axis=1)
        average = 0.5 * (input_field + offset)
        return dict(average=average)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        input_field = data_alloc.zero_field(grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
        result = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim, extend={dims.KDim: 1})
        return dict(
            input_field=input_field,
            average=result,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_edges),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )


class TestAverageTwoVerticalLevelsDownwardsOnCells(stencil_tests.StencilTest):
    PROGRAM = vertical_ops.average_two_vertical_levels_downwards_on_cells
    OUTPUTS = (
        stencil_tests.Output(
            "average",
            refslice=(slice(None), slice(None, -1)),
            gtslice=(slice(None), slice(None, -1)),
        ),
    )

    @staticmethod
    def reference(
        connectivities: dict[gtx.Dimension, np.ndarray],
        *,
        input_field: np.ndarray,
        **kwargs: Any,
    ) -> dict:
        shp = input_field.shape
        res = 0.5 * (input_field + np.roll(input_field, shift=-1, axis=1))[:, : shp[1] - 1]
        return dict(average=res)

    @pytest.fixture
    def input_data(self, grid: base.Grid) -> dict:
        input_field = data_alloc.random_field(grid, dims.CellDim, dims.KDim, extend={dims.KDim: 1})
        result = data_alloc.zero_field(grid, dims.CellDim, dims.KDim)
        return dict(
            input_field=input_field,
            average=result,
            horizontal_start=gtx.int32(0),
            horizontal_end=gtx.int32(grid.num_cells),
            vertical_start=gtx.int32(0),
            vertical_end=gtx.int32(grid.num_levels),
        )
