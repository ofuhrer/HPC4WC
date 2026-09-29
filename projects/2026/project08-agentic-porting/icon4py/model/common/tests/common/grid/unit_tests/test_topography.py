# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from icon4py.model.common import topography as topo
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.testing import test_utils
from icon4py.model.testing.fixtures import *  # noqa: F403


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.embedded_remap_error
@pytest.mark.datatest
def test_topography_smoothing_with_serialized_data(
    icon_grid: base_grid.Grid,
    grid_savepoint: sb.IconGridSavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    topography_savepoint: sb.TopographySavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    cell_geometry = grid_savepoint.construct_cell_geometry()
    assert cell_geometry.area is not None, (
        "Broken assumption: this test assumes it's running from a savepoint containing a 'cell_area' field."
    )
    geofac_n2s = interpolation_savepoint.geofac_n2s()

    num_iterations = 25
    topography = topography_savepoint.topo_c()
    topography_smoothed_ref = topography_savepoint.topo_smt_c().asnumpy()

    topography_smoothed = topo.smooth_topography(
        topography=topography.ndarray,
        cell_areas=cell_geometry.area.ndarray,
        geofac_n2s=geofac_n2s.ndarray,
        c2e2co=icon_grid.get_connectivity("C2E2CO").ndarray,
        num_iterations=num_iterations,
        exchange=decomposition.single_node_exchange,
    )

    assert test_utils.dallclose(topography_smoothed_ref, topography_smoothed, atol=1.0e-14)
