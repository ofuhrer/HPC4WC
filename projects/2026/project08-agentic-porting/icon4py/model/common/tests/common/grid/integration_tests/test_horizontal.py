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
from gt4py import next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.testing import definitions
from icon4py.model.testing.fixtures import download_ser_data, experiment, experiment_description

from ...fixtures import *  # noqa: F403
from .. import utils


if TYPE_CHECKING:
    import numpy as np

    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
@pytest.mark.parametrize("dim", dims.horizontal_dims())
def test_map_icon_start_end_index(
    experiment: definitions.Experiment, dim: gtx.Dimension, grid_savepoint: sb.IconGridSavepoint
) -> None:
    end_indices = grid_savepoint.end_index()
    start_indices = grid_savepoint.start_index()
    start_map, end_map = h_grid.get_start_end_idx_from_icon_arrays(dim, start_indices, end_indices)
    _assert_domain_map(start_map, start_indices[dim])
    _assert_domain_map(end_map, end_indices[dim])


def _assert_domain_map(index_map: dict[h_grid.Domain, gtx.int32], index_array: np.ndarray) -> None:  # noqa: PLR0912
    same_index = False
    for d, index in index_map.items():
        if d.zone == h_grid.Zone.INTERIOR:
            same_index = index == index_array[h_grid._ICON_INTERIOR[d.dim]]
        if d.zone == h_grid.Zone.LOCAL:
            same_index = index == index_array[h_grid.ICON_LOCAL[d.dim]]
        if d.zone == h_grid.Zone.END:
            same_index = index == index_array[h_grid._ICON_END[d.dim]]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim]]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 1]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 2]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 3]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_5:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 4]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_6:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 5]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_7:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 6]
        if d.zone == h_grid.Zone.LATERAL_BOUNDARY_LEVEL_8:
            same_index = index == index_array[h_grid._ICON_LATERAL_BOUNDARY[d.dim] + 7]
        if d.zone == h_grid.Zone.NUDGING:
            same_index = index == index_array[h_grid._ICON_NUDGING[d.dim]]
        if d.zone == h_grid.Zone.NUDGING_LEVEL_2:
            same_index = index == index_array[h_grid._ICON_NUDGING[d.dim] + 1]
        if d.zone == h_grid.Zone.HALO:
            same_index = index == index_array[h_grid._ICON_HALO[d.dim]]
        if d.zone == h_grid.Zone.HALO_LEVEL_2:
            same_index = index == index_array[h_grid._ICON_HALO[d.dim] - 1]
        if not same_index:
            raise AssertionError(f"Wrong index for {d.zone} zone in dimension {d.dim}: ")
