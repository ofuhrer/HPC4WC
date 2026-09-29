# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, ClassVar

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import icon as icon_grid
from icon4py.model.common.utils import data_allocation as data_alloc


if TYPE_CHECKING:
    from icon4py.model.common.grid import grid_manager as gm


@dataclasses.dataclass
class FlatTopographyConfig:
    # Empty config class, used as "tag type" for the match/case statement in
    # topography.config
    fortran_name_map: ClassVar[dict[str, str]] = {}


def flat_topography(
    *,
    grid_manager: gm.GridManager,
) -> data_alloc.NDArray:

    match grid_manager.grid.geometry_type:
        case icon_grid.GeometryType.ICOSAHEDRON:
            cell_x = grid_manager.coordinates[dims.CellDim]["lon"].ndarray
        case icon_grid.GeometryType.TORUS:
            cell_x = grid_manager.coordinates[dims.CellDim]["x"].ndarray

    array_ns = data_alloc.array_namespace(cell_x)

    return array_ns.zeros_like(cell_x)
