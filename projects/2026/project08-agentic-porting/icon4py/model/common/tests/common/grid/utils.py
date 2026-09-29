# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Iterator

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import grid_manager as gm
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, grid_utils as gridtest_utils


managers: dict[str, gm.GridManager] = {}


def horizontal_offsets() -> Iterator[gtx.FieldOffset]:
    for d in vars(dims).values():
        if isinstance(d, gtx.FieldOffset) and len(d.target) == 2:
            yield d


def run_grid_manager(
    grid: test_defs.GridDescription,
    keep_skip_values: bool,
    backend: gtx_typing.Backend | None,
) -> gm.GridManager:
    key = "_".join(
        (grid.name, data_alloc.backend_name(backend), "skip" if keep_skip_values else "no_skip")
    )
    if (manager := managers.get(key)) is not None:
        return manager
    else:
        manager = gridtest_utils.get_grid_manager_from_identifier(
            grid,
            keep_skip_values=keep_skip_values,
            num_levels=1,
            allocator=model_backends.get_allocator(backend),
        )
        managers[key] = manager
        return manager


GRID_REFERENCE_VALUES: dict[str, dict[str, float]] = {
    test_defs.Grids.R01B01_GLOBAL.name: {
        "num_cells": 80,
        "num_vertices": 42,
        "num_edges": 120,
    },
    test_defs.Grids.R02B04_GLOBAL.name: {
        "num_cells": 20480,
        "num_vertices": 10242,
        "num_edges": 30720,
        "mean_edge_length": 239835.16092638028,
        "mean_dual_edge_length": 138468.89472198288,
        "mean_cell_area": 24907282236.708576,
        "mean_dual_area": 49804836966.19719,
    },
    test_defs.Grids.R02B06_GLOBAL.name: {
        "num_cells": 327680,
        "num_vertices": 163842,
        "num_edges": 491520,
    },
    test_defs.Grids.R02B07_GLOBAL.name: {
        "num_cells": 1310720,
        "num_vertices": 655362,
        "num_edges": 1966080,
    },
    test_defs.Grids.R19_B07_MCH_LOCAL.name: {
        "num_cells": 283876,
        "num_vertices": 142724,
        "num_edges": 426599,
    },
    test_defs.Grids.MCH_OPR_R04B07_DOMAIN01.name: {
        "num_cells": 10700,
        "num_vertices": 5510,
        "num_edges": 16209,
    },
    test_defs.Grids.MCH_OPR_R19B08_DOMAIN01.name: {
        "num_cells": 44528,
        "num_vertices": 22569,
        "num_edges": 67096,
    },
    test_defs.Grids.MCH_CH_R04B09_DSL.name: {
        "num_cells": 20896,
        "num_vertices": 10663,
        "num_edges": 31558,
        "mean_edge_length": 3747.424389474692,
        "mean_dual_edge_length": 2163.5764800309826,
        "mean_cell_area": 6080879.45232143,
        "mean_dual_area": 12161758.324725032,
    },
    test_defs.Grids.TORUS_100X116_1000M.name: {
        "num_cells": 23200,
        "num_vertices": 11600,
        "num_edges": 34800,
    },
    test_defs.Grids.TORUS_50000x5000.name: {
        "num_cells": 1056,
        "num_vertices": 528,
        "num_edges": 1584,
        "mean_edge_length": 757.5757575757576,
        "mean_dual_edge_length": 437.3865675678984,
        "mean_cell_area": 248515.09520903317,
        "mean_dual_area": 497030.1904180664,
    },
    test_defs.Grids.TORUS_1000X1000_250M.name: {
        "num_cells": 24,
        "num_vertices": 12,
        "num_edges": 36,
    },
}
