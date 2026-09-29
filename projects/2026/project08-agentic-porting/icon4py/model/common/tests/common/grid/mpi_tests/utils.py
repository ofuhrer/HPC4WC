# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import pathlib

import gt4py.next as gtx

from icon4py.model.common.decomposition import decomposer as decomp, definitions as decomp_defs
from icon4py.model.common.grid import grid_manager as gm, vertical as v_grid


def _grid_manager(file: pathlib.Path, num_levels: int) -> gm.GridManager:
    return gm.GridManager(
        grid_file=str(file), config=v_grid.VerticalGridConfig(num_levels=num_levels)
    )


NUM_LEVELS = 10


def run_grid_manager_for_single_rank(
    file: pathlib.Path,
    allocator: gtx.typing.Allocator,
    num_levels: int = NUM_LEVELS,
) -> gm.GridManager:
    manager = _grid_manager(file, num_levels)
    manager(
        keep_skip_values=True,
        process_props=decomp_defs.SingleNodeProcessProperties(),
        decomposer=decomp.SingleNodeDecomposer(),
        allocator=allocator,
    )
    return manager


def run_grid_manager_for_multi_rank(
    file: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    decomposer: decomp.Decomposer,
    allocator: gtx.typing.Allocator,
    num_levels: int = NUM_LEVELS,
) -> gm.GridManager:
    manager = _grid_manager(file, num_levels)
    manager(
        keep_skip_values=True,
        allocator=allocator,
        process_props=process_props,
        decomposer=decomposer,
    )
    return manager
