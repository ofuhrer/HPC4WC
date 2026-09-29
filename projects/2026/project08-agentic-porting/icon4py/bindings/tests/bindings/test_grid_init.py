# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from unittest import mock

import cffi
import numpy as np
import pytest

from icon4py.bindings import (
    common as wrapper_common,
    debug_utils as wrapper_debug_utils,
    grid_wrapper,
)
from icon4py.model.common import dimension as dims
from icon4py.tools.py2fgen import test_utils


def _grid_init_kwargs(grid_savepoint, experiment, *, parallel_arrays, comm_id):
    """Build the kwargs dict for ``grid_wrapper.grid_init`` from savepoint data."""
    num_vertices = grid_savepoint.num(dims.VertexDim)
    num_cells = grid_savepoint.num(dims.CellDim)
    num_edges = grid_savepoint.num(dims.EdgeDim)
    vertical_size = grid_savepoint.num(dims.KDim)
    limited_area = grid_savepoint.get_metadata("limited_area").get("limited_area")
    cfg = experiment.config.vertical_grid

    return {
        "cell_starts": test_utils.array_to_array_info(grid_savepoint._read_int32("c_start_index")),
        "cell_ends": test_utils.array_to_array_info(grid_savepoint._read_int32("c_end_index")),
        "vertex_starts": test_utils.array_to_array_info(
            grid_savepoint._read_int32("v_start_index")
        ),
        "vertex_ends": test_utils.array_to_array_info(grid_savepoint._read_int32("v_end_index")),
        "edge_starts": test_utils.array_to_array_info(grid_savepoint._read_int32("e_start_index")),
        "edge_ends": test_utils.array_to_array_info(grid_savepoint._read_int32("e_end_index")),
        "c2e": test_utils.array_to_array_info(grid_savepoint._read_int32("c2e")),
        "e2c": test_utils.array_to_array_info(grid_savepoint._read_int32("e2c")),
        "c2e2c": test_utils.array_to_array_info(grid_savepoint._read_int32("c2e2c")),
        "e2c2e": test_utils.array_to_array_info(grid_savepoint._read_int32("e2c2e")),
        "e2v": test_utils.array_to_array_info(grid_savepoint._read_int32("e2v")),
        "v2e": test_utils.array_to_array_info(grid_savepoint._read_int32("v2e")),
        "v2c": test_utils.array_to_array_info(grid_savepoint._read_int32("v2c")),
        "e2c2v": test_utils.array_to_array_info(grid_savepoint._read_int32("e2c2v")),
        "c2v": test_utils.array_to_array_info(grid_savepoint._read_int32("c2v")),
        "num_vertices": num_vertices,
        "num_cells": num_cells,
        "num_edges": num_edges,
        "vertical_size": vertical_size,
        "limited_area": limited_area,
        "tangent_orientation": test_utils.array_to_array_info(
            grid_savepoint.tangent_orientation().ndarray
        ),
        "inverse_primal_edge_lengths": test_utils.array_to_array_info(
            grid_savepoint.inverse_primal_edge_lengths().ndarray
        ),
        "inv_dual_edge_length": test_utils.array_to_array_info(
            grid_savepoint.inv_dual_edge_length().ndarray
        ),
        "inv_vert_vert_length": test_utils.array_to_array_info(
            grid_savepoint.inv_vert_vert_length().ndarray
        ),
        "edge_areas": test_utils.array_to_array_info(grid_savepoint.edge_areas().ndarray),
        "f_e": test_utils.array_to_array_info(grid_savepoint.f_e().ndarray),
        "cell_center_lat": test_utils.array_to_array_info(grid_savepoint.cell_center_lat().ndarray),
        "cell_center_lon": test_utils.array_to_array_info(grid_savepoint.cell_center_lon().ndarray),
        "cell_areas": test_utils.array_to_array_info(grid_savepoint.cell_areas().ndarray),
        "primal_normal_vert_x": test_utils.array_to_array_info(
            grid_savepoint.primal_normal_vert_x().ndarray
        ),
        "primal_normal_vert_y": test_utils.array_to_array_info(
            grid_savepoint.primal_normal_vert_y().ndarray
        ),
        "dual_normal_vert_x": test_utils.array_to_array_info(
            grid_savepoint.dual_normal_vert_x().ndarray
        ),
        "dual_normal_vert_y": test_utils.array_to_array_info(
            grid_savepoint.dual_normal_vert_y().ndarray
        ),
        "primal_normal_cell_x": test_utils.array_to_array_info(
            grid_savepoint.primal_normal_cell_x().ndarray
        ),
        "primal_normal_cell_y": test_utils.array_to_array_info(
            grid_savepoint.primal_normal_cell_y().ndarray
        ),
        "dual_normal_cell_x": test_utils.array_to_array_info(
            grid_savepoint.dual_normal_cell_x().ndarray
        ),
        "dual_normal_cell_y": test_utils.array_to_array_info(
            grid_savepoint.dual_normal_cell_y().ndarray
        ),
        "edge_center_lat": test_utils.array_to_array_info(grid_savepoint.edge_center_lat().ndarray),
        "edge_center_lon": test_utils.array_to_array_info(grid_savepoint.edge_center_lon().ndarray),
        "primal_normal_x": test_utils.array_to_array_info(
            grid_savepoint.primal_normal_v1().ndarray
        ),
        "primal_normal_y": test_utils.array_to_array_info(
            grid_savepoint.primal_normal_v2().ndarray
        ),
        "vct_a": test_utils.array_to_array_info(grid_savepoint.vct_a().ndarray),
        "lowest_layer_thickness": cfg.lowest_layer_thickness,
        "model_top_height": cfg.model_top_height,
        "stretch_factor": cfg.stretch_factor,
        "flat_height": cfg.flat_height,
        "rayleigh_damping_height": cfg.rayleigh_damping_height,
        "mean_cell_area": grid_savepoint.mean_cell_area(),
        **parallel_arrays,
        "comm_id": comm_id,
        "backend": wrapper_common.BackendIntEnum.DEFAULT,
    }


@pytest.fixture
def grid_init(grid_savepoint, experiment):
    # serial mode: parallel arrays unused inside grid_init
    _dummy_int = np.array([], dtype=np.int32)
    _dummy_bool = np.array([], dtype=np.bool_)
    parallel_arrays = {
        "c_glb_index": test_utils.array_to_array_info(_dummy_int),
        "e_glb_index": test_utils.array_to_array_info(_dummy_int),
        "v_glb_index": test_utils.array_to_array_info(_dummy_int),
        "c_owner_mask": test_utils.array_to_array_info(_dummy_bool),
        "e_owner_mask": test_utils.array_to_array_info(_dummy_bool),
        "v_owner_mask": test_utils.array_to_array_info(_dummy_bool),
    }
    grid_wrapper.grid_init(
        cffi.FFI(),
        perf_counters=None,
        **_grid_init_kwargs(
            grid_savepoint, experiment, parallel_arrays=parallel_arrays, comm_id=None
        ),
    )


@pytest.mark.datatest
@pytest.mark.parametrize("backend", [None])
def test_grid_init_calls_decomposition_helpers_with_kwargs(grid_savepoint, experiment, backend):
    """grid_init must call ``construct_decomposition`` and ``print_grid_decomp_info`` with
    kwargs — both are keyword-only since #1270 and a positional call TypeErrors at runtime."""
    num_cells = grid_savepoint.num(dims.CellDim)
    num_edges = grid_savepoint.num(dims.EdgeDim)
    num_vertices = grid_savepoint.num(dims.VertexDim)
    parallel_arrays = {
        "c_glb_index": test_utils.array_to_array_info(np.arange(num_cells, dtype=np.int32)),
        "e_glb_index": test_utils.array_to_array_info(np.arange(num_edges, dtype=np.int32)),
        "v_glb_index": test_utils.array_to_array_info(np.arange(num_vertices, dtype=np.int32)),
        "c_owner_mask": test_utils.array_to_array_info(np.ones(num_cells, dtype=np.bool_)),
        "e_owner_mask": test_utils.array_to_array_info(np.ones(num_edges, dtype=np.bool_)),
        "v_owner_mask": test_utils.array_to_array_info(np.ones(num_vertices, dtype=np.bool_)),
    }
    kwargs = _grid_init_kwargs(
        grid_savepoint, experiment, parallel_arrays=parallel_arrays, comm_id=42
    )

    # autospec=True makes the mocks enforce the real signatures — a positional
    # call to a keyword-only function raises TypeError on the mock too.
    with (
        mock.patch.object(wrapper_common, "construct_decomposition", autospec=True) as mock_cd,
        mock.patch.object(
            wrapper_debug_utils, "print_grid_decomp_info", autospec=True
        ) as mock_pgdi,
    ):
        mock_cd.return_value = (mock.MagicMock(), mock.MagicMock(), mock.MagicMock())
        grid_wrapper.grid_init(cffi.FFI(), perf_counters=None, **kwargs)

    mock_cd.assert_called_once()
    assert mock_cd.call_args.args == ()
    assert mock_cd.call_args.kwargs["comm_id"] == 42

    mock_pgdi.assert_called_once()
    assert mock_pgdi.call_args.args == ()
