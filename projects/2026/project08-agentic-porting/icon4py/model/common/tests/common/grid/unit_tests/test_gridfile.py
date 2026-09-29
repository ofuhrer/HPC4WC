# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import gridfile
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions,
    grid_utils as gridtest_utils,
)
from icon4py.model.testing.fixtures import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_description,
    grid_savepoint,
    process_props,
)

from .. import utils


if TYPE_CHECKING:
    from icon4py.model.testing import serialbox


@pytest.mark.with_netcdf
def test_grid_file_dimension() -> None:
    grid_description = definitions.Grids.R02B04_GLOBAL
    global_grid_file = str(dt_utils.get_grid_filepath(grid_description))
    parser = gridfile.GridFile(global_grid_file, offset_transformation=gridfile.NoTransformation())
    try:
        parser.open()
        ref = utils.GRID_REFERENCE_VALUES[grid_description.name]
        assert parser.dimension(gridfile.DynamicDimension.CELL_NAME) == ref["num_cells"]
        assert parser.dimension(gridfile.DynamicDimension.VERTEX_NAME) == ref["num_vertices"]
        assert parser.dimension(gridfile.DynamicDimension.EDGE_NAME) == ref["num_edges"]
    except Exception:
        pytest.fail()
    finally:
        parser.close()


@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_file_vertex_cell_edge_dimensions(
    experiment: definitions.Experiment, grid_savepoint: serialbox.IconGridSavepoint
) -> None:
    file = dt_utils.get_grid_filepath(experiment.grid)
    parser = gridfile.GridFile(str(file), gridfile.ToZeroBasedIndexTransformation())
    try:
        parser.open()
        assert parser.dimension(gridfile.DynamicDimension.CELL_NAME) == grid_savepoint.num(
            dims.CellDim
        )
        assert parser.dimension(gridfile.DynamicDimension.VERTEX_NAME) == grid_savepoint.num(
            dims.VertexDim
        )
        assert parser.dimension(gridfile.DynamicDimension.EDGE_NAME) == grid_savepoint.num(
            dims.EdgeDim
        )
    except Exception as error:
        pytest.fail(f"reading of dimension from netcdf failed: {error}")
    finally:
        parser.close()


@pytest.mark.parametrize("apply_offset", (True, False))
def test_int_variable(grid_description: definitions.GridDescription, apply_offset: bool) -> None:
    file = dt_utils.get_grid_filepath(grid_description)
    with gridfile.GridFile(str(file), gridfile.ToZeroBasedIndexTransformation()) as parser:
        edge_dim = parser.dimension(gridfile.DynamicDimension.EDGE_NAME)
        # use a test field that does not contain Pentagons
        test_field = parser.int_variable(gridfile.ConnectivityName.C2E, apply_offset=apply_offset)
        min_value = 0 if apply_offset else 1
        max_value = edge_dim - 1 if apply_offset else edge_dim
        assert min_value == np.min(test_field)
        assert max_value == np.max(test_field)


_index_selection: Iterable[list[int]] = [
    [0, 1, 2, 3, 4, 5],
    [],
    [0, 2, 4, 6, 7, 8, 24, 57],
    [1, 2, 12, 13, 23, 24, 513],
]


@pytest.mark.parametrize(
    "selection",
    _index_selection,
)
def test_index_read_for_1d_fields(
    grid_description: definitions.GridDescription, selection: list[int]
) -> None:
    file = dt_utils.get_grid_filepath(grid_description)
    with gridfile.GridFile(str(file), gridfile.ToZeroBasedIndexTransformation()) as parser:
        indices_to_read = np.asarray(selection) if len(selection) > 0 else None
        full_field = parser.variable(gridfile.CoordinateName.CELL_LATITUDE)
        selective_field = parser.variable(
            gridfile.CoordinateName.CELL_LATITUDE, indices=indices_to_read
        )
        assert np.allclose(full_field[indices_to_read], selective_field)


@pytest.mark.parametrize(
    "selection",
    _index_selection,
)
@pytest.mark.parametrize(
    "field",
    (
        gridfile.ConnectivityName.V2E,
        gridfile.ConnectivityName.V2C,
        gridfile.ConnectivityName.E2V,
    ),
)
@pytest.mark.parametrize("apply_offset", (True, False))
def test_index_read_for_2d_connectivity(
    grid_description: definitions.GridDescription,
    selection: list[int],
    field: gridfile.FieldName,
    apply_offset: bool,
) -> None:
    file = dt_utils.get_grid_filepath(grid_description)
    with gridfile.GridFile(str(file), gridfile.ToZeroBasedIndexTransformation()) as parser:
        indices_to_read = np.asarray(selection) if len(selection) > 0 else None
        full_field = parser.int_variable(field, transpose=True, apply_offset=apply_offset)
        selective_field = parser.int_variable(
            field,
            indices=indices_to_read,
            transpose=True,
            apply_offset=apply_offset,
        )
        assert np.allclose(full_field[indices_to_read], selective_field)
