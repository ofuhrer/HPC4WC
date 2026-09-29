# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from datetime import datetime, timedelta

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims
from icon4py.model.common.grid import base as grid_def, vertical as v_grid
from icon4py.model.common.io import cf_utils, utils, writers
from icon4py.model.common.states import data, metadata
from icon4py.model.common.utils import data_allocation as data_alloc

from ...fixtures import random_name, test_path
from .. import utils as test_io_utils


@pytest.mark.parametrize("value", ["air_density", "upward_air_velocity"])
def test_filter_by_standard_name(value):
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    assert writers.filter_by_standard_name(state, value) == {value: state[value]}


def test_filter_by_standard_name_key_differs_from_name():
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    assert writers.filter_by_standard_name(state, "virtual_potential_temperature") == {
        "theta_v": state["theta_v"]
    }


def test_filter_by_standard_name_non_existing_name():
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    assert writers.filter_by_standard_name(state, "does_not_exist") == {}


def initialized_writer(
    test_path, random_name, grid=test_io_utils.simple_grid
) -> tuple[writers.NETCDFWriter, grid_def.Grid]:
    num_levels = grid.config.vertical_size
    heights = np.linspace(start=12000.0, stop=0.0, num=num_levels + 1)
    vertical_config = v_grid.VerticalGridConfig(num_levels=num_levels)
    vertical_params = v_grid.VerticalGrid(
        vertical_config,
        vct_a=gtx.as_field((dims.KDim,), heights),
        vct_b=None,
    )
    horizontal = grid.config.horizontal_config
    fname = str(test_path.absolute()) + "/" + random_name + ".nc"
    writer = writers.NETCDFWriter(
        file_name=fname,
        vertical=vertical_params,
        horizontal=horizontal,
        time_properties=writers.TimeProperties(
            cf_utils.DEFAULT_TIME_UNIT, cf_utils.DEFAULT_CALENDAR
        ),
        global_attrs={"title": "test", "institution": "EXCLAIM - ETH Zurich"},
    )
    writer.initialize_dataset()
    return writer, grid


def test_initialize_writer_time_var(test_path, random_name):
    dataset, _ = initialized_writer(test_path, random_name)
    time_var = dataset.variables[writers.TIME]
    assert time_var.dimensions == ("time",)
    assert time_var.units == "seconds since 1970-01-01 00:00:00"
    assert time_var.calendar == "proleptic_gregorian"
    assert time_var.long_name == "time"
    assert time_var.standard_name == "time"
    assert len(time_var) == 0


def test_initialize_writer_vertical_model_levels(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    vertical = dataset.variables[writers.MODEL_LEVEL]
    assert vertical.units == "1"
    assert vertical.dimensions == (writers.MODEL_LEVEL,)
    assert vertical.long_name == "model full level index"
    assert vertical.standard_name == cf_utils.LEVEL_STANDARD_NAME
    assert vertical.datatype == np.int32
    assert len(vertical) == grid.num_levels
    assert np.all(vertical == np.arange(grid.num_levels))


def test_initialize_writer_half_levels(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    half_levels = dataset.variables[writers.MODEL_HALF_LEVEL]
    assert half_levels.units == "1"
    assert half_levels.datatype == np.int32
    assert half_levels.long_name == "model half level index"
    assert half_levels.standard_name == metadata.INTERFACE_LEVEL_STANDARD_NAME
    assert len(half_levels) == grid.num_levels + 1
    assert np.all(half_levels == np.arange(grid.num_levels + 1))


def test_initialize_writer_heights(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    heights = dataset.variables["height"]
    assert heights.units == "m"
    assert heights.datatype == np.float64
    assert heights.long_name == "height value of half levels without topography"
    assert heights.standard_name == metadata.INTERFACE_LEVEL_HEIGHT_STANDARD_NAME
    assert len(heights) == grid.num_levels + 1
    assert heights[0] == 12000.0
    assert heights[-1] == 0.0


def test_writer_append_timeslice(test_path, random_name):
    writer, _ = initialized_writer(test_path, random_name)
    time = datetime.now()
    assert len(writer.variables[writers.TIME]) == 0
    slice1 = {}
    writer.append(slice1, time)
    assert len(writer.variables[writers.TIME]) == 1
    time1 = time + timedelta(hours=1)
    writer.append(slice1, time1)
    assert len(writer.variables[writers.TIME]) == 2
    time2 = time1 + timedelta(hours=1)
    writer.append(slice1, time2)
    assert len(writer.variables[writers.TIME]) == 3
    time_units = writer.variables[writers.TIME].units
    cal = writer.variables[writers.TIME].calendar
    assert np.all(
        writer.variables[writers.TIME][:]
        == np.array(cf_utils.date2num((time, time1, time2), units=time_units, calendar=cal))
    )


def test_writer_append_timeslice_create_new_var(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    time = datetime.now()
    assert len(dataset.variables[writers.TIME]) == 0
    assert "air_density" not in dataset.variables

    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    dataset.append(state, time)
    assert len(dataset.variables[writers.TIME]) == 1
    assert "air_density" in dataset.variables
    assert dataset.variables["air_density"].dimensions == (
        writers.TIME,
        writers.MODEL_LEVEL,
        writers.CELL,
    )
    assert dataset.variables["air_density"].shape == (
        1,
        grid.num_levels,
        grid.num_cells,
    )
    assert np.allclose(dataset.variables["air_density"][0], state["air_density"].data.T)


def test_writer_append_timeslice_to_existing_var(test_path, random_name):
    dataset, grid = initialized_writer(test_path, random_name)
    time = datetime.now()
    state = dict(air_density=test_io_utils.model_state(grid)["air_density"])
    dataset.append(state, time)
    assert len(dataset.variables[writers.TIME]) == 1
    assert "air_density" in dataset.variables

    new_rho = data_alloc.random_field(grid, dims.CellDim, dims.KDim, dtype=np.float32)
    state["air_density"] = utils.to_data_array(
        new_rho, data.PROGNOSTIC_CF_ATTRIBUTES["air_density"]
    )

    new_time = time + timedelta(hours=1)
    dataset.append(state, new_time)

    assert len(dataset.variables[writers.TIME]) == 2
    assert dataset.variables["air_density"].shape == (
        2,
        grid.num_levels,
        grid.num_cells,
    )
    assert np.allclose(dataset.variables["air_density"][1], new_rho.ndarray.T)


def test_initialize_writer_create_dimensions(
    test_path,
    random_name,
):
    writer, grid = initialized_writer(test_path, random_name)

    assert writer["title"] == "test"
    assert writer["institution"] == "EXCLAIM - ETH Zurich"
    assert len(writer.dims) == 6
    assert writer.dims[writers.MODEL_LEVEL].size == grid.num_levels
    assert writer.dims[writers.MODEL_HALF_LEVEL].size == grid.num_levels + 1
    assert writer.dims[writers.CELL].size == grid.num_cells
    assert writer.dims[writers.VERTEX].size == grid.num_vertices
    assert writer.dims[writers.EDGE].size == grid.num_edges
    assert writer.dims[writers.TIME].size == 0
    assert writer.dims[writers.TIME].isunlimited

    assert writer.variables[writers.TIME].units == cf_utils.DEFAULT_TIME_UNIT
    assert writer.variables[writers.TIME].calendar == cf_utils.DEFAULT_CALENDAR
