# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import datetime as dt
import pathlib
import re
import uuid
from typing import Any

import gt4py.next as gtx
import numpy as np
import pytest
import uxarray as ux  # type: ignore[import-untyped]  # uxarray has no type hints

import icon4py.model.common.exceptions as errors
from icon4py.model.common import dimension as dims, time
from icon4py.model.common.grid import base, vertical as v_grid
from icon4py.model.common.io import ugrid
from icon4py.model.common.io.io import (
    FieldGroupIOConfig,
    FieldGroupMonitor,
    IOConfig,
    IOMonitor,
    OutputInterval,
    generate_name,
)
from icon4py.model.common.states import data
from icon4py.model.testing import datatest_utils, definitions, grid_utils

from ...fixtures import test_path
from .. import utils as test_io_utils


# setting backend to fieldview embedded here.
backend = None


@pytest.mark.parametrize(
    "name, expected",
    [
        ("output.nc", "output_0002.nc"),
        ("outxxput_20220101.xc", "outxxput_20220101_0002.nc"),
        ("output_20220101T000000_x", "output_20220101T000000_x_0002.nc"),
    ],
)
def test_generate_name(name: str, expected: str) -> None:
    counter = 2
    assert expected == generate_name(name, counter)


def is_valid_uxgrid(file: pathlib.Path | str) -> bool:
    import uxarray as ux  # noqa: PLC0415 [import-outside-top-level]

    grid = ux.open_grid(file)
    try:
        grid.validate()
        return True
    except RuntimeError:
        return False


def test_io_monitor_create_output_path(test_path: pathlib.Path) -> None:
    path_name = test_path.absolute().as_posix() + "/output"
    vertical_config = v_grid.VerticalGridConfig(num_levels=test_io_utils.simple_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field(
            (dims.KDim,),
            np.linspace(12000.0, 0.0, test_io_utils.simple_grid.num_levels + 1),  # type: ignore[arg-type]
        ),
        vct_b=None,
    )
    config = IOConfig(field_groups=[], output_path=path_name)
    monitor = IOMonitor(
        config=config,
        vertical_size=vertical_params,
        horizontal_size=test_io_utils.simple_grid.config.horizontal_config,
        grid_file_name=test_io_utils.grid_file,
        grid_id=uuid.UUID(test_io_utils.simple_grid.id),
        dtime=time.RelativeTime(hours=1),
    )
    assert monitor.path.exists()
    assert monitor.path.is_dir()


def test_io_monitor_write_ugrid_file(test_path: pathlib.Path) -> None:
    path_name = test_path.absolute().as_posix() + "/output"
    vertical_config = v_grid.VerticalGridConfig(num_levels=test_io_utils.simple_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field(
            (dims.KDim,),
            np.linspace(12000.0, 0.0, test_io_utils.simple_grid.num_levels + 1),  # type: ignore[arg-type]
        ),
        vct_b=None,
    )

    config = IOConfig(field_groups=[], output_path=path_name)
    monitor = IOMonitor(
        config=config,
        vertical_size=vertical_params,
        horizontal_size=test_io_utils.simple_grid.config.horizontal_config,
        grid_file_name=test_io_utils.grid_file,
        grid_id=uuid.UUID(test_io_utils.simple_grid.id),
        dtime=time.RelativeTime(hours=1),
    )
    ugrid_file = monitor.path.iterdir().__next__().absolute()
    assert "ugrid.nc" in ugrid_file.name
    assert is_valid_uxgrid(ugrid_file)


@pytest.mark.parametrize(
    "variables",
    (
        ["air_density", "exner_function", "upward_air_velocity"],
        ["normal_velocity", "upward_air_velocity", "theta_v"],
    ),
)
def test_io_monitor_write_and_read_ugrid_dataset(
    test_path: pathlib.Path, variables: list[str]
) -> None:
    path_name = test_path.absolute().as_posix() + "/output"
    grid = grid_utils.get_grid_manager_from_identifier(
        definitions.Experiments.EXCLAIM_APE.grid,
        num_levels=60,
        keep_skip_values=True,
        allocator=backend,  # type: ignore[arg-type]  # None selects the embedded backend
    ).grid
    vertical_config = v_grid.VerticalGridConfig(num_levels=grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field((dims.KDim,), np.linspace(12000.0, 0.0, grid.num_levels + 1)),  # type: ignore[arg-type]
        vct_b=None,
    )

    state = test_io_utils.model_state(grid)
    field_configs = [
        FieldGroupIOConfig(
            output_interval=time.NumTimeSteps(1),
            filename="icon4py_dummy_output",
            variables=variables,
            nc_comment="Writing dummy data from icon4py for testing.",
        )
    ]
    config = IOConfig(field_groups=field_configs, output_path=path_name)
    monitor = IOMonitor(
        config=config,
        vertical_size=vertical_params,
        horizontal_size=grid.config.horizontal_config,
        grid_file_name=test_io_utils.grid_file,
        grid_id=uuid.UUID(grid.id),
        dtime=time.RelativeTime(hours=1),
    )
    current_time = dt.datetime.fromisoformat("2024-01-01T12:00:00")
    for _ in range(3):
        monitor.store(state, current_time)
        current_time = current_time + dt.timedelta(hours=1)
    monitor.close()

    assert len([f for f in monitor.path.iterdir() if f.is_file()]) == 1 + len(field_configs)
    uxds = read_back_as_uxarray(monitor.path)
    for var in variables:
        assert var in uxds.variables
        if var in ["air_density", "exner_function", "theta_v"]:
            assert uxds[var].shape == (3, grid.num_levels, grid.num_cells)
        elif var == "upward_air_velocity":
            assert uxds[var].shape == (3, grid.num_levels + 1, grid.num_cells)
        elif var == "normal_velocity":
            assert uxds[var].shape == (3, grid.num_levels, grid.num_edges)


def test_fieldgroup_monitor_write_dataset_file_roll(test_path: pathlib.Path) -> None:
    grid = grid_utils.get_grid_manager_from_identifier(
        definitions.Experiments.EXCLAIM_APE.grid,
        num_levels=60,
        keep_skip_values=True,
        allocator=backend,  # type: ignore[arg-type]  # None selects the embedded backend
    ).grid
    vertical_config = v_grid.VerticalGridConfig(num_levels=grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field((dims.KDim,), np.linspace(12000.0, 0.0, grid.num_levels + 1)),  # type: ignore[arg-type]
        vct_b=None,
    )

    state = test_io_utils.model_state(grid)
    filename_stub = "icon4py_dummy_output"
    config = FieldGroupIOConfig(
        output_interval=time.NumTimeSteps(1),
        filename=filename_stub,
        variables=["air_density", "exner_function", "upward_air_velocity"],
        timesteps_per_file=1,
    )
    monitor = FieldGroupMonitor(
        config=config,
        vertical=vertical_params,
        horizontal=grid.config.horizontal_config,
        grid_id=uuid.UUID(grid.id),
        output_path=test_path,
        dtime=time.RelativeTime(hours=1),
    )
    current_time = dt.datetime.fromisoformat("2024-01-01T12:00:00")
    for _ in range(4):
        monitor.store(state, current_time)
        current_time = current_time + dt.timedelta(hours=1)
    assert len([f for f in monitor.output_path.iterdir() if f.is_file()]) == 4
    expected_name = re.compile(filename_stub + "_\\d{4}.nc")
    for f in monitor.output_path.iterdir():
        if f.is_file():
            assert expected_name.match(f.name)

            with ugrid.load_data_file(f) as ds:
                assert ds.sizes["time"] == 1
                assert ds.sizes["level"] == grid.num_levels
                assert ds.sizes["cell"] == grid.num_cells
                assert ds.sizes["half_level"] == grid.num_levels + 1
                assert ds.variables["air_density"].shape == (
                    1,
                    grid.num_levels,
                    grid.num_cells,
                )
                assert ds.variables["exner_function"].shape == (
                    1,
                    grid.num_levels,
                    grid.num_cells,
                )
                assert ds.variables["upward_air_velocity"].shape == (
                    1,
                    grid.num_levels + 1,
                    grid.num_cells,
                )


def test_fieldgroup_monitor_refuses_to_overwrite_existing_output(test_path: pathlib.Path) -> None:
    # a first run writes ..._0001.nc; a second run sharing the directory must not
    # silently overwrite it -- the per-run file counter restarts at 0.
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    current_time = dt.datetime.fromisoformat("2024-01-01T00:00:00")

    _, first_monitor = create_field_group_monitor(test_path, test_io_utils.simple_grid)
    first_monitor.store(state, current_time)
    first_monitor.close()

    _, second_monitor = create_field_group_monitor(test_path, test_io_utils.simple_grid)
    with pytest.raises(errors.InvalidConfigError, match="already exists"):
        second_monitor.store(state, current_time)


def read_back_as_uxarray(path: pathlib.Path) -> Any:
    ugrid_file = None
    data_files = []
    for f in path.iterdir():
        if f.is_file():
            if "_ugrid.nc" in f.name:
                ugrid_file = f.absolute()
            else:
                data_files.append(f.absolute())
    uxds = ux.open_dataset(ugrid_file, data_files[0])
    return uxds


def test_fieldgroup_monitor_no_output_between_step_intervals(test_path: pathlib.Path) -> None:
    # output every 3rd step: the first two stores must not produce any output
    _, group_monitor = create_field_group_monitor(
        test_path, test_io_utils.simple_grid, output_interval=time.NumTimeSteps(3)
    )
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    step_time = dt.datetime.fromisoformat("2024-01-01T00:00:00")
    group_monitor.store(state, step_time)
    group_monitor.store(state, step_time + dt.timedelta(hours=1))
    group_monitor.close()
    assert len([f for f in group_monitor.output_path.iterdir() if f.is_file()]) == 0


def create_field_group_monitor(
    test_path: pathlib.Path,
    grid: base.Grid,
    output_interval: OutputInterval = time.NumTimeSteps(1),
    dtime: time.RelativeTime = time.RelativeTime(hours=1),
) -> tuple[FieldGroupIOConfig, FieldGroupMonitor]:
    config = FieldGroupIOConfig(
        filename="test_empty.nc",
        output_interval=output_interval,
        variables=["exner_function", "air_density"],
    )
    vertical_config = v_grid.VerticalGridConfig(num_levels=test_io_utils.simple_grid.num_levels)
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=gtx.as_field(
            (dims.KDim,),
            np.linspace(12000.0, 0.0, test_io_utils.simple_grid.num_levels + 1),  # type: ignore[arg-type]
        ),
        vct_b=None,
    )

    group_monitor = FieldGroupMonitor(
        config=config,
        vertical=vertical_params,
        horizontal=grid.config.horizontal_config,
        grid_id=uuid.UUID(grid.id),
        output_path=test_path,
        dtime=dtime,
    )
    return config, group_monitor


@pytest.mark.parametrize(
    "filename, output_interval, variables, message",
    [
        (
            "",
            1,
            ["exner_function", "air_density"],
            "Output filename is missing.",
        ),
        (
            "/vars/prognostics.nc",
            1,
            ["exner_function", "air_density"],
            "absolute path",
        ),
        (
            "vars/prognostics.nc",
            1,
            [],
            "No variables provided for output.",
        ),
        (
            "vars/prognostics.nc",
            0,
            ["air_density, exner_function"],
            "Output interval must be positive",
        ),
    ],
)
def test_fieldgroup_config_validate_filename(
    filename: str, output_interval: OutputInterval, variables: list[str], message: str
) -> None:
    with pytest.raises(errors.InvalidConfigError) as err:
        FieldGroupIOConfig(
            filename=filename,
            output_interval=output_interval,
            variables=variables,
        )
    assert message in str(err.value)


def test_fieldgroup_monitor_constructs_output_path_and_filepattern(test_path: pathlib.Path) -> None:
    config = FieldGroupIOConfig(
        filename="vars/prognostics.nc",
        output_interval=time.NumTimeSteps(1),
        variables=["exner_function", "air_density"],
    )
    vertical_size = test_io_utils.simple_grid.config.vertical_size
    horizontal_size = test_io_utils.simple_grid.config.horizontal_config
    group_monitor = FieldGroupMonitor(
        config=config,
        vertical=vertical_size,  # type: ignore[arg-type]  # vertical is unused in this test
        horizontal=horizontal_size,
        grid_id=uuid.UUID(test_io_utils.simple_grid.id),
        output_path=test_path,
        dtime=time.RelativeTime(hours=1),
    )
    assert group_monitor.output_path == test_path.joinpath("vars")
    assert group_monitor.output_path.exists()
    assert group_monitor.output_path.is_dir()
    assert "prognostics" in group_monitor._file_name_pattern


def test_fieldgroup_monitor_throw_exception_on_missing_field(test_path: pathlib.Path) -> None:
    config = FieldGroupIOConfig(
        filename="vars/prognostics.nc",
        output_interval=time.NumTimeSteps(1),
        variables=["exner_function", "air_density", "foo"],
    )
    vertical_size = test_io_utils.simple_grid.config.vertical_size
    horizontal_size = test_io_utils.simple_grid.config.horizontal_config
    group_monitor = FieldGroupMonitor(
        config=config,
        vertical=vertical_size,  # type: ignore[arg-type]  # vertical is unused in this test
        horizontal=horizontal_size,
        grid_id=uuid.UUID(test_io_utils.simple_grid.id),
        output_path=test_path,
        dtime=time.RelativeTime(hours=1),
    )
    with pytest.raises(errors.IncompleteStateError, match="Field 'foo' is missing"):
        group_monitor.store(
            test_io_utils.model_state(test_io_utils.simple_grid),
            dt.datetime.fromisoformat("2023-04-04T11:00:00"),
        )


def test_fieldgroup_config_rejects_invalid_interval() -> None:
    # a string interval is no longer supported: only int (steps) or timedelta
    with pytest.raises(errors.InvalidConfigError, match="must be of type"):
        FieldGroupIOConfig(
            filename="a.nc",
            variables=["air_density"],
            output_interval="1 HOUR",  # type: ignore[arg-type]
        )


def test_fieldgroup_monitor_time_interval_normalized_to_steps(test_path: pathlib.Path) -> None:
    # a 3-hour interval with a 1-hour time step fires every 3rd step
    _, group_monitor = create_field_group_monitor(
        test_path,
        test_io_utils.simple_grid,
        output_interval=time.RelativeTime(hours=3),
        dtime=time.RelativeTime(hours=1),
    )
    state = test_io_utils.model_state(test_io_utils.simple_grid)
    step_time = dt.datetime.fromisoformat("2024-01-01T00:00:00")
    # first two steps: no output
    group_monitor.store(state, step_time)
    group_monitor.store(state, step_time + dt.timedelta(hours=1))
    assert len([f for f in group_monitor.output_path.iterdir() if f.is_file()]) == 0
    # third step: output is written
    group_monitor.store(state, step_time + dt.timedelta(hours=2))
    group_monitor.close()
    assert len([f for f in group_monitor.output_path.iterdir() if f.is_file()]) == 1


def test_fieldgroup_monitor_interval_shorter_than_dtime_raises(test_path: pathlib.Path) -> None:
    with pytest.raises(errors.InvalidConfigError, match="shorter than the model time step"):
        create_field_group_monitor(
            test_path,
            test_io_utils.simple_grid,
            output_interval=time.RelativeTime(minutes=30),
            dtime=time.RelativeTime(hours=1),
        )
