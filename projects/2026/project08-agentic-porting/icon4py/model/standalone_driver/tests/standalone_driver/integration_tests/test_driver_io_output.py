# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Integration test for single-node output from the standalone driver.

Runs the Jablonowski-Williamson testcase for one step with output enabled and asserts
that valid CF/UGRID NetCDF files are produced. It exercises the full ``store -> file``
path, not just the in-memory bridge.

Being a datatest, it requires the JW grid and experiment configuration.
"""

import pathlib

import gt4py.next.typing as gtx_typing
import netCDF4 as nc
import pytest

from icon4py.model.common import model_backends, time
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.standalone_driver import (
    config as driver_config,
    driver_io,
    driver_utils,
    standalone_driver,
)
from icon4py.model.testing import datatest_utils as dt_utils, definitions as test_defs, grid_utils

from ..fixtures import *  # noqa: F403


def _find_one(directory: pathlib.Path, pattern: str) -> pathlib.Path:
    matches = sorted(directory.rglob(pattern))
    assert matches, f"no file matching {pattern!r} under {directory}"
    return matches[0]


@pytest.mark.datatest
@pytest.mark.embedded_remap_error
@pytest.mark.parametrize("experiment_description", [test_defs.Experiments.JW])
def test_standalone_driver_writes_output(
    experiment_description: test_defs.ExperimentDescription,
    *,
    download_ser_data: None,
    tmp_path: pathlib.Path,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx_typing.Backend,
) -> None:
    allocator = model_backends.get_allocator(backend)
    grid_file_path = grid_utils._download_grid_file(experiment_description.grid)
    config_file_path = dt_utils.get_path_for_experiment(experiment_description, process_props)

    config = driver_config.read_experiment_config_from_fortran(config_file_path)
    config = config.with_overrides(
        driver={
            "output_path": tmp_path / "io_driver_output",
            "enable_output": True,
            "end_of_simulation": time.NumTimeSteps(1),
        }
    )

    grid_manager = driver_utils.create_grid_manager(
        grid_file_path=grid_file_path,
        vertical_grid_config=config.vertical_grid,
        allocator=allocator,
        process_props=process_props,
    )
    standalone_driver.run_driver(
        config=config,
        grid_manager=grid_manager,
        process_props=process_props,
        backend=backend,
    )

    # the UGRID grid file is written on monitor init
    _find_one(tmp_path, "*_ugrid.nc")

    # single data file: all prognostic AND diagnostic fields together. Two time slices:
    # the initial state (always written) plus the one integrated step.
    output_file = _find_one(tmp_path, f"{driver_io.DEFAULT_OUTPUT_FILENAME}_*.nc")
    with nc.Dataset(output_file) as ds:
        assert ds.Conventions == "CF-1.7"
        for name in driver_io.DEFAULT_OUTPUT_VARIABLES:
            assert name in ds.variables, f"{name} missing from output"
            var = ds.variables[name]
            assert var.dimensions[0] == "time"
            assert var.shape[0] == 2
        # vertical placement: w on half (interface) levels, the rest on full levels
        assert "half_level" in ds.variables["upward_air_velocity"].dimensions
        assert "edge" in ds.variables["normal_velocity"].dimensions
        # diagnostics live on cells/full levels
        assert "cell" in ds.variables["temperature"].dimensions
        assert len(ds.dimensions["time"]) == 2
