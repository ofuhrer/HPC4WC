# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""Unit tests for the driver-side IO bridge (``driver_io``).

These tests are data-free: they use the ``simple_grid`` and a zero-initialised
``PrognosticState`` so they need no serialized/grid test data.
"""

import copy
import datetime
import pathlib
import uuid

import gt4py.next as gtx
import numpy as np
import pytest
import xarray as xr

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.grid import base, simple
from icon4py.model.common.io import io as common_io
from icon4py.model.common.states import (
    data as state_data,
    model as state_model,
    prognostic_state as prognostics,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.standalone_driver import driver_io
from icon4py.model.testing.fixtures import backend


@pytest.fixture
def grid() -> base.Grid:
    return simple.simple_grid()


def _make_prognostic_state(
    grid: base.Grid, allocator: gtx.typing.Backend | None = None
) -> prognostics.PrognosticState:
    # Constructed directly (instead of `initialize_prognostic_state`) so it works with
    # the generic `simple_grid`.
    def _cell_k(extend: dict[gtx.Dimension, int] | None = None) -> gtx.Field:
        return data_alloc.zero_field(
            grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, extend=extend, allocator=allocator
        )

    return prognostics.PrognosticState(
        rho=_cell_k(),
        w=_cell_k(extend={dims.KDim: 1}),
        vn=data_alloc.zero_field(
            grid, dims.EdgeDim, dims.KDim, dtype=ta.wpfloat, allocator=allocator
        ),
        exner=_cell_k(),
        theta_v=_cell_k(),
    )


@pytest.fixture
def prognostic_state(grid: base.Grid) -> prognostics.PrognosticState:
    return _make_prognostic_state(grid)


def _expected(
    cf_key: str, horizontal_dim: gtx.Dimension, *, is_on_half_levels: bool = False
) -> state_model.FieldMetaData:
    """Expected output metadata: the shared CF entry plus the expected dims and vertical
    placement. ``standard_name``/``units`` come from the shared table rather than being
    re-spelled here; ``dims`` and ``is_on_half_levels`` are stated independently of the
    production code so the assertions stay a genuine check."""
    return {
        **state_data.PROGNOSTIC_CF_ATTRIBUTES[cf_key],
        "dims": (horizontal_dim, dims.KDim),
        "is_on_half_levels": is_on_half_levels,
    }


#: Expected output metadata per output variable (keyed by CF name).
_EXPECTED: dict[str, state_model.FieldMetaData] = {
    "air_density": _expected("air_density", dims.CellDim),
    "exner_function": _expected("exner_function", dims.CellDim),
    "virtual_potential_temperature": _expected("virtual_potential_temperature", dims.CellDim),
    "upward_air_velocity": _expected("upward_air_velocity", dims.CellDim, is_on_half_levels=True),
    "normal_velocity": _expected("normal_velocity", dims.EdgeDim),
}

#: UGRID dimension names of the horizontal dimensions.
_UGRID_DIM_NAMES: dict[gtx.Dimension, str] = {
    dims.CellDim: "cell",
    dims.EdgeDim: "edge",
    dims.VertexDim: "vertex",
}


def _horizontal_size(grid: base.Grid, dim: gtx.Dimension) -> int:
    return {
        dims.CellDim: grid.num_cells,
        dims.EdgeDim: grid.num_edges,
        dims.VertexDim: grid.num_vertices,
    }[dim]


def test_assembles_all_default_variables(
    prognostic_state: prognostics.PrognosticState, grid: base.Grid
) -> None:
    state = driver_io.prognostic_state_to_dataarrays(prognostic_state)

    assert set(state.keys()) == set(driver_io.PROGNOSTIC_VARIABLES)
    for name, da in state.items():
        assert isinstance(da, xr.DataArray)
        expected = _EXPECTED[name]
        horizontal_dim = next(d for d in expected["dims"] if d.kind == gtx.DimensionKind.HORIZONTAL)
        on_half_levels = expected["is_on_half_levels"]

        vertical_name = "half_level" if on_half_levels else "level"
        assert da.dims == (_UGRID_DIM_NAMES[horizontal_dim], vertical_name)

        vertical_size = grid.num_levels + 1 if on_half_levels else grid.num_levels
        assert da.shape == (_horizontal_size(grid, horizontal_dim), vertical_size)


def test_dataarrays_carry_cf_and_ugrid_metadata(
    prognostic_state: prognostics.PrognosticState,
) -> None:
    state = driver_io.prognostic_state_to_dataarrays(prognostic_state)

    air_density = state["air_density"]
    # CF metadata from states.data
    assert air_density.attrs["standard_name"] == "air_density"
    assert air_density.attrs["units"] == "kg m-3"
    # UGRID metadata added by io.utils.to_data_array for the horizontal dimension
    assert air_density.attrs["location"] == "face"
    assert air_density.attrs["mesh"] == "mesh"
    assert air_density.attrs["coordinates"] == "clon clat"

    # edge field gets the edge location mapping
    assert state["normal_velocity"].attrs["location"] == "edge"


def test_does_not_mutate_shared_cf_attributes(
    prognostic_state: prognostics.PrognosticState,
) -> None:
    """`to_data_array` adds UGRID keys to the attrs it is handed; the shared
    module-level CF attribute table must be left untouched."""
    before = copy.deepcopy(state_data.PROGNOSTIC_CF_ATTRIBUTES)

    driver_io.prognostic_state_to_dataarrays(prognostic_state)

    assert before == state_data.PROGNOSTIC_CF_ATTRIBUTES
    # specifically, no UGRID keys leaked into the shared table
    for entry in state_data.PROGNOSTIC_CF_ATTRIBUTES.values():
        assert "location" not in entry
        assert "mesh" not in entry
        assert "coordinates" not in entry


def test_variables_subset(prognostic_state: prognostics.PrognosticState) -> None:
    subset = ["air_density", "normal_velocity"]
    state = driver_io.prognostic_state_to_dataarrays(prognostic_state, variables=subset)
    assert set(state.keys()) == set(subset)


def test_unknown_variable_raises(prognostic_state: prognostics.PrognosticState) -> None:
    with pytest.raises(ValueError, match="Unknown prognostic output variable"):
        driver_io.prognostic_state_to_dataarrays(prognostic_state, variables=["not_a_field"])


def test_data_is_host_numpy(
    grid: base.Grid,
    backend: gtx.typing.Backend | None,
) -> None:
    """The buffer handed to netCDF4 must be a host numpy array, not a device array.

    Parameterized on the backend (``--backend``) so that with a GPU backend the inputs
    really are device buffers and the host transfer is exercised.
    """
    prognostic_state = _make_prognostic_state(grid, allocator=backend)
    state = driver_io.prognostic_state_to_dataarrays(prognostic_state)
    for da in state.values():
        assert isinstance(da.data, np.ndarray)


def test_create_io_monitor_builds_single_field_group(
    grid: base.Grid, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The monitor holds one field group with all output fields, capturing every step.

    ``IOMonitor`` is replaced by a recorder so the test needs no real grid file.
    """
    recorded: dict[str, object] = {}

    class _RecordingMonitor:
        def __init__(
            self,
            *,
            config: common_io.IOConfig,
            vertical_size: object,
            horizontal_size: object,
            grid_file_name: pathlib.Path,
            grid_id: uuid.UUID,
            dtime: datetime.timedelta,
        ) -> None:
            recorded["config"] = config
            recorded["grid_file_name"] = grid_file_name
            recorded["grid_id"] = grid_id

    monkeypatch.setattr(common_io, "IOMonitor", _RecordingMonitor)

    driver_io.create_io_monitor(
        output_path=tmp_path,
        grid_file_path=tmp_path / "grid.nc",
        grid=grid,
        vertical_grid=None,  # type: ignore[arg-type] # not used by the recorder
        dtime=datetime.timedelta(seconds=1),
    )

    config = recorded["config"]
    assert isinstance(config, common_io.IOConfig)
    assert len(config.field_groups) == 1
    field_group = config.field_groups[0]
    # default cadence: capture on every model step
    assert field_group.output_interval == 1
    # a single group holding all fields, prognostic + diagnostic, in one file
    assert list(field_group.variables) == driver_io.DEFAULT_OUTPUT_VARIABLES
    assert list(field_group.variables) == [
        *driver_io.PROGNOSTIC_VARIABLES,
        *driver_io.DIAGNOSTIC_VARIABLES,
    ]
    assert field_group.filename == driver_io.DEFAULT_OUTPUT_FILENAME
    # output is written directly into the run output directory
    assert config.output_path == str(tmp_path)
    # the string grid id is converted to a UUID at the IO boundary
    assert recorded["grid_id"] == uuid.UUID(grid.id)


def test_create_io_monitor_has_no_separate_diagnostic_group(
    grid: base.Grid, tmp_path: pathlib.Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prognostics and diagnostics are not split: there is exactly one field group."""
    recorded: dict[str, common_io.IOConfig] = {}

    class _RecordingMonitor:
        def __init__(self, *, config: common_io.IOConfig, **kwargs: object) -> None:
            recorded["config"] = config

    monkeypatch.setattr(common_io, "IOMonitor", _RecordingMonitor)

    driver_io.create_io_monitor(
        output_path=tmp_path,
        grid_file_path=tmp_path / "grid.nc",
        grid=grid,
        vertical_grid=None,  # type: ignore[arg-type] # not used by the recorder
        dtime=datetime.timedelta(seconds=1),
    )

    groups = recorded["config"].field_groups
    assert len(groups) == 1
    assert set(driver_io.DIAGNOSTIC_VARIABLES) <= set(groups[0].variables)
    assert set(driver_io.PROGNOSTIC_VARIABLES) <= set(groups[0].variables)


def test_diagnostic_fields_to_dataarrays(grid: base.Grid) -> None:
    """The diagnostic assembly mirrors the prognostic one: correct dims/metadata, host
    numpy buffers, and the shared CF table is not mutated."""
    before = copy.deepcopy(state_data.DIAGNOSTIC_CF_ATTRIBUTES)

    # cell/full-level fields, like what compute_diagnostics returns
    def _zero_cell_k() -> gtx.Field:
        return data_alloc.zero_field(grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat)

    fields = {name: _zero_cell_k() for name in driver_io.DIAGNOSTIC_VARIABLES}
    state = driver_io.diagnostic_fields_to_dataarrays(fields)

    assert set(state.keys()) == set(driver_io.DIAGNOSTIC_VARIABLES)
    for da in state.values():
        assert da.dims == ("cell", "level")
        assert da.shape == (grid.num_cells, grid.num_levels)
        assert isinstance(da.data, np.ndarray)

    assert state["temperature"].attrs["standard_name"] == "air_temperature"
    assert state["pressure"].attrs["units"] == "Pa"
    assert state["eastward_wind"].attrs["location"] == "face"

    # shared diagnostic CF table must be untouched
    assert before == state_data.DIAGNOSTIC_CF_ATTRIBUTES
