# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import functools
from types import ModuleType
from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims, utils as common_utils
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import horizontal as h_grid, icon, simple, vertical as v_grid
from icon4py.model.common.math import (
    coordinate_transformations as coord_trans,
    vertical_operations as vertical_ops,
)
from icon4py.model.common.states import factory, model, utils as state_utils
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, serialbox
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    metrics_savepoint,
    process_props,
)


if TYPE_CHECKING:
    from collections.abc import Generator

    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb

cell_domain = h_grid.domain(dims.CellDim)
k_domain = v_grid.domain(dims.KDim)


class SimpleFieldSource(factory.FieldSource):
    def __init__(
        self,
        data_: dict[str, tuple[state_utils.GTXFieldType, model.FieldMetaData]],
        backend: gtx_typing.Backend | None,
        grid: icon.IconGrid,
        vertical_grid: v_grid.VerticalGrid | None = None,
    ):
        self._providers = {}
        self._backend = backend
        self._grid = grid
        self._vertical_grid = vertical_grid
        self._metadata = {}
        self._initial_data = data_
        self._exchange: decomposition.ExchangeRuntime = decomposition.single_node_exchange

        for key, value in data_.items():
            self.register_provider(factory.PrecomputedFieldProvider(fields={key: value[0]}))
            self._metadata[key] = value[1]

    def _register_initial_fields(self) -> None:
        for key, value in self._initial_data.items():
            self.register_provider(factory.PrecomputedFieldProvider(fields={key: value[0]}))
            self._metadata[key] = value[1]

    def reset(self) -> None:
        self._providers = {}
        self._metadata = {}
        self._register_initial_fields()

    @common_utils.chainable
    def with_metadata(self, metadata: dict) -> None:
        self._metadata.update(metadata)

    @property
    def metadata(self) -> dict:
        return self._metadata

    @property
    def _sources(self) -> factory.FieldSource:
        return self

    @property
    def grid(self) -> icon.IconGrid:
        return self._grid

    @property
    def vertical_grid(self) -> v_grid.VerticalGrid | None:
        return self._vertical_grid

    @property
    def backend(self) -> gtx_typing.Backend | None:
        return self._backend


# TODO(): this reads lat lon from the grid_savepoint, which could be read from the grid file/geometry, to make it non datatests
@pytest.fixture(scope="function")
def cell_coordinate_source(
    grid_savepoint: sb.IconGridSavepoint, backend: gtx_typing.Backend
) -> Generator[SimpleFieldSource, None, None]:
    grid = grid_savepoint.construct_icon_grid(backend=backend)
    lat = grid_savepoint.lat(dims.CellDim)
    lon = grid_savepoint.lon(dims.CellDim)
    data: dict[str, tuple[state_utils.GTXFieldType, model.FieldMetaData]] = {
        "lat": (lat, {"standard_name": "lat", "units": ""}),
        "lon": (lon, {"standard_name": "lon", "units": ""}),
        "x": (
            data_alloc.random_field(grid, dims.CellDim, dims.KDim),
            {"standard_name": "x", "units": ""},
        ),
        "y": (
            data_alloc.random_field(grid, dims.CellDim, dims.KDim),
            {"standard_name": "y", "units": ""},
        ),
        "z": (
            data_alloc.random_field(grid, dims.CellDim, dims.KDim),
            {"standard_name": "z", "units": ""},
        ),
    }

    coordinate_source = SimpleFieldSource(data_=data, backend=backend, grid=grid)
    yield coordinate_source
    coordinate_source.reset()


@pytest.fixture(scope="function")
def height_coordinate_source(
    metrics_savepoint: sb.MetricSavepoint,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> Generator[SimpleFieldSource, None, None]:
    grid = grid_savepoint.construct_icon_grid(backend=backend)
    z_ifc = metrics_savepoint.z_ifc()
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    data: dict[str, tuple[state_utils.GTXFieldType, model.FieldMetaData]] = {
        "height_coordinate": (z_ifc, {"standard_name": "height_coordinate", "units": ""})
    }
    vertical_grid = v_grid.VerticalGrid(
        v_grid.VerticalGridConfig(num_levels=experiment.config.vertical_grid.num_levels),
        vct_a,
        vct_b,
    )
    field_source = SimpleFieldSource(
        data_=data, backend=backend, grid=grid, vertical_grid=vertical_grid
    )
    yield field_source
    field_source.reset()


@pytest.mark.datatest
def test_field_operator_provider(cell_coordinate_source: SimpleFieldSource) -> None:
    field_op = coord_trans.geographical_to_cartesian_on_cells.with_backend(None)

    domain = {dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.LOCAL))}
    deps = {"lat": "lat", "lon": "lon"}
    fields = {"x": "x", "y": "y", "z": "z"}

    provider = factory.EmbeddedFieldOperatorProvider(
        func=field_op, domain=domain, fields=fields, deps=deps, do_exchange=False
    )
    provider(
        field_name="x",
        field_src=cell_coordinate_source,
        backend=cell_coordinate_source.backend,
        grid=cell_coordinate_source,
        exchange=decomposition.single_node_exchange,
    )
    x = provider.fields["x"]
    assert isinstance(x, gtx.Field)
    assert dims.CellDim in x.domain.dims


@pytest.mark.datatest
def test_program_provider(height_coordinate_source: SimpleFieldSource) -> None:
    program = vertical_ops.average_two_vertical_levels_downwards_on_cells
    domain = {
        dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.LOCAL)),
        dims.KDim: (k_domain(v_grid.Zone.TOP), k_domain(v_grid.Zone.BOTTOM)),
    }
    deps = {
        "input_field": "height_coordinate",
    }
    fields = {"average": "output_f"}
    provider = factory.ProgramFieldProvider(
        func=program, domain=domain, fields=fields, deps=deps, do_exchange=False
    )
    provider(
        field_name="output_f",
        field_src=height_coordinate_source,
        backend=height_coordinate_source.backend,
        grid=height_coordinate_source,
        exchange=decomposition.single_node_exchange,
    )
    x = provider.fields["output_f"]
    assert isinstance(x, gtx.Field)
    assert dims.CellDim in x.domain.dims


@pytest.mark.datatest
def test_field_source_raise_error_on_register(cell_coordinate_source: SimpleFieldSource) -> None:
    program = vertical_ops.average_two_vertical_levels_downwards_on_cells
    domain = {
        dims.CellDim: (cell_domain(h_grid.Zone.LOCAL), cell_domain(h_grid.Zone.LOCAL)),
        dims.KDim: (k_domain(v_grid.Zone.TOP), k_domain(v_grid.Zone.BOTTOM)),
    }
    deps = {
        "input_field": "height_coordinate",
    }
    fields = {"result": "output_f"}
    provider = factory.ProgramFieldProvider(
        func=program, domain=domain, fields=fields, deps=deps, do_exchange=False
    )
    with pytest.raises(ValueError, match="Missing dependency: 'height_coordinate'"):
        cell_coordinate_source.register_provider(provider)


@pytest.mark.datatest
def test_composite_field_source_contains_all_metadata(
    cell_coordinate_source: SimpleFieldSource, height_coordinate_source: SimpleFieldSource
) -> None:
    backend = cell_coordinate_source.backend
    grid = cell_coordinate_source.grid
    foo = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
    bar = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
    data: dict[str, tuple[state_utils.GTXFieldType, model.FieldMetaData]] = {
        "foo": (foo, {"standard_name": "foo", "units": ""}),
        "bar": (bar, {"standard_name": "bar", "units": ""}),
    }

    test_source = SimpleFieldSource(data_=data, grid=grid, backend=backend)
    composite = factory.CompositeSource(
        me=test_source, others=(cell_coordinate_source, height_coordinate_source)
    )

    assert composite.backend == test_source.backend
    assert composite.grid.id == test_source.grid.id
    assert test_source.metadata.items() <= composite.metadata.items()
    assert height_coordinate_source.metadata.items() <= composite.metadata.items()
    assert cell_coordinate_source.metadata.items() <= composite.metadata.items()


@pytest.mark.datatest
def test_composite_field_source_get_all_fields(
    cell_coordinate_source: SimpleFieldSource, height_coordinate_source: SimpleFieldSource
) -> None:
    backend = cell_coordinate_source.backend
    grid = cell_coordinate_source.grid
    foo = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
    bar = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
    data: dict[str, tuple[state_utils.GTXFieldType, model.FieldMetaData]] = {
        "foo": (foo, {"standard_name": "foo", "units": ""}),
        "bar": (bar, {"standard_name": "bar", "units": ""}),
    }

    test_source = SimpleFieldSource(data_=data, grid=grid, backend=backend)
    composite = factory.CompositeSource(
        me=test_source, others=(cell_coordinate_source, height_coordinate_source)
    )
    foo = composite.get("foo")
    assert isinstance(foo, gtx.Field)
    assert {dims.CellDim, dims.KDim}.issubset(foo.domain.dims)

    bar = composite.get("bar")
    assert len(bar.domain.dims) == 2
    assert isinstance(bar, gtx.Field)
    assert {dims.EdgeDim, dims.KDim}.issubset(bar.domain.dims)

    lon = composite.get("lon")
    assert isinstance(lon, gtx.Field)
    assert dims.CellDim in lon.domain.dims
    assert len(lon.domain.dims) == 1

    lat = composite.get("height_coordinate")
    assert isinstance(lat, gtx.Field)
    assert dims.KDim in lat.domain.dims
    assert len(lat.domain.dims) == 2


@pytest.mark.datatest
def test_composite_field_source_raises_upon_get_unknown_field(
    cell_coordinate_source: SimpleFieldSource, height_coordinate_source: SimpleFieldSource
) -> None:
    backend = cell_coordinate_source.backend
    grid = cell_coordinate_source.grid
    foo = data_alloc.random_field(grid, dims.CellDim, dims.KDim)
    bar = data_alloc.random_field(grid, dims.EdgeDim, dims.KDim)
    data: dict[str, tuple[state_utils.GTXFieldType, model.FieldMetaData]] = {
        "foo": (foo, {"standard_name": "foo", "units": ""}),
        "bar": (bar, {"standard_name": "bar", "units": ""}),
    }

    test_source = SimpleFieldSource(data_=data, grid=grid, backend=backend)
    composite = factory.CompositeSource(
        me=test_source, others=(cell_coordinate_source, height_coordinate_source)
    )
    with pytest.raises(ValueError, match="Field 'alice' not provided by the source"):
        composite.get("alice")


def reduce_scalar_min(ar: data_alloc.NDArray, xp: ModuleType) -> gtx.float:
    return xp.min(ar).item()


@pytest.mark.datatest
def test_compute_scalar_value_from_numpy_provider(
    height_coordinate_source: factory.FieldSource,
    metrics_savepoint: serialbox.MetricSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    value_ref = np.min(metrics_savepoint.z_ifc().asnumpy())
    sample_func = functools.partial(reduce_scalar_min, xp=data_alloc.import_array_ns(backend))
    provider = factory.NumpyDataProvider(
        func=sample_func, deps={"ar": "height_coordinate"}, domain=(), fields=("minimal_height",)
    )
    height_coordinate_source.register_provider(provider)
    value = height_coordinate_source.get("minimal_height", factory.RetrievalType.FIELD)
    assert np.isscalar(value)
    assert value_ref == value
