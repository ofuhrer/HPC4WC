# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import pathlib
import random
from collections.abc import Generator

import gt4py.next.typing as gtx_typing
import pytest

from icon4py.model.common.grid import (
    geometry,
    geometry_attributes,
    geometry_config,
    gridfile,
    vertical,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.testing import serialbox
from icon4py.model.testing.fixtures.datatest import (
    backend,
    backend_like,
    data_provider,
    decomposition,
    decomposition_info,
    definitions,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    linit,
    metrics_savepoint,
    process_props,
    topography_savepoint,
)


@pytest.fixture
def random_name() -> str:
    return "test" + str(random.randint(0, 100000))


@pytest.fixture
def test_path(tmp_path: pathlib.Path) -> Generator[pathlib.Path, None, None]:
    base_path = tmp_path.joinpath("io_tests")
    base_path.mkdir(exist_ok=True, parents=True, mode=0o777)
    yield base_path
    _delete_recursive(base_path)


def _delete_recursive(p: pathlib.Path) -> None:
    for child in p.iterdir():
        if child.is_file():
            child.unlink()
        else:
            _delete_recursive(child)
    p.rmdir()


@pytest.fixture
def geometry_from_savepoint(
    grid_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend,
    decomposition_info: decomposition.DecompositionInfo,
    process_props: decomposition.ProcessProperties,
) -> Generator[geometry.GridGeometry]:
    grid = grid_savepoint.construct_icon_grid(backend, with_repeated_index=False)
    coordinates = grid_savepoint.coordinates()
    extra_fields = {
        gridfile.GeometryName.CELL_AREA: grid_savepoint.cell_areas(),
        gridfile.GeometryName.EDGE_LENGTH: grid_savepoint.primal_edge_length(),
        gridfile.GeometryName.DUAL_EDGE_LENGTH: grid_savepoint.dual_edge_length(),
        gridfile.GeometryName.EDGE_CELL_DISTANCE: grid_savepoint.edge_cell_length(),
        gridfile.GeometryName.EDGE_VERTEX_DISTANCE: grid_savepoint.edge_vert_length(),
        gridfile.GeometryName.DUAL_AREA: grid_savepoint.vertex_dual_area(),
        gridfile.GeometryName.TANGENT_ORIENTATION: grid_savepoint.tangent_orientation(),
        gridfile.GeometryName.CELL_NORMAL_ORIENTATION: grid_savepoint.edge_orientation(),
        gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX: grid_savepoint.vertex_edge_orientation(),
    }

    exchange = decomposition.create_exchange(process_props, decomposition_info)
    global_reductions = decomposition.create_reduction(process_props, decomposition_info)
    grid_geometry = geometry.GridGeometry(
        grid=grid,
        decomposition_info=decomposition_info,
        backend=backend,
        metadata=geometry_attributes.attrs,
        coordinates=coordinates,
        extra_fields=extra_fields,
        config=geometry_config.GeometryConfig(),
        process_props=process_props,
        exchange=exchange,
        global_reductions=global_reductions,
    )
    yield grid_geometry


@pytest.fixture
def interpolation_factory_from_savepoint(
    *,
    experiment: definitions.Experiment,
    grid_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend,
    decomposition_info: decomposition.DecompositionInfo,
    process_props: decomposition.ProcessProperties,
    geometry_from_savepoint: geometry.GridGeometry,
) -> Generator[interpolation_factory.InterpolationFieldsFactory]:
    geometry_source = geometry_from_savepoint
    exchange = decomposition.create_exchange(process_props, decomposition_info)
    intp_factory = interpolation_factory.InterpolationFieldsFactory(
        grid=geometry_source.grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_source,
        config=experiment.config.interpolation,
        backend=backend,
        metadata=interpolation_attributes.attrs,
        exchange=exchange,
    )
    yield intp_factory


@pytest.fixture
def metrics_factory_from_savepoint(
    *,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
    grid_savepoint: serialbox.IconGridSavepoint,
    topography_savepoint: serialbox.TopographySavepoint,
    decomposition_info: decomposition.DecompositionInfo,
    process_props: decomposition.ProcessProperties,
    geometry_from_savepoint: geometry.GridGeometry,
    interpolation_factory_from_savepoint: interpolation_factory.InterpolationFieldsFactory,
) -> Generator[metrics_factory.MetricsFieldsFactory]:
    exchange = decomposition.create_exchange(process_props, decomposition_info)
    global_reductions = decomposition.create_reduction(process_props, decomposition_info)
    geometry_source = geometry_from_savepoint
    interpolation_field_source = interpolation_factory_from_savepoint
    topography = topography_savepoint.topo_c()
    vertical_config = experiment.config.vertical_grid
    vertical_grid = vertical.VerticalGrid(
        vertical_config, grid_savepoint.vct_a(), grid_savepoint.vct_b()
    )
    factory = metrics_factory.MetricsFieldsFactory(
        grid=geometry_source.grid,
        vertical_grid=vertical_grid,
        decomposition_info=decomposition_info,
        geometry_source=geometry_source,
        topography=topography,
        interpolation_source=interpolation_field_source,
        config=experiment.config.metrics,
        backend=backend,
        metadata=metrics_attributes.attrs,
        exchange=exchange,
        global_reductions=global_reductions,
    )

    yield factory
