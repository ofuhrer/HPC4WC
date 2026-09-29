# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from collections.abc import Generator

import gt4py.next as gtx
import pytest

import icon4py.model.common.dimension as dims
from icon4py.model.common import model_backends, model_options, topography
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import (
    geometry as grid_geometry,
    geometry_attributes as geometry_meta,
    geometry_config,
    grid_manager as gm,
    vertical as v_grid,
)
from icon4py.model.common.interpolation import interpolation_attributes, interpolation_factory
from icon4py.model.common.metrics import metrics_attributes, metrics_factory
from icon4py.model.common.topography.analytical import jablonowski_williamson as jw_topo


@pytest.fixture(
    scope="session",
)
def geometry_field_source(
    grid_manager: gm.GridManager | None,
    backend_like: model_backends.BackendLike,
) -> Generator[grid_geometry.GridGeometry, None, None]:
    if not grid_manager:
        pytest.skip("Incomplete grid Information for test, are you running with `simple_grid`?")
    mesh = grid_manager.grid

    generic_concrete_backend = model_options.customize_backend(None, backend_like)
    decomposition_info = grid_manager.decomposition_info

    geometry_field_source = grid_geometry.GridGeometry(
        grid=mesh,
        decomposition_info=decomposition_info,
        backend=generic_concrete_backend,
        coordinates=grid_manager.coordinates,
        extra_fields=grid_manager.geometry_fields,
        metadata=geometry_meta.attrs,
        config=geometry_config.GeometryConfig(),
        process_props=decomposition.SingleNodeProcessProperties(),
        exchange=decomposition.single_node_exchange,
    )
    yield geometry_field_source


@pytest.fixture(
    scope="session",
)
def interpolation_field_source(
    grid_manager: gm.GridManager,
    geometry_field_source: grid_geometry.GridGeometry,
    backend_like: model_backends.BackendLike,
) -> Generator[interpolation_factory.InterpolationFieldsFactory, None, None]:
    mesh = grid_manager.grid

    generic_concrete_backend = model_options.customize_backend(None, backend_like)
    decomposition_info = grid_manager.decomposition_info

    interpolation_field_source = interpolation_factory.InterpolationFieldsFactory(
        config=interpolation_factory.InterpolationConfig(),
        grid=mesh,
        decomposition_info=decomposition_info,
        exchange=decomposition.single_node_exchange,
        geometry_source=geometry_field_source,
        backend=generic_concrete_backend,
        metadata=interpolation_attributes.attrs,
    )
    yield interpolation_field_source


@pytest.fixture(
    scope="session",
)
def metrics_field_source(
    grid_manager: gm.GridManager,
    geometry_field_source: grid_geometry.GridGeometry,
    interpolation_field_source: interpolation_factory.InterpolationFieldsFactory,
    backend_like: model_backends.BackendLike,
) -> Generator[metrics_factory.MetricsFieldsFactory, None, None]:
    mesh = grid_manager.grid

    allocator = model_backends.get_allocator(backend_like)
    generic_concrete_backend = model_options.customize_backend(None, backend_like)
    decomposition_info = grid_manager.decomposition_info

    vertical_config = v_grid.VerticalGridConfig(
        mesh.num_levels,
        lowest_layer_thickness=50,
        model_top_height=23500.0,
        stretch_factor=1.0,
        rayleigh_damping_height=1.0,
    )
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, allocator)

    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )

    config = topography.TopographyConfig(
        config=jw_topo.JablonowskiWilliamsonConfig(),
    )
    topo_c = topography.create(
        config=config,
        grid_manager=grid_manager,
        backend=generic_concrete_backend,
        exchange=decomposition.single_node_exchange,
    )

    metrics_field_source = metrics_factory.MetricsFieldsFactory(
        grid=mesh,
        vertical_grid=vertical_grid,
        decomposition_info=decomposition_info,
        exchange=decomposition.single_node_exchange,
        geometry_source=geometry_field_source,
        topography=gtx.as_field((dims.CellDim,), data=topo_c),  # type: ignore[arg-type]  # NDArrayObject is not exported from gt4py
        interpolation_source=interpolation_field_source,
        config=metrics_factory.MetricsConfig(),
        backend=generic_concrete_backend,
        metadata=metrics_attributes.attrs,
    )
    yield metrics_field_source
