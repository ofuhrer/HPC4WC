# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import math
from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import vertical as v_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, test_utils
from icon4py.model.testing.fixtures import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    metrics_savepoint,
    process_props,
    topography_savepoint,
)


if TYPE_CHECKING:
    from collections.abc import Iterator

    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid
    from icon4py.model.testing import serialbox as sb


@pytest.mark.parametrize(
    "max_h,damping_height,delta,flat_height",
    [(60000, 34000, 612, 50000), (12000, 10000, 100, 11000), (109050, 45000, 123, 80000)],
)
def test_damping_layer_calculation(
    max_h: float, damping_height: float, delta: float, flat_height: float
) -> None:
    vct_a = np.arange(0, max_h, delta)
    vct_a_field = gtx.as_field((dims.KDim,), data=vct_a[::-1])  # type: ignore[arg-type] # TODO(havogt): needs fix in GT4Py
    vertical_config = v_grid.VerticalGridConfig(
        num_levels=1000,
        flat_height=flat_height,
        rayleigh_damping_height=damping_height,
    )
    vertical_params = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a_field,
        vct_b=None,
    )
    assert (
        vertical_params.end_index_of_damping_layer
        == vct_a.shape[0] - math.ceil(damping_height / delta) - 1
    )


@pytest.mark.datatest
def test_damping_layer_calculation_from_icon_input(
    grid_savepoint: sb.IconGridSavepoint, experiment: definitions.Experiment
) -> None:
    a = grid_savepoint.vct_a()
    b = grid_savepoint.vct_b()
    nrdmax = grid_savepoint.nrdmax()
    vertical_config = experiment.config.vertical_grid
    vertical_grid = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=a,
        vct_b=b,
    )
    assert nrdmax == vertical_grid.end_index_of_damping_layer
    a_array = a.ndarray
    damping_height = min(vertical_config.rayleigh_damping_height, a_array[0])
    assert a_array[nrdmax] >= damping_height
    assert a_array[nrdmax + 1] < damping_height
    assert vertical_grid.index(v_grid.Domain(dims.KDim, v_grid.Zone.DAMPING)) == nrdmax


@pytest.mark.datatest
def test_grid_size(
    experiment: definitions.Experiment, grid_savepoint: sb.IconGridSavepoint
) -> None:
    config = v_grid.VerticalGridConfig(num_levels=grid_savepoint.num(dims.KDim))
    vertical_grid = v_grid.VerticalGrid(
        config=config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
    )

    assert experiment.config.vertical_grid.num_levels == vertical_grid.size(dims.KDim)
    assert experiment.config.vertical_grid.num_levels + 1 == vertical_grid.size(dims.KHalfDim)


@pytest.mark.parametrize(
    "dim", (dims.CellDim, dims.VertexDim, dims.EdgeDim, dims.C2EDim, dims.C2VDim, dims.E2VDim)
)
@pytest.mark.datatest
def test_grid_size_raises_for_non_vertical_dim(
    grid_savepoint: sb.IconGridSavepoint, dim: gtx.Dimension
) -> None:
    vertical_grid = configure_vertical_grid(grid_savepoint)
    with pytest.raises(AssertionError):
        vertical_grid.size(dim)


@pytest.mark.datatest
def test_grid_size_raises_for_unknown_vertical_dim(grid_savepoint: sb.IconGridSavepoint) -> None:
    vertical_grid = configure_vertical_grid(grid_savepoint)
    j_dim = gtx.Dimension("J", kind=gtx.DimensionKind.VERTICAL)
    with pytest.raises(ValueError):
        vertical_grid.size(j_dim)


def configure_vertical_grid(
    grid_savepoint: sb.IconGridSavepoint, top_moist_threshold: float = 22500.0
) -> v_grid.VerticalGrid:
    config = v_grid.VerticalGridConfig(
        num_levels=grid_savepoint.num(dims.KDim), htop_moist_proc=top_moist_threshold
    )
    vertical_grid = v_grid.VerticalGrid(
        config=config,
        vct_a=grid_savepoint.vct_a(),
        vct_b=grid_savepoint.vct_b(),
    )

    return vertical_grid


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description, expected_moist_level",
    [(definitions.Experiments.MCH_CH_R04B09, 0), (definitions.Experiments.EXCLAIM_APE, 25)],
)
def test_moist_level_calculation(
    grid_savepoint: sb.IconGridSavepoint, expected_moist_level: int
) -> None:
    threshold = 22500.0
    vertical_grid = configure_vertical_grid(grid_savepoint, top_moist_threshold=threshold)
    assert expected_moist_level == vertical_grid.kstart_moist
    assert expected_moist_level == vertical_grid.index(v_grid.Domain(dims.KDim, v_grid.Zone.MOIST))


@pytest.mark.datatest
def test_interface_physical_height(grid_savepoint: sb.IconGridSavepoint) -> None:
    vertical_grid = configure_vertical_grid(grid_savepoint)
    assert test_utils.dallclose(
        grid_savepoint.vct_a().asnumpy(), vertical_grid.interface_physical_height.asnumpy()
    )


@pytest.mark.datatest
def test_flat_level_calculation(grid_savepoint: sb.IconGridSavepoint) -> None:
    vertical_grid = configure_vertical_grid(grid_savepoint)

    assert grid_savepoint.nflatlev() == vertical_grid.nflatlev
    assert grid_savepoint.nflatlev() == vertical_grid.index(
        v_grid.Domain(dims.KDim, v_grid.Zone.FLAT)
    )


def offsets() -> Iterator[int]:
    yield from range(5)


def vertical_zones() -> Iterator[v_grid.Zone]:
    yield from v_grid.Zone.__members__.values()


@pytest.mark.parametrize("zone", vertical_zones())
@pytest.mark.parametrize("kind", (gtx.DimensionKind.LOCAL, gtx.DimensionKind.HORIZONTAL))
def test_domain_raises_for_non_vertical_dim(zone: v_grid.Zone, kind: gtx.DimensionKind) -> None:
    dim = gtx.Dimension("I", kind=kind)
    with pytest.raises(AssertionError):
        v_grid.Domain(dim, zone)


@pytest.mark.datatest
@pytest.mark.parametrize("dim", dims.vertical_dims())
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_top(
    grid_savepoint: sb.IconGridSavepoint, dim: gtx.Dimension, offset: int
) -> None:
    vertical_grid = configure_vertical_grid(grid_savepoint)
    assert offset == vertical_grid.index(v_grid.Domain(dim, v_grid.Zone.TOP, offset))


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE])
@pytest.mark.parametrize("dim", dims.vertical_dims())
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_damping(
    grid_savepoint: sb.IconGridSavepoint, dim: gtx.Dimension, offset: int
) -> None:
    vertical_grid = configure_vertical_grid(grid_savepoint)
    upwards = -offset
    downwards = offset
    zone = v_grid.Zone.DAMPING
    domain = v_grid.Domain(dim, zone, upwards)
    assert vertical_grid.end_index_of_damping_layer + upwards == vertical_grid.index(domain)
    domain = v_grid.Domain(dim, zone, downwards)
    assert vertical_grid.end_index_of_damping_layer + downwards == vertical_grid.index(domain)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE])
@pytest.mark.parametrize("dim", dims.vertical_dims())
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_moist(
    grid_savepoint: sb.IconGridSavepoint, dim: gtx.Dimension, offset: int
) -> None:
    vertical_grid = configure_vertical_grid(grid_savepoint)
    upwards = -offset
    downwards = offset
    zone = v_grid.Zone.MOIST
    domain = v_grid.Domain(dim, zone, upwards)
    assert vertical_grid.kstart_moist + upwards == vertical_grid.index(domain)
    domain = v_grid.Domain(dim, zone, downwards)
    assert vertical_grid.kstart_moist + downwards == vertical_grid.index(domain)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE])
@pytest.mark.parametrize("dim", dims.vertical_dims())
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_flat(
    grid_savepoint: sb.IconGridSavepoint, dim: gtx.Dimension, offset: int
) -> None:
    vertical_grid = configure_vertical_grid(grid_savepoint)
    upwards = -offset
    downwards = offset
    zone = v_grid.Zone.FLAT
    domain = v_grid.Domain(dim, zone, upwards)
    assert vertical_grid.nflatlev + upwards == vertical_grid.index(domain)
    domain = v_grid.Domain(dim, zone, downwards)
    assert vertical_grid.nflatlev + downwards == vertical_grid.index(domain)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.MCH_CH_R04B09, definitions.Experiments.EXCLAIM_APE],
)
@pytest.mark.parametrize("dim", dims.vertical_dims())
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_bottom(
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
    dim: gtx.Dimension,
    offset: int,
) -> None:
    valid_offset = -offset
    vertical_grid = configure_vertical_grid(grid_savepoint)
    num_levels = (
        experiment.config.vertical_grid.num_levels
        if dim == dims.KDim
        else experiment.config.vertical_grid.num_levels + 1
    )
    domain = v_grid.Domain(dim, v_grid.Zone.BOTTOM, valid_offset)
    assert num_levels + valid_offset == vertical_grid.index(domain)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE])
@pytest.mark.parametrize("zone", vertical_zones())
@pytest.mark.parametrize("dim", dims.vertical_dims())
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_raises_if_index_above_num_levels(
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
    zone: v_grid.Zone,
    dim: gtx.Dimension,
    offset: int,
) -> None:
    vertical_size = (
        experiment.config.vertical_grid.num_levels
        if dim == dims.KDim
        else experiment.config.vertical_grid.num_levels + 1
    )
    invalid_offset = vertical_size + 1 + offset
    vertical_grid = configure_vertical_grid(grid_savepoint)
    domain = v_grid.Domain(dim, zone, invalid_offset)
    with pytest.raises(expected_exception=AssertionError):
        vertical_grid.index(domain)


@pytest.mark.datatest
@pytest.mark.parametrize("experiment_description", [definitions.Experiments.EXCLAIM_APE])
@pytest.mark.parametrize("zone", vertical_zones())
@pytest.mark.parametrize("dim", dims.vertical_dims())
@pytest.mark.parametrize("offset", offsets())
def test_grid_index_raises_if_index_below_zero(
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
    zone: v_grid.Zone,
    dim: gtx.Dimension,
    offset: int,
) -> None:
    vertical_size = (
        experiment.config.vertical_grid.num_levels
        if dim == dims.KDim
        else experiment.config.vertical_grid.num_levels + 1
    )
    invalid_offset = -(vertical_size + 1 + offset)
    vertical_grid = configure_vertical_grid(grid_savepoint)
    with pytest.raises(expected_exception=AssertionError):
        domain = v_grid.Domain(dim, zone, invalid_offset)
        vertical_grid.index(domain)


@pytest.mark.datatest
def test_vct_a_vct_b_calculation_from_icon_input(
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    vertical_config = experiment.config.vertical_grid
    vct_a, vct_b = v_grid.get_vct_a_and_vct_b(vertical_config, backend)

    assert test_utils.dallclose(vct_a.asnumpy(), grid_savepoint.vct_a().asnumpy())
    assert test_utils.dallclose(vct_b.asnumpy(), grid_savepoint.vct_b().asnumpy())


@pytest.mark.level("unit")
@pytest.mark.datatest
@pytest.mark.parametrize(
    "experiment_description",
    [
        definitions.Experiments.MCH_CH_R04B09,
        definitions.Experiments.GAUSS3D,
        definitions.Experiments.EXCLAIM_APE,
    ],
)
def test_compute_vertical_coordinate(  # noqa: PLR0917 [too-many-positional-arguments]
    grid_savepoint: sb.IconGridSavepoint,
    metrics_savepoint: sb.MetricSavepoint,
    topography_savepoint: sb.TopographySavepoint,
    interpolation_savepoint: sb.InterpolationSavepoint,
    icon_grid: base_grid.Grid,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    vct_a = grid_savepoint.vct_a()
    vct_b = grid_savepoint.vct_b()
    cell_geometry = grid_savepoint.construct_cell_geometry()

    vertical_config = experiment.config.vertical_grid

    vertical_geometry = v_grid.VerticalGrid(
        config=vertical_config,
        vct_a=vct_a,
        vct_b=vct_b,
    )
    assert vertical_geometry.nflatlev == grid_savepoint.nflatlev()

    topography = None
    if experiment.description in (
        definitions.Experiments.MCH_CH_R04B09,
        definitions.Experiments.GAUSS3D,
    ):
        topography = topography_savepoint.topo_c()
    elif experiment.description == definitions.Experiments.EXCLAIM_APE:
        topography = data_alloc.zero_field(
            icon_grid, dims.CellDim, allocator=backend, dtype=ta.wpfloat
        )

    geofac_n2s = interpolation_savepoint.geofac_n2s()

    assert cell_geometry.area is not None
    assert topography is not None

    vertical_coordinates_on_half_levels = v_grid.compute_vertical_coordinate(
        vct_a=vct_a.ndarray,
        topography=topography.ndarray,
        cell_areas=cell_geometry.area.ndarray,
        geofac_n2s=geofac_n2s.ndarray,
        c2e2co=icon_grid.get_connectivity("C2E2CO").ndarray,
        nflatlev=vertical_geometry.nflatlev,
        model_top_height=vertical_config.model_top_height,
        SLEVE_decay_scale_1=vertical_config.SLEVE_decay_scale_1,
        SLEVE_decay_exponent=vertical_config.SLEVE_decay_exponent,
        SLEVE_decay_scale_2=vertical_config.SLEVE_decay_scale_2,
        SLEVE_minimum_layer_thickness_1=vertical_config._SLEVE_minimum_layer_thickness_1,
        SLEVE_minimum_relative_layer_thickness_1=vertical_config._SLEVE_minimum_relative_layer_thickness_1,
        SLEVE_minimum_layer_thickness_2=vertical_config._SLEVE_minimum_layer_thickness_2,
        SLEVE_minimum_relative_layer_thickness_2=vertical_config._SLEVE_minimum_relative_layer_thickness_2,
        lowest_layer_thickness=vertical_config.lowest_layer_thickness,
        exchange=decomposition.single_node_exchange,
    )

    assert test_utils.dallclose(
        data_alloc.as_numpy(vertical_coordinates_on_half_levels),
        metrics_savepoint.z_ifc().asnumpy(),
        rtol=1e-9,
        atol=1e-13,
    )
