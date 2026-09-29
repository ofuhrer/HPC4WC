# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import functools
import re
from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

from icon4py.model.common import constants, dimension as dims, model_backends
from icon4py.model.common.grid import base, gridfile, horizontal as h_grid, icon
from icon4py.model.testing import definitions, grid_utils as gridtest_utils
from icon4py.model.testing.fixtures import (
    backend,
    cpu_allocator,
    data_provider,
    download_ser_data,
    experiment,
    grid_savepoint,
    icon_grid,
    process_props,
)

from .. import utils


if TYPE_CHECKING:
    from collections.abc import Iterator

    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import base as base_grid


@functools.cache
def grid_from_limited_area_grid_file() -> icon.IconGrid:
    return gridtest_utils.get_grid_manager_from_identifier(
        definitions.Experiments.MCH_CH_R04B09.grid,
        num_levels=65,
        keep_skip_values=True,
        allocator=model_backends.get_allocator(None),
    ).grid


def lateral_boundary() -> Iterator[h_grid.Zone]:
    for marker in h_grid.Zone.__members__.values():
        if "lb" in marker.value:
            yield marker


def nudging() -> Iterator[h_grid.Zone]:
    for marker in h_grid.Zone.__members__.values():
        if "nudging" in marker.value:
            yield marker


LATERAL_BOUNDARY_IDX = {
    dims.CellDim: [0, 850, 1688, 2511, 3316, 4104],
    dims.EdgeDim: [0, 428, 1278, 1700, 2538, 2954, 3777, 4184, 4989, 5387, 6176],
    dims.VertexDim: [0, 428, 850, 1266, 1673, 2071],
}

NUDGING_IDX = {
    dims.CellDim: [3316, 4104],
    dims.EdgeDim: [4989, 5387, 6176],
    dims.VertexDim: [1673, 2071],
}
HALO_IDX = {
    dims.CellDim: [20896, 20896],
    dims.EdgeDim: [31558, 31558],
    dims.VertexDim: [10663, 10663],
}
INTERIOR_IDX = {
    dims.CellDim: [4104, HALO_IDX[dims.CellDim][0]],
    dims.EdgeDim: [6176, HALO_IDX[dims.EdgeDim][0]],
    dims.VertexDim: [2071, HALO_IDX[dims.VertexDim][0]],
}


@pytest.fixture(
    params=[definitions.Experiments.MCH_CH_R04B09],
    ids=lambda r: r.name,
)
def experiment_description(
    request: pytest.FixtureRequest,
) -> definitions.ExperimentDescription:
    return request.param


@pytest.fixture(params=["serialbox", "file"])
def grid(
    icon_grid: base_grid.Grid,
    request: pytest.FixtureRequest,
) -> base_grid.Grid:
    if request.param == "serialbox":
        return icon_grid
    else:
        return grid_from_limited_area_grid_file()


@pytest.mark.datatest
@pytest.mark.parametrize("dim", dims.horizontal_dims())
@pytest.mark.parametrize("marker", [h_grid.Zone.HALO, h_grid.Zone.HALO_LEVEL_2])
def test_halo(grid: base_grid.Grid, dim: gtx.Dimension, marker: h_grid.Zone) -> None:
    # For single node this returns an empty region - start and end index are the same see  also ./mpi_tests/test_icon.py
    domain = h_grid.domain(dim)(marker)
    assert grid.start_index(domain) == HALO_IDX[dim][0]
    assert grid.end_index(domain) == HALO_IDX[dim][1]


@pytest.mark.datatest
@pytest.mark.parametrize("dim", dims.horizontal_dims())
def test_local(dim: gtx.Dimension, grid: base_grid.Grid) -> None:
    domain = h_grid.domain(dim)(h_grid.Zone.LOCAL)
    assert grid.start_index(domain) == 0
    assert grid.end_index(domain) == grid.size[dim]


@pytest.mark.datatest
@pytest.mark.parametrize("dim", dims.horizontal_dims())
@pytest.mark.parametrize("marker", lateral_boundary())
def test_lateral_boundary(grid: base_grid.Grid, dim: gtx.Dimension, marker: h_grid.Zone) -> None:
    num = int(next(iter(re.findall(r"\d+", marker.value))))
    if num > 4 and dim in (dims.VertexDim, dims.CellDim):
        with pytest.raises(AssertionError, match=f"Invalid zone {marker} for dimension"):
            h_grid.domain(dim)(marker)
    else:
        domain = h_grid.domain(dim)(marker)
        start_index = grid.start_index(domain)
        end_index = grid.end_index(domain)
        assert start_index == LATERAL_BOUNDARY_IDX[dim][num - 1]
        assert end_index == LATERAL_BOUNDARY_IDX[dim][num]


@pytest.mark.datatest
@pytest.mark.parametrize("dim", dims.horizontal_dims())
def test_end(grid: base_grid.Grid, dim: gtx.Dimension) -> None:
    domain = h_grid.domain(dim)(h_grid.Zone.END)
    assert grid.start_index(domain) == grid.size[dim]
    assert grid.end_index(domain) == grid.size[dim]


@pytest.mark.datatest
@pytest.mark.parametrize("marker", nudging())
@pytest.mark.parametrize("dim", dims.horizontal_dims())
def test_nudging(grid: base_grid.Grid, dim: gtx.Dimension, marker: h_grid.Zone) -> None:
    num = int(next(iter(re.findall(r"\d+", marker.value))))
    if dim in (dims.VertexDim, dims.CellDim) and num > 1:
        with pytest.raises(AssertionError, match=f"Invalid zone {marker} for dimension"):
            h_grid.domain(dim)(marker)
    else:
        domain = h_grid.domain(dim)(marker)
        start_index = grid.start_index(domain)
        end_index = grid.end_index(domain)
        assert start_index == NUDGING_IDX[dim][num - 1]
        assert end_index == NUDGING_IDX[dim][num]


@pytest.mark.datatest
@pytest.mark.parametrize("dim", dims.horizontal_dims())
def test_interior(grid: base_grid.Grid, dim: gtx.Dimension) -> None:
    domain = h_grid.domain(dim)(h_grid.Zone.INTERIOR)
    start_index = grid.start_index(domain)
    end_index = grid.end_index(domain)
    assert start_index == INTERIOR_IDX[dim][0]
    assert end_index == INTERIOR_IDX[dim][1]


@pytest.mark.datatest
def test_grid_size(icon_grid: base_grid.Grid) -> None:
    assert icon_grid.size[dims.VertexDim] == 10663
    assert icon_grid.size[dims.CellDim] == 20896
    assert icon_grid.size[dims.EdgeDim] == 31558


@pytest.mark.parametrize(
    "grid_description",
    (definitions.Grids.MCH_CH_R04B09_DSL, definitions.Grids.R02B04_GLOBAL),
)
@pytest.mark.parametrize("offset", (utils.horizontal_offsets()), ids=lambda x: x.value)
def test_when_keep_skip_value_then_neighbor_table_matches_config(
    grid_description: definitions.GridDescription,
    offset: gtx.FieldOffset,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(grid_description, keep_skip_values=True, backend=backend).grid
    connectivity = grid.get_connectivity(offset)

    assert (
        np.any(connectivity.asnumpy() == gridfile.GridFile.INVALID_INDEX).item()
    ) == icon._has_skip_values(offset, grid.config.limited_area)
    if not icon._has_skip_values(offset, grid.config.limited_area):
        assert connectivity.skip_value is None, f"skip value for offset {offset} should be None"
    else:
        assert connectivity.skip_value == gridfile.GridFile.INVALID_INDEX, (
            f"skip for offset {offset} value should be {gridfile.GridFile.INVALID_INDEX}"
        )


@pytest.mark.parametrize(
    "grid_description",
    (definitions.Grids.MCH_CH_R04B09_DSL, definitions.Grids.R02B04_GLOBAL),
)
@pytest.mark.parametrize("dim", (dims.local_dims()))
def test_when_replace_skip_values_then_only_pentagon_points_remain(
    grid_description: definitions.GridDescription,
    dim: gtx.Dimension,
    backend: gtx_typing.Backend,
) -> None:
    if dim == dims.V2E2VDim:
        pytest.skip("V2E2VDim is not supported in the current grid configuration.")
    if dim == dims.LsqUnkDim:
        pytest.skip("LsqUnkDim is not an offset dimension.")
    grid = utils.run_grid_manager(grid_description, keep_skip_values=False, backend=backend).grid
    connectivity = grid.get_connectivity(dim.value)
    if dim in icon.CONNECTIVITIES_ON_PENTAGONS and not grid.limited_area:
        assert np.any(connectivity.asnumpy() == gridfile.GridFile.INVALID_INDEX).item(), (
            f"Connectivity {dim.value} for {grid_description.name} should have skip values."
        )
        assert connectivity.skip_value == gridfile.GridFile.INVALID_INDEX
    else:
        assert not np.any(connectivity.asnumpy() == gridfile.GridFile.INVALID_INDEX).item(), (
            f"Connectivity {dim.value} for {grid_description.name} contains skip values, but none are expected."
        )
        assert connectivity.skip_value is None


@pytest.mark.parametrize(
    "geometry_type,grid_root,grid_level",
    [
        (icon.GeometryType.ICOSAHEDRON, 1, 0),
        (icon.GeometryType.ICOSAHEDRON, 1, 1),
        (icon.GeometryType.ICOSAHEDRON, 1, 2),
        (icon.GeometryType.ICOSAHEDRON, 2, 4),
        (icon.GeometryType.ICOSAHEDRON, 4, 9),
        (icon.GeometryType.TORUS, None, None),
    ],
)
def test_grid_params(
    geometry_type: icon.GeometryType,
    grid_root: int | None,
    grid_level: int | None,
) -> None:
    match geometry_type:
        case icon.GeometryType.ICOSAHEDRON:
            params = icon.GridParams(
                icon.IcosahedronParams(
                    subdivision=icon.GridSubdivision(root=grid_root, level=grid_level),  # type: ignore[arg-type]
                ),
            )
        case icon.GeometryType.TORUS:
            params = icon.GridParams(
                icon.TorusParams(
                    domain_length=42.0,
                    domain_height=100.5,
                ),
            )

    assert geometry_type == params.geometry_type
    match geometry_type:
        case icon.GeometryType.ICOSAHEDRON:
            assert params.subdivision == icon.GridSubdivision(root=grid_root, level=grid_level)  # type: ignore[arg-type]
            assert pytest.approx(params.radius) == constants.EARTH_RADIUS
            assert params.domain_length is None
            assert params.domain_height is None
        case icon.GeometryType.TORUS:
            assert params.subdivision is None
            assert params.radius is None
            assert params.domain_length == 42.0
            assert params.domain_height == 100.5


@pytest.mark.parametrize(
    "grid_root,grid_level",
    [
        (0, 0),
    ],
)
def test_icosahedron_params_fail(grid_root: int, grid_level: int) -> None:
    with pytest.raises(ValueError):
        _ = icon.IcosahedronParams(
            subdivision=icon.GridSubdivision(root=grid_root, level=grid_level),
        )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "grid_description, geometry_type, subdivision, radius, domain_length, domain_height, num_cells",
    [
        (
            definitions.Grids.R02B04_GLOBAL,
            icon.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=2, level=4),
            constants.EARTH_RADIUS,
            None,
            None,
            20480,
        ),
        (
            definitions.Grids.MCH_OPR_R04B07_DOMAIN01,
            icon.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=4, level=7),
            constants.EARTH_RADIUS,
            None,
            None,
            10700,
        ),
        (
            definitions.Grids.MCH_OPR_R19B08_DOMAIN01,
            icon.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=19, level=8),
            constants.EARTH_RADIUS,
            None,
            None,
            44528,
        ),
        (
            definitions.Grids.MCH_CH_R04B09_DSL,
            icon.GeometryType.ICOSAHEDRON,
            icon.GridSubdivision(root=4, level=9),
            constants.EARTH_RADIUS,
            None,
            None,
            20896,
        ),
        (
            definitions.Grids.TORUS_100X116_1000M,
            icon.GeometryType.TORUS,
            None,
            None,
            100000.0,
            100458.94683899487,
            23200,
        ),
        (
            definitions.Grids.TORUS_50000x5000,
            icon.GeometryType.TORUS,
            None,
            None,
            50000.0,
            5248.638810814779,
            1056,
        ),
    ],
)
def test_grid_params_from_grid_manager(  # noqa: PLR0917 [too-many-positional-arguments]
    grid_description: definitions.GridDescription,
    backend: gtx_typing.Backend,
    geometry_type: icon.GeometryType,
    subdivision: icon.GridSubdivision,
    radius: float,
    domain_length: float,
    domain_height: float,
    num_cells: int,
) -> None:
    grid = utils.run_grid_manager(grid_description, keep_skip_values=True, backend=backend).grid
    params = grid.grid_params
    assert params is not None
    assert params.geometry_type == geometry_type
    match geometry_type:
        case icon.GeometryType.ICOSAHEDRON:
            assert params.subdivision == subdivision
        case icon.GeometryType.TORUS:
            assert params.subdivision is None
    assert pytest.approx(params.radius) == radius
    assert pytest.approx(params.domain_length) == domain_length
    assert pytest.approx(params.domain_height) == domain_height
    assert grid.config.num_cells == num_cells
