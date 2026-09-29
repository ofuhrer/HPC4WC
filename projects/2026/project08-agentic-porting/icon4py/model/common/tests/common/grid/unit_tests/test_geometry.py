# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import dataclasses
import functools
from typing import TYPE_CHECKING

import numpy as np
import pytest

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import (
    base,
    geometry,
    geometry_attributes as attrs,
    horizontal as h_grid,
    icon as icon_grid,
    simple,
)
from icon4py.model.common.grid.geometry import as_sparse_field
from icon4py.model.common.math import vector_operations as vector_ops
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions, grid_utils, test_utils
from icon4py.model.testing.fixtures import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    process_props,
)

from .. import utils


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox as sb


@pytest.mark.datatest
def test_geometry_raises_for_unknown_field(
    backend: gtx_typing.Backend, experiment: definitions.Experiment
) -> None:
    geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    with pytest.raises(ValueError, match="Field 'foo' not provided by the source"):
        geometry.get("foo")


@pytest.mark.parametrize(
    "experiment_description, rtol",
    [
        (definitions.Experiments.MCH_CH_R04B09, 1e-7),
        (definitions.Experiments.EXCLAIM_APE, 3e-12),
        (definitions.Experiments.GAUSS3D, 1e-13),
    ],
)
@pytest.mark.datatest
def test_edge_control_area(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
    rtol: float,
) -> None:
    expected = grid_savepoint.edge_areas()
    geometry_source = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    result = geometry_source.get(attrs.EDGE_AREA)
    assert test_utils.dallclose(expected.asnumpy(), result.asnumpy(), rtol=rtol)


@pytest.mark.datatest
def test_coriolis_parameter(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    geometry_source = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    expected = grid_savepoint.f_e()

    result = geometry_source.get(attrs.CORIOLIS_PARAMETER)
    assert test_utils.dallclose(expected.asnumpy(), result.asnumpy())


@pytest.mark.datatest
def test_compute_edge_length(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    geometry_source = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    expected = grid_savepoint.primal_edge_length()
    result = geometry_source.get(attrs.EDGE_LENGTH)
    assert test_utils.dallclose(result.asnumpy(), expected.asnumpy())


@pytest.mark.datatest
def test_compute_inverse_edge_length(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    expected = grid_savepoint.inverse_primal_edge_lengths()
    geometry_source = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    computed = geometry_source.get(f"inverse_of_{attrs.EDGE_LENGTH}")

    assert test_utils.dallclose(computed.asnumpy(), expected.asnumpy())


@pytest.mark.datatest
def test_compute_dual_edge_length(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)

    expected = grid_savepoint.dual_edge_length()
    result = grid_geometry.get(attrs.DUAL_EDGE_LENGTH)
    assert test_utils.dallclose(result.asnumpy(), expected.asnumpy())


@pytest.mark.datatest
def test_compute_inverse_dual_edge_length(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    expected = grid_savepoint.inv_dual_edge_length()
    result = grid_geometry.get(f"inverse_of_{attrs.DUAL_EDGE_LENGTH}")

    # compared to ICON we overcompute, so we only compare the values from LATERAL_BOUNDARY_LEVEL_2
    level = h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    start_index = grid_geometry.grid.start_index(level)
    assert test_utils.dallclose(result.asnumpy()[start_index:], expected.asnumpy()[start_index:])


@pytest.mark.parametrize(
    "experiment_description, rtol",
    [
        (definitions.Experiments.MCH_CH_R04B09, 5e-10),
        (definitions.Experiments.EXCLAIM_APE, 1e-12),
        (definitions.Experiments.GAUSS3D, 1e-14),
    ],
)
@pytest.mark.datatest
def test_compute_inverse_vertex_vertex_length(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
    rtol: float,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)

    expected = grid_savepoint.inv_vert_vert_length().asnumpy()
    result = grid_geometry.get(attrs.INVERSE_VERTEX_VERTEX_LENGTH).asnumpy()
    assert test_utils.dallclose(result, expected, rtol=rtol)


@pytest.mark.datatest
def test_compute_coordinates_of_edge_tangent_and_normal(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    x_normal = grid_geometry.get(attrs.EDGE_NORMAL_X)
    y_normal = grid_geometry.get(attrs.EDGE_NORMAL_Y)
    z_normal = grid_geometry.get(attrs.EDGE_NORMAL_Z)
    x_tangent = grid_geometry.get(attrs.EDGE_TANGENT_X)
    y_tangent = grid_geometry.get(attrs.EDGE_TANGENT_Y)
    z_tangent = grid_geometry.get(attrs.EDGE_TANGENT_Z)

    x_normal_ref = grid_savepoint.primal_cart_normal_x()
    y_normal_ref = grid_savepoint.primal_cart_normal_y()
    z_normal_ref = grid_savepoint.primal_cart_normal_z()
    x_tangent_ref = grid_savepoint.dual_cart_normal_x()
    y_tangent_ref = grid_savepoint.dual_cart_normal_y()
    z_tangent_ref = grid_savepoint.dual_cart_normal_z()
    assert test_utils.dallclose(x_tangent.asnumpy(), x_tangent_ref.asnumpy(), atol=1e-12)
    assert test_utils.dallclose(y_tangent.asnumpy(), y_tangent_ref.asnumpy(), atol=1e-12)
    assert test_utils.dallclose(z_tangent.asnumpy(), z_tangent_ref.asnumpy(), atol=1e-12)
    assert test_utils.dallclose(x_normal.asnumpy(), x_normal_ref.asnumpy(), atol=1e-13)  # 1e-16
    assert test_utils.dallclose(z_normal.asnumpy(), z_normal_ref.asnumpy(), atol=1e-13)
    assert test_utils.dallclose(y_normal.asnumpy(), y_normal_ref.asnumpy(), atol=1e-12)


@pytest.mark.datatest
def test_compute_primal_normals(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    primal_normal_u = grid_geometry.get(attrs.EDGE_NORMAL_U)
    primal_normal_v = grid_geometry.get(attrs.EDGE_NORMAL_V)

    primal_normal_u_ref = grid_savepoint.primal_normal_v1()
    primal_normal_v_ref = grid_savepoint.primal_normal_v2()

    assert test_utils.dallclose(
        primal_normal_u.asnumpy(), primal_normal_u_ref.asnumpy(), atol=1e-12
    )
    assert test_utils.dallclose(
        primal_normal_v.asnumpy(), primal_normal_v_ref.asnumpy(), atol=1e-12
    )


@pytest.mark.datatest
def test_tangent_orientation(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    result = grid_geometry.get(attrs.TANGENT_ORIENTATION)
    expected = grid_savepoint.tangent_orientation()

    assert test_utils.dallclose(result.asnumpy(), expected.asnumpy())


@pytest.mark.datatest
def test_cell_area(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    result = grid_geometry.get(attrs.CELL_AREA)
    expected = grid_savepoint.cell_areas()

    assert test_utils.dallclose(result.asnumpy(), expected.asnumpy())


@pytest.mark.datatest
def test_primal_normal_cell(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    primal_normal_cell_u_ref = grid_savepoint.primal_normal_cell_x().asnumpy()
    primal_normal_cell_v_ref = grid_savepoint.primal_normal_cell_y().asnumpy()
    primal_normal_cell_u = grid_geometry.get(attrs.EDGE_NORMAL_CELL_U)
    primal_normal_cell_v = grid_geometry.get(attrs.EDGE_NORMAL_CELL_V)

    assert test_utils.dallclose(
        primal_normal_cell_u.asnumpy(), primal_normal_cell_u_ref, atol=1e-12
    )
    assert test_utils.dallclose(
        primal_normal_cell_v.asnumpy(), primal_normal_cell_v_ref, atol=1e-12
    )


@pytest.mark.datatest
def test_dual_normal_cell(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    dual_normal_cell_u_ref = grid_savepoint.dual_normal_cell_x().asnumpy()
    dual_normal_cell_v_ref = grid_savepoint.dual_normal_cell_y().asnumpy()
    dual_normal_cell_u = grid_geometry.get(attrs.EDGE_TANGENT_CELL_U)
    dual_normal_cell_v = grid_geometry.get(attrs.EDGE_TANGENT_CELL_V)

    assert test_utils.dallclose(dual_normal_cell_u.asnumpy(), dual_normal_cell_u_ref, atol=1e-12)
    assert test_utils.dallclose(dual_normal_cell_v.asnumpy(), dual_normal_cell_v_ref, atol=1e-12)


@pytest.mark.datatest
def test_primal_normal_vert(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    primal_normal_vert_u_ref = grid_savepoint.primal_normal_vert_x().asnumpy()
    primal_normal_vert_v_ref = grid_savepoint.primal_normal_vert_y().asnumpy()
    primal_normal_vert_u = grid_geometry.get(attrs.EDGE_NORMAL_VERTEX_U)
    primal_normal_vert_v = grid_geometry.get(attrs.EDGE_NORMAL_VERTEX_V)

    assert test_utils.dallclose(
        primal_normal_vert_u.asnumpy(), primal_normal_vert_u_ref, atol=1e-12
    )
    assert test_utils.dallclose(
        primal_normal_vert_v.asnumpy(), primal_normal_vert_v_ref, atol=1e-12
    )


@pytest.mark.datatest
def test_dual_normal_vert(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    dual_normal_vert_u_ref = grid_savepoint.dual_normal_vert_x().asnumpy()
    dual_normal_vert_v_ref = grid_savepoint.dual_normal_vert_y().asnumpy()
    dual_normal_vert_u = grid_geometry.get(attrs.EDGE_TANGENT_VERTEX_U)
    dual_normal_vert_v = grid_geometry.get(attrs.EDGE_TANGENT_VERTEX_V)

    assert test_utils.dallclose(dual_normal_vert_u.asnumpy(), dual_normal_vert_u_ref, atol=1e-12)
    assert test_utils.dallclose(dual_normal_vert_v.asnumpy(), dual_normal_vert_v_ref, atol=1e-12)


@pytest.mark.datatest
def test_cartesian_centers_edge(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    grid = grid_geometry.grid
    x = grid_geometry.get(attrs.EDGE_CENTER_X)
    y = grid_geometry.get(attrs.EDGE_CENTER_Y)
    z = grid_geometry.get(attrs.EDGE_CENTER_Z)

    ser_x = grid_savepoint.edges_center_cart_x()
    ser_y = grid_savepoint.edges_center_cart_y()
    ser_z = grid_savepoint.edges_center_cart_z()

    assert x.ndarray.shape == (grid.num_edges,)
    assert y.ndarray.shape == (grid.num_edges,)
    assert z.ndarray.shape == (grid.num_edges,)

    assert test_utils.dallclose(x.asnumpy(), ser_x.asnumpy(), atol=1e-15)
    assert test_utils.dallclose(y.asnumpy(), ser_y.asnumpy(), atol=1e-15)
    assert test_utils.dallclose(z.asnumpy(), ser_z.asnumpy(), atol=1e-15)

    match grid.geometry_type:
        case icon_grid.GeometryType.ICOSAHEDRON:
            # those are coordinates on the unit sphere: hence norm should be 1
            norm = data_alloc.zero_field(grid, dims.EdgeDim, dtype=x.dtype, allocator=backend)
            vector_ops.norm2_on_edges(x, z, y, out=norm, offset_provider={})
            assert test_utils.dallclose(norm.asnumpy(), 1.0)
        case icon_grid.GeometryType.TORUS:
            # on a torus coordinates should be within the domain
            assert all(x.asnumpy() >= 0.0)
            assert all(x.asnumpy() <= grid.grid_params.domain_length)
            assert all(y.asnumpy() >= 0.0)
            assert all(y.asnumpy() <= grid.grid_params.domain_height)
            assert all(z.asnumpy() == 0.0)


@pytest.mark.datatest
def test_cartesian_centers_cell(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    grid = grid_geometry.grid
    x = grid_geometry.get(attrs.CELL_CENTER_X)
    y = grid_geometry.get(attrs.CELL_CENTER_Y)
    z = grid_geometry.get(attrs.CELL_CENTER_Z)

    ser_x = grid_savepoint.cell_center_cart_x()
    ser_y = grid_savepoint.cell_center_cart_y()
    ser_z = grid_savepoint.cell_center_cart_z()

    assert x.ndarray.shape == (grid.num_cells,)
    assert y.ndarray.shape == (grid.num_cells,)
    assert z.ndarray.shape == (grid.num_cells,)

    assert test_utils.dallclose(x.asnumpy(), ser_x.asnumpy(), atol=1e-15)
    assert test_utils.dallclose(y.asnumpy(), ser_y.asnumpy(), atol=1e-15)
    assert test_utils.dallclose(z.asnumpy(), ser_z.asnumpy(), atol=1e-15)

    match grid.geometry_type:
        case icon_grid.GeometryType.ICOSAHEDRON:
            # those are coordinates on the unit sphere: hence norm should be 1
            norm = data_alloc.zero_field(grid, dims.CellDim, dtype=x.dtype, allocator=backend)
            vector_ops.norm2_on_cells(x, z, y, out=norm, offset_provider={})
            assert test_utils.dallclose(norm.asnumpy(), 1.0)
        case icon_grid.GeometryType.TORUS:
            # on a torus coordinates should be within the domain
            assert all(x.asnumpy() >= 0.0)
            assert all(x.asnumpy() <= grid.grid_params.domain_length)
            assert all(y.asnumpy() >= 0.0)
            assert all(y.asnumpy() <= grid.grid_params.domain_height)
            assert all(z.asnumpy() == 0.0)


@pytest.mark.datatest
def test_vertex(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    grid = grid_geometry.grid
    x = grid_geometry.get(attrs.VERTEX_X)
    y = grid_geometry.get(attrs.VERTEX_Y)
    z = grid_geometry.get(attrs.VERTEX_Z)

    ser_x = grid_savepoint.verts_vertex_cart_x()
    ser_y = grid_savepoint.verts_vertex_cart_y()
    ser_z = grid_savepoint.verts_vertex_cart_z()

    assert x.ndarray.shape == (grid.num_vertices,)
    assert y.ndarray.shape == (grid.num_vertices,)
    assert z.ndarray.shape == (grid.num_vertices,)

    assert test_utils.dallclose(x.asnumpy(), ser_x.asnumpy(), atol=1e-15)
    assert test_utils.dallclose(y.asnumpy(), ser_y.asnumpy(), atol=1e-15)
    assert test_utils.dallclose(z.asnumpy(), ser_z.asnumpy(), atol=1e-15)

    match grid.geometry_type:
        case icon_grid.GeometryType.ICOSAHEDRON:
            # those are coordinates on the unit sphere: hence norm should be 1
            norm = data_alloc.zero_field(grid, dims.VertexDim, dtype=x.dtype, allocator=backend)
            vector_ops.norm2_on_vertices(x, z, y, out=norm, offset_provider={})
            assert test_utils.dallclose(norm.asnumpy(), 1.0)
        case icon_grid.GeometryType.TORUS:
            # on a torus coordinates should be within the domain
            assert all(x.asnumpy() >= 0.0)
            assert all(x.asnumpy() <= grid.grid_params.domain_length)
            assert all(y.asnumpy() >= 0.0)
            assert all(y.asnumpy() <= grid.grid_params.domain_height)
            assert all(z.asnumpy() == 0.0)


def test_sparse_fields_creator() -> None:
    grid = simple.simple_grid()
    f1 = data_alloc.random_field(grid, dims.EdgeDim)
    f2 = data_alloc.random_field(grid, dims.EdgeDim)
    g1 = data_alloc.random_field(grid, dims.EdgeDim)
    g2 = data_alloc.random_field(grid, dims.EdgeDim)

    sparse = as_sparse_field((dims.EdgeDim, dims.E2CDim), [(f1, f2), (g1, g2)])
    sparse_e2c = functools.partial(as_sparse_field, (dims.EdgeDim, dims.E2CDim))
    sparse2 = sparse_e2c(((f1, f2), (g1, g2)))
    assert sparse[0].asnumpy().shape == (grid.num_edges, 2)
    assert test_utils.dallclose(sparse[0].asnumpy(), sparse2[0].asnumpy())
    assert test_utils.dallclose(sparse[1].asnumpy(), sparse2[1].asnumpy())


@pytest.mark.datatest
def test_create_auxiliary_orientation_coordinates(
    backend: gtx_typing.Backend, experiment: definitions.Experiment
) -> None:
    gm = grid_utils.get_grid_manager_from_identifier(
        experiment.grid,
        num_levels=1,
        keep_skip_values=True,
        allocator=backend,
    )
    grid = gm.grid
    coordinates = gm.coordinates

    cell_lat = coordinates[dims.CellDim]["lat"]
    cell_lon = coordinates[dims.CellDim]["lon"]
    edge_lat = coordinates[dims.EdgeDim]["lat"]
    edge_lon = coordinates[dims.EdgeDim]["lon"]
    lat_0, lon_0, lat_1, lon_1 = geometry.create_auxiliary_coordinate_arrays_for_orientation(
        grid=grid,
        cell_lat=cell_lat,
        cell_lon=cell_lon,
        edge_lat=edge_lat,
        edge_lon=edge_lon,
        allocator=backend,
    )
    connectivity = grid.get_connectivity(dims.E2C).asnumpy()
    has_boundary_edges = np.count_nonzero(connectivity == -1)
    if has_boundary_edges == 0:
        assert test_utils.dallclose(lat_0.asnumpy(), cell_lat.asnumpy()[connectivity[:, 0]])
        assert test_utils.dallclose(lat_1.asnumpy(), cell_lat.asnumpy()[connectivity[:, 1]])
        assert test_utils.dallclose(lon_0.asnumpy(), cell_lon.asnumpy()[connectivity[:, 0]])
        assert test_utils.dallclose(lon_1.asnumpy(), cell_lon.asnumpy()[connectivity[:, 1]])

    edge_coordinates_0 = np.nonzero(connectivity[:, 0] < 0)
    edge_coordinates_1 = np.nonzero(connectivity[:, 1] < 0)
    cell_coordinates_0 = np.nonzero(connectivity[:, 0] >= 0)
    cell_coordinates_1 = np.nonzero(connectivity[:, 1] >= 0)
    assert test_utils.dallclose(
        lat_0.asnumpy()[edge_coordinates_0], edge_lat.asnumpy()[edge_coordinates_0]
    )
    assert test_utils.dallclose(
        lat_0.asnumpy()[cell_coordinates_0], cell_lat.asnumpy()[connectivity[cell_coordinates_0, 0]]
    )

    assert test_utils.dallclose(
        lon_0.asnumpy()[edge_coordinates_0], edge_lon.asnumpy()[edge_coordinates_0]
    )
    assert test_utils.dallclose(
        lon_0.asnumpy()[cell_coordinates_0], cell_lon.asnumpy()[connectivity[cell_coordinates_0, 0]]
    )

    assert test_utils.dallclose(
        lat_1.asnumpy()[edge_coordinates_1], edge_lat.asnumpy()[edge_coordinates_1]
    )
    assert test_utils.dallclose(
        lat_1.asnumpy()[cell_coordinates_1], cell_lat.asnumpy()[connectivity[cell_coordinates_1, 1]]
    )
    assert test_utils.dallclose(
        lon_1.asnumpy()[edge_coordinates_1], edge_lon.asnumpy()[edge_coordinates_1]
    )
    assert test_utils.dallclose(
        lon_1.asnumpy()[cell_coordinates_1], cell_lon.asnumpy()[connectivity[cell_coordinates_1, 1]]
    )


@pytest.mark.datatest
@pytest.mark.parametrize(
    "attr_name", ["mean_edge_length", "mean_dual_edge_length", "mean_cell_area", "mean_dual_area"]
)
def test_geometry_mean_fields(
    backend: gtx_typing.Backend,
    grid_savepoint: sb.IconGridSavepoint,
    experiment: definitions.Experiment,
    attr_name: str,
) -> None:
    grid_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    value_ref = utils.GRID_REFERENCE_VALUES[experiment.grid.name][attr_name]
    value = grid_geometry.get(attr_name)
    assert value == pytest.approx(value_ref)


@pytest.mark.datatest
@pytest.mark.parametrize(
    "attr_name", ["mean_edge_length", "mean_dual_edge_length", "mean_cell_area", "mean_dual_area"]
)
def test_analytical_and_global_reduction_mean_fields_agree(
    backend: gtx_typing.Backend,
    # TODO(msimberg): This would in principle not need the full experiment, but
    # only the grid (description). Currently we go through the grid manager,
    # which does require e.g. num_levels from experiment. This should be cleaned
    # up when fixtures in https://github.com/C2SM/icon4py/issues/1134.
    experiment: definitions.Experiment,
    attr_name: str,
) -> None:
    analytical_config = dataclasses.replace(
        experiment.config,
        geometry=dataclasses.replace(experiment.config.geometry, use_analytical_means=True),
    )
    reduction_config = dataclasses.replace(
        experiment.config,
        geometry=dataclasses.replace(experiment.config.geometry, use_analytical_means=False),
    )
    analytical_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, analytical_config)
    reduction_geometry = grid_utils.get_grid_geometry(backend, experiment.grid, reduction_config)
    analytical_value = analytical_geometry.get(attr_name)
    reduction_value = reduction_geometry.get(attr_name)
    match experiment.grid.params.geometry_type:
        case icon_grid.GeometryType.TORUS:
            rtol = 1e-15
        case icon_grid.GeometryType.ICOSAHEDRON:
            rtol = 3e-2 if experiment.grid.limited_area else 2e-3
        case _ as geometry_type:
            raise ValueError(f"Unsupported geometry type '{geometry_type}'.")
    assert analytical_value == pytest.approx(reduction_value, rel=rtol)
