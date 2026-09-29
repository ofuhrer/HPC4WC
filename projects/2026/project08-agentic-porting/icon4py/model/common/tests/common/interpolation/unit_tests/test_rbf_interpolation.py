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

import numpy as np
import pytest

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.grid import geometry_attributes as geometry_attrs, horizontal as h_grid
from icon4py.model.common.grid.gridfile import GridFile
from icon4py.model.common.interpolation import rbf_interpolation as rbf
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    definitions,
    grid_utils as gridtest_utils,
    test_utils as test_helpers,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    interpolation_savepoint,
    process_props,
)


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.testing import serialbox

RBF_TOLERANCES = {
    dims.CellDim: {
        definitions.Experiments.EXCLAIM_APE.name: 3.1e-9,
        definitions.Experiments.MCH_CH_R04B09.name: 4e-2,
        definitions.Experiments.GAUSS3D.name: 1e-14,
    },
    dims.EdgeDim: {
        definitions.Experiments.EXCLAIM_APE.name: 8e-14,
        definitions.Experiments.MCH_CH_R04B09.name: 2e-9,
        definitions.Experiments.GAUSS3D.name: 0,
    },
    dims.VertexDim: {
        definitions.Experiments.EXCLAIM_APE.name: 3e-10,
        definitions.Experiments.MCH_CH_R04B09.name: 3e-3,
        definitions.Experiments.GAUSS3D.name: 1e-15,
    },
}


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_construct_rbf_matrix_offsets_tables_for_cells(
    experiment: definitions.Experiment,
    grid_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    grid_manager = gridtest_utils.get_grid_manager_from_identifier(
        experiment.grid, 1, True, model_backends.get_allocator(backend)
    )
    grid = grid_manager.grid
    offset_table = data_alloc.as_numpy(rbf.construct_rbf_matrix_offsets_tables_for_cells(grid))
    assert offset_table.shape == (
        grid.num_cells,
        rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.CELL],
    )
    assert np.max(offset_table) == grid.num_edges - 1

    offset_table_savepoint = data_alloc.as_numpy(grid_savepoint.c2e2c2e())
    assert offset_table.shape == offset_table_savepoint.shape

    # Savepoint neighbors before start index may not be populated correctly,
    # ignore them.
    start_index = grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    for i in range(start_index, offset_table.shape[0]):
        # Neighbors may not be in the same order. Ignore differences in order.
        assert (np.sort(offset_table[i]) == np.sort(offset_table_savepoint[i])).all()


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_construct_rbf_matrix_offsets_tables_for_edges(
    experiment: definitions.Experiment,
    grid_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    grid_manager = gridtest_utils.get_grid_manager_from_identifier(
        experiment.grid, 1, True, model_backends.get_allocator(backend)
    )
    grid = grid_manager.grid
    offset_table = data_alloc.as_numpy(rbf.construct_rbf_matrix_offsets_tables_for_edges(grid))
    assert offset_table.shape == (
        grid.num_edges,
        rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.EDGE],
    )
    assert np.max(offset_table) == grid.num_edges - 1

    offset_table_savepoint = data_alloc.as_numpy(grid_savepoint.e2c2e())
    assert offset_table.shape == offset_table_savepoint.shape

    start_index = grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    for i in range(start_index, offset_table.shape[0]):
        # Neighbors may not be in the same order. Ignore differences in order.
        assert (np.sort(offset_table[i]) == np.sort(offset_table_savepoint[i])).all()


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_construct_rbf_matrix_offsets_tables_for_vertices(
    experiment: definitions.Experiment,
    grid_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend | None,
) -> None:
    grid_manager = gridtest_utils.get_grid_manager_from_identifier(
        experiment.grid, 1, True, model_backends.get_allocator(backend)
    )
    grid = grid_manager.grid
    offset_table = data_alloc.as_numpy(rbf.construct_rbf_matrix_offsets_tables_for_vertices(grid))
    assert offset_table.shape == (
        grid.num_vertices,
        rbf.RBF_STENCIL_SIZE[rbf.RBFDimension.VERTEX],
    )
    assert np.max(offset_table) == grid.num_edges - 1

    offset_table_savepoint = data_alloc.as_numpy(grid_savepoint.v2e())
    assert offset_table.shape == offset_table_savepoint.shape

    start_index = grid.start_index(
        h_grid.domain(dims.VertexDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )

    for i in range(start_index, offset_table.shape[0]):
        # Make sure invalid neighbors are represented the same way.
        _, index = np.unique(offset_table_savepoint[i, :], return_index=True)
        offset_table_savepoint[i, max(index) + 1 :] = GridFile.INVALID_INDEX

        # Neighbors may not be in the same order. Ignore differences in order.
        assert (np.sort(offset_table[i]) == np.sort(offset_table_savepoint[i])).all()


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_rbf_interpolation_coeffs_cell(
    grid_savepoint: serialbox.IconGridSavepoint,
    interpolation_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend | None,
    experiment: definitions.Experiment,
) -> None:
    geometry = gridtest_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    grid = geometry.grid
    rbf_dim = rbf.RBFDimension.CELL

    horizontal_start = grid.start_index(
        h_grid.domain(dims.CellDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    horizontal_end = grid.end_index(h_grid.domain(dims.CellDim)(h_grid.Zone.LOCAL))
    assert horizontal_start < horizontal_end <= grid.num_cells

    geometry_type = (
        grid.grid_params.geometry_type
        if grid.grid_params.geometry_type
        else pytest.fail("geometry_type cannot be None")
    )

    rbf_vec_coeff_c1, rbf_vec_coeff_c2 = rbf.compute_rbf_interpolation_coeffs_cell(  # type: ignore[misc] # function returns two vars
        cell_center_lat=geometry.get(geometry_attrs.CELL_LAT).ndarray,
        cell_center_lon=geometry.get(geometry_attrs.CELL_LON).ndarray,
        cell_center_x=geometry.get(geometry_attrs.CELL_CENTER_X).ndarray,
        cell_center_y=geometry.get(geometry_attrs.CELL_CENTER_Y).ndarray,
        cell_center_z=geometry.get(geometry_attrs.CELL_CENTER_Z).ndarray,
        edge_center_x=geometry.get(geometry_attrs.EDGE_CENTER_X).ndarray,
        edge_center_y=geometry.get(geometry_attrs.EDGE_CENTER_Y).ndarray,
        edge_center_z=geometry.get(geometry_attrs.EDGE_CENTER_Z).ndarray,
        edge_normal_x=geometry.get(geometry_attrs.EDGE_NORMAL_X).ndarray,
        edge_normal_y=geometry.get(geometry_attrs.EDGE_NORMAL_Y).ndarray,
        edge_normal_z=geometry.get(geometry_attrs.EDGE_NORMAL_Z).ndarray,
        rbf_offset=rbf.construct_rbf_matrix_offsets_tables_for_cells(grid),
        rbf_kernel=rbf.DEFAULT_RBF_KERNEL[rbf_dim],
        geometry_type=geometry_type.value,
        scale_factor=rbf.compute_default_rbf_scale_cell(
            geometry_type.value,
            geometry.get_wpfloat(geometry_attrs.CHARACTERISTIC_LENGTH),
            geometry.get_wpfloat(geometry_attrs.MEAN_DUAL_EDGE_LENGTH),
        ),
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        domain_length=grid.grid_params.domain_length,  # type: ignore[arg-type] # test would fail if None
        domain_height=grid.grid_params.domain_height,  # type: ignore[arg-type] # test would fail if None
    )

    rbf_vec_coeff_c1_ref = interpolation_savepoint.rbf_vec_coeff_c1().asnumpy()
    rbf_vec_coeff_c2_ref = interpolation_savepoint.rbf_vec_coeff_c2().asnumpy()

    assert rbf_vec_coeff_c1.shape == rbf_vec_coeff_c1_ref.shape
    assert rbf_vec_coeff_c2.shape == rbf_vec_coeff_c2_ref.shape
    assert rbf_vec_coeff_c1_ref.shape == (
        grid.num_cells,
        rbf.RBF_STENCIL_SIZE[rbf_dim],
    )
    assert rbf_vec_coeff_c2_ref.shape == (
        grid.num_cells,
        rbf.RBF_STENCIL_SIZE[rbf_dim],
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_c1[horizontal_start:],
        rbf_vec_coeff_c1_ref[horizontal_start:],
        atol=RBF_TOLERANCES[dims.CellDim][experiment.name],
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_c2[horizontal_start:],
        rbf_vec_coeff_c2_ref[horizontal_start:],
        atol=RBF_TOLERANCES[dims.CellDim][experiment.name],
    )


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_rbf_interpolation_coeffs_vertex(
    grid_savepoint: serialbox.IconGridSavepoint,
    interpolation_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend | None,
    experiment: definitions.Experiment,
) -> None:
    geometry = gridtest_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    grid = geometry.grid
    rbf_dim = rbf.RBFDimension.VERTEX

    horizontal_start = grid.start_index(
        h_grid.domain(dims.VertexDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    horizontal_end = grid.end_index(h_grid.domain(dims.VertexDim)(h_grid.Zone.LOCAL))
    assert horizontal_start < horizontal_end <= grid.num_vertices

    geometry_type = (
        grid.grid_params.geometry_type
        if grid.grid_params.geometry_type
        else pytest.fail("geometry_type cannot be None")
    )

    rbf_vec_coeff_v1, rbf_vec_coeff_v2 = rbf.compute_rbf_interpolation_coeffs_vertex(
        vertex_lat=geometry.get(geometry_attrs.VERTEX_LAT).ndarray,
        vertex_lon=geometry.get(geometry_attrs.VERTEX_LON).ndarray,
        vertex_x=geometry.get(geometry_attrs.VERTEX_X).ndarray,
        vertex_y=geometry.get(geometry_attrs.VERTEX_Y).ndarray,
        vertex_z=geometry.get(geometry_attrs.VERTEX_Z).ndarray,
        edge_center_x=geometry.get(geometry_attrs.EDGE_CENTER_X).ndarray,
        edge_center_y=geometry.get(geometry_attrs.EDGE_CENTER_Y).ndarray,
        edge_center_z=geometry.get(geometry_attrs.EDGE_CENTER_Z).ndarray,
        edge_normal_x=geometry.get(geometry_attrs.EDGE_NORMAL_X).ndarray,
        edge_normal_y=geometry.get(geometry_attrs.EDGE_NORMAL_Y).ndarray,
        edge_normal_z=geometry.get(geometry_attrs.EDGE_NORMAL_Z).ndarray,
        rbf_offset=rbf.construct_rbf_matrix_offsets_tables_for_vertices(grid),
        rbf_kernel=rbf.DEFAULT_RBF_KERNEL[rbf_dim],
        geometry_type=geometry_type.value,
        scale_factor=rbf.compute_default_rbf_scale_vertex(
            geometry_type.value,
            geometry.get_wpfloat(geometry_attrs.CHARACTERISTIC_LENGTH),
            geometry.get_wpfloat(geometry_attrs.MEAN_DUAL_EDGE_LENGTH),
        ),
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        domain_length=grid.grid_params.domain_length,  # type: ignore[arg-type] # test would fail if None
        domain_height=grid.grid_params.domain_height,  # type: ignore[arg-type] # test would fail if None
    )

    rbf_vec_coeff_v1_ref = interpolation_savepoint.rbf_vec_coeff_v1()
    rbf_vec_coeff_v2_ref = interpolation_savepoint.rbf_vec_coeff_v2()

    assert rbf_vec_coeff_v1.shape == rbf_vec_coeff_v1_ref.shape
    assert rbf_vec_coeff_v2.shape == rbf_vec_coeff_v2_ref.shape
    assert rbf_vec_coeff_v1_ref.shape == (
        grid.num_vertices,
        rbf.RBF_STENCIL_SIZE[rbf_dim],
    )
    assert rbf_vec_coeff_v2_ref.shape == (
        grid.num_vertices,
        rbf.RBF_STENCIL_SIZE[rbf_dim],
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_v1[horizontal_start:],
        rbf_vec_coeff_v1_ref.asnumpy()[horizontal_start:],
        atol=RBF_TOLERANCES[dims.VertexDim][experiment.name],
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_v2[horizontal_start:],
        rbf_vec_coeff_v2_ref.asnumpy()[horizontal_start:],
        atol=RBF_TOLERANCES[dims.VertexDim][experiment.name],
    )


@pytest.mark.level("unit")
@pytest.mark.datatest
def test_rbf_interpolation_coeffs_edge(
    grid_savepoint: serialbox.IconGridSavepoint,
    interpolation_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend | None,
    experiment: definitions.Experiment,
) -> None:
    geometry = gridtest_utils.get_grid_geometry(backend, experiment.grid, experiment.config)
    grid = geometry.grid
    rbf_dim = rbf.RBFDimension.EDGE

    horizontal_start = grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    horizontal_end = grid.end_index(h_grid.domain(dims.EdgeDim)(h_grid.Zone.LOCAL))
    assert horizontal_start < horizontal_end <= grid.num_edges

    geometry_type = (
        grid.grid_params.geometry_type
        if grid.grid_params.geometry_type
        else pytest.fail("geometry_type cannot be None")
    )

    rbf_vec_coeff_e = rbf.compute_rbf_interpolation_coeffs_edge(
        edge_lat=geometry.get(geometry_attrs.EDGE_LAT).ndarray,
        edge_lon=geometry.get(geometry_attrs.EDGE_LON).ndarray,
        edge_center_x=geometry.get(geometry_attrs.EDGE_CENTER_X).ndarray,
        edge_center_y=geometry.get(geometry_attrs.EDGE_CENTER_Y).ndarray,
        edge_center_z=geometry.get(geometry_attrs.EDGE_CENTER_Z).ndarray,
        edge_normal_x=geometry.get(geometry_attrs.EDGE_NORMAL_X).ndarray,
        edge_normal_y=geometry.get(geometry_attrs.EDGE_NORMAL_Y).ndarray,
        edge_normal_z=geometry.get(geometry_attrs.EDGE_NORMAL_Z).ndarray,
        edge_dual_normal_u=geometry.get(geometry_attrs.EDGE_DUAL_U).ndarray,
        edge_dual_normal_v=geometry.get(geometry_attrs.EDGE_DUAL_V).ndarray,
        # NOTE: Neighbors are not in the same order. Use savepoint to make sure
        # order of coefficients computed by icon4py matches order of
        # coefficients in savepoint.
        rbf_offset=grid_savepoint.e2c2e(),
        rbf_kernel=rbf.DEFAULT_RBF_KERNEL[rbf_dim],
        geometry_type=geometry_type.value,
        scale_factor=rbf.compute_default_rbf_scale_edge(
            geometry_type.value,
            geometry.get_wpfloat(geometry_attrs.CHARACTERISTIC_LENGTH),
            geometry.get_wpfloat(geometry_attrs.MEAN_DUAL_EDGE_LENGTH),
        ),
        horizontal_start=horizontal_start,
        horizontal_end=horizontal_end,
        domain_length=grid.grid_params.domain_length,  # type: ignore[arg-type] # test would fail if None
        domain_height=grid.grid_params.domain_height,  # type: ignore[arg-type] # test would fail if None
    )

    rbf_vec_coeff_e_ref = interpolation_savepoint.rbf_vec_coeff_e()

    assert rbf_vec_coeff_e.shape == rbf_vec_coeff_e_ref.shape
    assert rbf_vec_coeff_e_ref.shape == (
        grid.num_edges,
        rbf.RBF_STENCIL_SIZE[rbf_dim],
    )
    assert test_helpers.dallclose(
        rbf_vec_coeff_e[horizontal_start:],
        rbf_vec_coeff_e_ref.asnumpy()[horizontal_start:],
        atol=RBF_TOLERANCES[dims.EdgeDim][experiment.name],
    )
