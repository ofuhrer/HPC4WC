# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

import logging
import typing
from collections.abc import Iterator

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np
import pytest

import icon4py.model.common.grid.gridfile
from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import decomposer as decomp, definitions as decomp_defs
from icon4py.model.common.grid import (
    base as base_grid,
    grid_manager as gm,
    grid_refinement as refin,
    gridfile,
    horizontal as h_grid,
    icon,
    vertical as v_grid,
)
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions,
    definitions as test_defs,
    grid_utils,
    test_utils,
)


if typing.TYPE_CHECKING:
    import netCDF4

    from icon4py.model.testing import serialbox

try:
    import netCDF4
except ImportError:
    pytest.skip("optional netcdf dependency not installed", allow_module_level=True)


from icon4py.model.testing.fixtures import (
    backend,
    backend_like,
    cpu_allocator,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_description,
    grid_savepoint,
    process_props,
)

from ...decomposition import utils as decomp_utils
from .. import utils


# TODO @magdalena add test cases for hexagon vertices v2e2v
# v2e2v: grid,???


# v2e: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_v2e(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    seralized_v2e = data_alloc.as_numpy(grid_savepoint.v2e())
    # there are vertices at the boundary of a local domain or at a pentagon point that have less than
    # 6 neighbors hence there are "Missing values" in the grid file
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(seralized_v2e)
    v2e_table = grid.get_connectivity("V2E").asnumpy()
    # Torus grids have no pentagon points and no boundaries hence no invalid
    # indexes (while REGIONAL and GLOBAL grids can have)
    assert (
        not has_invalid_index(v2e_table)
        if experiment.grid.params.geometry_type == icon.GeometryType.TORUS
        else has_invalid_index(v2e_table)
    )
    _reset_invalid_index(seralized_v2e)
    assert np.allclose(v2e_table, seralized_v2e)


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize("dim", dims.horizontal_dims())
def test_grid_manager_refin_ctrl(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    dim: gtx.Dimension,
    backend: gtx_typing.Backend,
) -> None:
    refin_ctrl = utils.run_grid_manager(
        experiment.grid, keep_skip_values=True, backend=backend
    ).grid.refinement_control
    refin_ctrl_serialized = grid_savepoint.refin_ctrl(dim)
    assert np.all(
        refin_ctrl_serialized.ndarray
        == refin.convert_to_non_nested_refinement_values(refin_ctrl[dim].ndarray, dim)
    )


# v2c: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_v2c(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    serialized_v2c = data_alloc.as_numpy(grid_savepoint.v2c())
    v2c_table = grid.get_connectivity("V2C").asnumpy()
    # there are vertices that have less than 6 neighboring cells: either pentagon points or
    # vertices at the boundary of the domain for a limited area mode
    # hence in the grid file there are "missing values"
    # they get substituted by the "last valid index" in preprocessing step in icon.
    assert not has_invalid_index(serialized_v2c)
    # Torus grids have no pentagon points and no boundaries hence no invalid
    # indexes (while REGIONAL and GLOBAL grids can have)
    assert (
        not has_invalid_index(v2c_table)
        if experiment.grid.params.geometry_type == icon.GeometryType.TORUS
        else has_invalid_index(v2c_table)
    )
    _reset_invalid_index(serialized_v2c)
    assert np.allclose(v2c_table, serialized_v2c)


def _reset_invalid_index(index_array: np.ndarray) -> None:
    """
    Revert changes from mo_model_domimp_patches.

    Helper function to revert mo_model_domimp_patches.f90: move_dummies_to_end_idxblk.
    see:
    # ! Checks for the pentagon case and moves dummy cells to end.
    #  ! The dummy entry is either set to 0 or duplicated from the last one
    #  SUBROUTINE move_dummies_to_end(array, array_size, max_connectivity, duplicate)

    After reading the grid file ICON moves all invalid indices (neighbors not existing in the
    grid file) to the end of the neighbor list and replaces them with the "last valid neighbor index"
    it is up to the user then to ensure that any coefficients in neighbor some multiplied with
    these values are zero in order to "remove" them again from the sum.

    For testing we resubstitute those to the GridFile.INVALID_INDEX
    Args:
        index_array: the array where values the invalid values have to be reset

    Returns: an array where the spurious "last valid index" are replaced by GridFile.INVALID_INDEX

    """
    for i in range(0, index_array.shape[0]):
        _, index = np.unique(index_array[i, :], return_index=True)
        index_array[i, max(index) + 1 :] = gridfile.GridFile.INVALID_INDEX


# e2v: exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_e2v(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid

    serialized_e2v = data_alloc.as_numpy(grid_savepoint.e2v())
    e2v_table = grid.get_connectivity("E2V").asnumpy()
    # all vertices in the system have to neighboring edges, there no edges that point nowhere
    # hence this connectivity has no "missing values" in the grid file
    assert not has_invalid_index(serialized_e2v)
    assert not has_invalid_index(e2v_table)
    assert np.allclose(e2v_table, serialized_e2v)


def has_invalid_index(ar: np.ndarray) -> np.bool_:
    return np.any(np.isin(ar, gridfile.GridFile.INVALID_INDEX))


# e2c : exists in serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_e2c(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid

    serialized_e2c = data_alloc.as_numpy(grid_savepoint.e2c())
    e2c_table = grid.get_connectivity("E2C").asnumpy()
    assert has_invalid_index(serialized_e2c) == grid.limited_area
    assert has_invalid_index(e2c_table) == grid.limited_area
    assert np.allclose(e2c_table, serialized_e2c)


# c2e: serial, simple, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_c2e(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid

    serialized_c2e = data_alloc.as_numpy(grid_savepoint.c2e())
    c2e_table = grid.get_connectivity("C2E").asnumpy()
    # no cells with less than 3 neighboring edges exist, otherwise the cell is not there in the
    # first place
    # hence there are no "missing values" in the grid file
    assert not has_invalid_index(serialized_c2e)
    assert not has_invalid_index(c2e_table)
    assert np.allclose(c2e_table, serialized_c2e)


# c2e2c: exists in  serial, simple_mesh, grid
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_c2e2c(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    assert np.allclose(
        grid.get_connectivity("C2E2C").asnumpy(),
        data_alloc.as_numpy(grid_savepoint.c2e2c()),
    )


@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_c2e2cO(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    serialized_grid = grid_savepoint.construct_icon_grid(backend=backend)
    assert np.allclose(
        grid.get_connectivity("C2E2CO").asnumpy(),
        serialized_grid.get_connectivity("C2E2CO").asnumpy(),
    )


# e2c2e (e2c2eo) - diamond: exists in serial, simple_mesh
@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_e2c2e(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    serialized_grid = grid_savepoint.construct_icon_grid(backend=backend)
    serialized_e2c2e = serialized_grid.get_connectivity("E2C2E").asnumpy()
    serialized_e2c2eO = serialized_grid.get_connectivity("E2C2EO").asnumpy()
    assert has_invalid_index(serialized_e2c2e) == grid.limited_area

    e2c2e_table = grid.get_connectivity("E2C2E").asnumpy()
    e2c2eO_table = grid.get_connectivity("E2C2EO").asnumpy()
    assert has_invalid_index(e2c2e_table) == grid.limited_area
    # ICON calculates diamond edges only from rl_start = 2 (lateral_boundary(dims.EdgeDim) + 1 for
    # boundaries all values are INVALID even though the half diamond exists (see mo_model_domimp_setup.f90 ll 163ff.)
    start_index = grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_3)
    )
    assert np.allclose(e2c2e_table[start_index:, :], serialized_e2c2e[start_index:, :])
    assert np.allclose(e2c2eO_table[start_index:, :], serialized_e2c2eO[start_index:, :])


@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_e2c2v(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    serialized_ref = data_alloc.as_numpy(grid_savepoint.e2c2v())
    # the "far" (adjacent to edge normal ) is not always there, because ICON only calculates those starting from
    #   (lateral_boundary(dims.EdgeDim) + 1) to end(dims.EdgeDim)  (see mo_intp_coeffs.f90) and only for owned cells
    table = grid.get_connectivity("E2C2V").asnumpy()
    start_index = grid.start_index(
        h_grid.domain(dims.EdgeDim)(h_grid.Zone.LATERAL_BOUNDARY_LEVEL_2)
    )
    # e2c2e in ICON (quad_idx) has a different neighbor ordering than the e2c2e constructed in grid_manager.py
    assert_up_to_order(table, serialized_ref, start_index)
    assert np.allclose(table[:, :2], grid.get_connectivity("E2V").asnumpy())


@pytest.mark.datatest
@pytest.mark.with_netcdf
def test_grid_manager_eval_c2v(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    c2v = grid.get_connectivity("C2V").asnumpy()
    assert np.allclose(c2v, data_alloc.as_numpy(grid_savepoint.c2v()))


@pytest.mark.with_netcdf
def test_grid_manager_grid_size(
    backend: gtx_typing.Backend, grid_description: definitions.GridDescription
) -> None:
    grid = utils.run_grid_manager(grid_description, keep_skip_values=True, backend=backend).grid
    ref = utils.GRID_REFERENCE_VALUES[grid_description.name]
    assert ref["num_cells"] == grid.size[dims.CellDim]
    assert ref["num_edges"] == grid.size[dims.EdgeDim]
    assert ref["num_vertices"] == grid.size[dims.VertexDim]


def assert_up_to_order(
    table: np.ndarray,
    reference_table: np.ndarray,
    start_index: gtx.int32 = 0,
) -> None:
    assert table.shape == reference_table.shape, "arrays need to have the same shape"
    reduced_table = table[start_index:, :]
    reduced_reference = reference_table[start_index:, :]
    for n in range(reduced_table.shape[0]):
        assert np.all(np.isin(reduced_table[n, :], reduced_reference[n, :])), (
            f"values in row {n + start_index} are not equal: {reduced_table[n, :]} vs ref= {reduced_reference[n, :]}."
        )


@pytest.mark.with_netcdf
def test_gridmanager_given_file_not_found_then_abort(
    cpu_allocator: gtx_typing.Allocator,
) -> None:
    fname = "./unknown_grid.nc"
    with pytest.raises(FileNotFoundError):
        manager = gm.GridManager(
            grid_file=fname,
            config=v_grid.VerticalGridConfig(num_levels=80),
            offset_transformation=icon4py.model.common.grid.gridfile.NoTransformation(),
        )
        manager(allocator=cpu_allocator, keep_skip_values=True)


@pytest.mark.parametrize("size", [100, 1500, 20000])
@pytest.mark.with_netcdf
def test_gt4py_transform_offset_by_1_where_valid(size: int) -> None:
    trafo = gridfile.ToZeroBasedIndexTransformation()
    rng = np.random.default_rng()
    input_field = rng.integers(-1, size, size)
    offset = trafo(input_field)
    expected = np.where(input_field >= 0, -1, 0)
    assert np.allclose(expected, offset)


@pytest.mark.parametrize(
    "grid_description, expected_subdivision",
    [
        (
            definitions.Grids.R02B04_GLOBAL,
            icon.GridSubdivision(root=2, level=4),
        ),
        (
            definitions.Grids.MCH_CH_R04B09_DSL,
            icon.GridSubdivision(root=4, level=9),
        ),
    ],
)
def test_grid_manager_grid_level_and_root(
    grid_description: definitions.GridDescription,
    expected_subdivision: icon.GridSubdivision,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(grid_description, keep_skip_values=True, backend=backend).grid
    assert expected_subdivision == grid.grid_params.subdivision


@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize(
    "experiment_description",
    [definitions.Experiments.JW],
)
def test_grid_manager_eval_c2e2c2e(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid
    serialized_grid = grid_savepoint.construct_icon_grid(backend=backend)
    assert np.allclose(
        grid.get_connectivity("C2E2C2E").asnumpy(),
        serialized_grid.get_connectivity("C2E2C2E").asnumpy(),
    )
    assert grid.get_connectivity("C2E2C2E").asnumpy().shape == (grid.num_cells, 9)


# TODO (halungge): check EXCOAIM APE with new serialized data ( standard grid, start_idx/end_idx arrays
@pytest.mark.datatest
@pytest.mark.with_netcdf
@pytest.mark.parametrize("dim", dims.horizontal_dims())
def test_grid_manager_start_end_index_compare_with_serialized_data(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    dim: gtx.Dimension,
    backend: gtx_typing.Backend,
) -> None:
    serialized_grid = grid_savepoint.construct_icon_grid(backend=backend)
    grid = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend).grid

    for domain in h_grid.get_domains_for_dim(dim):
        if not (
            experiment.description == definitions.Experiments.EXCLAIM_APE
            and domain.dim == dims.EdgeDim
        ):
            # serialized start indices for EdgeDim are all zero
            assert grid.start_index(domain) == serialized_grid.start_index(domain), (
                f"start index wrong for domain {domain}"
            )
        if not grid.limited_area and domain.zone in [h_grid.Zone.END, h_grid.Zone.INTERIOR]:
            assert grid.end_index(domain) == grid.size[domain.dim]
        else:
            assert grid.end_index(domain) == serialized_grid.end_index(domain), (
                f"end index wrong for domain {domain}"
            )


@pytest.mark.datatest
def test_read_geometry_fields(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    cell_area = manager.geometry_fields[gridfile.GeometryName.CELL_AREA]
    tangent_orientation = manager.geometry_fields[gridfile.GeometryName.TANGENT_ORIENTATION]

    assert test_utils.dallclose(cell_area.asnumpy(), grid_savepoint.cell_areas().asnumpy())
    assert test_utils.dallclose(
        tangent_orientation.asnumpy(), grid_savepoint.tangent_orientation().asnumpy()
    )


@pytest.mark.datatest
@pytest.mark.parametrize("dim", dims.horizontal_dims())
def test_coordinates(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    dim: gtx.Dimension,
    backend: gtx_typing.Backend,
) -> None:
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    lat = manager.coordinates[dim]["lat"]
    lon = manager.coordinates[dim]["lon"]
    assert test_utils.dallclose(lat.asnumpy(), grid_savepoint.lat(dim).asnumpy())
    assert test_utils.dallclose(lon.asnumpy(), grid_savepoint.lon(dim).asnumpy())


@pytest.mark.datatest
def test_tangent_orientation(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    expected = data_alloc.as_numpy(grid_savepoint.tangent_orientation())
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    assert test_utils.dallclose(
        manager.geometry_fields[gridfile.GeometryName.TANGENT_ORIENTATION].asnumpy(),
        expected,
    )


@pytest.mark.datatest
def test_edge_orientation_on_vertex(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    expected = data_alloc.as_numpy(grid_savepoint.vertex_edge_orientation())
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    assert test_utils.dallclose(
        manager.geometry_fields[gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX].asnumpy(),
        expected,
    )


@pytest.mark.datatest
def test_dual_area(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    expected = grid_savepoint.vertex_dual_area()
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    assert test_utils.dallclose(
        manager.geometry_fields[gridfile.GeometryName.DUAL_AREA].asnumpy(), expected.asnumpy()
    )


@pytest.mark.datatest
def test_edge_cell_distance(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    expected = grid_savepoint.edge_cell_length()
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)

    assert test_utils.dallclose(
        manager.geometry_fields[gridfile.GeometryName.EDGE_CELL_DISTANCE].asnumpy(),
        expected.asnumpy(),
        equal_nan=True,
    )


@pytest.mark.datatest
def test_cell_normal_orientation(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    expected = data_alloc.as_numpy(grid_savepoint.edge_orientation())
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)
    assert test_utils.dallclose(
        manager.geometry_fields[gridfile.GeometryName.CELL_NORMAL_ORIENTATION].asnumpy(),
        expected,
    )


@pytest.mark.datatest
def test_edge_vertex_distance(
    grid_savepoint: serialbox.IconGridSavepoint,
    experiment: definitions.Experiment,
    backend: gtx_typing.Backend,
) -> None:
    expected = data_alloc.as_numpy(grid_savepoint.edge_vert_length())
    manager = utils.run_grid_manager(experiment.grid, keep_skip_values=True, backend=backend)

    assert test_utils.dallclose(
        manager.geometry_fields[gridfile.GeometryName.EDGE_VERTEX_DISTANCE].asnumpy(),
        expected,
        equal_nan=True,
    )


@pytest.mark.parametrize(
    "grid_description, expected",
    [
        (definitions.Grids.MCH_CH_R04B09_DSL, True),
        (definitions.Grids.R02B04_GLOBAL, False),
    ],
)
def test_limited_area_on_grid(
    grid_description: definitions.GridDescription, expected: bool
) -> None:
    grid = utils.run_grid_manager(grid_description, keep_skip_values=True, backend=None).grid
    assert expected == grid.limited_area


@pytest.mark.datatest
@pytest.mark.parametrize("dim", dims.horizontal_dims())
def test_decomposition_info_single_rank(
    dim: gtx.Dimension,
    experiment: definitions.Experiment,
    grid_savepoint: serialbox.IconGridSavepoint,
    backend: gtx_typing.Backend,
) -> None:
    expected = grid_savepoint.construct_decomposition_info()
    grid_file = experiment.grid
    gm = utils.run_grid_manager(grid_file, keep_skip_values=True, backend=backend)
    result = gm.decomposition_info
    assert (result.local_index(dim) == expected.local_index(dim)).all()
    assert (result.global_index(dim) == expected.global_index(dim)).all()
    assert (result.owner_mask(dim) == expected.owner_mask(dim)).all()
    assert (result.halo_levels(dim) == expected.halo_levels(dim)).all()


@pytest.mark.parametrize("rank", (0, 1, 2, 3), ids=lambda rank: f"rank{rank}")
@pytest.mark.parametrize(
    "field_offset",
    [
        dims.C2V,
        dims.E2V,
        dims.V2C,
        dims.E2C,
        dims.C2E,
        dims.V2E,
        dims.C2E2C,
        dims.V2E2V,
        dims.C2E2CO,
        dims.C2E2C2E,
        dims.C2E2C2E2C,
        dims.E2C2V,
        dims.E2C2E,
        dims.E2C2EO,
    ],
    ids=lambda offset: offset.value,
)
def test_local_connectivity(
    rank: int,
    caplog: Iterator,
    field_offset: gtx.FieldOffset,
    backend_like: model_backends.BackendLike,
) -> None:
    process_props = decomp_utils.DummyProps(rank=rank)
    caplog.set_level(logging.INFO)
    partitioner = decomp.MetisDecomposer()
    allocator = model_backends.get_allocator(backend_like)
    file = dt_utils.get_grid_filepath(test_defs.Grids.R02B04_GLOBAL)
    manager = gm.GridManager(config=v_grid.VerticalGridConfig(num_levels=10), grid_file=file)
    manager(
        decomposer=partitioner,
        allocator=allocator,
        keep_skip_values=True,
        process_props=process_props,
    )
    grid = manager.grid

    decomposition_info = manager.decomposition_info
    connectivity = grid.get_connectivity(field_offset).asnumpy()

    assert (
        connectivity.shape[0]
        == decomposition_info.global_index(
            field_offset.target[0], decomp_defs.DecompositionInfo.EntryType.ALL
        ).size
    ), "connectivity shapes do not match"

    # all neighbor indices are valid local indices
    max_local_index = np.max(
        decomposition_info.local_index(
            field_offset.source, decomp_defs.DecompositionInfo.EntryType.ALL
        )
    )
    assert np.max(connectivity) == max_local_index, (
        f"max value in the connectivity is {np.max(connectivity)} is larger than the local patch size {max_local_index}"
    )
    # - outer halo entries have SKIP_VALUE neighbors (depends on offsets)
    neighbor_dim = field_offset.target[1]  # type: ignore [misc]
    dim = field_offset.target[0]
    last_halo_level = (
        decomp_defs.DecompositionFlag.THIRD_HALO_LEVEL
        if dim == dims.EdgeDim
        else decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL
    )
    level_index = np.nonzero(
        data_alloc.as_numpy(decomposition_info.halo_levels(dim)) == last_halo_level.value
    )
    if (
        neighbor_dim in icon.CONNECTIVITIES_ON_BOUNDARIES
        or neighbor_dim in icon.CONNECTIVITIES_ON_PENTAGONS
    ):
        assert np.count_nonzero(
            (connectivity[level_index] == gridfile.GridFile.INVALID_INDEX) > 0
        ), f"missing invalid index in {dim} - offset {field_offset}"
    else:
        assert np.count_nonzero(
            (connectivity[level_index] == gridfile.GridFile.INVALID_INDEX) == 0
        ), f"have invalid index in {dim} - offset {field_offset} when none expected"


@pytest.mark.parametrize("ranks", (2, 3, 4))
def test_decomposition_size(
    ranks: int,
    grid_description: test_defs.GridDescription,
) -> None:
    if grid_description.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    decomposer = decomp.MetisDecomposer()
    file = dt_utils.get_grid_filepath(grid_description)
    with gridfile.GridFile(str(file), gridfile.ToZeroBasedIndexTransformation()) as parser:
        partitions = decomposer(parser.int_variable(gridfile.ConnectivityName.C2E2C), ranks)
        sizes = [np.count_nonzero(partitions == r) for r in range(ranks)]
        # Verify that sizes are close to each other. This is not a hard
        # requirement, but simply a sanity check to make sure that partitions
        # are relatively balanced.
        assert max(sizes) - min(sizes) <= 2
