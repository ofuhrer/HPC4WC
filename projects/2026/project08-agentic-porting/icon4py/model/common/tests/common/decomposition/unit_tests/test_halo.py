# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import numpy as np
import pytest
from gt4py.next import common as gtx_common

from icon4py.model.common import dimension as dims, exceptions, model_backends
from icon4py.model.common.decomposition import decomposer as decomp, definitions, halo
from icon4py.model.common.grid import base as base_grid, simple
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import test_utils

from ...fixtures import backend_like, process_props
from .. import utils
from ..fixtures import simple_neighbor_tables
from ..utils import dummy_four_ranks
from .test_definitions import get_neighbor_tables_for_simple_grid, offsets


@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_halo_constructor_owned_cells(rank, simple_neighbor_tables, backend_like):
    process_props = utils.DummyProps(rank=rank)
    allocator = model_backends.get_allocator(backend_like)
    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        process_props=process_props,
        allocator=allocator,
    )
    xp = data_alloc.import_array_ns(allocator)
    my_owned_cells = data_alloc.as_numpy(
        halo_generator.owned_cells(xp.asarray(utils.SIMPLE_DISTRIBUTION))
    )

    print(f"rank {process_props.rank} owns {my_owned_cells} ")
    assert my_owned_cells.size == len(utils._CELL_OWN[process_props.rank])
    assert np.setdiff1d(my_owned_cells, utils._CELL_OWN[process_props.rank]).size == 0


@pytest.mark.parametrize("dim", [dims.CellDim, dims.VertexDim, dims.EdgeDim])
@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_halo_constructor_decomposition_info_global_indices(rank, simple_neighbor_tables, dim):
    process_props = utils.dummy_four_ranks(rank=rank)
    if process_props.comm_size != 4:
        pytest.skip(
            f"This test requires exactly 4 MPI ranks, current run has {process_props.comm_size}"
        )

    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        process_props=process_props,
    )

    decomp_info = halo_generator(utils.SIMPLE_DISTRIBUTION)
    my_halo = decomp_info.global_index(dim, definitions.DecompositionInfo.EntryType.HALO)
    print(f"rank = {process_props.rank}: has halo {dim} : {my_halo}")
    expected = len(utils.HALO[dim][process_props.rank])
    assert my_halo.size == expected, (
        f" rank = {process_props.rank}: total halo size does not match for dim {dim}- expected {expected} bot was {my_halo.size}"
    )
    assert (
        missing := np.setdiff1d(
            my_halo, utils.HALO[dim][process_props.rank], assume_unique=True
        ).size
        == 0
    ), f"missing halo elements are {missing}"
    my_owned = decomp_info.global_index(dim, definitions.DecompositionInfo.EntryType.OWNED)
    print(f"rank {process_props.rank} owns {dim} : {my_owned} ")
    utils.assert_same_entries(dim, my_owned, utils.OWNED, process_props.rank)


@pytest.mark.parametrize("dim", [dims.EdgeDim])
@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_halo_constructor_decomposition_info_halo_levels(rank, dim, simple_neighbor_tables):
    process_props = utils.DummyProps(rank=rank)
    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        process_props=process_props,
    )
    decomp_info = halo_generator(utils.SIMPLE_DISTRIBUTION)
    my_halo_levels = decomp_info.halo_levels(dim)
    print(f"{dim.value}: rank {process_props.rank} has halo levels {my_halo_levels} ")
    assert np.all(my_halo_levels != definitions.DecompositionFlag.UNDEFINED), (
        "All indices should have a defined DecompositionFlag"
    )

    assert np.nonzero(my_halo_levels == definitions.DecompositionFlag.OWNED)[0].size == len(
        utils.OWNED[dim][process_props.rank]
    )
    owned_local_indices = decomp_info.local_index(
        dim, definitions.DecompositionInfo.EntryType.OWNED
    )
    assert np.all(my_halo_levels[owned_local_indices] == definitions.DecompositionFlag.OWNED), (
        "owned local indices should have DecompositionFlag.OWNED"
    )
    first_halo_level_local_index = np.nonzero(
        my_halo_levels == definitions.DecompositionFlag.FIRST_HALO_LEVEL
    )[0]
    first_halo_level_global_index = decomp_info.global_index(
        dim, definitions.DecompositionInfo.EntryType.ALL
    )[first_halo_level_local_index]
    utils.assert_same_entries(
        dim, first_halo_level_global_index, utils.FIRST_HALO_LINE, process_props.rank
    )
    second_halo_level_local_index = np.nonzero(
        my_halo_levels == definitions.DecompositionFlag.SECOND_HALO_LEVEL
    )[0]
    second_halo_level_global_index = decomp_info.global_index(
        dim, definitions.DecompositionInfo.EntryType.ALL
    )[second_halo_level_local_index]
    utils.assert_same_entries(
        dim, second_halo_level_global_index, utils.SECOND_HALO_LINE, process_props.rank
    )
    third_halo_level_index = np.nonzero(
        my_halo_levels == definitions.DecompositionFlag.THIRD_HALO_LEVEL
    )[0]
    third_halo_level_global_index = decomp_info.global_index(
        dim, definitions.DecompositionInfo.EntryType.ALL
    )[third_halo_level_index]
    utils.assert_same_entries(
        dim, third_halo_level_global_index, utils.THIRD_HALO_LINE, process_props.rank
    )


def test_no_halo():
    grid_size = base_grid.HorizontalGridSize(num_cells=9, num_edges=14, num_vertices=6)
    halo_generator = halo.NoHalos(horizontal_size=grid_size, allocator=None)
    decomposition = decomp.SingleNodeDecomposer()
    decomposition_info = halo_generator(decomposition(np.arange(grid_size.num_cells), 1))
    # cells
    test_utils.assert_dallclose(
        np.arange(grid_size.num_cells), decomposition_info.global_index(dims.CellDim)
    )
    assert np.all(decomposition_info.owner_mask(dims.CellDim))
    assert np.all(
        decomposition_info.halo_levels(dims.CellDim) == definitions.DecompositionFlag.OWNED
    )
    # edges
    test_utils.assert_dallclose(
        np.arange(grid_size.num_edges), decomposition_info.global_index(dims.EdgeDim)
    )
    assert np.all(
        decomposition_info.halo_levels(dims.EdgeDim) == definitions.DecompositionFlag.OWNED
    )
    assert np.all(decomposition_info.owner_mask(dims.EdgeDim))
    # vertices
    test_utils.assert_dallclose(
        np.arange(grid_size.num_vertices),
        decomposition_info.global_index(dims.VertexDim),
    )
    assert np.all(
        decomposition_info.halo_levels(dims.VertexDim) == definitions.DecompositionFlag.OWNED
    )
    assert np.all(decomposition_info.owner_mask(dims.VertexDim))


def test_halo_constructor_validate_rank_mapping_wrong_shape(simple_neighbor_tables):
    process_props = utils.DummyProps(rank=2)
    num_cells = simple_neighbor_tables["C2E2C"].shape[0]
    with pytest.raises(exceptions.ValidationError) as e:
        halo_generator = halo.IconLikeHaloConstructor(
            connectivities=simple_neighbor_tables,
            process_props=process_props,
        )
        halo_generator(np.zeros((num_cells, 3), dtype=int))
    assert f"should have shape ({num_cells},)" in e.value.args[0]


@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_halo_constructor_validate_number_of_node_mismatch(rank, simple_neighbor_tables):
    process_props = utils.DummyProps(rank=rank)
    num_cells = simple_neighbor_tables["C2E2C"].shape[0]
    distribution = np.full(num_cells, process_props.comm_size + 1, dtype=int)
    with pytest.raises(expected_exception=exceptions.ValidationError) as e:
        halo_generator = halo.IconLikeHaloConstructor(
            connectivities=simple_neighbor_tables,
            process_props=process_props,
        )
        halo_generator(distribution)
    assert "The distribution assumes more nodes than the current run" in e.value.args[0]


@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_owned_halo_mask_contiguous(rank):
    simple_neighbor_tables = get_neighbor_tables_for_simple_grid()
    process_props = dummy_four_ranks(rank)
    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        process_props=process_props,
    )
    decomp_info = halo_generator(utils.SIMPLE_DISTRIBUTION)

    for dim in dims.horizontal_dims():
        owner_mask = decomp_info.owner_mask(dim)
        owned_indices = np.nonzero(owner_mask)[0]
        # NOTE: These assumptions may change once limited area grids are
        # supported for icon4py domain decomposition.
        assert test_utils.is_sorted(decomp_info.halo_levels(dim)), (
            f"Halo levels for {dim} should be sorted, but are {decomp_info.halo_levels(dim)}"
        )
        assert (owned_indices == np.arange(owned_indices.size)).all(), (
            f"Owned indices for {dim} should be contiguous and start at 0, but are {owned_indices=}"
        )


@pytest.mark.parametrize("offset", offsets)
@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_global_to_local_index(offset, rank):
    grid = simple.simple_grid()
    neighbor_tables = {
        k: v.ndarray for k, v in grid.connectivities.items() if gtx_common.is_neighbor_table(v)
    }
    process_props = dummy_four_ranks(rank)
    halo_constructor = halo.IconLikeHaloConstructor(process_props, neighbor_tables)
    decomposition_info = halo_constructor(utils.SIMPLE_DISTRIBUTION)
    source_indices_on_local_grid = decomposition_info.global_index(offset.target[0])

    offset_full_grid = grid.connectivities[offset.value].ndarray[source_indices_on_local_grid]
    neighbor_dim = offset.source
    neighbor_index_full_grid = decomposition_info.global_index(neighbor_dim)

    local_offset = halo.global_to_local(
        decomposition_info.global_index(neighbor_dim), offset_full_grid
    )

    ## assert by backmapping

    for i in range(local_offset.shape[0]):
        for k in range(local_offset.shape[1]):
            k_ = local_offset[i][k]
            if k_ == -1:
                # global index is not on this local patch:
                assert not np.isin(offset_full_grid[i][k], neighbor_index_full_grid)
            else:
                assert neighbor_index_full_grid[k_] == offset_full_grid[i][k], (
                    f"failed to map [{offset_full_grid[i]}] to local: [{local_offset[i]}]"
                )


@pytest.mark.parametrize("rank", (0, 1, 2, 3))
def test_horizontal_size(rank):
    simple_neighbor_tables = get_neighbor_tables_for_simple_grid()
    process_props = dummy_four_ranks(rank)
    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        process_props=process_props,
    )
    decomp_info = halo_generator(utils.SIMPLE_DISTRIBUTION)
    horizontal_size = decomp_info.get_horizontal_size()
    expected_verts = len(utils.OWNED[dims.VertexDim][rank]) + len(utils.HALO[dims.VertexDim][rank])
    assert horizontal_size.num_vertices == expected_verts, (
        f"local size mismatch on rank={rank} for {dims.VertexDim}: expected {expected_verts}, but was {horizontal_size.num_vertices}"
    )
    expected_edges = len(utils.OWNED[dims.EdgeDim][rank]) + len(utils.HALO[dims.EdgeDim][rank])
    assert horizontal_size.num_edges == expected_edges, (
        f"local size mismatch on rank={rank} for {dims.EdgeDim}: expected {expected_edges}, but was {horizontal_size.num_edges}"
    )
    expected_cells = len(utils.OWNED[dims.CellDim][rank]) + len(utils.HALO[dims.CellDim][rank])
    assert horizontal_size.num_cells == expected_cells, (
        f"local size mismatch on rank={rank}  for {dims.CellDim}: expected {expected_cells}, but was {horizontal_size.num_cells}"
    )
