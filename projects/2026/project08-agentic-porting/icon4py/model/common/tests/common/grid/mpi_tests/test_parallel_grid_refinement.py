# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import gt4py.next as gtx
import pytest

from icon4py.model.common import dimension as dims, model_backends
from icon4py.model.common.decomposition import decomposer as decomp, definitions as decomp_defs
from icon4py.model.common.grid import grid_refinement, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import (
    datatest_utils as dt_utils,
    definitions,
    grid_utils,
    serialbox,
    test_utils,
)
from icon4py.model.testing.fixtures.datatest import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    process_props,
)

from .. import utils
from . import utils as mpi_test_utils


_log = logging.getLogger(__name__)


def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    if "zone" in metafunc.fixturenames:
        params = [
            (dim, zone) for dim in dims.horizontal_dims() for zone in h_grid._get_zones_for_dim(dim)
        ]
        ids = [f"{dim.value}-{zone}" for dim, zone in params]
        metafunc.parametrize("dim,zone", params, ids=ids)
    elif "dim" in metafunc.fixturenames:
        ids = [dim.value for dim in dims.horizontal_dims()]
        metafunc.parametrize("dim", dims.horizontal_dims(), ids=ids)


@pytest.fixture
def domain(dim: gtx.Dimension, zone: h_grid.Zone) -> h_grid.Domain:
    return h_grid.domain(dim)(zone)


@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_compute_domain_bounds(  # noqa: PLR0917 [too-many-positional-arguments]
    dim: gtx.Dimension,
    zone: h_grid.Zone,
    domain: h_grid.Domain,
    experiment: definitions.Experiment,
    grid_savepoint: serialbox.IconGridSavepoint,
    process_props: decomp_defs.ProcessProperties,
    backend: gtx.typing.Backend | None,
) -> None:
    if (
        process_props.is_single_rank()
        and experiment.description == definitions.Experiments.EXCLAIM_APE
        and dim == dims.EdgeDim
    ):
        pytest.xfail(
            "end index data for single node APE are all 0 - re- serialization should fix that (patch%cells%end_index vs patch%cells%end_idx)"
        )

    ref_grid = grid_savepoint.construct_icon_grid(backend=backend, keep_skip_values=True)
    decomposition_info = grid_savepoint.construct_decomposition_info()
    refin_ctrl = {dim: grid_savepoint.refin_ctrl(dim) for dim in dims.horizontal_dims()}
    start_indices, end_indices = grid_refinement.compute_domain_bounds(
        dim,
        refin_ctrl,
        decomposition_info,
    )
    if (
        experiment.description == definitions.Experiments.GAUSS3D
        and dim == dims.EdgeDim
        and zone in (h_grid.Zone.LOCAL, h_grid.Zone.INTERIOR, h_grid.Zone.HALO)
    ):
        pytest.xfail(
            f"start or end index is known to be inconsistent with {experiment.name=} for {dim=} and {zone=}"
        )

    ref_start_index = ref_grid.start_index(domain)
    ref_end_index = ref_grid.end_index(domain)
    computed_start = start_indices[domain]
    computed_end = end_indices[domain]
    _log.info(
        f"rank = {process_props.rank}/{process_props.comm_size}: domain={domain} : start = {computed_start} end = {computed_end} "
    )
    assert computed_start == ref_start_index, (
        f"rank={process_props.rank}/{process_props.comm_size} - experiment = {experiment.name}: start_index for {domain} does not match: is {computed_start}, expected {ref_start_index}"
    )
    assert computed_end == ref_end_index, (
        f"rank={process_props.rank}/{process_props.comm_size} - experiment = {experiment.name}: end_index for {domain} does not match: is {computed_end}, expected {ref_end_index}"
    )


@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_bounds_decomposition(
    process_props: decomp_defs.ProcessProperties,
    backend: gtx.typing.Backend | None,
    experiment: definitions.Experiment,
    dim: gtx.Dimension,
) -> None:
    if experiment.grid.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    file = dt_utils.get_grid_filepath(experiment.grid)
    _log.info(f"running on {process_props.comm} with {process_props.comm_size} ranks")

    grid_manager = mpi_test_utils.run_grid_manager_for_multi_rank(
        file=file,
        process_props=process_props,
        decomposer=decomp.MetisDecomposer(),
        allocator=model_backends.get_allocator(backend),
    )
    _log.info(
        f"rank = {process_props.rank} : {grid_manager.decomposition_info.get_horizontal_size()!r}"
    )
    _log.info(
        f"rank = {process_props.rank}: halo size for 'CellDim' "
        f"(1: {grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.FIRST_HALO_LEVEL)}), "
        f"(2: {grid_manager.decomposition_info.get_halo_size(dims.CellDim, decomp_defs.DecompositionFlag.SECOND_HALO_LEVEL)})"
    )

    decomposition_info = grid_manager.decomposition_info
    start_index = grid_manager.grid.start_index
    end_index = grid_manager.grid.end_index
    domain = h_grid.domain(dim)

    assert test_utils.is_sorted(decomposition_info.halo_levels(dim)), (
        f"Halo levels for {dim} should be sorted, but are {decomposition_info.halo_levels(dim)}"
    )

    local_owned_size = decomposition_info.local_index(
        dim, decomp_defs.DecompositionInfo.EntryType.OWNED
    ).shape[0]
    local_all_size = decomposition_info.local_index(
        dim, decomp_defs.DecompositionInfo.EntryType.ALL
    ).shape[0]
    local_halo_size = decomposition_info.local_index(
        dim, decomp_defs.DecompositionInfo.EntryType.HALO
    ).shape[0]
    global_owned_size = decomposition_info.global_index(
        dim, decomp_defs.DecompositionInfo.EntryType.OWNED
    ).shape[0]
    global_all_size = decomposition_info.global_index(
        dim, decomp_defs.DecompositionInfo.EntryType.ALL
    ).shape[0]
    global_halo_size = decomposition_info.global_index(
        dim, decomp_defs.DecompositionInfo.EntryType.HALO
    ).shape[0]

    # NOTE: These assumptions may change once limited area grids are supported
    # for icon4py domain decomposition.
    assert start_index(domain(h_grid.Zone.LOCAL)) == 0
    assert end_index(domain(h_grid.Zone.LOCAL)) == local_owned_size
    assert end_index(domain(h_grid.Zone.LOCAL)) == global_owned_size

    assert start_index(domain(h_grid.Zone.INTERIOR)) == 0
    assert end_index(domain(h_grid.Zone.INTERIOR)) == local_owned_size
    assert end_index(domain(h_grid.Zone.INTERIOR)) == global_owned_size

    assert start_index(domain(h_grid.Zone.HALO)) == local_owned_size
    assert start_index(domain(h_grid.Zone.HALO)) == global_owned_size
    assert end_index(domain(h_grid.Zone.END)) == local_all_size
    assert end_index(domain(h_grid.Zone.END)) == global_all_size
    assert (
        end_index(domain(h_grid.Zone.END)) - start_index(domain(h_grid.Zone.HALO))
        == local_halo_size
    )
    assert (
        end_index(domain(h_grid.Zone.END)) - start_index(domain(h_grid.Zone.HALO))
        == global_halo_size
    )
