# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest
from gt4py import next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, model_backends
from icon4py.model.common.decomposition import (
    decomposer as decomp,
    definitions as decomp_defs,
    mpi_decomposition,
)
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.states import factory
from icon4py.model.common.utils import data_allocation as data_alloc
from icon4py.model.testing import definitions as test_defs, grid_utils, parallel_helpers

from ...grid.mpi_tests import utils as mpi_tests_utils
from ..fixtures import backend, grid_description, process_props
from ..unit_tests.test_factory import SimpleFieldSource


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing


if mpi_decomposition.mpi4py is None:
    pytest.skip("Skipping parallel tests on single node installation", allow_module_level=True)


@gtx.field_operator
def _fill_op(field: fa.EdgeField[gtx.int32], value: gtx.int32) -> fa.EdgeField[gtx.int32]:
    return field + value


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def _fill_edges(
    field: fa.EdgeField[gtx.int32],
    value: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    _fill_op(field, value, out=field, domain={dims.EdgeDim: (horizontal_start, horizontal_end)})


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize("do_exchange", [True, False])
def test_program_provider_exchange(
    do_exchange: bool,
    process_props: decomp_defs.ProcessProperties,
    grid_description: test_defs.GridDescription,
    backend: gtx_typing.Backend | None,
) -> None:
    if grid_description.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    parallel_helpers.check_comm_size(process_props, sizes=(2, 4))
    grid_file = grid_utils._download_grid_file(grid_description)
    allocator = model_backends.get_allocator(backend)
    grid_manager = mpi_tests_utils.run_grid_manager_for_multi_rank(
        file=grid_file,
        process_props=process_props,
        decomposer=decomp.MetisDecomposer(),
        allocator=allocator,
        num_levels=1,
    )
    grid = grid_manager.grid
    decomposition_info = grid_manager.decomposition_info
    exchange = decomp_defs.create_exchange(process_props, decomposition_info)

    number = process_props.rank + 10
    source = SimpleFieldSource(
        data_={},
        backend=backend,
        grid=grid,
    ).with_metadata({"out": {"dtype": np.int32, "standard_name": "out", "units": ""}})
    source._exchange = exchange
    edge_domain = h_grid.domain(dims.EdgeDim)
    provider = factory.ProgramFieldProvider(
        func=_fill_edges.with_backend(backend),
        domain={
            dims.EdgeDim: (edge_domain(h_grid.Zone.LOCAL), edge_domain(h_grid.Zone.END)),
        },
        fields={"field": "out"},
        deps={},
        params={"value": number},
        do_exchange=do_exchange,
    )
    source.register_provider(provider)
    field = source.get("out")

    halo_points = decomposition_info.local_index(
        dims.EdgeDim, decomp_defs.DecompositionInfo.EntryType.HALO
    )
    owned_points = decomposition_info.local_index(
        dims.EdgeDim, decomp_defs.DecompositionInfo.EntryType.OWNED
    )
    xp = data_alloc.import_array_ns(backend)
    valid_values = xp.asarray(
        [r + 10 for r in range(process_props.comm_size) if r != process_props.rank]
    )
    arr = field.ndarray

    assert (arr[owned_points] == number).all()  # type: ignore[attr-defined]
    if do_exchange:
        assert xp.all(xp.isin(arr[halo_points], valid_values))
    else:
        assert (arr[halo_points] == number).all()  # type: ignore[attr-defined]


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize("do_exchange", [True, False])
def test_numpy_provider_exchange(
    do_exchange: bool,
    process_props: decomp_defs.ProcessProperties,
    grid_description: test_defs.GridDescription,
    backend: gtx_typing.Backend | None,
) -> None:
    if grid_description.limited_area:
        pytest.xfail("Limited-area grids not yet supported")

    parallel_helpers.check_comm_size(process_props, sizes=(2, 4))
    grid_file = grid_utils._download_grid_file(grid_description)
    allocator = model_backends.get_allocator(backend)
    grid_manager = mpi_tests_utils.run_grid_manager_for_multi_rank(
        file=grid_file,
        process_props=process_props,
        decomposer=decomp.MetisDecomposer(),
        allocator=allocator,
        num_levels=1,
    )
    grid = grid_manager.grid
    decomposition_info = grid_manager.decomposition_info
    exchange = decomp_defs.create_exchange(process_props, decomposition_info)

    number = process_props.rank + 10
    source = SimpleFieldSource(
        data_={},
        backend=backend,
        grid=grid,
    )
    source._exchange = exchange

    xp = data_alloc.import_array_ns(backend)

    def make_constant() -> data_alloc.NDArray:
        return xp.full(grid.size[dims.EdgeDim], number)

    provider = factory.NumpyDataProvider(
        func=make_constant,
        domain=(dims.EdgeDim,),
        fields=("out",),
        deps={},
        params={},
        do_exchange=do_exchange,
    )
    source.register_provider(provider)
    field = source.get("out")

    halo_points = decomposition_info.local_index(
        dims.EdgeDim, decomp_defs.DecompositionInfo.EntryType.HALO
    )
    owned_points = decomposition_info.local_index(
        dims.EdgeDim, decomp_defs.DecompositionInfo.EntryType.OWNED
    )
    valid_values = xp.asarray(
        [r + 10 for r in range(process_props.comm_size) if r != process_props.rank]
    )
    arr = field.ndarray

    assert (arr[owned_points] == number).all()  # type: ignore[attr-defined]
    if do_exchange:
        assert xp.all(xp.isin(arr[halo_points], valid_values))
    else:
        assert (arr[halo_points] == number).all()  # type: ignore[attr-defined]
