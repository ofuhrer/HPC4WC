# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.dimension as dims
import icon4py.model.common.grid.horizontal as h_grid
from icon4py.model.common.decomposition import definitions as decomp_defs
from icon4py.model.common.decomposition.decomposer import MetisDecomposer
from icon4py.model.common.grid import base as base_grid, gridfile, horizontal as h_grid
from icon4py.model.testing import definitions as test_defs, grid_utils, parallel_helpers

from ...fixtures import (
    backend,
    data_provider,
    download_ser_data,
    experiment,
    experiment_description,
    grid_savepoint,
    icon_grid,
    process_props,
)
from .. import utils
from . import utils as parallel_utils


if TYPE_CHECKING:
    import gt4py.next as gtx

    from icon4py.model.common.grid import base as base_grid

_log = logging.getLogger(__name__)


@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_props(process_props: decomp_defs.ProcessProperties) -> None:
    """dummy test to check whether the MPI initialization and GHEX setup works."""
    import ghex.context as ghex  # type: ignore[import-not-found]  # noqa: PLC0415 [import-outside-top-level]
    from mpi4py import MPI  # noqa: PLC0415 [import-outside-top-level]

    assert process_props.comm

    assert isinstance(process_props.comm, MPI.Comm), "comm needs to be an instance of MPI.Comm"
    ghex.make_context(process_props.comm)


LOCAL_IDX_2 = {
    dims.CellDim: {0: 10454, 1: 10454},
    dims.EdgeDim: {0: 15830, 1: 15754},
    dims.VertexDim: {0: 5375, 1: 5296},
}

LOCAL_IDX_4 = {
    dims.CellDim: {0: 5238, 1: 5222, 2: 5231, 3: 5230},
    dims.EdgeDim: {0: 7929, 1: 7838, 2: 7955, 3: 7889},
    dims.VertexDim: {0: 2688, 1: 2612, 2: 2723, 3: 2656},
}

LOCAL_IDX = {4: LOCAL_IDX_4, 2: LOCAL_IDX_2}


@pytest.mark.datatest
@pytest.mark.mpi
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.parametrize(
    "experiment_description",
    [
        test_defs.Experiments.MCH_CH_R04B09,
    ],
)
@pytest.mark.parametrize("dim", dims.horizontal_dims())
def test_start_index_end_index_local_zone_on_distributed_lam_grid(
    process_props: decomp_defs.ProcessProperties,
    dim: gtx.Dimension,
    icon_grid: base_grid.Grid,
    experiment: test_defs.Experiment,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    domain = h_grid.domain(dim)(h_grid.Zone.LOCAL)
    _log.info(
        f"rank {process_props.rank}/{process_props.comm_size} dim = {dim} : {icon_grid.size[dim]}"
    )
    # local still runs entire field:
    assert icon_grid.start_index(domain) == 0
    _log.info(
        f"rank {process_props.rank}/{process_props.comm_size} dim = {dim}  LOCAL : ({icon_grid.start_index(domain)}, {icon_grid.end_index(domain)})"
    )
    assert (
        icon_grid.end_index(domain) == LOCAL_IDX[process_props.comm_size][dim][process_props.rank]
    )


HALO_IDX_4 = {
    dims.CellDim: {
        0: (5238, 5340, 5446),
        1: (5222, 5325, 5433),
        2: (5231, 5334, 5442),
        3: (5230, 5334, 5444),
    },
    dims.EdgeDim: {
        0: (7929, 7966, 8173),
        1: (7838, 7940, 8150),
        2: (7955, 7955, 8165),
        3: (7889, 7954, 8167),
    },
    dims.VertexDim: {
        0: (2688, 2727, 2834),
        1: (2612, 2717, 2825),
        2: (2723, 2723, 2832),
        3: (2656, 2723, 2832),
    },
}
HALO_IDX_2 = {
    dims.CellDim: {
        0: (10454, 10531, 10611),
        1: (10454, 10531, 10612),
    },
    dims.EdgeDim: {
        0: (15830, 15830, 15986),
        1: (15754, 15830, 15987),
    },
    dims.VertexDim: {
        0: (5375, 5375, 5455),
        1: (5296, 5375, 5456),
    },
}

HALO_IDX = {4: HALO_IDX_4, 2: HALO_IDX_2}


@pytest.mark.datatest
@pytest.mark.parametrize("process_props", [True], indirect=True)
@pytest.mark.mpi
@pytest.mark.parametrize("dim", dims.horizontal_dims())
@pytest.mark.parametrize(
    "experiment_description",
    [
        test_defs.Experiments.MCH_CH_R04B09,
    ],
)
@pytest.mark.parametrize("zone, level", [(h_grid.Zone.HALO, 1), (h_grid.Zone.HALO_LEVEL_2, 2)])
def test_start_index_end_index_halo_zones_on_distributed_lam_grid(  # noqa: PLR0917 [too-many-positional-arguments]
    process_props: decomp_defs.ProcessProperties,
    dim: gtx.Dimension,
    zone: h_grid.Zone,
    icon_grid: base_grid.Grid,
    experiment: test_defs.Experiment,
    level: int,
) -> None:
    parallel_helpers.check_comm_size(process_props)
    domain = h_grid.domain(dim)(zone)
    start_index = icon_grid.start_index(domain)
    end_index = icon_grid.end_index(domain)
    rank = process_props.rank
    _log.info(
        f"rank {rank}/{process_props.comm_size} dim = {dim}  {zone} : ({start_index}, {end_index})"
    )
    expected = HALO_IDX[process_props.comm_size][dim][rank][level - 1]
    assert start_index == expected, f"expected start index {expected}, but was {start_index}"
    expected = HALO_IDX[process_props.comm_size][dim][rank][level]
    assert end_index == expected, f"expected start index {1}, but was {start_index}"
