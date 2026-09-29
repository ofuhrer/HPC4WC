# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import logging

import gt4py.next as gtx
import numpy as np
import pytest

import icon4py.model.common.dimension as dims
from icon4py.model.common.decomposition import definitions as decomp_defs, halo
from icon4py.model.common.grid import simple
from icon4py.model.testing import parallel_helpers
from icon4py.model.testing.fixtures import process_props

from .. import utils
from ..fixtures import simple_neighbor_tables


_log = logging.getLogger(__name__)

backend = None


def global_indices(dim: gtx.Dimension) -> np.ndarray:
    mesh = simple.simple_grid()
    return np.arange(mesh.size[dim], dtype=gtx.int32)


@pytest.mark.parametrize("dim", dims.horizontal_dims())
@pytest.mark.mpi(min_size=4)
@pytest.mark.parametrize("process_props", [True], indirect=True)
def test_element_ownership_is_unique(
    dim,
    process_props,
    simple_neighbor_tables,
):
    parallel_helpers.check_comm_size(process_props, sizes=[4])

    halo_generator = halo.IconLikeHaloConstructor(
        connectivities=simple_neighbor_tables,
        process_props=process_props,
        allocator=backend,
    )

    decomposition_info = halo_generator(utils.SIMPLE_DISTRIBUTION)
    owned = decomposition_info.global_index(dim, decomp_defs.DecompositionInfo.EntryType.OWNED)
    _log.info(f"\nrank {process_props.rank} owns {dim} : {owned} ")
    # assert that each cell is only owned by one rank
    comm = process_props.comm

    my_size = owned.shape[0]
    local_sizes = np.array(comm.gather(my_size, root=0))
    buffer_size = 27
    send_buf = np.full(buffer_size, -1, dtype=int)
    send_buf[:my_size] = owned
    _log.info(f"rank {process_props.rank} send_buf: {send_buf}")
    if process_props.rank == 0:
        _log.info(f"local_sizes: {local_sizes}")
        recv_buffer = np.full((4, buffer_size), -1, dtype=int)
        _log.info(f"{recv_buffer.shape}")
    else:
        recv_buffer = None
    # Gatherv does not work if one of the buffers has size-0 (VertexDim)
    comm.Gather(sendbuf=send_buf, recvbuf=recv_buffer, root=0)
    if process_props.rank == 0:
        _log.info(f"global indices: {recv_buffer}")
        # check there are no duplicates
        values = recv_buffer[recv_buffer != -1]
        assert values.size == len(np.unique(values))
        # check the buffer has all global indices
        assert np.all(np.sort(values) == global_indices(dim))
