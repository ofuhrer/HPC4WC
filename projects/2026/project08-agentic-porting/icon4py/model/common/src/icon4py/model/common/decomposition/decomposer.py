# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Protocol, runtime_checkable

import gt4py.next as gtx
import numpy as np

from icon4py.model.common.utils import data_allocation as data_alloc


@runtime_checkable
class Decomposer(Protocol):
    def __call__(
        self, adjacency_matrix: data_alloc.NDArray, num_partitions: int
    ) -> data_alloc.NDArray:
        """
        Call the decomposition.

        Args:
            adjacency_matrix: cell-to-cell connectivity matrix on the global (undecomposed) grid. In the Icon4py context this C2E2C
            n_part: number of nodes
        """
        ...


class MetisDecomposer(Decomposer):
    """
    A simple decomposer using METIS for partitioning a grid topology.
    """

    def __call__(
        self, adjacency_matrix: data_alloc.NDArray, num_partitions: int
    ) -> data_alloc.NDArray:
        """
        Generate partition labels for this grid topology using METIS:
        https://github.com/KarypisLab/METIS

        This method utilizes the pymetis Python bindings:
        https://github.com/inducer/pymetis

        Args:
            n_part: int, number of partitions to create
            adjacency_matrix: nd array: neighbor table describing of the main dimension object to be distributed: for example cell -> cell neighbors
        Returns: data_alloc.NDArray: array with partition label (int, rank number) for each cell
        """

        import pymetis  # noqa: PLC0415 [import-outside-top-level]

        # Invalid indices are not allowed here. Metis will segfault or fail if
        # there are any invalid indices in the adjacency matrix.
        assert (adjacency_matrix >= 0).all()

        # Explicitly move to CPU before passing to Metis, Metis only runs on
        # CPU.
        adjacency_matrix_np = data_alloc.as_numpy(adjacency_matrix)

        # The partitioning is done on all ranks, and this assumes that the
        # partitioning is deterministic.
        #
        # Passes the adjacency matrix in CSR format (xadj/adjncy) to pymetis,
        # which avoids Python-side iteration and copies in pymetis._prepare_graph.
        #
        # adjncy is flattened array of the cell neighbors. The first three
        # entries contain the neighbors of cell 0, the next three neighbors of
        # cell 1, and so on. xadj contains the indices where the ith cell's
        # neighbors start in adjncy. Since we only support global/torus grids at
        # the moment, every cell always has three neighbors. xadj will always be
        # [0, 3, 6, 9, ...] because the first cell's neighbors are at indices 0
        # to 2, the second cell's neighbors are at indices 3 to 5, and so on.
        _, partition_index = pymetis.part_graph(
            nparts=num_partitions,
            xadj=np.arange(adjacency_matrix_np.shape[0] + 1, dtype=np.int32)
            * adjacency_matrix_np.shape[1],
            adjncy=adjacency_matrix_np.ravel(),
        )
        return data_alloc.array_namespace(adjacency_matrix).array(partition_index)


class SingleNodeDecomposer(Decomposer):
    def __call__(
        self, adjacency_matrix: data_alloc.NDArray, num_partitions: int
    ) -> data_alloc.NDArray:
        """Dummy decomposer for single node: assigns all cells to rank = 0"""
        if num_partitions != 1:
            raise ValueError(
                f"SingleNodeDecomposer can only be used for num_partitions=1, but got {num_partitions}"
            )

        return data_alloc.array_namespace(adjacency_matrix).zeros(
            adjacency_matrix.shape[0],
            dtype=gtx.int32,
        )
