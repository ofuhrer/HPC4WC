# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
from typing import Any

import numpy as np
from gt4py import next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions


"""
TESTDATA using the
[SimpleGrid](../../../src/icon4py/model/common/grid/simple.py) The distribution
maps all of the 18 cells of the simple grid to ranks 0..3.

The dictionaries contain the mapping from rank to global (in the simple grid)
index of the dimension:
- _CELL_OWN: rank -> owned cells, essentially the inversion of the
  SIMPLE_DISTRIBUTION
- _EDGE_OWN: rank -> owned edges
 - _VERTEX_OWN: rank -> owned vertices

The decision as to whether a "secondary" dimension (edge, vertices) is owned by
a rank are made according to the rules and conventions described in
(../../../src/icon4py/model/common/decomposition/halo.py).


_CELL_FIRST_HALO_LINE:
_CELL_SECOND_HALO_LINE:
_EDGE_FIRST_HALO_LINE:
_EDGE_SECOND_HALO_LINE:
_VERTEX_FIRST_HALO_LINE:
_VERTEX_SECOND_HALO_LINE: :mapping of rank to global indices that belongs to a ranks halo lines.
"""


SIMPLE_DISTRIBUTION = np.asarray(
    [
        0,  # 0c
        1,  # 1c
        1,  # 2c
        0,  # 3c
        0,  # 4c
        1,  # 5c
        0,  # 6c
        0,  # 7c
        2,  # 8c
        2,  # 9c
        0,  # 10c
        2,  # 11c
        3,  # 12c
        3,  # 13c
        1,  # 14c
        3,  # 15c
        3,  # 16c
        1,  # 17c
    ]
)
_CELL_OWN = {0: [0, 3, 4, 6, 7, 10], 1: [1, 2, 5, 14, 17], 2: [8, 9, 11], 3: [12, 13, 15, 16]}
_CELL_FIRST_HALO_LINE = {
    0: [1, 11, 13, 9, 2, 15],
    1: [3, 8, 4, 11, 16, 13, 15],
    2: [5, 7, 6, 12, 14],
    3: [9, 10, 17, 14, 0, 1],
}
_CELL_SECOND_HALO_LINE = {
    0: [17, 5, 12, 14, 8, 16],
    1: [0, 7, 6, 9, 10, 12],
    2: [2, 1, 4, 3, 10, 15, 16, 17, 13],
    3: [6, 7, 8, 2, 3, 4, 5, 11],
}
_CELL_HALO = {
    0: _CELL_FIRST_HALO_LINE[0] + _CELL_SECOND_HALO_LINE[0],
    1: _CELL_FIRST_HALO_LINE[1] + _CELL_SECOND_HALO_LINE[1],
    2: _CELL_FIRST_HALO_LINE[2] + _CELL_SECOND_HALO_LINE[2],
    3: _CELL_FIRST_HALO_LINE[3] + _CELL_SECOND_HALO_LINE[3],
}
_EDGE_OWN = {
    0: [1, 5, 12, 13, 14, 9],
    1: [8, 7, 6, 25, 4, 2],
    2: [16, 11, 15, 17, 10, 24],
    3: [19, 23, 22, 26, 0, 3, 20, 18, 21],
}
_EDGE_FIRST_HALO_LINE = {0: [0, 4, 17, 21, 10, 2], 1: [3, 15, 20, 26, 24], 2: [18], 3: []}
_EDGE_SECOND_HALO_LINE = {
    0: [3, 6, 7, 8, 15, 24, 25, 26, 16, 22, 23, 18, 19, 20, 11],
    1: [0, 1, 5, 9, 12, 11, 10, 13, 16, 17, 18, 19, 21, 22, 23],
    2: [2, 9, 12, 4, 8, 7, 14, 21, 13, 19, 20, 22, 23, 25, 26],
    3: [11, 10, 14, 13, 16, 17, 24, 25, 6, 2, 1, 5, 4, 8, 7],
}
_EDGE_THIRD_HALO_LINE = {
    0: [],
    1: [14],
    2: [0, 1, 3, 5, 6],
    3: [9, 12, 15],
}
_EDGE_HALO = {
    0: _EDGE_FIRST_HALO_LINE[0] + _EDGE_SECOND_HALO_LINE[0] + _EDGE_THIRD_HALO_LINE[0],
    1: _EDGE_FIRST_HALO_LINE[1] + _EDGE_SECOND_HALO_LINE[1] + _EDGE_THIRD_HALO_LINE[1],
    2: _EDGE_FIRST_HALO_LINE[2] + _EDGE_SECOND_HALO_LINE[2] + _EDGE_THIRD_HALO_LINE[2],
    3: _EDGE_FIRST_HALO_LINE[3] + _EDGE_SECOND_HALO_LINE[3] + _EDGE_THIRD_HALO_LINE[3],
}
_VERTEX_OWN = {
    0: [4],
    1: [],
    2: [3, 5],
    3: [
        0,
        1,
        2,
        6,
        7,
        8,
    ],
}
_VERTEX_FIRST_HALO_LINE = {
    0: [0, 1, 5, 8, 7, 3],
    1: [1, 2, 0, 5, 3, 8, 6],
    2: [
        6,
        8,
        7,
    ],
    3: [],
}
_VERTEX_SECOND_HALO_LINE = {
    0: [2, 6],
    1: [7, 4],
    2: [4, 0, 2, 1],
    3: [3, 4, 5],
}
_VERTEX_HALO = {
    0: _VERTEX_FIRST_HALO_LINE[0] + _VERTEX_SECOND_HALO_LINE[0],
    1: _VERTEX_FIRST_HALO_LINE[1] + _VERTEX_SECOND_HALO_LINE[1],
    2: _VERTEX_FIRST_HALO_LINE[2] + _VERTEX_SECOND_HALO_LINE[2],
    3: _VERTEX_FIRST_HALO_LINE[3] + _VERTEX_SECOND_HALO_LINE[3],
}
OWNED = {dims.CellDim: _CELL_OWN, dims.EdgeDim: _EDGE_OWN, dims.VertexDim: _VERTEX_OWN}
HALO = {dims.CellDim: _CELL_HALO, dims.EdgeDim: _EDGE_HALO, dims.VertexDim: _VERTEX_HALO}
FIRST_HALO_LINE = {
    dims.CellDim: _CELL_FIRST_HALO_LINE,
    dims.VertexDim: _VERTEX_FIRST_HALO_LINE,
    dims.EdgeDim: _EDGE_FIRST_HALO_LINE,
}
SECOND_HALO_LINE = {
    dims.CellDim: _CELL_SECOND_HALO_LINE,
    dims.VertexDim: _VERTEX_SECOND_HALO_LINE,
    dims.EdgeDim: _EDGE_SECOND_HALO_LINE,
}
THIRD_HALO_LINE = {
    dims.CellDim: {0: [], 1: [], 2: [], 3: []},
    dims.VertexDim: {0: [], 1: [], 2: [], 3: []},
    dims.EdgeDim: _EDGE_THIRD_HALO_LINE,
}


def assert_same_entries(
    dim: gtx.Dimension, my_owned: np.ndarray, reference: dict[gtx.Dimension, dict], rank: int
) -> None:
    print(f"myowned {my_owned}")
    print(reference[dim][rank])
    assert my_owned.size == len(reference[dim][rank]), (
        f"{dim}(rank = {rank}) : wrong size expected {len(reference[dim][rank])} but was {my_owned.size}"
    )
    assert np.setdiff1d(my_owned, reference[dim][rank], assume_unique=True).size == 0


@dataclasses.dataclass(frozen=True, init=False)
class DummyProps(definitions.ProcessProperties):
    comm: Any
    comm_name: str
    comm_size: int
    rank: int

    def __init__(self, rank: int):
        object.__setattr__(self, "rank", rank % 4)
        object.__setattr__(self, "comm", None)
        object.__setattr__(self, "comm_name", "dummy on 4")
        object.__setattr__(self, "comm_size", 4)


def dummy_four_ranks(rank: int) -> definitions.ProcessProperties:
    return DummyProps(rank)
