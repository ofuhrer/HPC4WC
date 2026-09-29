# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import Iterator

import gt4py.next as gtx


KDim = gtx.Dimension("K", kind=gtx.DimensionKind.VERTICAL)
KHalfDim = gtx.Dimension("KHalf", kind=gtx.DimensionKind.VERTICAL)
EdgeDim = gtx.Dimension("Edge")
CellDim = gtx.Dimension("Cell")
VertexDim = gtx.Dimension("Vertex")
LsqUnkDim = gtx.Dimension("LsqUnk", gtx.DimensionKind.LOCAL)
E2CDim = gtx.Dimension("E2C", gtx.DimensionKind.LOCAL)
E2VDim = gtx.Dimension("E2V", gtx.DimensionKind.LOCAL)
C2EDim = gtx.Dimension("C2E", gtx.DimensionKind.LOCAL)
V2CDim = gtx.Dimension("V2C", gtx.DimensionKind.LOCAL)
C2VDim = gtx.Dimension("C2V", gtx.DimensionKind.LOCAL)
V2EDim = gtx.Dimension("V2E", gtx.DimensionKind.LOCAL)
V2E2VDim = gtx.Dimension("V2E2V", gtx.DimensionKind.LOCAL)
E2C2VDim = gtx.Dimension("E2C2V", gtx.DimensionKind.LOCAL)
C2E2CODim = gtx.Dimension("C2E2CO", gtx.DimensionKind.LOCAL)
E2C2EODim = gtx.Dimension("E2C2EO", gtx.DimensionKind.LOCAL)
E2C2EDim = gtx.Dimension("E2C2E", gtx.DimensionKind.LOCAL)
C2E2CDim = gtx.Dimension("C2E2C", gtx.DimensionKind.LOCAL)
C2E2C2EDim = gtx.Dimension("C2E2C2E", gtx.DimensionKind.LOCAL)
C2E2C2E2CDim = gtx.Dimension("C2E2C2E2C", gtx.DimensionKind.LOCAL)
E2C = gtx.FieldOffset("E2C", source=CellDim, target=(EdgeDim, E2CDim))
C2E = gtx.FieldOffset("C2E", source=EdgeDim, target=(CellDim, C2EDim))
V2C = gtx.FieldOffset("V2C", source=CellDim, target=(VertexDim, V2CDim))
C2V = gtx.FieldOffset("C2V", source=VertexDim, target=(CellDim, C2VDim))
V2E = gtx.FieldOffset("V2E", source=EdgeDim, target=(VertexDim, V2EDim))
E2V = gtx.FieldOffset("E2V", source=VertexDim, target=(EdgeDim, E2VDim))
E2C2V = gtx.FieldOffset("E2C2V", source=VertexDim, target=(EdgeDim, E2C2VDim))
C2E2CO = gtx.FieldOffset("C2E2CO", source=CellDim, target=(CellDim, C2E2CODim))
E2C2EO = gtx.FieldOffset("E2C2EO", source=EdgeDim, target=(EdgeDim, E2C2EODim))
E2C2E = gtx.FieldOffset("E2C2E", source=EdgeDim, target=(EdgeDim, E2C2EDim))
C2E2C = gtx.FieldOffset("C2E2C", source=CellDim, target=(CellDim, C2E2CDim))
C2E2C2E = gtx.FieldOffset("C2E2C2E", source=EdgeDim, target=(CellDim, C2E2C2EDim))
C2E2C2E2C = gtx.FieldOffset("C2E2C2E2C", source=CellDim, target=(CellDim, C2E2C2E2CDim))
V2E2V = gtx.FieldOffset("V2E2V", source=VertexDim, target=(VertexDim, V2E2VDim))
Koff = gtx.FieldOffset("Koff", source=KDim, target=(KDim,))
KHalfOff = gtx.FieldOffset("KHalfOff", source=KHalfDim, target=(KHalfDim,))


def horizontal_dims() -> Iterator[gtx.Dimension]:
    return iter(
        tuple(
            d
            for d in globals().values()
            if isinstance(d, gtx.Dimension) and d.kind == gtx.DimensionKind.HORIZONTAL
        )
    )


def non_horizontal_dims() -> Iterator[gtx.Dimension]:
    yield from vertical_dims()
    yield from local_dims()


def local_dims() -> Iterator[gtx.Dimension]:
    for d in globals().values():
        if isinstance(d, gtx.Dimension) and d.kind == gtx.DimensionKind.LOCAL:
            yield d


def vertical_dims() -> Iterator[gtx.Dimension]:
    return iter(
        tuple(
            d
            for d in globals().values()
            if isinstance(d, gtx.Dimension) and d.kind == gtx.DimensionKind.VERTICAL
        )
    )
