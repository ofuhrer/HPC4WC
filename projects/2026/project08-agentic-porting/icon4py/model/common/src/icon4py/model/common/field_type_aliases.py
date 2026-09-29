# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import TypeAlias, TypeVar

import gt4py.next as gtx
from gt4py.next import Dims, Field

from icon4py.model.common import dimension as dims
from icon4py.model.common.type_alias import vpfloat, wpfloat


T = TypeVar("T", wpfloat, vpfloat, float, bool, gtx.int32, gtx.int64)

CellField: TypeAlias = Field[Dims[dims.CellDim], T]  # noqa: UP040
EdgeField: TypeAlias = Field[Dims[dims.EdgeDim], T]  # noqa: UP040
VertexField: TypeAlias = Field[Dims[dims.VertexDim], T]  # noqa: UP040
KField: TypeAlias = Field[Dims[dims.KDim], T]  # noqa: UP040
KHalfField: TypeAlias = Field[Dims[dims.KHalfDim], T]  # noqa: UP040

CellKField: TypeAlias = Field[Dims[dims.CellDim, dims.KDim], T]  # noqa: UP040
EdgeKField: TypeAlias = Field[Dims[dims.EdgeDim, dims.KDim], T]  # noqa: UP040
VertexKField: TypeAlias = Field[Dims[dims.VertexDim, dims.KDim], T]  # noqa: UP040
