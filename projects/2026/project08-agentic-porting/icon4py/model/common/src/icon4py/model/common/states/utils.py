# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from collections.abc import MutableMapping
from typing import TypeAlias, TypeVar

import gt4py.next as gtx
import xarray as xa
from gt4py.next.common import DimsT

from icon4py.model.common import type_alias as ta
from icon4py.model.common.utils import data_allocation as data_alloc


FloatType: TypeAlias = ta.wpfloat | ta.vpfloat | float  # noqa: UP040
IntegerType: TypeAlias = gtx.int32 | gtx.int64 | int  # noqa: UP040
ScalarType: TypeAlias = FloatType | bool | IntegerType  # noqa: UP040

T = TypeVar("T", ta.wpfloat, ta.vpfloat, float, bool, gtx.int32, gtx.int64)

GTXFieldType: TypeAlias = gtx.Field[DimsT, T]  # noqa: UP040
FieldType: TypeAlias = gtx.Field[DimsT, T] | data_alloc.NDArray  # noqa: UP040


def to_data_array(field: FieldType, attrs: MutableMapping[str, ...]):
    data = data_alloc.as_numpy(field)
    return xa.DataArray(data, attrs=attrs)
