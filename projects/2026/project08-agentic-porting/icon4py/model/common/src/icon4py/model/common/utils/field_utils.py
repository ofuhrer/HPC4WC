# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import array_api_compat
from gt4py import next as gtx
from gt4py.next import typing as gtx_typing


def flip(field: gtx.Field, dim: gtx.Dimension, allocator: gtx_typing.Allocator) -> gtx.Field:
    """Flip a field along a given dimension.

    Args:
        field: The field to flip.
        dim: The dimension along which to flip the field.
        allocator: Allocator to use for the output field.
    """
    # Note: `allocator` needs to be passed explicitly since GT4Py fields currently don't persist how they were allocated.
    xp = array_api_compat.array_namespace(field.ndarray)
    flipped_array = xp.flip(field.ndarray, axis=field.domain.dims.index(dim))
    return gtx.as_field(field.domain, flipped_array, allocator=allocator)


def index2offset(
    index_field: gtx.Field, dim: gtx.Dimension, allocator: gtx_typing.Allocator
) -> gtx.Field:
    """Convert an index field to an offset field.

    Note: Additionally clips negative indices to become zero offset as Fortran initializes some indices with `0` (which corresponds to `-1` in Python) to indicate that they are not used.
    As GT4Py's unstructured domain inference is incomplete and runs over the whole domain we might use out-of-bounds offsets in intermediate computations.

    Args:
        index_field: Index field in Python indexing (0-based).
        dim: The dimension along which to convert indices to offsets.
        allocator: Allocator to use for the output field.

    Example:
        >>> import numpy as np
        >>> from gt4py import next as gtx
        >>> from icon4py.model.common import dimension as dims
        >>> index_field = gtx.as_field(
        ...     {dims.CellDim: range(2, 6)}, np.array([5, 3, 4, 2], dtype=gtx.int32)
        ... )
        >>> result = index2offset(index_field, dims.CellDim, allocator=np)
        >>> result.domain
        Domain(dims=(Dimension(value='Cell', kind=<DimensionKind.HORIZONTAL: 'horizontal'>),), ranges=(UnitRange(2, 6),))
        >>> result.ndarray
        array([ 3,  0,  0, -3], dtype=int32)
    """
    # Note: `allocator` needs to be passed explicitly since GT4Py fields currently don't persist how they were allocated.
    xp = array_api_compat.array_namespace(index_field.ndarray)

    current_index = gtx.as_field(
        gtx.Domain(index_field.domain[dim]),
        xp.arange(
            index_field.domain[dim].unit_range.start,
            index_field.domain[dim].unit_range.stop,
            dtype=index_field.ndarray.dtype,
        ),
        allocator=allocator,
    )
    # use GT4Py's broadcasting and field arithmetic (includes clipping)
    offset_field = gtx.where(index_field >= 0, index_field - current_index, 0)

    # if GT4Py embedded would propagate the allocator, we could avoid this extra conversion.
    return gtx.as_field(offset_field.domain, offset_field.ndarray, allocator=allocator)
