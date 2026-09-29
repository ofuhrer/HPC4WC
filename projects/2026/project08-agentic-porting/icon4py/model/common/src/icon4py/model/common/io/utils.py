# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import xarray as xa
from gt4py import next as gtx
from gt4py._core import definitions as gt_coredefs
from gt4py.next import common as gt_common

from icon4py.model.common.io.ugrid import dimension_mapping, ugrid_attributes
from icon4py.model.common.states.model import FieldMetaData
from icon4py.model.common.utils import data_allocation as data_alloc


def to_data_array(
    field: gtx.Field[gtx.Dims[gt_common.DimsT], gt_coredefs.ScalarT],
    attrs: FieldMetaData | dict | None = None,
    is_on_half_levels: bool = False,
    to_host: bool = False,
) -> xa.DataArray:
    """Convert a gt4py field to a xarray data array.

    Args:
        field: gt4py field,
        attrs: optional dictionary of metadata attributes to be added to the data array, empty by default.
            The dictionary is copied, the caller's instance is left untouched.
        is_on_half_levels: optional boolean flag indicating if the 2d field is defined on the half (interface) levels, False by default.
        to_host: if True, copy the data buffer to host (numpy). netCDF4 cannot consume
            device arrays, so set this when the result is written from a GPU backend.

    Note:
        The data buffer is not copied: the result usually references ``field`` (a view).
        ``to_host=True`` is a copy only on GPU; on CPU it still references ``field``.
        Callers that keep the DataArray past the next mutation of ``field`` must copy it.
    """
    attrs = {} if attrs is None else dict(attrs)
    dims = tuple(dimension_mapping(d, is_on_half_levels) for d in field.domain.dims)
    horizontal_dim = next(d for d in field.domain.dims if _is_horizontal(d))
    uxgrid_attrs = ugrid_attributes(horizontal_dim)
    attrs.update(uxgrid_attrs)  # type: ignore [typeddict-item] # mypy does not accept the dict types flexibility
    data = data_alloc.as_numpy(field) if to_host else field.ndarray
    return xa.DataArray(data=data, dims=dims, attrs=attrs)


def _is_horizontal(dim: gtx.Dimension) -> bool:
    return dim.kind == gtx.DimensionKind.HORIZONTAL
