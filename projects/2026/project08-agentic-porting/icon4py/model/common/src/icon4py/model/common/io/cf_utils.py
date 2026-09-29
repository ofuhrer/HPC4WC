# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from typing import Final

import cftime  # type: ignore[import-untyped] # mypy cannot find the corresponding cftime type hints
import xarray

from icon4py.model.common.states import metadata


#: from standard name table https://cfconventions.org/Data/cf-standard-names/current/build/cf-standard-name-table.html
SLEVE_COORD_STANDARD_NAME: Final[str] = "atmosphere_sleve_coordinate"
LEVEL_STANDARD_NAME: Final[str] = "model_level_number"


DEFAULT_CALENDAR: Final[str] = "proleptic_gregorian"
DEFAULT_TIME_UNIT: Final[str] = "seconds since 1970-01-01 00:00:00"

#: icon4py specific CF extensions:


COARDS_T_POS: Final[int] = 0
COARDS_Z_POS: Final[int] = 1
HORIZONTAL_POS: Final[int] = 2
"""
CF conventions encourage to use the COARDS conventions for the order of the dimensions:
    `T` (time),
    `Z` (height or depth),
    `Y` (latitude),
    `X` (longitude).
In the unstructured case `Y` and `X`  combine to the horizontal dimension.
"""

COARDS_VERTICAL_COORDINATE_NAME: Final[str] = "Z"
COARDS_TIME_COORDINATE_NAME: Final[str] = "T"
COARDS_LONGITUDE_COORDINATE_NAME: Final[str] = "X"
COARDS_LATITUDE_COORDINATE_NAME: Final[str] = "Y"


def date2num(
    date: cftime.datetime,
    units: str = DEFAULT_TIME_UNIT,
    calendar: str = DEFAULT_CALENDAR,
) -> float:
    """

    Convert a datetime object to a number.

    Convenience method that makes units and calendar optional and uses the default values.
    """
    return cftime.date2num(date, units=units, calendar=calendar)


def to_canonical_dim_order(data: xarray.DataArray) -> xarray.DataArray | None:
    """Check for spatial dimensions being in canonical order ('T', 'Z', 'Y', 'X') and return them in this order."""
    dims = data.dims
    if len(dims) >= 2:
        if dims[0] in ("cell", "edge", "vertex") and dims[1] in (
            metadata.INTERFACE_LEVEL_HEIGHT_STANDARD_NAME,
            "level",
            "half_level",
        ):
            return data.transpose(dims[1], dims[0], *dims[2:], transpose_coords=True)
        else:
            return data
    return None
