# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


"""
This module handles several aspects of the horizontal grid in ICON.

Among which most importantly:

Horizontal domain zones
-------------------------
ICON provides three routines `get_indices_c`, `get_indices_e` and `get_indices_v` which return indices into Fields of the given dimension
that mark the start and end of specific horizontal grid domains such as the lateral boundaries, nudging zones etc.

Those routines get passed an integer value normally called `rl_start` or `rl_end`. The values ranges over a custom index range
for each dimension, some of which are denoted by constants defined in `mo_impl_constants.f90` and `mo_impl_constants_grf.f90`.

Internally ICON uses a double indexing scheme for those start and end indices. They are
stored in arrays `start_idx` and `end_idx` originally read from the grid file ICON accesses those indices by a custom index range
denoted by the constants mentioned above. However, some entries into these arrays contain invalid Field indices and must not
be used ever.

horizontal.py provides an interface to a Python port of constants wrapped in a custom `Domain` class, which takes care of the
custom index range and makes sure that for each dimension only legal values can be passed.

The horizontal domain zones are denoted by a set of named enums for the different zones:
see Fig. 9.2 in the official [ICON tutorial](https://www.dwd.de/DE/leistungen/nwv_icon_tutorial/pdf_einzelbaende/icon_tutorial2024.html).


"""

import dataclasses
import enum
import functools
from collections.abc import Callable, Iterator
from typing import Any, Final

import gt4py.next as gtx
import numpy as np

from icon4py.model.common import dimension as dims


_EDGE_GRF: Final[int] = 24
_CELL_GRF: Final[int] = 14
_VERTEX_GRF: Final[int] = 13

_ICON_CONSTANTS_BOUNDS = {
    dims.CellDim: (0, _CELL_GRF - 1),
    dims.EdgeDim: (0, _EDGE_GRF - 1),
    dims.VertexDim: (0, _VERTEX_GRF - 1),
}

_ICON_LATERAL_BOUNDARY: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: 9,
    dims.EdgeDim: 14,
    dims.VertexDim: 8,
}
ICON_LOCAL = {
    dims.CellDim: 4,
    dims.EdgeDim: 5,
    dims.VertexDim: 3,
}
_ICON_HALO = {
    dims.CellDim: 3,
    dims.EdgeDim: 4,
    dims.VertexDim: 2,
}
_ICON_INTERIOR = {
    dims.CellDim: 8,
    dims.EdgeDim: 13,
    dims.VertexDim: 7,
}
_ICON_NUDGING = {
    dims.CellDim: 13,
    dims.EdgeDim: 22,
    dims.VertexDim: 12,
}
_ICON_END = {
    dims.CellDim: 0,
    dims.EdgeDim: 0,
    dims.VertexDim: 0,
}
"""
Indices used to look up start index and end index in the arrays int `start_idx_[c,e,v]` from ICON.

ICON uses constants defined in `mo_impl_constants.f90` and `mo_impl_constants_grf.f90` to index into these arrays, where
for each dimension they index from start_idx(-n, m) such that what we call `Zone.INTERIOR` is at  start_idx[0]. (see tables below)

The values here are translations of these indices taking into account that all arrays in Python are zero based.

"""


def _icon_domain_index(value_dict: dict, dim: gtx.Dimension, offset: int = 0) -> int:
    index = value_dict[dim] + offset
    assert index <= _ICON_CONSTANTS_BOUNDS[dim][1], (
        f"Index {index} out of bounds for {dim}:  {_ICON_CONSTANTS_BOUNDS[dim]}"
    )
    assert index >= _ICON_CONSTANTS_BOUNDS[dim][0], (
        f"Index {index} out of bounds for {dim}: {_ICON_CONSTANTS_BOUNDS[dim]}"
    )
    return index


class Zone(enum.Enum):
    """
    Enum of different zones on the horizontal ICON grid.
    The mapping to the constant used in ICON is as follows: (note that not all values exist for all dimensions


    ## CellDim
    | ICON constant or value                | python| ICON4py Name               |
    | from mo_impl_constants.f90            | index |                            |
    |:--------------------------------------|:------|:---------------------------|
    | `min_rlcell_int-3`, `min_rlcell` (-8) |  0    | `END`                      |
    | `min_rlcell_int-3` (-7)               |  1    |                            |
    | `min_rlcell_int-2`, (-6)              |  2    | `HALO_LEVEL_2`             |
    | `min_rlcell_int-1` (-5)               |  3    | `HALO`                     |
    | `min_rlcell_int`(-4)                  |  4    | `LOCAL`                    |
    | (-3)                                  |  5    |                            | unused in icon4py (relevant for nesting)
    | (-2)                                  |  6    |                            | unused in icon4py (relevant for nesting)
    | (-1)                                  |  7    |                            | unused in icon4py (relevant for nesting)
    | `0`                                   |  8    | `INTERIOR`                 |
    | `1`                                   |  9    | `LATERAL_BOUNDARY`         |
    | `2`                                   | 10    | `LATERAL_BOUNDARY_LEVEL_2` |
    | `3`                                   | 11    | `LATERAL_BOUNDARY_LEVEL_3` |
    | `grf_bdywidth_c` (4)                  | 12    | `LATERAL_BOUNDARY_LEVEL_4` |
    | `grf_bdywith_c +1`,max_rlcell (5)     | 13    | `NUDGING`                  |


    Lateral boundary and nudging are only relevant for LAM runs, halo lines only for distributed domains.
    The constants are defined in `mo_impl_constants.f90` and `mo_impl_constants_grf.f90`

    ## VertexDim

    | ICON constant or value                  | python| ICON4Py Name               |
    | from mo_impl_constants.f90              | index |                            |
    |:----------------------------------------|:------|:---------------------------|
    | `min_rlvert` (-7)                       |   0   | `END`                      |
    | `min_rlvert+1`, `min_rlvert_int-2` (-6) |   1   | `HALO_LEVEL_2`             |
    | `min_rlvert_int-1` (-5)                 |   2   | `HALO`                     |
    | `min_rlvert_int` (-4)                   |   3   | `LOCAL`                    |
    | (-3)                                    |   4   |                            | unused in icon4py (relevant for nesting)
    | (-2)                                    |   5   |                            | unused in icon4py (relevant for nesting)
    | (-1)                                    |   6   |                            | unused in icon4py (relevant for nesting)
    | `0`                                     |   7   | `INTERIOR`                 |
    | `1`                                     |   8   | `LATERAL_BOUNDARY`         |
    | `2`                                     |   9   | `LATERAL_BOUNDARY_LEVEL_2` |
    | `3`                                     |  10   | `LATERAL_BOUNDARY_LEVEL_3` |
    | `4`                                     |  11   | `LATERAL_BOUNDARY_LEVEL_4` |
    | `max_rlvert` (5)                        |  12   | `NUDGING`                  |

    For the meaning see above.

    ## EdgeDim


    | ICON constant or value                 | python | ICON4Py Name               |
    | from mo_impl_constants.f90             | index  |                            |
    |:---------------------------------------|:-------|:---------------------------|
    | `min_rledge` (-13)                     |   0    | `END`                      |
    | (-12)                                  |   1    |                            |
    | (-11)                                  |   2    |                            |
    | `min_rledge_int-2` (-10)               |   3    | `HALO_LEVEL_2`             |
    | `min_rledge_int-1` (-9)                |   4    | `HALO`                     |
    | `min_rledge_int`   (-8)                |   5    | `LOCAL`                    |
    | (-7)                                   |   6    |                            | unused in icon4py (relevant for nesting)
    | (-6)                                   |   7    |                            | unused in icon4py (relevant for nesting)
    | (-5)                                   |   8    |                            | unused in icon4py (relevant for nesting)
    | (-4)                                   |   9    |                            | unused in icon4py (relevant for nesting)
    | (-3)                                   |  10    |                            | unused in icon4py (relevant for nesting)
    | (-2)                                   |  11    |                            | unused in icon4py (relevant for nesting)
    | (-1)                                   |  12    |                            | unused in icon4py (relevant for nesting)
    | `0`                                    |  13    | `INTERIOR`                 |
    | `1`                                    |  14    | `LATERAL_BOUNDARY`         |
    | `2`                                    |  15    | `LATERAL_BOUNDARY_LEVEL_2` |
    | `3`                                    |  16    | `LATERAL_BOUNDARY_LEVEL_3` |
    | `4`                                    |  17    | `LATERAL_BOUNDARY_LEVEL_4` |
    | `5`                                    |  18    | `LATERAL_BOUNDARY_LEVEL_5` |
    | `6`                                    |  19    | `LATERAL_BOUNDARY_LEVEL_6` |
    | `7`                                    |  20    | `LATERAL_BOUNDARY_LEVEL_7` |
    | `8`                                    |  21    | `LATERAL_BOUNDARY_LEVEL_8` |
    | `grf_bdywidth_e`   (9)                 |  22    | `NUDGING`                  |
    | `grf_bdywidth_e+1`, `max_rledge`  (10) |  23    | `NUDGING_LEVEL_2`          |


    """

    def __init__(self, name: str, level: int) -> None:
        self._name = name  # Use _name to avoid conflict with Enum's name
        self.level = level
        self._value_str = f"{name}_{level}" if level > 0 else name

    #: points to the number of entries in a local grid
    END = ("end", 0)

    #: interior unordered prognostic entries
    INTERIOR = ("interior", 0)

    #: first halo line
    HALO = ("halo_level", 1)

    #: 2nd halo line
    HALO_LEVEL_2 = ("halo_level", 2)

    #: all entries owned on the local grid, that is all entries excluding halo lines
    LOCAL = ("local", 0)

    #: lateral boundary (row 1) in LAM model
    LATERAL_BOUNDARY = ("lb_level", 1)

    #: lateral boundary (row 2) in LAM model
    LATERAL_BOUNDARY_LEVEL_2 = ("lb_level", 2)

    # ; lateral boundary (row 3) in LAM model
    LATERAL_BOUNDARY_LEVEL_3 = ("lb_level", 3)

    #: lateral boundary (row 4) in LAM model
    LATERAL_BOUNDARY_LEVEL_4 = ("lb_level", 4)

    #: lateral boundary (row 5) in LAM model
    LATERAL_BOUNDARY_LEVEL_5 = ("lb_level", 5)

    #: lateral boundary (row 6) in LAM model
    LATERAL_BOUNDARY_LEVEL_6 = ("lb_level", 6)

    #: lateral boundary (row 7) in LAM model
    LATERAL_BOUNDARY_LEVEL_7 = ("lb_level", 7)

    #: lateral boundary (row 8) in LAM model
    LATERAL_BOUNDARY_LEVEL_8 = ("lb_level", 8)

    #: nudging level in LAM model
    NUDGING = ("nudging_level", 1)

    #: 2nd nudging level in LAM model
    NUDGING_LEVEL_2 = ("nudging_level", 2)

    @property
    def value(self) -> str:
        return self._value_str

    def __str__(self) -> str:
        return self._value_str

    def __hash__(self) -> int:
        """Generate a hash based on the zone name and level."""
        return hash((self.name, self.level))

    def __eq__(self, other: Any) -> bool:
        """Check equality based on zone name and level."""
        if not isinstance(other, Zone):
            return False
        return (self.name, self.level) == (other.name, other.level)

    def is_halo(self) -> bool:
        return self in (Zone.HALO, Zone.HALO_LEVEL_2)

    def is_lateral_boundary(self) -> bool:
        return self in (
            Zone.LATERAL_BOUNDARY,
            Zone.LATERAL_BOUNDARY_LEVEL_2,
            Zone.LATERAL_BOUNDARY_LEVEL_3,
            Zone.LATERAL_BOUNDARY_LEVEL_4,
            Zone.LATERAL_BOUNDARY_LEVEL_5,
            Zone.LATERAL_BOUNDARY_LEVEL_6,
            Zone.LATERAL_BOUNDARY_LEVEL_7,
            Zone.LATERAL_BOUNDARY_LEVEL_8,
        )

    def is_nudging(self) -> bool:
        return self in (Zone.NUDGING, Zone.NUDGING_LEVEL_2)

    def is_local(self) -> bool:
        return self == Zone.LOCAL


VERTEX_AND_CELL_ZONES = (
    Zone.END,
    Zone.INTERIOR,
    Zone.HALO,
    Zone.HALO_LEVEL_2,
    Zone.LOCAL,
    Zone.LATERAL_BOUNDARY,
    Zone.LATERAL_BOUNDARY_LEVEL_2,
    Zone.LATERAL_BOUNDARY_LEVEL_3,
    Zone.LATERAL_BOUNDARY_LEVEL_4,
    Zone.NUDGING,
)


EDGE_ZONES = tuple(Zone)


def max_boundary_level(dim: gtx.Dimension) -> int:
    return max((d.level for d in _get_zones_for_dim(dim) if d.is_lateral_boundary()), default=1)


_ZONE_TO_INDEX_MAPPING = {
    Zone.END: lambda dim: _icon_domain_index(_ICON_END, dim),
    Zone.INTERIOR: lambda dim: _icon_domain_index(_ICON_INTERIOR, dim),
    Zone.HALO: lambda dim: _icon_domain_index(_ICON_HALO, dim),
    Zone.HALO_LEVEL_2: lambda dim: _icon_domain_index(_ICON_HALO, dim, -1),
    Zone.LOCAL: lambda dim: _icon_domain_index(ICON_LOCAL, dim),
    Zone.LATERAL_BOUNDARY: lambda dim: _icon_domain_index(_ICON_LATERAL_BOUNDARY, dim),
    Zone.LATERAL_BOUNDARY_LEVEL_2: lambda dim: _icon_domain_index(_ICON_LATERAL_BOUNDARY, dim, 1),
    Zone.LATERAL_BOUNDARY_LEVEL_3: lambda dim: _icon_domain_index(_ICON_LATERAL_BOUNDARY, dim, 2),
    Zone.LATERAL_BOUNDARY_LEVEL_4: lambda dim: _icon_domain_index(_ICON_LATERAL_BOUNDARY, dim, 3),
    Zone.LATERAL_BOUNDARY_LEVEL_5: lambda dim: _icon_domain_index(_ICON_LATERAL_BOUNDARY, dim, 4),
    Zone.LATERAL_BOUNDARY_LEVEL_6: lambda dim: _icon_domain_index(_ICON_LATERAL_BOUNDARY, dim, 5),
    Zone.LATERAL_BOUNDARY_LEVEL_7: lambda dim: _icon_domain_index(_ICON_LATERAL_BOUNDARY, dim, 6),
    Zone.LATERAL_BOUNDARY_LEVEL_8: lambda dim: _icon_domain_index(_ICON_LATERAL_BOUNDARY, dim, 7),
    Zone.NUDGING: lambda dim: _icon_domain_index(_ICON_NUDGING, dim),
    Zone.NUDGING_LEVEL_2: lambda dim: _icon_domain_index(_ICON_NUDGING, dim, 1),
}


def _map_zone_to_icon_array_index(dim: gtx.Dimension, zone: Zone) -> int:
    return _ZONE_TO_INDEX_MAPPING[zone](dim)


@dataclasses.dataclass(frozen=True)
class Domain:
    """
    Domain Description on the horizontal grid
    Used to access domain bounds in concrete the ICON grid.
    """

    dim: gtx.Dimension
    zone: Zone

    def __eq__(self, other: Any) -> bool:
        if isinstance(other, Domain):
            return self.dim == other.dim and self.zone == other.zone
        return False

    def __hash__(self) -> int:
        return hash((self.dim, self.zone))

    def __str__(self) -> str:
        return f"Domain (dim = {self.dim}: zone = {self.zone} /ICON index[ {_map_zone_to_icon_array_index(self.dim, self.zone)} ])"

    def __post_init__(self) -> None:
        assert _validate(self.dim, self.zone), (
            f"Invalid zone {self.zone} for dimension {self.dim}. Valid zones are: {_get_zones_for_dim(self.dim)}"
        )

    @functools.cached_property
    def is_local(self) -> bool:
        return self.zone.is_local()


def domain(dim: gtx.Dimension) -> Callable[[Zone], Domain]:
    """
    Factory function to create a domain object for a given dimension.

    This is the main entry point to create a domain object for a given dimension. In order to access the start or end index for
    `INTERIOR` (unordered prognostic) cells in ICON one would call:

    >>> import icon4py.model.common.grid.icon as icon_grid
    >>> grid = icon_grid.IconGrid()
    >>> domain = domain(dims.CellDim)(Zone.INTERIOR)
    >>> start_index = grid.start_index(domain)



    """

    def _domain(marker: Zone) -> Domain:
        return Domain(dim, marker)

    assert dim.kind == gtx.DimensionKind.HORIZONTAL, "Only defined for horizontal dimensions"
    return _domain


vertex_domain = domain(dims.VertexDim)
edge_domain = domain(dims.EdgeDim)
cell_domain = domain(dims.CellDim)


def _validate(dim: gtx.Dimension, marker: Zone) -> bool:
    return marker in _get_zones_for_dim(dim)


def _get_zones_for_dim(dim: gtx.Dimension) -> tuple[Zone, ...]:
    """
    Get the grid zones valid for a given horizontal dimension in ICON .
    """
    match dim:
        case dims.CellDim | dims.VertexDim:
            return VERTEX_AND_CELL_ZONES
        case dims.EdgeDim:
            return EDGE_ZONES
        case _:
            raise ValueError(
                f"Dimension should be one of {list(dims.horizontal_dims())} but was {dim}"
            )


def get_domains_for_dim(dim: gtx.Dimension) -> Iterator[Domain]:
    """
    Generate all grid Domains for a given dimension
    Args:
        dim: Dimension, one of CelLDim, EdgeDim, VertexDim

    Returns:

    """
    assert dim.kind == gtx.DimensionKind.HORIZONTAL, "Only horizontal dimension are allowed."
    get_domain = domain(dim)
    domains = (get_domain(zone) for zone in _get_zones_for_dim(dim))
    return domains


def get_halo_domains(dim: gtx.Dimension) -> Iterator[Domain]:
    get_domain = domain(dim)
    domains = (get_domain(zone) for zone in _get_zones_for_dim(dim) if zone.is_halo())
    return domains


def get_ordered_domains(dim: gtx.Dimension) -> Iterator[Domain]:
    get_domain = domain(dim)
    domains = (
        get_domain(zone)
        for zone in _get_zones_for_dim(dim)
        if zone.is_lateral_boundary() or zone.is_nudging()
    )
    return domains


def get_last_nudging(dim):
    zone = Zone.NUDGING if dim in (dims.VertexDim, dims.CellDim) else Zone.NUDGING_LEVEL_2
    return domain(dim)(zone)


def get_start_end_idx_from_icon_arrays(
    dim: gtx.Dimension,
    start_indices: dict[gtx.Dimension, np.ndarray],
    end_indices: dict[gtx.Dimension, np.ndarray],
) -> tuple[dict[Domain, gtx.int32], dict[Domain, gtx.int32]]:  # type: ignore [name-defined]
    """
    Translates ICON type start_idx and end_idx arrays to mapping of Domains to index values
    Args:
        dim: dimsension
        start_indices: icon type index arrays for start_idx
        end_indices: icon type index array for end_idx

    Returns: dict[Domain, gtx.int32] that can be used with the [base.py](Grid) for relevant domains for dimension dim

    """
    start = start_indices[dim]
    end = end_indices[dim]
    return _map_icon_array_to_domains(dim, start), _map_icon_array_to_domains(dim, end)


def _map_icon_array_to_domains(
    dim: gtx.Dimension, pre_computed_bounds: np.ndarray
) -> dict[Domain, gtx.int32]:  # type: ignore [name-defined]
    domains = get_domains_for_dim(dim)
    return {
        d: gtx.int32(pre_computed_bounds[_map_zone_to_icon_array_index(dim, d.zone)].item())  # type: ignore [attr-defined]
        for d in domains
    }
