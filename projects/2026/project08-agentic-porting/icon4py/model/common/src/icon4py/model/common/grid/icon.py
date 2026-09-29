# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import dataclasses
import enum
import logging
from collections.abc import Callable

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing

from icon4py.model.common import constants, dimension as dims
from icon4py.model.common.grid import base, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc


log = logging.getLogger(__name__)


class GeometryType(enum.Enum):
    """Define geometries of the horizontal domain supported by the ICON grid.

    Values are the same as mo_grid_geometry_info.f90.
    """

    ICOSAHEDRON = 1
    TORUS = 2


CONNECTIVITIES_ON_BOUNDARIES = (
    dims.C2E2C2EDim,
    dims.E2CDim,
    dims.C2E2CDim,
    dims.C2E2CODim,
    dims.E2C2VDim,
    dims.E2C2EDim,
    dims.E2C2EODim,
    dims.C2E2C2E2CDim,
)
CONNECTIVITIES_ON_PENTAGONS = (dims.V2EDim, dims.V2CDim, dims.V2E2VDim)


@dataclasses.dataclass(frozen=True, kw_only=True)
class GridSubdivision:
    root: int
    level: int


@dataclasses.dataclass(frozen=True, kw_only=True)
class IcosahedronParams:
    subdivision: GridSubdivision
    radius: float

    def __init__(
        self,
        *,
        subdivision: GridSubdivision,
        radius: float | None = None,
    ) -> None:
        object.__setattr__(self, "subdivision", subdivision)
        object.__setattr__(
            self,
            "radius",
            radius if radius is not None else constants.EARTH_RADIUS,
        )
        self.__post_init__()

    def __post_init__(self) -> None:
        if self.subdivision.root < 1 or self.subdivision.level < 0:
            raise ValueError(
                f"root must be >= 1 and level must be >= 0, got {self.subdivision.root=} and {self.subdivision.level=}"
            )

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.ICOSAHEDRON


@dataclasses.dataclass(frozen=True, kw_only=True)
class TorusParams:
    domain_length: float
    domain_height: float

    @property
    def geometry_type(self) -> GeometryType:
        return GeometryType.TORUS


@dataclasses.dataclass(frozen=True)
class GridParams:
    _params: IcosahedronParams | TorusParams | None = None

    @property
    def geometry_type(self) -> GeometryType | None:
        return self._params.geometry_type if self._params is not None else None

    @property
    def radius(self) -> float | None:
        return self._params.radius if isinstance(self._params, IcosahedronParams) else None

    @property
    def domain_length(self) -> float | None:
        return self._params.domain_length if isinstance(self._params, TorusParams) else None

    @property
    def domain_height(self) -> float | None:
        return self._params.domain_height if isinstance(self._params, TorusParams) else None

    @property
    def subdivision(self) -> GridSubdivision | None:
        return self._params.subdivision if isinstance(self._params, IcosahedronParams) else None


@dataclasses.dataclass(frozen=True)
class IconGrid(base.Grid):
    grid_params: GridParams = dataclasses.field(kw_only=True)
    refinement_control: dict[gtx.Dimension, gtx.Field] = dataclasses.field(
        default=None, kw_only=True
    )

    @property
    def geometry_type(self) -> GeometryType | None:
        return self.grid_params.geometry_type


def _has_skip_values(offset: gtx.FieldOffset, limited_area_or_distributed: bool) -> bool:
    """
    For the icosahedral global grid skip values are only present for the pentagon points.

    In the local area model or a distributed grid there are also skip values at the boundaries or halos when
    accessing neighbouring cells or edges from vertices.
    """
    dimension = offset.target[1]
    assert dimension.kind == gtx.DimensionKind.LOCAL, "only local dimensions can have skip values"
    return dimension in CONNECTIVITIES_ON_PENTAGONS or (
        limited_area_or_distributed and dimension in CONNECTIVITIES_ON_BOUNDARIES
    )


def _should_replace_skip_values(
    offset: gtx.FieldOffset, keep_skip_values: bool, limited_area_or_distributed: bool
) -> bool:
    """
    Check if the skip_values in a neighbor table  should be replaced.

    There are various reasons for skip_values in neighbor tables depending on the type of grid:
        - pentagon points (icosahedral grid),
        - boundary layers of limited area grids,
        - halos for distributed grids.

    There is config flag to evaluate whether skip_value replacement should be done at all.
    If so, we replace skip_values for halos and boundary layers of limited area grids.

    Even though by specifying the correct output domain of a stencil, access to
    invalid indices is avoided in the output fields, temporary computations
    inside a stencil do run over the entire data buffer including halos and boundaries
    as the output domain is unknown at that point.

    Args:
        dim: The (local) dimension for which the neighbor table is checked.
    Returns:
        bool: True if the skip values in the neighbor table should be replaced, False otherwise.

    """
    return not keep_skip_values and (
        limited_area_or_distributed or not _has_skip_values(offset, limited_area_or_distributed)
    )


def icon_grid(
    *,
    id_: str,
    allocator: gtx_typing.Allocator | None,
    config: base.GridConfig,
    neighbor_tables: dict[gtx.FieldOffset, data_alloc.NDArray],
    start_index: Callable[[h_grid.Domain], gtx.int32],
    end_index: Callable[[h_grid.Domain], gtx.int32],
    grid_params: GridParams,
    refinement_control: dict[gtx.Dimension, gtx.Field] | None = None,
) -> IconGrid:
    limited_area_or_distributed = config.limited_area or config.distributed
    connectivities = {
        offset.value: base.construct_connectivity(
            offset,
            data_alloc.import_array_ns(allocator).asarray(table),
            skip_value=-1 if _has_skip_values(offset, limited_area_or_distributed) else None,
            allocator=allocator,
            replace_skip_values=_should_replace_skip_values(
                offset, config.keep_skip_values, limited_area_or_distributed
            ),
        )
        for offset, table in neighbor_tables.items()
    }
    return IconGrid(
        id=id_,
        config=config,
        connectivities=connectivities,
        start_index=start_index,
        end_index=end_index,
        grid_params=grid_params,
        refinement_control=refinement_control or {},
    )


def get_start_and_end_index(
    constructor: Callable[
        [gtx.Dimension], tuple[dict[h_grid.Domain, gtx.int32], dict[h_grid.Domain, gtx.int32]]
    ],
) -> tuple[Callable[[h_grid.Domain], gtx.int32], Callable[[h_grid.Domain], gtx.int32]]:
    """
    Return start_index and end_index functions to be passed to the Grid constructor.

    This function defines a version of `start_index` and `end_index` that looks up the indeces in an internal map from [Domain](horizontal.py::Domain) -> gtx.int32
    It takes the constructor function of this map as input.

    Args:
        constructor: function that takes a dimension as argument and constructs  a lookup table
        dict[Domain, gtx.int32] for all domains for a given dimension

    Returns:
        tuple of functions `start_index` and `end_index` to be passed to the [Grid](./base.py::Grid)

    """
    start_indices = {}
    end_indices = {}
    for dim in dims.horizontal_dims():
        start_map, end_map = constructor(dim)
        start_indices.update(start_map)
        end_indices.update(end_map)

    def start_index(domain: h_grid.Domain) -> gtx.int32:
        return start_indices[domain]

    def end_index(domain: h_grid.Domain) -> gtx.int32:
        return end_indices[domain]

    return start_index, end_index
