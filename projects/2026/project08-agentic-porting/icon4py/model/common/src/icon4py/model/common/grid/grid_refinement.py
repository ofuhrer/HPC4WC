# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import logging
from typing import Final

from gt4py import next as gtx

from icon4py.model.common import dimension as dims
from icon4py.model.common.decomposition import definitions as decomposition
from icon4py.model.common.grid import gridfile, horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc


"""
Refinement control for ICON grid.

Grid refinement is used in the context of
- local area grids to determine the type of a grid point: its distance from the boundary
- nested horizontal grids to determine the nested overlap regions for feedback
- domain decomposition to order grid points

See Zaengl et al. Grid Refinement in ICON v2.6.4 (Geosci. Model Dev., 15, 7153-7176, 202)

In ICON grid refinement is largely concerned with the nesting of grids an reordering of fields
as a consequence, ICON4Py does not support nested grids, so we we ignore this aspect here and focus on
functionality needed for single grid ordering and decomposition.

"""
_log = logging.getLogger(__name__)

_MAX_ORDERED: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: gridfile.FixedSizeDimension.CELL_GRF.size,
    dims.EdgeDim: gridfile.FixedSizeDimension.EDGE_GRF.size,
    dims.VertexDim: gridfile.FixedSizeDimension.VERTEX_GRF.size,
}
r"""
Grid points in the grid refinement fields are labeled with their distance to the lateral boundary.
The distance is counted in "rows" and distances up to _MAX_BOUNDARY_DISTANCE are computed, elements
at distances larger than _MAX_BOUNDARY_DISTANCE are labeled as _UNORDERED

**Rows** can be defined like:
### CELLS:
1. row: describes all cells that share *at least* 1 vertex with the lateral boundary
2. row: shares at least 1 vertex with the 1. row
....

          xxxx  xxxx  xxxx    ← Lateral boundary
        /\ 1  /\ 1  /\ 1  /    ← Row 1
       /1 \  /1 \  /1 \  /     ← Row 1
      /____\/____\/____\/
     /\ 2  /\ 2  /\ 2  /      ← Row 2 (downward triangles)
    /2 \  /2 \  /2 \  /       ← Row 2 (updward triangles)
   /____\/____\/____\/
  /\    /\    /\    /
 /  \  /  \  /  \  /
/____\/____\/____\/


### VERTEX:
1. row: vertex on the lateral boundary
2. row: shared between cell of 1. row and 2. row cell.
...

         1 xxxx 1 xxxxx 1 xxxxx  ← Lateral boundary, row 1 VERTICES
        / \    / \     / \    /
       /   \  /   \   /   \  /   <- row 1 CELLS
      /     \/     \ /     \/
      2------ 2------2------2  <- row 2 VERTICES
     / \    / \    / \    /
    /   \  /   \  /   \  /    <- row 2 cells
   /     \/     \/     \/
   3------3------3------3 <- row 3 VERTICES
  / \    / \    / \    /
 /   \  /   \  /   \  /
/_____\/_____\/_____\/

## EDGES:
1. row: edge on boundary
2. row: edge only between cells of 1. row
3. row: edge between cells of 1. row and 2. row
4. row: edges only between 2. row cells
...


         xx 1 xx xx 1 xx xx 1 xx ← Lateral boundary, row 1 EDGES
        / \    / \     / \    /
       2   2  2   2   2   2  2   <- row 2 EDGES on row 1 cells
      /     \/     \ /     \/
      ---3-------3------3---     <- row 3 EDGES between row 1 and row 2 cells
     / \    / \    / \    /
    4   4  4   4  4   4  4       <- row 4 EDGES
   /     \/     \/     \/
   ---5-------5-----5---         <- row 5 EDGES
  / \    / \    / \    /
 6   6  6   6  6   6  6
/__7__\/__7__\/__7__\/

As a consequence there are twice as many edge-rows than cells and vertices and
equal edge rows (2, 4, ...) tend to contain twice as many elements as unequal rows


Grid points can be ordered (moved to the beginning of an field array) up to any distance <= MAX_BOUNDARY_DISTANCE in the grid-generators, the exact number of
ordered rows is a parameter to the grid generation.
"""

_GRID_REFINEMENT_BOUNDARY_WIDTH: Final[dict[gtx.Dimension, int]] = {
    dims.CellDim: 4,
    dims.EdgeDim: 9,
    dims.VertexDim: 4,
}
"""Number of rows used in grid refinement in the ICON model:
The ICON fields are ordered up to this BOUNDARY_DISTANCE.
Elements with a boundary distance larger than _GRID_REFINEMENT_BOUNDARY_WITH[dim] are interspersed with the _UNORDERED
elements in the data buffers. The memory layout of ICON fields looks roughly like this

data buffer index:                        |0, 1, ... |start_index(Domain(dim, LATERAL_BOUNDARY_LEVEL_2) .... | ... ...   | start_index(Domain(dim, INTERIOR) ... | start_index(dim, HALO) ...|
refinement value of elements in buffer:   |1, 1, ... |2, 2, ...                                              | ... ...   | _GRID_REFINEMENT_BOUNDARY_WITH[dim]   | (0, i > _GRID_REFINEMENT_BOUNDARY_WITH[dim]) | (0, i > _GRID_REFINEMENT_BOUNDARY_WITH[dim])

"""

_UNORDERED: Final[dict[gtx.Dimension, tuple[int, int]]] = {
    dims.CellDim: (0, -4),
    dims.EdgeDim: (0, -8),
    dims.VertexDim: (0, -4),
}
"""Refinement value indicating a point is in the unordered interior (fully prognostic) of the grid: this is encoded by 0 or -4 in coarser parent grid."""

_MIN_ORDERED: Final[dict[gtx.Dimension, int]] = {
    dim: value[1] + 1 for dim, value in _UNORDERED.items()
}
"""For coarse parent grids the refinement control value of overlapping boundary regions are counted with negative values.
"""


DEFAULT_GRF_NUDGEZONE_WIDTH: Final[int] = 8


_LAST_NUDGING: dict[gtx.Dimension, h_grid.Zone] = {
    dims.EdgeDim: h_grid.Zone.NUDGING_LEVEL_2,
    dims.CellDim: h_grid.Zone.NUDGING,
    dims.VertexDim: h_grid.Zone.NUDGING,
}
_LAST_BOUNDARY: dict[gtx.Dimension, h_grid.Zone] = {
    dims.EdgeDim: h_grid.Zone.LATERAL_BOUNDARY_LEVEL_8,
    dims.CellDim: h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4,
    dims.VertexDim: h_grid.Zone.LATERAL_BOUNDARY_LEVEL_4,
}


def _refinement_level_placed_with_halo(domain: h_grid.Domain) -> int:
    """There is a speciality in the setup of the ICON halos: generally halo points are located at the end of the arrays after all
    points owned by a node. This is true for global grids and for a local area model for all points with a
    refinement control value larger than (6, 2) for HALO level 1 and (4,1) for HALO_LEVEL_2.

    That is for the local area grid some (but not all!) halo points that are lateral boundary points are placed with the lateral boundary
    domains rather than the halo. The reason for this is mysterious (to me) as well as what advantage it might have.

    Disadvantage clearly is that in the LAM case the halos are **not contigous**.
    """
    assert domain.zone.is_halo(), "Domain must be a halo Zone."
    match domain.dim:
        case dims.EdgeDim:
            return 6 if domain.zone == h_grid.Zone.HALO else 4
        case dims.CellDim | dims.VertexDim:
            return 2 if domain.zone == h_grid.Zone.HALO else 1
        case _:
            raise ValueError(f"Invalid domain: {domain}, must be a HALO domain")


def compute_domain_bounds(
    dim: gtx.Dimension,
    refinement_fields: dict[gtx.Dimension, gtx.Field],
    decomposition_info: decomposition.DecompositionInfo,
) -> tuple[dict[h_grid.Domain, gtx.int32], dict[h_grid.Domain, gtx.int32]]:  # type: ignore   [name-defined]
    """
    Compute the domain bounds (start_index, end_index) based on a grid Domain.

    In a local area model, ICON orders the field arrays according to their distance from the boundary. For each
    dimension (cell, vertex, edge) points are "ordered" (moved to the beginning of the array) up to
    the values defined in _GRID_REFINEMENT_BOUNDARY_WIDTH. We call these distance a Grid Zone. The `Dimension`
    and the `Zone` determine a Grid `Domain`. For a given field values for a domain are located
    in a contiguous section of the field array.

    This function can be used to deterine the start_index and end_index of a `Domain`in the arrays.

    For a global model grid all points are unordered. `Zone` that do not exist for a global model grid
    return empty domains.

    For distributed grids halo points build their own `Domain`and are located at the end of the field arrays, with the exception of
    some points in the lateral boundary as described in (_refinement_level_placed_with_halo)

    Args:
        dim: Dimension one of `CellDim`. `VertexDim`, `EdgeDim`
        refinement_fields: dict[Dimension, ndarray] containing the refinement_control values for each dimension
        decomposition_info: DecompositionInfo needed to determine the HALO `Zone`s

    """
    assert dim.kind == gtx.DimensionKind.HORIZONTAL, (
        f"Dimension must be one of {list(dims.horizontal_dims())}"
    )
    refinement_ctrl = convert_to_non_nested_refinement_values(refinement_fields[dim].ndarray, dim)
    array_ns = data_alloc.array_namespace(refinement_ctrl)
    owned = decomposition_info.owner_mask(dim)
    halo_level_1 = decomposition_info.halo_level_mask(
        dim, decomposition.DecompositionFlag.FIRST_HALO_LEVEL
    )
    halo_level_2 = decomposition_info.halo_level_mask(
        dim, decomposition.DecompositionFlag.SECOND_HALO_LEVEL
    )

    start_indices = {}
    end_indices = {}

    end_domain = h_grid.domain(dim)(h_grid.Zone.END)
    start_indices[end_domain] = gtx.int32(refinement_ctrl.shape[0])
    end_indices[end_domain] = gtx.int32(refinement_ctrl.shape[0])

    halo_domains = h_grid.get_halo_domains(dim)
    upper_boundary_level_1 = _refinement_level_placed_with_halo(
        h_grid.domain(dim)(h_grid.Zone.HALO)
    )
    not_lateral_boundary_1 = (refinement_ctrl < 1) | (refinement_ctrl > upper_boundary_level_1)
    halo_region_1 = array_ns.nonzero(halo_level_1 & not_lateral_boundary_1)[0]
    not_lateral_boundary_2 = (refinement_ctrl < 1) | (
        refinement_ctrl
        > _refinement_level_placed_with_halo(h_grid.domain(dim)(h_grid.Zone.HALO_LEVEL_2))
    )

    halo_region_2 = array_ns.nonzero(halo_level_2 & not_lateral_boundary_2)[0]
    start_halo_2, end_halo_2 = (
        (array_ns.min(halo_region_2).item(), array_ns.max(halo_region_2).item() + 1)
        if halo_region_2.size > 0
        else (refinement_ctrl.size, refinement_ctrl.size)
    )
    for domain in halo_domains:
        my_flag = decomposition.DecompositionFlag(domain.zone.level)
        if my_flag == h_grid.Zone.HALO.level:
            start_index = (
                array_ns.min(halo_region_1).item() if halo_region_1.size > 0 else start_halo_2
            )
            end_index = start_halo_2
        else:
            start_index = start_halo_2
            end_index = end_halo_2
        start_indices[domain] = gtx.int32(start_index)  # type: ignore [attr-defined]
        end_indices[domain] = gtx.int32(end_index)  # type: ignore [attr-defined]

    ordered_domains = h_grid.get_ordered_domains(dim)
    for domain in ordered_domains:
        value = (
            domain.zone.level
            if domain.zone.is_lateral_boundary()
            else _LAST_BOUNDARY[dim].level + domain.zone.level
        )
        found = array_ns.nonzero((refinement_ctrl == value) & owned)[0]
        start_index, end_index = (
            (array_ns.min(found).item(), array_ns.max(found).item() + 1)
            if found.size > 0
            else (0, 0)
        )
        start_indices[domain] = gtx.int32(start_index)
        end_indices[domain] = gtx.int32(end_index)

    interior_domain = h_grid.domain(dim)(h_grid.Zone.INTERIOR)
    # for the Vertex and Edges the level after the nudging zones are not ordered anymore, so
    # we rely on using the end index of the nudging zone for INTERIOR
    nudging = h_grid.get_last_nudging(dim)
    start_indices[interior_domain] = end_indices[nudging]
    halo_1 = h_grid.domain(dim)(h_grid.Zone.HALO)
    end_indices[interior_domain] = start_indices[halo_1]

    local_domain = h_grid.domain(dim)(h_grid.Zone.LOCAL)
    start_indices[local_domain] = gtx.int32(0)
    end_indices[local_domain] = start_indices[halo_1]

    return start_indices, end_indices


def get_nudging_refinement_value(dim: gtx.Dimension) -> int:
    return _LAST_BOUNDARY[dim].level + _LAST_NUDGING[dim].level


def is_unordered_field(field: data_alloc.NDArray, dim: gtx.Dimension) -> data_alloc.NDArray:
    assert field.dtype in (gtx.int32, gtx.int64), f"not an integer type {field.dtype}"  # type: ignore [attr-defined]
    array_ns = data_alloc.array_namespace(field)
    return array_ns.isin(field, _UNORDERED[dim])


def convert_to_non_nested_refinement_values(
    field: data_alloc.NDArray, dim: gtx.Dimension
) -> data_alloc.NDArray:
    """Convenience function that converts the grid refinement value from a coarser
    parent grid to the canonical values used in an unnested setup.

    The nested values are used for example in the radiation grids.
    """
    assert field.dtype in (gtx.int32, gtx.int64), f"not an integer type {field.dtype}"  # type: ignore [attr-defined]
    array_ns = data_alloc.array_namespace(field)
    return array_ns.where(field == _UNORDERED[dim][1], 0, array_ns.where(field < 0, -field, field))


def is_limited_area_grid(
    refinement_field: data_alloc.NDArray,
) -> bool:
    """Check if the grid is a local area grid.

    This is done by checking whether there are Boundary points (Refinement value > 0) in the grid.
    The .item() call is needed to get a scalar return for cupy arrays.

    It only operates works for non nested grids. It should be used like this

    >>> refinement_edges: fa.EdgeField[int] = grid.refinement_control[dims.EdgeDim]
    >>> non_nested_edge_refinement = convert_to_non_nested_refinement_values(
    ...     refinement_edges, dims.EdgeDim
    ... )
    >>> assert is_limited_area_grid(non_nested_edge_refinement)
    """
    return (refinement_field > 0).any().item()
