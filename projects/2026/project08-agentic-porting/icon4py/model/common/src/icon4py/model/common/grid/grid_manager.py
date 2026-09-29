# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import functools
import logging
import pathlib
from types import ModuleType
from typing import Literal

import gt4py.next as gtx
import gt4py.next.typing as gtx_typing
import numpy as np

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.decomposition import (
    decomposer as decomp,
    definitions as decomposition,
    halo,
)
from icon4py.model.common.exceptions import InvalidConfigError
from icon4py.model.common.grid import (
    base,
    grid_refinement as refinement,
    gridfile,
    icon,
    vertical as v_grid,
)
from icon4py.model.common.utils import data_allocation as data_alloc


_log = logging.getLogger(__name__)
_single_node_decomposer = decomp.SingleNodeDecomposer()
_single_process_props = decomposition.SingleNodeProcessProperties()
_fortran_to_python_transformer = gridfile.ToZeroBasedIndexTransformation()


class IconGridError(RuntimeError):
    pass


type CoordinateDict = dict[gtx.Dimension, dict[Literal["lat", "lon", "x", "y", "z"], gtx.Field]]
# TODO (halungge): use a TypeDict for that
type GeometryDict = dict[gridfile.GeometryName, gtx.Field]


class GridManager:
    """
    Read ICON grid file and set up grid topology, refinement information and geometry fields.

    It handles the reading of the ICON grid file and extracts information such as:
    - topology (connectivity arrays)
    - refinement information: association of field positions to specific zones in the horizontal grid like boundaries, inner prognostic cells, etc.
    - geometry fields present in the grid file


    """

    def __init__(
        self,
        grid_file: pathlib.Path | str,
        config: v_grid.VerticalGridConfig,  # TODO(msimberg): remove to separate vertical and horizontal grid
        offset_transformation: gridfile.IndexTransformation = _fortran_to_python_transformer,
    ):
        self._offset_transformation = offset_transformation
        self._file_name = str(grid_file)
        self._vertical_config = config
        # Output
        self._grid: icon.IconGrid | None = None
        self._decomposition_info: decomposition.DecompositionInfo | None = None
        self._geometry: GeometryDict = {}
        self._coordinates: CoordinateDict = {}
        self._reader = None

    def open(self):
        """Open the gridfile resource for reading."""
        self._reader = gridfile.GridFile(self._file_name, self._offset_transformation)
        self._reader.open()

    def close(self):
        """close the gridfile resource."""
        self._reader.close()

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        if exc_type is not None:
            _log.debug(
                f"Exception '{exc_type}: {exc_val}' while reading the grid file {self._file_name}"
            )
        if exc_type is FileNotFoundError:
            raise FileNotFoundError(f"gridfile {self._file_name} not found, aborting")

    def __call__(
        self,
        allocator: gtx_typing.Allocator | None,
        keep_skip_values: bool,
        decomposer: decomp.Decomposer = _single_node_decomposer,
        process_props: decomposition.ProcessProperties = _single_process_props,
    ) -> None:
        if not process_props.is_single_rank() and isinstance(
            decomposer, decomp.SingleNodeDecomposer
        ):
            raise InvalidConfigError("Need a Decomposer for multi node run")

        if not self._reader:
            self.open()

        if geometry_type := self._reader.try_attribute(gridfile.MPIMPropertyName.GEOMETRY):
            geometry_type = icon.GeometryType(geometry_type)
        else:
            geometry_type = icon.GeometryType.ICOSAHEDRON

        self._construct_decomposed_grid(
            allocator=allocator,
            keep_skip_values=keep_skip_values,
            geometry_type=geometry_type,
            decomposer=decomposer,
            process_props=process_props,
        )
        self._coordinates = self._read_coordinates(allocator, geometry_type)
        self._geometry = self._read_geometry_fields(allocator)

        self.close()

    def _read_coordinates(
        self,
        allocator: gtx_typing.Allocator,
        geometry_type: icon.GeometryType,
    ) -> CoordinateDict:
        my_cell_indices = self._decomposition_info.global_index(dims.CellDim)
        my_edge_indices = self._decomposition_info.global_index(dims.EdgeDim)
        my_vertex_indices = self._decomposition_info.global_index(dims.VertexDim)
        coordinates = {
            dims.CellDim: {
                "lat": gtx.as_field(
                    (dims.CellDim,),
                    self._reader.variable(
                        gridfile.CoordinateName.CELL_LATITUDE, indices=my_cell_indices
                    ),
                    dtype=ta.wpfloat,
                    allocator=allocator,
                ),
                "lon": gtx.as_field(
                    (dims.CellDim,),
                    self._reader.variable(
                        gridfile.CoordinateName.CELL_LONGITUDE, indices=my_cell_indices
                    ),
                    dtype=ta.wpfloat,
                    allocator=allocator,
                ),
            },
            dims.EdgeDim: {
                "lat": gtx.as_field(
                    (dims.EdgeDim,),
                    self._reader.variable(
                        gridfile.CoordinateName.EDGE_LATITUDE, indices=my_edge_indices
                    ),
                    dtype=ta.wpfloat,
                    allocator=allocator,
                ),
                "lon": gtx.as_field(
                    (dims.EdgeDim,),
                    self._reader.variable(
                        gridfile.CoordinateName.EDGE_LONGITUDE, indices=my_edge_indices
                    ),
                    dtype=ta.wpfloat,
                    allocator=allocator,
                ),
            },
            dims.VertexDim: {
                "lat": gtx.as_field(
                    (dims.VertexDim,),
                    self._reader.variable(
                        gridfile.CoordinateName.VERTEX_LATITUDE, indices=my_vertex_indices
                    ),
                    allocator=allocator,
                    dtype=ta.wpfloat,
                ),
                "lon": gtx.as_field(
                    (dims.VertexDim,),
                    self._reader.variable(
                        gridfile.CoordinateName.VERTEX_LONGITUDE, indices=my_vertex_indices
                    ),
                    allocator=allocator,
                    dtype=ta.wpfloat,
                ),
            },
        }

        if geometry_type == icon.GeometryType.TORUS:
            coordinates[dims.CellDim]["x"] = gtx.as_field(
                (dims.CellDim,),
                self._reader.variable(gridfile.CoordinateName.CELL_X, indices=my_cell_indices),
                dtype=ta.wpfloat,
                allocator=allocator,
            )
            coordinates[dims.CellDim]["y"] = gtx.as_field(
                (dims.CellDim,),
                self._reader.variable(gridfile.CoordinateName.CELL_Y, indices=my_cell_indices),
                dtype=ta.wpfloat,
                allocator=allocator,
            )
            coordinates[dims.CellDim]["z"] = gtx.as_field(
                (dims.CellDim,),
                self._reader.variable(gridfile.CoordinateName.CELL_Z, indices=my_cell_indices),
                dtype=ta.wpfloat,
                allocator=allocator,
            )
            coordinates[dims.EdgeDim]["x"] = gtx.as_field(
                (dims.EdgeDim,),
                self._reader.variable(gridfile.CoordinateName.EDGE_X, indices=my_edge_indices),
                dtype=ta.wpfloat,
                allocator=allocator,
            )
            coordinates[dims.EdgeDim]["y"] = gtx.as_field(
                (dims.EdgeDim,),
                self._reader.variable(gridfile.CoordinateName.EDGE_Y, indices=my_edge_indices),
                dtype=ta.wpfloat,
                allocator=allocator,
            )
            coordinates[dims.EdgeDim]["z"] = gtx.as_field(
                (dims.EdgeDim,),
                self._reader.variable(gridfile.CoordinateName.EDGE_Z, indices=my_edge_indices),
                dtype=ta.wpfloat,
                allocator=allocator,
            )
            coordinates[dims.VertexDim]["x"] = gtx.as_field(
                (dims.VertexDim,),
                self._reader.variable(gridfile.CoordinateName.VERTEX_X, indices=my_vertex_indices),
                dtype=ta.wpfloat,
                allocator=allocator,
            )
            coordinates[dims.VertexDim]["y"] = gtx.as_field(
                (dims.VertexDim,),
                self._reader.variable(gridfile.CoordinateName.VERTEX_Y, indices=my_vertex_indices),
                dtype=ta.wpfloat,
                allocator=allocator,
            )
            coordinates[dims.VertexDim]["z"] = gtx.as_field(
                (dims.VertexDim,),
                self._reader.variable(gridfile.CoordinateName.VERTEX_Z, indices=my_vertex_indices),
                dtype=ta.wpfloat,
                allocator=allocator,
            )

        return coordinates

    def _read_geometry_fields(
        self,
        allocator: gtx_typing.Allocator,
    ) -> GeometryDict:
        my_cell_indices = self._decomposition_info.global_index(dims.CellDim)
        my_edge_indices = self._decomposition_info.global_index(dims.EdgeDim)
        my_vertex_indices = self._decomposition_info.global_index(dims.VertexDim)
        return {
            # TODO(halungge): still needs to ported, values from "our" grid files contains (wrong) values:
            #   based on bug in generator fixed with this [PR40](https://gitlab.dkrz.de/dwd-sw/dwd_icon_tools/-/merge_requests/40) .
            gridfile.GeometryName.CELL_AREA.value: gtx.as_field(
                (dims.CellDim,),
                self._reader.variable(gridfile.GeometryName.CELL_AREA, indices=my_cell_indices),
                allocator=allocator,
            ),
            # TODO(halungge): easily computed from a neighbor_sum V2C over the cell areas?
            gridfile.GeometryName.DUAL_AREA.value: gtx.as_field(
                (dims.VertexDim,),
                self._reader.variable(gridfile.GeometryName.DUAL_AREA, indices=my_vertex_indices),
                allocator=allocator,
            ),
            gridfile.GeometryName.EDGE_LENGTH.value: gtx.as_field(
                (dims.EdgeDim,),
                self._reader.variable(gridfile.GeometryName.EDGE_LENGTH, indices=my_edge_indices),
                allocator=allocator,
            ),
            gridfile.GeometryName.DUAL_EDGE_LENGTH.value: gtx.as_field(
                (dims.EdgeDim,),
                self._reader.variable(
                    gridfile.GeometryName.DUAL_EDGE_LENGTH, indices=my_edge_indices
                ),
                allocator=allocator,
            ),
            gridfile.GeometryName.EDGE_CELL_DISTANCE.value: gtx.as_field(
                (dims.EdgeDim, dims.E2CDim),
                self._reader.variable(
                    gridfile.GeometryName.EDGE_CELL_DISTANCE,
                    transpose=True,
                    indices=my_edge_indices,
                ),
                allocator=allocator,
            ),
            gridfile.GeometryName.EDGE_VERTEX_DISTANCE.value: gtx.as_field(
                (dims.EdgeDim, dims.E2VDim),
                self._reader.variable(
                    gridfile.GeometryName.EDGE_VERTEX_DISTANCE,
                    transpose=True,
                    indices=my_edge_indices,
                ),
                allocator=allocator,
            ),
            # TODO(halungge): recompute from coordinates? field in gridfile contains NaN on boundary edges
            gridfile.GeometryName.TANGENT_ORIENTATION.value: gtx.as_field(
                (dims.EdgeDim,),
                self._reader.variable(
                    gridfile.GeometryName.TANGENT_ORIENTATION, indices=my_edge_indices
                ),
                allocator=allocator,
            ),
            gridfile.GeometryName.CELL_NORMAL_ORIENTATION.value: gtx.as_field(
                (dims.CellDim, dims.C2EDim),
                self._reader.variable(
                    gridfile.GeometryName.CELL_NORMAL_ORIENTATION,
                    transpose=True,
                    indices=my_cell_indices,
                ),
                allocator=allocator,
            ),
            gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX.value: gtx.as_field(
                (dims.VertexDim, dims.V2EDim),
                self._reader.int_variable(
                    gridfile.GeometryName.EDGE_ORIENTATION_ON_VERTEX,
                    transpose=True,
                    apply_offset=False,
                    indices=my_vertex_indices,
                ),
                allocator=allocator,
            ),
        }

    def _read_grid_refinement_fields(
        self,
        allocator: gtx_typing.Allocator,
    ) -> dict[gtx.Dimension, gtx.Field]:
        """
        Reads the refinement control fields from the grid file.

        Refinement control contains the classification of each entry in a field to predefined horizontal grid zones as for example the distance to the boundaries,
        see [grid_refinement.py](grid_refinement.py)

        Args:
            allocator: Allocator to use for reading the fields.
        Returns:
            dict[gtx.Dimension, gtx.Field]: A dictionary containing the refinement control fields for each dimension.
        """
        refinement_control_names = {
            dims.CellDim: gridfile.GridRefinementName.CONTROL_CELLS,
            dims.EdgeDim: gridfile.GridRefinementName.CONTROL_EDGES,
            dims.VertexDim: gridfile.GridRefinementName.CONTROL_VERTICES,
        }
        refinement_control_fields = {
            dim: gtx.as_field(
                (dim,),
                self._reader.int_variable(
                    name,
                    indices=self._decomposition_info.global_index(dim),
                    transpose=False,
                    apply_offset=False,
                ),
                allocator=allocator,
            )
            for dim, name in refinement_control_names.items()
        }
        return refinement_control_fields

    @property
    def file_path(self) -> str:
        """Path of the ICON grid file this manager reads from."""
        return self._file_name

    @property
    def grid(self) -> icon.IconGrid:
        return self._grid

    @property
    def geometry_fields(self) -> GeometryDict:
        return self._geometry

    @property
    def coordinates(self) -> CoordinateDict:
        return self._coordinates

    @property
    def decomposition_info(self) -> decomposition.DecompositionInfo:
        return self._decomposition_info

    def _construct_decomposed_grid(
        self,
        allocator: gtx_typing.Allocator | None,
        keep_skip_values: bool,
        geometry_type: icon.GeometryType,
        decomposer: decomp.Decomposer,
        process_props: decomposition.ProcessProperties,
    ) -> None:
        """Construct the grid topology from the icon grid file.

        Reads connectivity fields from the grid file and constructs derived
        connectivities needed in Icon4py from them. Adds constructed start/end
        index information to the grid. The grid will be distributed or not based
        on process_props.

        """
        xp = data_alloc.import_array_ns(allocator)
        ## FULL GRID PROPERTIES
        cell_refinement = xp.asarray(
            self._reader.variable(gridfile.GridRefinementName.CONTROL_CELLS)
        )
        global_size = self._read_full_grid_size()
        global_params = self._construct_grid_params(geometry_type)
        limited_area = refinement.is_limited_area_grid(cell_refinement)

        if limited_area and not process_props.is_single_rank():
            raise NotImplementedError("Limited-area grids are not supported in distributed runs")

        cell_to_cell_neighbors = self._get_index_field(gridfile.ConnectivityName.C2E2C, array_ns=xp)
        global_neighbor_tables = {
            dims.C2E2C: cell_to_cell_neighbors,
            dims.C2E: self._get_index_field(gridfile.ConnectivityName.C2E, array_ns=xp),
            dims.E2C: self._get_index_field(gridfile.ConnectivityName.E2C, array_ns=xp),
            dims.V2E: self._get_index_field(gridfile.ConnectivityName.V2E, array_ns=xp),
            dims.V2C: self._get_index_field(gridfile.ConnectivityName.V2C, array_ns=xp),
            dims.C2V: self._get_index_field(gridfile.ConnectivityName.C2V, array_ns=xp),
            dims.V2E2V: self._get_index_field(gridfile.ConnectivityName.V2E2V, array_ns=xp),
            dims.E2V: self._get_index_field(gridfile.ConnectivityName.E2V, array_ns=xp),
        }

        cells_to_rank_mapping = decomposer(cell_to_cell_neighbors, process_props.comm_size)
        # HALO CONSTRUCTION
        # TODO(halungge): reduce the set of neighbor tables used in the halo construction
        # TODO(halungge): figure out where to do the host to device copies (xp.asarray...)
        halo_constructor = halo.get_halo_constructor(
            process_props=process_props,
            full_grid_size=global_size,
            connectivities=global_neighbor_tables,
            allocator=allocator,
        )

        self._decomposition_info = halo_constructor(cells_to_rank_mapping)
        distributed_size = self._decomposition_info.get_horizontal_size()

        neighbor_tables = self._get_local_connectivities(global_neighbor_tables)

        # COMPUTE remaining derived connectivities
        neighbor_tables.update(_get_derived_connectivities(neighbor_tables))

        refinement_fields = self._read_grid_refinement_fields(allocator)

        domain_bounds_constructor = functools.partial(
            refinement.compute_domain_bounds,
            refinement_fields=refinement_fields,
            decomposition_info=self._decomposition_info,
        )
        start_index, end_index = icon.get_start_and_end_index(domain_bounds_constructor)

        grid_config = base.GridConfig(
            horizontal_config=distributed_size,
            vertical_size=self._vertical_config.num_levels,
            limited_area=limited_area,
            distributed=not process_props.is_single_rank(),
            keep_skip_values=keep_skip_values,
        )

        self._grid = icon.icon_grid(
            id_=self._reader.attribute(gridfile.MandatoryPropertyName.GRID_UUID),
            allocator=allocator,
            config=grid_config,
            neighbor_tables=neighbor_tables,
            start_index=start_index,
            end_index=end_index,
            grid_params=global_params,
            refinement_control=refinement_fields,
        )

    def _get_local_connectivities(
        self,
        neighbor_tables_global: dict[gtx.FieldOffset, data_alloc.NDArray],
    ) -> dict[gtx.FieldOffset, data_alloc.NDArray]:
        if self.decomposition_info.is_distributed():
            return {
                k: halo.global_to_local(
                    self._decomposition_info.global_index(k.source),
                    v[self._decomposition_info.global_index(k.target[0])],
                )
                for k, v in neighbor_tables_global.items()
            }
        else:
            return neighbor_tables_global

    def _construct_grid_params(self, geometry_type: icon.GeometryType):
        grid_root = self._reader.attribute(gridfile.MandatoryPropertyName.ROOT)
        grid_level = self._reader.attribute(gridfile.MandatoryPropertyName.LEVEL)
        sphere_radius = self._reader.try_attribute(gridfile.MPIMPropertyName.SPHERE_RADIUS)
        domain_length = self._reader.try_attribute(gridfile.MPIMPropertyName.DOMAIN_LENGTH)
        domain_height = self._reader.try_attribute(gridfile.MPIMPropertyName.DOMAIN_HEIGHT)

        match geometry_type:
            case icon.GeometryType.ICOSAHEDRON:
                return icon.GridParams(
                    icon.IcosahedronParams(
                        subdivision=icon.GridSubdivision(root=grid_root, level=grid_level),
                        radius=sphere_radius,
                    ),
                )
            case icon.GeometryType.TORUS:
                return icon.GridParams(
                    icon.TorusParams(
                        domain_length=domain_length,
                        domain_height=domain_height,
                    ),
                )

    def _read_full_grid_size(self) -> base.HorizontalGridSize:
        """
        Read the grid size propertes (cells, edges, vertices) from the grid file.

        As the grid file contains the _full_ (non-distributed) grid, these are the sizes of prior to distribution.

        """
        num_cells = self._reader.dimension(gridfile.DynamicDimension.CELL_NAME)
        num_edges = self._reader.dimension(gridfile.DynamicDimension.EDGE_NAME)
        num_vertices = self._reader.dimension(gridfile.DynamicDimension.VERTEX_NAME)
        full_grid_size = base.HorizontalGridSize(
            num_vertices=num_vertices, num_edges=num_edges, num_cells=num_cells
        )
        return full_grid_size

    def _get_index_field(
        self,
        field: gridfile.GridFileName,
        indices: data_alloc.NDArray | None = None,
        transpose=True,
        apply_offset=True,
        array_ns: ModuleType = np,
    ):
        return array_ns.asarray(
            self._reader.int_variable(
                field, indices=indices, transpose=transpose, apply_offset=apply_offset
            )
        )


def _get_derived_connectivities(
    neighbor_tables: dict[gtx.FieldOffset, data_alloc.NDArray],
) -> dict[gtx.FieldOffset, data_alloc.NDArray]:
    array_ns = data_alloc.array_namespace(next(iter(neighbor_tables.values())))
    e2v_table = neighbor_tables[dims.E2V]
    c2v_table = neighbor_tables[dims.C2V]
    e2c_table = neighbor_tables[dims.E2C]
    c2e_table = neighbor_tables[dims.C2E]
    c2e2c_table = neighbor_tables[dims.C2E2C]
    e2c2v = _construct_diamond_vertices(
        e2v_table,
        c2v_table,
        e2c_table,
    )
    e2c2e = _construct_diamond_edges(e2c_table, c2e_table)
    e2c2e0 = array_ns.column_stack((array_ns.asarray(range(e2c2e.shape[0])), e2c2e))

    c2e2c2e = _construct_triangle_edges(c2e2c_table, c2e_table)
    c2e2c0 = array_ns.column_stack(
        (
            array_ns.asarray(range(c2e2c_table.shape[0])),
            (c2e2c_table),
        )
    )
    c2e2c2e2c = _construct_butterfly_cells(c2e2c_table)

    return {
        dims.C2E2CO: c2e2c0,
        dims.C2E2C2E: c2e2c2e,
        dims.C2E2C2E2C: c2e2c2e2c,
        dims.E2C2V: e2c2v,
        dims.E2C2E: e2c2e,
        dims.E2C2EO: e2c2e0,
    }


def _construct_diamond_vertices(
    e2v: data_alloc.NDArray,
    c2v: data_alloc.NDArray,
    e2c: data_alloc.NDArray,
) -> data_alloc.NDArray:
    r"""
    Construct the connectivity table for the vertices of a diamond in the ICON triangular grid.

    Starting from the e2v and c2v connectivity the connectivity table for e2c2v is built up.

             v0
            /  \
           /    \
          /      \
         /        \
        v1---e0---v3
         \        /
          \      /
           \    /
            \  /
             v2

    For example for this diamond: e0 -> (v0, v1, v2, v3)
    Ordering is the same as ICON uses.

    Args:
        e2v: ndarray containing the connectivity table for edge-to-vertex
        c2v: ndarray containing the connectivity table for cell-to-vertex
        e2c: ndarray containing the connectivity table for edge-to-cell

    Returns: ndarray containing the connectivity table for edge-to-vertex on the diamond
    """
    array_ns = data_alloc.array_namespace(e2v)
    e2c_c2v = _patch_with_dummy_lastline(c2v)[e2c, :]
    # `flat` includes duplicated e2v vertices (v1, v3), shape (n_edges, 6).
    flat = e2c_c2v.reshape(e2c_c2v.shape[0], -1)

    far_indices_mask = (flat != e2v[:, 0, array_ns.newaxis]) & (flat != e2v[:, 1, array_ns.newaxis])

    neighbor_idx = array_ns.arange(flat.shape[1], dtype=gtx.int32)
    # identify v0 and v2 positions with the mask, and replace v1 and v3 positions with 6
    far_indices_pos = array_ns.where(
        far_indices_mask, neighbor_idx, neighbor_idx.shape[0]
    )  # (n_edges, 6)
    far_indices_pos = array_ns.sort(far_indices_pos, axis=1)[:, :2]
    e2v_far = array_ns.take_along_axis(flat, far_indices_pos, axis=1)
    return array_ns.hstack((e2v, e2v_far))


def _determine_center_position(
    centers: data_alloc.NDArray, neighbors: data_alloc.NDArray
) -> data_alloc.NDArray:
    """Determine the position of the values in `center` in the local neighbor array `neighbors`
    Args:
        centers: 1d array with shape (n, )
        neighbors: 2d array with shape (n, x)

    Returns:
         array of shape (n, ) for each row containing either the position of the `center` value along the second axis
         of neighbors or 0

    """
    array_ns = data_alloc.array_namespace(centers)
    center_idx = array_ns.where(neighbors == centers)
    me_cell = array_ns.zeros(centers.shape[0], dtype=gtx.int32)
    me_cell[center_idx[0]] = center_idx[1]
    return me_cell


def _construct_diamond_edges(
    e2c: data_alloc.NDArray, c2e: data_alloc.NDArray
) -> data_alloc.NDArray:
    r"""
    Construct the connectivity table for the edges of a diamond in the ICON triangular grid.

    Starting from the e2c and c2e connectivity the connectivity table for e2c2e is built up.

            /  \
           /    \
          e2 c0 e1
         /        \
         ----e0----
         \        /
          e3 c1 e4
           \    /
            \  /

    For example, for this diamond for e0 -> (e1, e2, e3, e4)


    Args:
        e2c: ndarray containing the connectivity table for edge-to-cell
        c2e: ndarray containing the connectivity table for cell-to-edge

    Returns: ndarray containing the connectivity table for central edge-to- boundary edges
             on the diamond
    """
    array_ns = data_alloc.array_namespace(e2c)
    # used to make sure that the local neighborhood is ordered in the same way as in ICON. At least
    # the compute_e_flx_avg function depends on that.
    icon_edge_order = array_ns.asarray([[1, 2], [2, 0], [0, 1]])

    dummy_c2e = _patch_with_dummy_lastline(c2e)
    expanded = dummy_c2e[e2c[:, :], :]
    n_edges, n_e2c, n_c2e = expanded.shape
    flattened = expanded.reshape(n_edges, n_e2c * n_c2e)

    centers = array_ns.arange(n_edges, dtype=gtx.int32)[:, None]
    me_cell1 = _determine_center_position(centers, expanded[:, 0, :])
    me_cell2 = _determine_center_position(centers, expanded[:, 1, :])
    ordered_local_index = array_ns.hstack(
        (icon_edge_order[me_cell1], icon_edge_order[me_cell2] + n_c2e)
    )
    e2c2e = array_ns.take_along_axis(flattened, ordered_local_index, axis=1)
    return e2c2e


def _construct_triangle_edges(
    c2e2c: data_alloc.NDArray, c2e: data_alloc.NDArray
) -> data_alloc.NDArray:
    r"""Compute the connectivity from a central cell to all neighboring edges of its cell neighbors.

         ----e3----  ----e7----
         \        /  \        /
          e4 c1  /    \  c3 e8
           \    e2 c0 e1    /
            \  /        \  /
               ----e0----
               \        /
                e5 c2 e6
                 \    /
                  \  /


    For example, for the triangular shape above, c0 -> (e3, e4, e2, e0, e5, e6, e7, e1, e8).

    Args:
        c2e2c: shape (n_cells, 3) connectivity table from a central cell to its cell neighbors
        c2e: shape (n_cells, 3), connectivity table from a cell to its neighboring edges
    Returns:
        ndarray: shape(n_cells, 9) connectivity table from a central cell to all neighboring
            edges of its cell neighbors
    """
    array_ns = data_alloc.array_namespace(c2e2c)
    dummy_c2e = _patch_with_dummy_lastline(c2e)
    table = array_ns.reshape(dummy_c2e[c2e2c, :], (c2e2c.shape[0], 9))
    return table


def _construct_butterfly_cells(
    c2e2c: data_alloc.NDArray,
) -> data_alloc.NDArray:
    r"""Compute the connectivity from a central cell to all neighboring cells of its cell neighbors.

                  /  \        /  \
                 /    \      /    \
                /  c4  \    /  c5  \
               /        \  /        \
               ----e3----  ----e7----
            /  \        /  \        /  \
           /    e4 c1  /    \  c3 e8    \
          /  c9  \    e2 c0 e1    /  c6  \
         /        \  /        \  /        \
         ----------  ----e0----  ----------
                  /  \        /  \
                 /    e5 c2 e6    \
                /  c8  \    /  c7  \
               /        \  /        \
               ----------  ----------

    For example, for the shape above, c0 -> (c1, c4, c9, c2, c7, c8, c3, c5, c6).

    Args:
        c2e2c: shape (n_cells, 3) connectivity table from a central cell to its cell neighbors
    Returns:
        ndarray: shape(n_cells, 9) connectivity table from a central cell to all neighboring cells of its cell neighbors
    """
    array_ns = data_alloc.array_namespace(c2e2c)
    dummy_c2e2c = _patch_with_dummy_lastline(c2e2c)
    c2e2c2e2c = array_ns.reshape(dummy_c2e2c[c2e2c, :], (c2e2c.shape[0], 9))
    return c2e2c2e2c


def _patch_with_dummy_lastline(ar):
    """
    Patch an array for easy access with another offset containing invalid indices (-1).

    Enlarges this table to contain a fake last line to account for numpy wrap around when
    encountering a -1 = GridFile.INVALID_INDEX value

    Args:
        ar: ndarray connectivity array to be patched

    Returns: same array with an additional line containing only GridFile.INVALID_INDEX

    """
    array_ns = data_alloc.array_namespace(ar)
    patched_ar = array_ns.append(
        ar,
        array_ns.full((1, ar.shape[1]), gridfile.GridFile.INVALID_INDEX, dtype=gtx.int32),
        axis=0,
    )
    return patched_ar
