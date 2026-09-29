# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import logging
from typing import Any, Protocol

import numpy as np
from gt4py import next as gtx

from icon4py.model.common import exceptions
from icon4py.model.common.utils import data_allocation as data_alloc


_log = logging.getLogger(__name__)


try:
    from netCDF4 import Dataset
except ImportError:

    class Dataset:
        """Dummy class to make import run when (optional) netcdf dependency is not installed."""

        def __init__(self, *args, **kwargs):
            raise ModuleNotFoundError("NetCDF4 is not installed.")


class IndexTransformation(Protocol):
    """Return an offset field to be applied to index fields"""

    def __call__(
        self,
        array: data_alloc.NDArray,
    ) -> data_alloc.NDArray: ...


class NoTransformation(IndexTransformation):
    """Empty implementation of the Protocol. Just return zeros."""

    def __call__(self, array: data_alloc.NDArray) -> data_alloc.NDArray:
        return data_alloc.array_namespace(array).zeros_like(array)


class ToZeroBasedIndexTransformation(IndexTransformation):
    def __call__(self, array: data_alloc.NDArray) -> data_alloc.NDArray:
        """
        Calculate the index offset needed for usage with python.

        Fortran indices are 1-based, hence the offset is -1 for 0-based ness of python except for
        INVALID values which are marked with -1 in the grid file and are kept such.
        """
        xp = data_alloc.array_namespace(array)
        return xp.asarray(xp.where(array == GridFile.INVALID_INDEX, 0, -1), dtype=gtx.int32)


class GridFileName(enum.StrEnum):
    pass


class OptionalPropertyName(GridFileName):
    """Global grid file attributes hat are not present in all files."""

    HISTORY = "history"
    GRID_ID = "grid_ID"
    PARENT_GRID_ID = "parent_grid_ID"
    MAX_CHILD_DOMAINS = "max_child_dom"


class PropertyName(GridFileName): ...


class LAMPropertyName(PropertyName):
    """
    Properties only present in the LAM file from MCH that we use in mch_ch_r04_b09_dsl.
    The source of this file is currently unknown.
    """

    GLOBAL_GRID = "global_grid"


class MPIMPropertyName(PropertyName):
    """
    Properties only present in the [MPI-M generated](https://gitlab.dkrz.de/mpim-sw/grid-generator)
    [grid files](http://icon-downloads.mpimet.mpg.de/mpim_grids.xml)
    """

    REVISION = "revision"
    DATE = "date"
    USER = "user_name"
    OS = "os_name"
    NUMBER_OF_SUBGRIDS = "number_of_subgrids"
    START_SUBGRID = "start_subgrid_id"
    BOUNDARY_DEPTH = "boundary_depth_index"
    ROTATION = "rotation_vector"
    GEOMETRY = "grid_geometry"
    CELL_TYPE = "grid_cell_type"
    MEAN_EDGE_LENGTH = "mean_edge_length"
    MEAN_DUAL_EDGE_LENGTH = "mean_dual_edge_length"
    MEAN_CELL_AREA = "mean_cell_area"
    MEAN_DUAL_CELL_AREA = "mean_dual_cell_area"
    DOMAIN_LENGTH = "domain_length"
    DOMAIN_HEIGHT = "domain_height"
    SPHERE_RADIUS = "sphere_radius"
    CARTESIAN_CENTER = "domain_cartesian_center"


class MandatoryPropertyName(PropertyName):
    """
    File attributes present in all files.
    DWD generated (from [icon-tools](https://gitlab.dkrz.de/dwd-sw/dwd_icon_tools)
    [grid files](http://icon-downloads.mpimet.mpg.de/dwd_grids.xml) contain only those properties.
    """

    TITLE = "title"
    INSTITUTION = "institution"
    SOURCE = "source"
    GRID_UUID = "uuidOfHGrid"
    PARENT_GRID_ID = "uuidOfParHGrid"
    NUMBER_OF_GRID = "number_of_grid_used"
    URI = "ICON_grid_file_uri"
    CENTER = "center"
    SUB_CENTER = "subcenter"
    CRS_ID = "crs_id"
    CRS_NAME = "crs_name"
    GRID_MAPPING = "grid_mapping_name"
    ELLIPSOID = "ellipsoid_name"
    SEMI_MAJOR_AXIS = "semi_major_axis"
    INVERSE_FLATTENING = "inverse_flattening"
    LEVEL = "grid_level"
    ROOT = "grid_root"


class DimensionName(GridFileName): ...


class DynamicDimension(DimensionName):
    """Dimension values (sizes) used in grid file."""

    #: number of vertices
    VERTEX_NAME = "vertex"
    #: number of edges
    EDGE_NAME = "edge"
    #: number of cells
    CELL_NAME = "cell"

    #: number of child domains (for nesting)
    MAX_CHILD_DOMAINS = "max_chdom"


class FixedSizeDimension(DimensionName):
    size: int

    def __new__(cls, value: str, size_: int):
        obj = str.__new__(cls)
        obj._value_ = value
        obj.size = size_
        return obj

    #: number of edges in a diamond: 4
    DIAMOND_EDGE_SIZE = ("no", 4)

    #: number of edges/cells neighboring one vertex: 6 (for regular, non pentagons)
    NEIGHBORS_TO_VERTEX_SIZE = ("ne", 6)

    #: number of cells edges, vertices and cells neighboring a cell: 3
    NEIGHBORS_TO_CELL_SIZE = ("nv", 3)

    #: number of vertices/cells neighboring an edge: 2
    NEIGHBORS_TO_EDGE_SIZE = ("nc", 2)

    #: Grid refinement: maximal number in grid-refinement (refin_ctl) array for each dimension
    CELL_GRF = ("cell_grf", 14)
    EDGE_GRF = ("edge_grf", 28)
    VERTEX_GRF = ("vert_grf", 14)

    def __str__(self):
        return f"{self.name}({self.name}: {self.size})"

    def __hash__(self):
        return hash((self.name, self.size))

    def __eq__(self, other: Any) -> bool:
        """Check equality based on zone name and level."""
        if not isinstance(other, FixedSizeDimension):
            return False
        return (self.name, self.size) == (other.name, other.size)


class FieldName(GridFileName): ...


class ConnectivityName(FieldName):
    """Names for connectivities used in the grid file."""

    # e2c2e/e2c2eO: diamond edges (including origin) not present in grid file-> construct
    #               from e2c and c2e
    # e2c2v: diamond vertices: not present in grid file -> constructed from e2c and c2v

    #: name of C2E2C connectivity in grid file: dims(nv=3, cell)
    C2E2C = "neighbor_cell_index"

    #: name of V2E2V connectivity in gridfile: dims(ne=6, vertex),
    #: all vertices of a pentagon/hexagon, same as V2C2V
    V2E2V = "vertices_of_vertex"  # does not exist in simple.py

    #: name of V2E dimension in grid file: dims(ne=6, vertex)
    V2E = "edges_of_vertex"

    #: name fo V2C connectivity in grid file: dims(ne=6, vertex)
    V2C = "cells_of_vertex"

    #: name of E2V connectivity in grid file: dims(nc=2, edge)
    E2V = "edge_vertices"

    #: name of C2V connectivity in grid file: dims(nv=3, cell)
    C2V = "vertex_of_cell"  # does not exist in grid.simple.py

    #: name of E2C connectivity in grid file: dims(nc=2, edge)
    E2C = "adjacent_cell_of_edge"

    #: name of C2E connectivity in grid file: dims(nv=3, cell)
    C2E = "edge_of_cell"


class GeometryName(FieldName):
    CELL_NORMAL_ORIENTATION = "orientation_of_normal"
    TANGENT_ORIENTATION = "edge_system_orientation"
    EDGE_ORIENTATION_ON_VERTEX = "edge_orientation"

    # TODO(halungge): compute from coordinates
    CELL_AREA = "cell_area"
    DUAL_AREA = "dual_area"
    EDGE_LENGTH = "edge_length"
    DUAL_EDGE_LENGTH = "dual_edge_length"
    # TODO(halungge): compute from coordinates
    EDGE_CELL_DISTANCE = "edge_cell_distance"
    EDGE_VERTEX_DISTANCE = "edge_vert_distance"
    EDGE_NORMAL_X = "edge_primal_normal_cartesian_x"
    EDGE_NORMAL_Y = "edge_primal_normal_cartesian_y"
    EDGE_NORMAL_Z = "edge_primal_normal_cartesian_z"
    EDGE_TANGENT_X = "edge_dual_normal_cartesian_x"
    EDGE_TANGENT_Y = "edge_dual_normal_cartesian_y"
    EDGE_TANGENT_Z = "edge_dual_normal_cartesian_z"


class CoordinateName(FieldName):
    """
    Coordinates of cell centers, edge midpoints and vertices.
    Units: radian for both MPI-M and DWD
    """

    CELL_LONGITUDE = "clon"
    CELL_LATITUDE = "clat"
    EDGE_LONGITUDE = "elon"
    EDGE_LATITUDE = "elat"
    VERTEX_LONGITUDE = "vlon"
    VERTEX_LATITUDE = "vlat"

    CELL_X = "cell_circumcenter_cartesian_x"
    CELL_Y = "cell_circumcenter_cartesian_y"
    CELL_Z = "cell_circumcenter_cartesian_z"
    EDGE_X = "edge_middle_cartesian_x"
    EDGE_Y = "edge_middle_cartesian_y"
    EDGE_Z = "edge_middle_cartesian_z"
    VERTEX_X = "cartesian_x_vertices"
    VERTEX_Y = "cartesian_y_vertices"
    VERTEX_Z = "cartesian_z_vertices"


class GridRefinementName(FieldName):
    """Names of arrays in grid file defining the grid control, definition of boundaries layers, start and end indices of horizontal zones."""

    #: refine control value of cell indices
    CONTROL_CELLS = "refin_c_ctrl"

    #: refine control value of edge indices
    CONTROL_EDGES = "refin_e_ctrl"

    #: refine control value of vertex indices
    CONTROL_VERTICES = "refin_v_ctrl"

    #: start indices of horizontal grid zones for cell fields
    START_INDEX_CELLS = "start_idx_c"

    #: start indices of horizontal grid zones for edge fields
    START_INDEX_EDGES = "start_idx_e"

    #: start indices of horizontal grid zones for vertex fields
    START_INDEX_VERTICES = "start_idx_v"

    #: end indices of horizontal grid zones for cell fields
    END_INDEX_CELLS = "end_idx_c"

    #: end indices of horizontal grid zones for edge fields
    END_INDEX_EDGES = "end_idx_e"

    #: end indices of horizontal grid zones for vertex fields
    END_INDEX_VERTICES = "end_idx_v"


class GridFile:
    """Represent an ICON netcdf grid file."""

    INVALID_INDEX = -1

    def __init__(self, file_name: str, offset_transformation: IndexTransformation):
        self._filename = file_name
        self._offset_transformation = offset_transformation
        self._dataset = None

    def dimension(self, name: DimensionName) -> int:
        """Read a dimension with name 'name' from the grid file."""
        return self._dataset.dimensions[name].size

    def attribute(self, name: PropertyName) -> str | int | float:
        """Read a global attribute with name 'name' from the grid file."""
        return self._dataset.getncattr(name)

    def try_attribute(self, name: PropertyName) -> str | int | float | None:
        """Try reading a global attribute with name 'name' from the grid file.
        Return None if the attribute does not exist."""
        if name in self._dataset.ncattrs():
            return self._dataset.getncattr(name)
        else:
            return None

    def int_variable(
        self,
        name: FieldName,
        indices: data_alloc.NDArray | None = None,
        transpose: bool = True,
        apply_offset: bool = True,
    ) -> np.ndarray:
        """Read a integer field from the grid file.

        Reads as gtx.int32.

        Args:
            name: name of the field to read
            indices: list of indices to read
            transpose: flag to indicate whether the file should be transposed (for 2d fields)
            apply_offset: flag to indicate whether the offset should be applied
                to the indices, defaults to True
        Returns:
            NDArray: field data

        """
        _log.debug(f"reading {name}: transposing = {transpose} apply_offset={apply_offset}")
        variable = self.variable(name, indices, transpose=transpose, dtype=gtx.int32)
        if apply_offset:
            return variable + self._offset_transformation(variable)
        return variable

    def variable(
        self,
        name: FieldName,
        indices: data_alloc.NDArray | None = None,
        transpose: bool = False,
        dtype: np.dtype = gtx.float64,
    ) -> np.ndarray:
        """Read a field from the grid file.

        If an index array is given it only reads the values at those positions.
        Args:
            name: name of the field to read
            indices: indices to read if requesting a restricted set of indices. We assume this be a 1d array it will be applied to the 1. dimension (after transposition)
            transpose: flag indicateing whether the array needs to be transposed
                to match icon4py dimension ordering, defaults to False
            dtype: datatype of the field
        """

        assert indices is None or indices.ndim == 1, "indices must be 1 dimensional"

        try:
            variable = self._dataset.variables[name]
            variable_size = variable.ndim
            n = (variable.shape[0],) if variable_size > 1 else ()
            target_shape = (*n, -1)

            slicer = [slice(None) for _ in range(variable_size)]
            if indices is not None and indices.size > 0:
                # apply the slicing to the correct dimension
                slicer[(1 if transpose else 0)] = data_alloc.as_numpy(indices)
            _log.debug(f"reading {name}: transposing = {transpose}")
            data = np.asarray(variable[tuple(slicer)])
            data = np.array(data, dtype=dtype).ravel(order="K").reshape(target_shape)
            return np.transpose(data) if transpose else data
        except KeyError as err:
            msg = f"{name} does not exist in dataset"
            _log.warning(msg)
            _log.debug(f"Error: {err}")
            raise exceptions.IconGridError(msg) from err

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def close(self):
        self._dataset.close()

    def open(self):
        self._dataset = Dataset(self._filename, "r", format="NETCDF4")
        _log.debug(f"opened data set: {self._dataset}")
