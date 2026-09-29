# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from typing import Final

import gt4py.next as gtx

from icon4py.model.common import dimension as dims, type_alias as ta
from icon4py.model.common.states import model


"""
This modules contains CF like metadata for the geometry fields in ICON.

TODO @halungge: should be cross checked by domain scientist.

"""

EDGE_LON: Final[str] = "grid_longitude_of_edge_midpoint"
EDGE_LAT: Final[str] = "grid_latitude_of_edge_midpoint"
VERTEX_LON: Final[str] = "grid_longitude_of_vertex"
VERTEX_LAT: Final[str] = "grid_latitude_of_vertex"
VERTEX_X: Final[str] = "x_component_of_vertex"
VERTEX_Y: Final[str] = "y_component_of_vertex"
VERTEX_Z: Final[str] = "z_component_of_vertex"

CELL_LON: Final[str] = "grid_longitude_of_cell_center"
CELL_LAT: Final[str] = "grid_latitude_of_cell_center"
CELL_CENTER_X: Final[str] = "x_component_of_cell_center"
CELL_CENTER_Y: Final[str] = "y_component_of_cell_center"
CELL_CENTER_Z: Final[str] = "z_component_of_cell_center"

CELL_AREA: Final[str] = "cell_area"
EDGE_AREA: Final[str] = "edge_area"
DUAL_AREA: Final[str] = "dual_area"
EDGE_CELL_DISTANCE: Final[str] = "edge_midpoint_to_cell_center_distance"
EDGE_VERTEX_DISTANCE: Final[str] = "edge_midpoint_to_vertex_distance"
TANGENT_ORIENTATION: Final[str] = "edge_orientation"
CELL_NORMAL_ORIENTATION: Final[str] = "orientation_of_normal_to_cell_edges"
VERTEX_EDGE_ORIENTATION: Final[str] = "orientation_of_edges_around_vertex"


CORIOLIS_PARAMETER: Final[str] = "coriolis_parameter"
VERTEX_VERTEX_LENGTH: Final[str] = "vertex_vertex_length"
INVERSE_VERTEX_VERTEX_LENGTH: Final[str] = "inverse_of_vertex_vertex_length"
EDGE_LENGTH: Final[str] = "edge_length"
DUAL_EDGE_LENGTH: Final[str] = "length_of_dual_edge"

EDGE_TANGENT_VERTEX_U: Final[str] = "eastward_component_of_edge_tangent_on_vertex"
EDGE_TANGENT_VERTEX_V: Final[str] = "northward_component_of_edge_tangent_on_vertex"
EDGE_TANGENT_CELL_U: Final[str] = "eastward_component_of_edge_tangent_on_cell"
EDGE_TANGENT_CELL_V: Final[str] = "northward_component_of_edge_tangent_on_cell"

EDGE_NORMAL_X: Final[str] = "x_component_of_edge_normal_unit_vector"
EDGE_NORMAL_Y: Final[str] = "y_component_of_edge_normal_unit_vector"
EDGE_NORMAL_Z: Final[str] = "z_component_of_edge_normal_unit_vector"
EDGE_NORMAL_U: Final[str] = "eastward_component_of_edge_normal"
EDGE_NORMAL_V: Final[str] = "northward_component_of_edge_normal"
EDGE_TANGENT_X: Final[str] = "x_component_of_edge_tangential_unit_vector"
EDGE_TANGENT_Y: Final[str] = "y_component_of_edge_tangential_unit_vector"
EDGE_TANGENT_Z: Final[str] = "z_component_of_edge_tangential_unit_vector"
EDGE_DUAL_U: Final[str] = "eastward_component_of_edge_tangent"
EDGE_DUAL_V: Final[str] = "northward_component_of_edge_tangent"
EDGE_NORMAL_VERTEX_U: Final[str] = "eastward_component_of_edge_normal_on_vertex"
EDGE_NORMAL_VERTEX_V: Final[str] = "northward_component_of_edge_normal_on_vertex"
EDGE_NORMAL_CELL_U: Final[str] = "eastward_component_of_edge_normal_on_cell"
EDGE_NORMAL_CELL_V: Final[str] = "northward_component_of_edge_normal_on_cell"
EDGE_CENTER_X: Final[str] = "x_coordinate_of_edge_center"
EDGE_CENTER_Y: Final[str] = "y_coordinate_of_edge_center"
EDGE_CENTER_Z: Final[str] = "z_coordinate_of_edge_center"

MEAN_EDGE_LENGTH: Final[str] = "mean_edge_length"
MEAN_DUAL_EDGE_LENGTH: Final[str] = "mean_dual_edge_length"
MEAN_CELL_AREA: Final[str] = "mean_cell_area"
MEAN_DUAL_AREA: Final[str] = "mean_dual_area"
CHARACTERISTIC_LENGTH: Final[str] = "characteristic_length"

attrs: dict[str, model.FieldMetaData] = {
    CELL_LAT: dict(
        standard_name=CELL_LAT,
        units="radian",
        dims=(dims.CellDim,),
        icon_var_name="t_grid_cells%center%lat",
        dtype=ta.wpfloat,
    ),
    CELL_LON: dict(
        standard_name=CELL_LON,
        units="radian",
        dims=(dims.CellDim,),
        icon_var_name="t_grid_cells%center%lon",
        dtype=ta.wpfloat,
    ),
    VERTEX_LAT: dict(
        standard_name=VERTEX_LAT,
        units="radian",
        dims=(dims.VertexDim,),
        icon_var_name="t_grid_vertices%vertex%lat",
        dtype=ta.wpfloat,
    ),
    VERTEX_LON: dict(
        standard_name=VERTEX_LON,
        units="radian",
        dims=(dims.VertexDim,),
        icon_var_name="t_grid_vertices%vertex%lon",
        dtype=ta.wpfloat,
    ),
    VERTEX_X: dict(
        standard_name=VERTEX_X,
        long_name="x component of cartesian coordinates of vertex",
        units="1",
        dims=(dims.VertexDim,),
        icon_var_name="t_grid_vertices%vertex%x(1)",
        dtype=ta.wpfloat,
    ),
    VERTEX_Y: dict(
        standard_name=VERTEX_Y,
        long_name="y component of cartesian coordinates of vertex",
        units="1",
        dims=(dims.VertexDim,),
        icon_var_name="t_grid_vertices%vertex%x(2)",
        dtype=ta.wpfloat,
    ),
    VERTEX_Z: dict(
        standard_name=VERTEX_Z,
        long_name="z component of cartesian coordinates of vertex",
        units="1",
        dims=(dims.VertexDim,),
        icon_var_name="t_grid_vertices%vertex%x(3)",
        dtype=ta.wpfloat,
    ),
    EDGE_LAT: dict(
        standard_name=EDGE_LAT,
        units="radian",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%center%lat",
        dtype=ta.wpfloat,
    ),
    EDGE_LON: dict(
        standard_name=EDGE_LON,
        units="radian",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%center%lon",
        dtype=ta.wpfloat,
    ),
    EDGE_LENGTH: dict(
        standard_name=EDGE_LENGTH,
        long_name="edge length",
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%primal_edge_length",
        dtype=ta.wpfloat,
    ),
    CELL_NORMAL_ORIENTATION: dict(
        standard_name=CELL_NORMAL_ORIENTATION,
        units="",
        dims=(dims.CellDim, dims.C2EDim),
        icon_var_name="t_grid_cells%edge_orientation",
        dtype=gtx.int32,
    ),
    EDGE_CELL_DISTANCE: dict(
        standard_name=EDGE_CELL_DISTANCE,
        long_name="distances between edge midpoint and adjacent triangle midpoints",
        units="m",
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="t_grid_edges%edge_cell_length",
        dtype=ta.wpfloat,
    ),
    EDGE_VERTEX_DISTANCE: dict(
        standard_name=EDGE_VERTEX_DISTANCE,
        long_name="distances between edge midpoint and adjacent vertices",
        units="m",
        dims=(dims.EdgeDim, dims.E2VDim),
        icon_var_name="t_grid_edges%edge_vert_length",
        dtype=ta.wpfloat,
    ),
    DUAL_EDGE_LENGTH: dict(
        standard_name=DUAL_EDGE_LENGTH,
        long_name="length of the dual edge",
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%dual_edge_length",
        dtype=ta.wpfloat,
    ),
    VERTEX_VERTEX_LENGTH: dict(
        standard_name=VERTEX_VERTEX_LENGTH,
        long_name="distance between outer vertices of adjacent cells",
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%vert_vert_length",
        dtype=ta.wpfloat,
    ),
    EDGE_AREA: dict(
        standard_name=EDGE_AREA,
        long_name="area of quadrilateral spanned by edge and associated dual edge",
        units="m2",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%area_edge",
        dtype=ta.wpfloat,
    ),
    CELL_AREA: dict(
        standard_name=CELL_AREA,
        long_name="area of a triangular cell",
        units="m2",
        dims=(dims.CellDim,),
        icon_var_name="t_grid_cells%area",
        dtype=ta.wpfloat,
    ),
    CELL_CENTER_X: dict(
        standard_name=CELL_CENTER_X,
        long_name="x component of cartesian coordinates of cell center",
        units="",
        dims=(dims.CellDim,),
        icon_var_name="t_grid_cells%%cartesian_center%x(1)",
        dtype=ta.wpfloat,
    ),
    CELL_CENTER_Y: dict(
        standard_name=CELL_CENTER_Y,
        long_name="y component of cartesian coordinates of cell center",
        units="",
        dims=(dims.CellDim,),
        icon_var_name="t_grid_cells%%cartesian_center%x(2)",
        dtype=ta.wpfloat,
    ),
    CELL_CENTER_Z: dict(
        standard_name=CELL_CENTER_Z,
        long_name="z component of cartesian coordinates of cell center",
        units="",
        dims=(dims.CellDim,),
        icon_var_name="t_grid_cells%%cartesian_center%x(3)",
        dtype=ta.wpfloat,
    ),
    DUAL_AREA: dict(
        standard_name=DUAL_AREA,
        long_name="area of the dual grid cell (hexagon cell)",
        units="m2",
        dims=(dims.VertexDim,),
        icon_var_name="t_grid_verts%dual_area",
        dtype=ta.wpfloat,
    ),
    CORIOLIS_PARAMETER: dict(
        standard_name=CORIOLIS_PARAMETER,
        long_name="coriolis parameter at cell edges",
        units="s-1",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%f_e",
        dtype=ta.wpfloat,
    ),
    EDGE_TANGENT_X: dict(
        standard_name=EDGE_TANGENT_X,
        long_name=EDGE_TANGENT_X,
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%dual_cart_normal%x(1)",
        dtype=ta.wpfloat,
    ),
    EDGE_TANGENT_Y: dict(
        standard_name=EDGE_TANGENT_Y,
        long_name=EDGE_TANGENT_Y,
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%dual_cart_normal%x(2)",
        dtype=ta.wpfloat,
    ),
    EDGE_TANGENT_Z: dict(
        standard_name=EDGE_TANGENT_Z,
        long_name=EDGE_TANGENT_Z,
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%dual_cart_normal%x(3)",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_U: dict(
        standard_name=EDGE_NORMAL_U,
        long_name="eastward (zonal) component of edge normal",
        units="radian",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%primal_normal%v2",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_V: dict(
        standard_name=EDGE_NORMAL_V,
        long_name="northward (meridional) component of edge normal",
        units="radian",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%primal_normal%v1",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_X: dict(
        standard_name=EDGE_NORMAL_X,
        long_name=EDGE_NORMAL_X,
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%primal_cart_normal%x(1)",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_Y: dict(
        standard_name=EDGE_NORMAL_Y,
        long_name=EDGE_NORMAL_Y,
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%primal_cart_normal%x(2)",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_Z: dict(
        standard_name=EDGE_NORMAL_Z,
        long_name=EDGE_NORMAL_Z,
        units="m",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%primal_cart_normal%x(3)",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_VERTEX_U: dict(
        standard_name=EDGE_NORMAL_VERTEX_U,
        long_name="eastward (zonal) component of edge normal projected to vertex locations",
        units="radian",
        dims=(dims.EdgeDim, dims.E2C2VDim),
        icon_var_name="t_grid_edges%primal_normal_vert%v1",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_VERTEX_V: dict(
        standard_name=EDGE_NORMAL_VERTEX_V,
        long_name="northward (meridional) component of edge normal projected to vertex locations",
        units="radian",
        dims=(dims.EdgeDim, dims.E2C2VDim),
        icon_var_name="t_grid_edges%primal_normal_vert%v2",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_CELL_U: dict(
        standard_name=EDGE_NORMAL_CELL_U,
        long_name="eastward (zonal) component of edge normal projected to neighbor cell centers",
        units="radian",
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="t_grid_edges%primal_normal_cell%v1",
        dtype=ta.wpfloat,
    ),
    EDGE_NORMAL_CELL_V: dict(
        standard_name=EDGE_NORMAL_CELL_V,
        long_name="northward (meridional) component of edge normal projected to neighbor cell centers",
        units="radian",
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="t_grid_edges%primal_normal_cell%v2",
        dtype=ta.wpfloat,
    ),
    EDGE_TANGENT_CELL_U: dict(
        standard_name=EDGE_TANGENT_CELL_U,
        long_name="eastward (zonal) component of edge tangent projected to neighbor cell centers",
        units="radian",
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="t_grid_edges%dual_normal_cell%v1",
        dtype=ta.wpfloat,
    ),
    EDGE_TANGENT_CELL_V: dict(
        standard_name=EDGE_TANGENT_CELL_V,
        long_name="northward (meridional) component of edge tangent projected to neighbor cell centers",
        units="radian",
        dims=(dims.EdgeDim, dims.E2CDim),
        icon_var_name="t_grid_edges%dual_normal_cell%v2",
        dtype=ta.wpfloat,
    ),
    EDGE_TANGENT_VERTEX_U: dict(
        standard_name=EDGE_TANGENT_VERTEX_U,
        long_name="eastward (zonal) component of edge tangent projected to vertex locations",
        units="radian",
        icon_var_name="t_grid_edges%dual_normal_vert%v1",
        dims=(dims.EdgeDim, dims.E2C2VDim),
        dtype=ta.wpfloat,
    ),
    EDGE_TANGENT_VERTEX_V: dict(
        standard_name=EDGE_TANGENT_VERTEX_V,
        long_name="northward (meridional) component of edge tangent projected to vertex locations",
        units="radian",
        dims=(dims.EdgeDim, dims.E2C2VDim),
        icon_var_name="t_grid_edges%dual_normal_vert%v2",
        dtype=ta.wpfloat,
    ),
    TANGENT_ORIENTATION: dict(
        standard_name=TANGENT_ORIENTATION,
        long_name="orientation of tangent vector",
        units="1",
        dims=(dims.EdgeDim,),
        icon_var_name=f"t_grid_edges%{TANGENT_ORIENTATION}",
        dtype=ta.wpfloat,  # TODO(halungge): netcdf: int
    ),
    VERTEX_EDGE_ORIENTATION: dict(
        standard_name=VERTEX_EDGE_ORIENTATION,
        long_name="orientation of tangent vector",
        units="1",
        dims=(dims.VertexDim, dims.V2EDim),
        icon_var_name="t_grid_vertex%edge_orientation",
        dtype=ta.wpfloat,
    ),
    EDGE_DUAL_U: dict(
        standard_name=EDGE_DUAL_U,
        long_name="eastward component of the dual edge (edge tangent)",
        units="",  # TODO(): add this
        dims=(dims.EdgeDim,),
        icon_var_name="ptr_patch%edges%dual_normal%v1",
        dtype=ta.wpfloat,
    ),
    EDGE_DUAL_V: dict(
        standard_name="northward component of the dual edge (edge tangent)",
        long_name="ptr_patch%edges%dual_normal_vert_y",
        units="",  # TODO(): add this
        dims=(dims.EdgeDim,),
        icon_var_name="ptr_patch%edges%dual_normal%v2",
        dtype=ta.wpfloat,
    ),
    EDGE_CENTER_X: dict(
        standard_name=EDGE_CENTER_X,
        long_name="x component of cartesian coordinates of edge centers",
        units="1",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%cartesian_center%x(1)",
        dtype=ta.wpfloat,
    ),
    EDGE_CENTER_Y: dict(
        standard_name=EDGE_CENTER_Y,
        long_name="y component of cartesian coordinates of edge centers",
        units="1",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%cartesian_center%x(2)",
        dtype=ta.wpfloat,
    ),
    EDGE_CENTER_Z: dict(
        standard_name=EDGE_CENTER_Z,
        long_name="z component of cartesian coordinates of edge centers",
        units="1",
        dims=(dims.EdgeDim,),
        icon_var_name="t_grid_edges%cartesian_center%x(3)",
        dtype=ta.wpfloat,
    ),
    MEAN_EDGE_LENGTH: dict(
        standard_name=MEAN_EDGE_LENGTH,
        long_name="mean_edge_length",
        units="",
        icon_var_name="",
        dtype=ta.wpfloat,
    ),
    MEAN_DUAL_EDGE_LENGTH: dict(
        standard_name=MEAN_DUAL_EDGE_LENGTH,
        long_name="mean_dual_edge_length",
        units="",
        icon_var_name="",
        dtype=ta.wpfloat,
    ),
    MEAN_CELL_AREA: dict(
        standard_name=MEAN_CELL_AREA,
        long_name="mean_cell_area",
        units="",
        icon_var_name="",
        dtype=ta.wpfloat,
    ),
    MEAN_DUAL_AREA: dict(
        standard_name=MEAN_DUAL_AREA,
        long_name="mean_dual_area",
        units="",
        icon_var_name="",
        dtype=ta.wpfloat,
    ),
    CHARACTERISTIC_LENGTH: dict(
        standard_name=CHARACTERISTIC_LENGTH,
        long_name="characteristic_length",
        units="",
        icon_var_name="",
        dtype=ta.wpfloat,
    ),
}


def metadata_for_inverse(metadata: model.FieldMetaData) -> model.FieldMetaData:
    def inv_name(name: str):
        x = name.split("%", 1)
        x[-1] = f"inv_{x[-1]}"
        return "%".join(x)

    standard_name = f"inverse_of_{metadata['standard_name']}"
    units = f"{metadata['units']}-1"
    long_name = f"inverse of {metadata.get('long_name')}" if metadata.get("long_name") else ""
    inverse_meta = dict(
        standard_name=standard_name,
        units=units,
        dims=metadata.get("dims"),
        dtype=metadata.get("dtype"),
        long_name=long_name,
        icon_var_name=inv_name(metadata.get("icon_var_name")),
    )

    return inverse_meta
