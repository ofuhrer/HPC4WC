# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause


import gt4py.next.typing as gtx_typing
from gt4py import next as gtx
from gt4py.next import sin, where

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C, E2C2V, E2V, EdgeDim
from icon4py.model.common.math.coordinate_transformations import (
    geographical_to_cartesian_on_edges,
    geographical_to_cartesian_on_vertices,
    zonal_and_meridional_components_on_edges,
)
from icon4py.model.common.math.distance import (
    arc_length_on_edges,
    diff_on_edges_torus,
    distance_on_edges_torus,
)
from icon4py.model.common.math.vector_operations import (
    cross_product_on_edges,
    normalize_cartesian_vector_on_edges,
)
from icon4py.model.common.utils import data_allocation as data_alloc


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def cartesian_coordinates_of_edge_tangent(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    edge_orientation: fa.EdgeField[ta.wpfloat],
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    """
    Compute normalized cartesian vector tangential to an edge.

    That is: computes the distance between the two vertices adjacent to the edge:
    t = d(v1, v2)

    Args:
        vertex_lat: latitude of vertices
        vertex_lon: longitude of vertices
        edge_orientation: encoding of the edge orientation: (-1, +1) depending on whether the
            edge is directed from first to second neighbor of vice versa.
    Returns:
          x: x coordinate of normalized tangent vector
          y: y coordinate of normalized tangent vector
          z: z coordinate of normalized tangent vector
    """
    vertex_x, vertex_y, vertex_z = geographical_to_cartesian_on_vertices(vertex_lat, vertex_lon)

    x = edge_orientation * (vertex_x(E2V[1]) - vertex_x(E2V[0]))
    y = edge_orientation * (vertex_y(E2V[1]) - vertex_y(E2V[0]))
    z = edge_orientation * (vertex_z(E2V[1]) - vertex_z(E2V[0]))

    return normalize_cartesian_vector_on_edges(x, y, z)


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def cartesian_coordinates_of_edge_tangent_torus(
    vertex_x: fa.VertexField[ta.wpfloat],
    vertex_y: fa.VertexField[ta.wpfloat],
    edge_orientation: fa.EdgeField[ta.wpfloat],
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """
    Compute normalized cartesian vector tangential to an edge on a torus grid.

    This is done by taking the difference between the two vertices adjacent to the edge
    t = d(v1, v2)

    Args:
        vertex_x: x coordinates of vertices
        vertex_y: y coordinates of vertices
        edge_orientation: encoding of the edge orientation: (-1, +1) depending on whether the
            edge is directed from first to second neighbor of vice versa.
    Returns:
          x: x coordinate of normalized tangent vector
          y: y coordinate of normalized tangent vector
          z: z coordinate of normalized tangent vector
    """
    xdiff, ydiff = diff_on_edges_torus(
        vertex_x(E2V[0]),
        vertex_x(E2V[1]),
        vertex_y(E2V[0]),
        vertex_y(E2V[1]),
        domain_length,
        domain_height,
    )
    x = edge_orientation * xdiff
    y = edge_orientation * ydiff
    z = 0.0 * x
    # TODO(msimberg): This should use something like numpy.zeros_like if and
    # when that becomes available in gt4py.

    return normalize_cartesian_vector_on_edges(x, y, z)


@gtx.field_operator
def cartesian_coordinates_of_edge_normal(
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    edge_tangent_x: fa.EdgeField[ta.wpfloat],
    edge_tangent_y: fa.EdgeField[ta.wpfloat],
    edge_tangent_z: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """
    Compute the normal to the edge tangent vector.

    The normal is  the cross product of the edge center (cartesian vector) and the edge tangent (cartesian_vector): (edge_center) x (edge_tangent)

    Args:
        edge_lat: latitude of edge center
        edge_lon: longitude of edge center
        edge_tangent_x: x coordinate of edge tangent
        edge_tangent_y: y coordinate of edge tangent
        edge_tangent_z: z coordinate of edge tangent
    Returns:
        edge_normal_x: x coordinate of the normal
        edge_normal_y: y coordinate of the normal
        edge_normal_z: z coordinate of the normal
    """
    edge_center_x, edge_center_y, edge_center_z = geographical_to_cartesian_on_edges(
        edge_lat, edge_lon
    )
    x, y, z = cross_product_on_edges(
        edge_center_x, edge_tangent_x, edge_center_y, edge_tangent_y, edge_center_z, edge_tangent_z
    )
    return normalize_cartesian_vector_on_edges(x, y, z)


@gtx.field_operator
def cartesian_coordinates_of_edge_normal_torus(
    edge_tangent_x: fa.EdgeField[ta.wpfloat],
    edge_tangent_y: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """
    Compute the normal to the edge tangent vector on a torus grid.

    Args:
        edge_tangent_x: x coordinate of the tangent
        edge_tangent_y: y coordinate of the tangent
    Returns:
        edge_normal_x: x coordinate of the normal
        edge_normal_y: y coordinate of the normal
        edge_normal_z: y coordinate of the normal
    """
    z = 0.0 * edge_tangent_x
    # TODO(msimberg): This should use something like numpy.zeros_like if and
    # when that becomes available in gt4py.
    return normalize_cartesian_vector_on_edges(-edge_tangent_y, edge_tangent_x, z)


@gtx.field_operator
def cartesian_coordinates_edge_tangent_and_normal(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    edge_orientation: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """Compute normalized cartesian vectors of edge tangent and edge normal."""
    tangent_x, tangent_y, tangent_z = cartesian_coordinates_of_edge_tangent(
        vertex_lat, vertex_lon, edge_orientation
    )
    normal_x, normal_y, normal_z = cartesian_coordinates_of_edge_normal(
        edge_lat=edge_lat,
        edge_lon=edge_lon,
        edge_tangent_x=tangent_x,
        edge_tangent_y=tangent_y,
        edge_tangent_z=tangent_z,
    )

    return tangent_x, tangent_y, tangent_z, normal_x, normal_y, normal_z


@gtx.field_operator
def cartesian_coordinates_edge_tangent_and_normal_torus(  # noqa: PLR0917 [too-many-positional-arguments]
    vertex_x: fa.VertexField[ta.wpfloat],
    vertex_y: fa.VertexField[ta.wpfloat],
    edge_x: fa.EdgeField[ta.wpfloat],
    edge_y: fa.EdgeField[ta.wpfloat],
    edge_orientation: fa.EdgeField[ta.wpfloat],
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """Compute normalized cartesian vectors of edge tangent and edge normal."""
    tangent_x, tangent_y, tangent_z = cartesian_coordinates_of_edge_tangent_torus(
        vertex_x=vertex_x,
        vertex_y=vertex_y,
        edge_orientation=edge_orientation,
        domain_length=domain_length,
        domain_height=domain_height,
    )
    tangent_u = tangent_x
    tangent_v = tangent_y

    normal_x, normal_y, normal_z = cartesian_coordinates_of_edge_normal_torus(
        edge_tangent_x=tangent_x,
        edge_tangent_y=tangent_y,
    )
    normal_u = normal_x
    normal_v = normal_y

    return (
        tangent_x,
        tangent_y,
        tangent_z,
        tangent_u,
        tangent_v,
        normal_x,
        normal_y,
        normal_z,
        normal_u,
        normal_v,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cartesian_coordinates_of_edge_tangent_and_normal(  # noqa: PLR0917 [too-many-positional-arguments]
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    edge_lat: fa.EdgeField[ta.wpfloat],
    edge_lon: fa.EdgeField[ta.wpfloat],
    edge_orientation: fa.EdgeField[ta.wpfloat],
    tangent_x: fa.EdgeField[ta.wpfloat],
    tangent_y: fa.EdgeField[ta.wpfloat],
    tangent_z: fa.EdgeField[ta.wpfloat],
    normal_x: fa.EdgeField[ta.wpfloat],
    normal_y: fa.EdgeField[ta.wpfloat],
    normal_z: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    cartesian_coordinates_edge_tangent_and_normal(
        vertex_lat=vertex_lat,
        vertex_lon=vertex_lon,
        edge_lat=edge_lat,
        edge_lon=edge_lon,
        edge_orientation=edge_orientation,
        out=(tangent_x, tangent_y, tangent_z, normal_x, normal_y, normal_z),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cartesian_coordinates_of_edge_tangent_and_normal_torus(  # noqa: PLR0917 [too-many-positional-arguments]
    vertex_x: fa.VertexField[ta.wpfloat],
    vertex_y: fa.VertexField[ta.wpfloat],
    edge_x: fa.EdgeField[ta.wpfloat],
    edge_y: fa.EdgeField[ta.wpfloat],
    edge_orientation: fa.EdgeField[ta.wpfloat],
    tangent_x: fa.EdgeField[ta.wpfloat],
    tangent_y: fa.EdgeField[ta.wpfloat],
    tangent_z: fa.EdgeField[ta.wpfloat],
    tangent_u: fa.EdgeField[ta.wpfloat],
    tangent_v: fa.EdgeField[ta.wpfloat],
    normal_x: fa.EdgeField[ta.wpfloat],
    normal_y: fa.EdgeField[ta.wpfloat],
    normal_z: fa.EdgeField[ta.wpfloat],
    normal_u: fa.EdgeField[ta.wpfloat],
    normal_v: fa.EdgeField[ta.wpfloat],
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    cartesian_coordinates_edge_tangent_and_normal_torus(
        vertex_x=vertex_x,
        vertex_y=vertex_y,
        edge_x=edge_x,
        edge_y=edge_y,
        edge_orientation=edge_orientation,
        domain_length=domain_length,
        domain_height=domain_height,
        out=(
            tangent_x,
            tangent_y,
            tangent_z,
            tangent_u,
            tangent_v,
            normal_x,
            normal_y,
            normal_z,
            normal_u,
            normal_v,
        ),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def zonal_and_meridional_component_of_edge_field_at_vertex(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """
    Compute the zonal (u) an meridional (v) component of a cartesian vector (x, y, z) at the vertex position (lat, lon).

    The cartesian vector is defined on edges and it projection onto all 4 neighboring vertices of the diamond is computed.

    Args:
        vertex_lat: latitude of vertices
        vertex_lon: longitude of vertices
        x: x coordinate
        y: y coordinate
        z: z coordinate
    Returns:
        u_vertex_0: zonal (eastward positive) component at E2C2V[0]
        v_vertex_0: meridional (northward) component at E2C2V[0]
        u_vertex_1: zonal (eastward positive) component at E2C2V[1]
        v_vertex_1: meridional (northward) component at E2C2V[1]
        u_vertex_2: zonal (eastward positive) component at E2C2V[2]
        v_vertex_2: meridional (northward) component at E2C2V[2]
        u_vertex_3: zonal (eastward positive) component at E2C2V[3]
        v_vertex_3: meridional (northward) component at E2C2V[3]

    """
    vertex_lat_0 = vertex_lat(E2C2V[0])
    vertex_lon_0 = vertex_lon(E2C2V[0])
    u_vertex_0, v_vertex_0 = zonal_and_meridional_components_on_edges(
        vertex_lat_0, vertex_lon_0, x, y, z
    )
    vertex_lat_1 = vertex_lat(E2C2V[1])
    vertex_lon_1 = vertex_lon(E2C2V[1])
    u_vertex_1, v_vertex_1 = zonal_and_meridional_components_on_edges(
        vertex_lat_1, vertex_lon_1, x, y, z
    )
    vertex_lat_2 = vertex_lat(E2C2V[2])
    vertex_lon_2 = vertex_lon(E2C2V[2])
    u_vertex_2, v_vertex_2 = zonal_and_meridional_components_on_edges(
        vertex_lat_2, vertex_lon_2, x, y, z
    )
    vertex_lat_3 = vertex_lat(E2C2V[3])
    vertex_lon_3 = vertex_lon(E2C2V[3])
    u_vertex_3, v_vertex_3 = zonal_and_meridional_components_on_edges(
        vertex_lat_3, vertex_lon_3, x, y, z
    )
    return (
        u_vertex_0,
        v_vertex_0,
        u_vertex_1,
        v_vertex_1,
        u_vertex_2,
        v_vertex_2,
        u_vertex_3,
        v_vertex_3,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_zonal_and_meridional_component_of_edge_field_at_vertex(  # noqa: PLR0917 [too-many-positional-arguments]
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
    u_vertex_1: fa.EdgeField[ta.wpfloat],
    v_vertex_1: fa.EdgeField[ta.wpfloat],
    u_vertex_2: fa.EdgeField[ta.wpfloat],
    v_vertex_2: fa.EdgeField[ta.wpfloat],
    u_vertex_3: fa.EdgeField[ta.wpfloat],
    v_vertex_3: fa.EdgeField[ta.wpfloat],
    u_vertex_4: fa.EdgeField[ta.wpfloat],
    v_vertex_4: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    zonal_and_meridional_component_of_edge_field_at_vertex(
        vertex_lat=vertex_lat,
        vertex_lon=vertex_lon,
        x=x,
        y=y,
        z=z,
        out=(
            u_vertex_1,
            v_vertex_1,
            u_vertex_2,
            v_vertex_2,
            u_vertex_3,
            v_vertex_3,
            u_vertex_4,
            v_vertex_4,
        ),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def zonal_and_meridional_component_of_edge_field_at_cell_center(
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
) -> tuple[
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
    fa.EdgeField[ta.wpfloat],
]:
    """
    Compute zonal (U) and meridional (V) component of a vector (x, y, z) at cell centers (lat, lon)

    The vector is defined on edges and the projection is computed for the neighboring cell center s of the edge.

    Args:
        cell_lat: latitude of cell centers
        cell_lon: longitude of cell centers
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        u_cell_0: zonal (U) component at first cell neighbor of the edge E2C[0]
        v_cell_0: meridional (V) component at first cell neighbor of the edge E2C[1]
        u_cell_0: zonal (U) component at first cell neighbor of the edge E2C[0]
        v_cell_0: meridional (V) component at first cell neighbor of the edge E2C[1]

    """
    cell_lat_0 = cell_lat(E2C[0])
    cell_lon_0 = cell_lon(E2C[0])
    u_cell_0, v_cell_0 = zonal_and_meridional_components_on_edges(cell_lat_0, cell_lon_0, x, y, z)
    cell_lat_1 = cell_lat(E2C[1])
    cell_lon_1 = cell_lon(E2C[1])
    u_cell_1, v_cell_1 = zonal_and_meridional_components_on_edges(cell_lat_1, cell_lon_1, x, y, z)
    return (
        u_cell_0,
        v_cell_0,
        u_cell_1,
        v_cell_1,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_zonal_and_meridional_component_of_edge_field_at_cell_center(  # noqa: PLR0917 [too-many-positional-arguments]
    cell_lat: fa.CellField[ta.wpfloat],
    cell_lon: fa.CellField[ta.wpfloat],
    x: fa.EdgeField[ta.wpfloat],
    y: fa.EdgeField[ta.wpfloat],
    z: fa.EdgeField[ta.wpfloat],
    u_cell_1: fa.EdgeField[ta.wpfloat],
    v_cell_1: fa.EdgeField[ta.wpfloat],
    u_cell_2: fa.EdgeField[ta.wpfloat],
    v_cell_2: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    zonal_and_meridional_component_of_edge_field_at_cell_center(
        cell_lat=cell_lat,
        cell_lon=cell_lon,
        x=x,
        y=y,
        z=z,
        out=(
            u_cell_1,
            v_cell_1,
            u_cell_2,
            v_cell_2,
        ),
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def cell_center_arc_distance(
    lat_neighbor_0: fa.EdgeField[ta.wpfloat],
    lon_neighbor_0: fa.EdgeField[ta.wpfloat],
    lat_neighbor_1: fa.EdgeField[ta.wpfloat],
    lon_neighbor_1: fa.EdgeField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute the distance between to cell centers.

    Computes the distance between the cell center of edge adjacent cells. This is a edge of the dual grid.

    Args:
        lat_neighbor_0: auxiliary vector of latitudes: cell centler of E2C[0] or edge center for boundary edges
        lon_neighbor_0: auxiliary vector of longitudes: cell centler of E2C[0] or edge center for boundary edges
        lat_neighbor_1: auxiliary vector of latitudes: cell centler of E2C[1] or edge center for boundary edges
        lon_neighbor_1: auxiliary vector of longitudes: cell centler of E2C[1] or edge center for boundary edges
        radius: radius of the sphere

    Returns:
        dual edge length

    """
    x0, y0, z0 = geographical_to_cartesian_on_edges(lat_neighbor_0, lon_neighbor_0)
    x1, y1, z1 = geographical_to_cartesian_on_edges(lat_neighbor_1, lon_neighbor_1)
    # (xi, yi, zi) are normalized by construction
    arc = arc_length_on_edges(x0, x1, y0, y1, z0, z1, radius)
    return arc


@gtx.field_operator
def arc_distance_of_far_edges_in_diamond(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute the arc length between the "far" vertices of an edge.

    Neighboring edges of an edge span up a diamond with 4 edges (E2C2E)  and 4 vertices (E2C2V). Here we compute the
    arc length between the two vertices in this diamond that are not directly connected to the edge:
    between d(v2, v3)
    v2-------v1
    |       /|
    |      / |
    |    e   |
    |  /     |
    |/       |
    v0 ------v3



    Args:
        vertex_lat: vertex latitude
        vertex_lon: vertex longitude
        radius: sphere radius

    Returns:
        arc length between the "far" vertices in the diamond.

    """
    x, y, z = geographical_to_cartesian_on_vertices(vertex_lat, vertex_lon)
    return arc_length_on_edges(
        x(E2C2V[2]),
        x(E2C2V[3]),
        y(E2C2V[2]),
        y(E2C2V[3]),
        z(E2C2V[2]),
        z(E2C2V[3]),
        radius,
    )


@gtx.field_operator
def distance_of_far_edges_in_diamond_torus(
    vertex_x: fa.VertexField[ta.wpfloat],
    vertex_y: fa.VertexField[ta.wpfloat],
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute the distance between the "far" vertices of an edge on a torus grid.

    See arc_distance_of_far_edges_in_diamond for details.

    Args:
        vertex_x: x coordinate of vertices
        vertex_y: y coordinate of vertices
        domain_length: length of the domain
        domain_height: height of the domain

    Returns:
        distance between the "far" vertices in the diamond.

    """
    return distance_on_edges_torus(
        vertex_x(E2C2V[2]),
        vertex_x(E2C2V[3]),
        vertex_y(E2C2V[2]),
        vertex_y(E2C2V[3]),
        domain_length,
        domain_height,
    )


@gtx.field_operator
def edge_length(
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute the arc length of an edge.

    This stencil could easily be inlined with `compute_arc_distance_of_far_edges_in_diamond`
    by using all indices in the E2C2V connectivity.
    They are kept separate due to different compute bounds.

    Args:
        vertex_lat: vertex latitudes
        vertex_lon: vertex longituds
        radius: sphere redius

    Returns:
        edge length
    """
    x, y, z = geographical_to_cartesian_on_vertices(vertex_lat, vertex_lon)
    x0 = x(E2V[0])
    x1 = x(E2V[1])
    y0 = y(E2V[0])
    y1 = y(E2V[1])
    z0 = z(E2V[0])
    z1 = z(E2V[1])
    # (xi, yi, zi) are normalized by construction

    length = arc_length_on_edges(x0, x1, y0, y1, z0, z1, radius)
    return length


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_length(  # noqa: PLR0917 [too-many-positional-arguments]
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
    length: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    edge_length(
        vertex_lat=vertex_lat,
        vertex_lon=vertex_lon,
        radius=radius,
        out=length,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_cell_center_arc_distance(  # noqa: PLR0917 [too-many-positional-arguments]
    edge_neighbor_0_lat: fa.EdgeField[ta.wpfloat],
    edge_neighbor_0_lon: fa.EdgeField[ta.wpfloat],
    edge_neighbor_1_lat: fa.EdgeField[ta.wpfloat],
    edge_neighbor_1_lon: fa.EdgeField[ta.wpfloat],
    radius: ta.wpfloat,
    dual_edge_length: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    cell_center_arc_distance(
        lat_neighbor_0=edge_neighbor_0_lat,
        lon_neighbor_0=edge_neighbor_0_lon,
        lat_neighbor_1=edge_neighbor_1_lat,
        lon_neighbor_1=edge_neighbor_1_lon,
        radius=radius,
        out=dual_edge_length,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_arc_distance_of_far_edges_in_diamond(  # noqa: PLR0917 [too-many-positional-arguments]
    vertex_lat: fa.VertexField[ta.wpfloat],
    vertex_lon: fa.VertexField[ta.wpfloat],
    radius: ta.wpfloat,
    far_vertex_distance: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    arc_distance_of_far_edges_in_diamond(
        vertex_lat=vertex_lat,
        vertex_lon=vertex_lon,
        radius=radius,
        out=far_vertex_distance,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_distance_of_far_edges_in_diamond_torus(  # noqa: PLR0917 [too-many-positional-arguments]
    vertex_x: fa.VertexField[ta.wpfloat],
    vertex_y: fa.VertexField[ta.wpfloat],
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
    far_vertex_distance: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    distance_of_far_edges_in_diamond_torus(
        vertex_x=vertex_x,
        vertex_y=vertex_y,
        domain_length=domain_length,
        domain_height=domain_height,
        out=far_vertex_distance,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def edge_area(
    owner_mask: fa.EdgeField[bool],
    primal_edge_length: fa.EdgeField[ta.wpfloat],
    dual_edge_length: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute the area spanned by an edge and the its dual edge
    Args:
        owner_mask: owner mask for edges
        primal_edge_length: length of edge in primal grid
        dual_edge_length: length of edge in dual grid

    Returns:
        area

    """
    return where(owner_mask, primal_edge_length * dual_edge_length, 0.0)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_edge_area(  # noqa: PLR0917 [too-many-positional-arguments]
    owner_mask: fa.EdgeField[bool],
    primal_edge_length: fa.EdgeField[ta.wpfloat],
    dual_edge_length: fa.EdgeField[ta.wpfloat],
    area: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
):
    edge_area(
        owner_mask=owner_mask,
        primal_edge_length=primal_edge_length,
        dual_edge_length=dual_edge_length,
        out=area,
        domain={EdgeDim: (horizontal_start, horizontal_end)},
    )


@gtx.field_operator
def coriolis_parameter_on_edges(
    edge_center_lat: fa.EdgeField[ta.wpfloat],
    angular_velocity: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute the coriolis force on edges.
    Args:
       edge_center_lat: latitude of edge center
       angular_velocity: angular velocity

    Returns:
        coriolis parameter
    """
    return 2.0 * angular_velocity * sin(edge_center_lat)


def coriolis_parameter_on_edges_torus(
    coriolis_coefficient: float,
    num_edges: int,
    backend: gtx_typing.Backend,
) -> fa.EdgeField[ta.wpfloat]:
    """
    Create a coriolis parameter field on edges for a torus grid.
    Args:
       coriolis_coefficient: coriolis coefficient

    Returns:
        coriolis parameter
    """
    xp = data_alloc.import_array_ns(backend)
    coriolis_parameter = gtx.as_field(
        (dims.EdgeDim,),
        xp.full(num_edges, coriolis_coefficient),
        dtype=ta.wpfloat,
        allocator=backend,
    )
    return coriolis_parameter


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_coriolis_parameter_on_edges(
    edge_center_lat: fa.EdgeField[ta.wpfloat],
    angular_velocity: ta.wpfloat,
    coriolis_parameter: fa.EdgeField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
) -> None:
    coriolis_parameter_on_edges(
        edge_center_lat=edge_center_lat,
        angular_velocity=angular_velocity,
        out=coriolis_parameter,
        domain={dims.EdgeDim: (horizontal_start, horizontal_end)},
    )


def compute_primal_cart_normal(
    primal_cart_normal_x: data_alloc.NDArray,
    primal_cart_normal_y: data_alloc.NDArray,
    primal_cart_normal_z: data_alloc.NDArray,
) -> data_alloc.NDArray:
    array_ns = data_alloc.array_namespace(primal_cart_normal_x)
    primal_cart_normal = array_ns.transpose(
        array_ns.stack(
            (
                primal_cart_normal_x,
                primal_cart_normal_y,
                primal_cart_normal_z,
            )
        )
    )
    return primal_cart_normal
