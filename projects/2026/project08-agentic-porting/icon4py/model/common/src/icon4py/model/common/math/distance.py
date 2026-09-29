# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Distance and arc length operations on unstructured grid fields.

Contains arc length computation on spheres and distance/difference operations
on torus geometries.
"""

from gt4py import next as gtx
from gt4py.next import (
    abs,  # noqa: A004
    arccos,
    sqrt,
    where,
)

from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.math.vector_operations import dot_product_on_edges


@gtx.field_operator
def arc_length_on_edges(  # noqa: PLR0917 [too-many-positional-arguments]
    x0: fa.EdgeField[ta.wpfloat],
    x1: fa.EdgeField[ta.wpfloat],
    y0: fa.EdgeField[ta.wpfloat],
    y1: fa.EdgeField[ta.wpfloat],
    z0: fa.EdgeField[ta.wpfloat],
    z1: fa.EdgeField[ta.wpfloat],
    radius: ta.wpfloat,
):
    """
    Compute the arc length between two points on the sphere.

    Inputs are cartesian coordinates of the points.

    Args:
        x0: x coordinate of point_0
        x1: x coordinate of point_1
        y0: y coordinate of point_0
        y1: y coordinate of point_1
        z0: z coordinate of point_0
        z1: z coordinate of point_1
        radius: sphere radius

    Returns:
        arc length

    """
    return radius * arccos(dot_product_on_edges(x0, x1, y0, y1, z0, z1))


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def diff_on_edges_torus(  # noqa: PLR0917 [too-many-positional-arguments]
    x0: fa.EdgeField[ta.wpfloat],
    x1: fa.EdgeField[ta.wpfloat],
    y0: fa.EdgeField[ta.wpfloat],
    y1: fa.EdgeField[ta.wpfloat],
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    """
    Compute the difference between two points on the torus.

    Inputs are cartesian coordinates of the points. Z is assumed zero and is
    ignored. Distances are computed modulo the domain length and height.

    Args:
        x0: x coordinate of point_0
        x1: x coordinate of point_1
        y0: y coordinate of point_0
        y1: y coordinate of point_1
        domain_length: length of the domain
        domain_height: height of the domain

    Returns:
        dx, dy

    """
    x1 = where(
        abs(x1 - x0) <= 0.5 * domain_length,
        x1,
        where(x0 > x1, x1 + domain_length, x1 - domain_length),
    )

    y1 = where(
        abs(y1 - y0) <= 0.5 * domain_height,
        y1,
        where(y0 > y1, y1 + domain_height, y1 - domain_height),
    )

    return (x1 - x0, y1 - y0)


@gtx.field_operator(grid_type=gtx.GridType.UNSTRUCTURED)
def distance_on_edges_torus(  # noqa: PLR0917 [too-many-positional-arguments]
    x0: fa.EdgeField[ta.wpfloat],
    x1: fa.EdgeField[ta.wpfloat],
    y0: fa.EdgeField[ta.wpfloat],
    y1: fa.EdgeField[ta.wpfloat],
    domain_length: ta.wpfloat,
    domain_height: ta.wpfloat,
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute the distance between two points on the torus.

    Inputs are cartesian coordinates of the points. Z is assumed zero and is
    ignored. Distances are computed modulo the domain length and height.

    Args:
        x0: x coordinate of point_0
        x1: x coordinate of point_1
        y0: y coordinate of point_0
        y1: y coordinate of point_1
        domain_length: length of the domain
        domain_height: height of the domain

    Returns:
        distance

    """
    xdiff, ydiff = diff_on_edges_torus(
        x0=x0, x1=x1, y0=y0, y1=y1, domain_length=domain_length, domain_height=domain_height
    )
    return sqrt(xdiff**2 + ydiff**2)
