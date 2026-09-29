# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Vector operations on unstructured grid fields.

Contains dot product, cross product, norm and normalization operations
for vectors defined on cell, edge and vertex fields.
"""

from gt4py import next as gtx
from gt4py.next import sqrt

from icon4py.model.common import field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def dot_product_on_edges(  # noqa: PLR0917 [too-many-positional-arguments]
    x1: fa.EdgeField[ta.wpfloat],
    x2: fa.EdgeField[ta.wpfloat],
    y1: fa.EdgeField[ta.wpfloat],
    y2: fa.EdgeField[ta.wpfloat],
    z1: fa.EdgeField[ta.wpfloat],
    z2: fa.EdgeField[ta.wpfloat],
) -> fa.EdgeField[ta.wpfloat]:
    """Compute dot product of cartesian vectors (x1, y1, z1) * (x2, y2, z2)"""
    return x1 * x2 + y1 * y2 + z1 * z2


@gtx.field_operator
def dot_product_on_cells(  # noqa: PLR0917 [too-many-positional-arguments]
    x1: fa.CellField[ta.wpfloat],
    x2: fa.CellField[ta.wpfloat],
    y1: fa.CellField[ta.wpfloat],
    y2: fa.CellField[ta.wpfloat],
    z1: fa.CellField[ta.wpfloat],
    z2: fa.CellField[ta.wpfloat],
) -> fa.CellField[ta.wpfloat]:
    """Compute dot product of cartesian vectors (x1, y1, z1) * (x2, y2, z2)"""
    return x1 * x2 + y1 * y2 + z1 * z2


@gtx.field_operator
def dot_product_on_vertices(  # noqa: PLR0917 [too-many-positional-arguments]
    x1: fa.VertexField[ta.wpfloat],
    x2: fa.VertexField[ta.wpfloat],
    y1: fa.VertexField[ta.wpfloat],
    y2: fa.VertexField[ta.wpfloat],
    z1: fa.VertexField[ta.wpfloat],
    z2: fa.VertexField[ta.wpfloat],
) -> fa.VertexField[ta.wpfloat]:
    """Compute dot product of cartesian vectors (x1, y1, z1) * (x2, y2, z2)"""
    return x1 * x2 + y1 * y2 + z1 * z2


@gtx.field_operator
def cross_product_on_edges(  # noqa: PLR0917 [too-many-positional-arguments]
    x1: fa.EdgeField[ta.wpfloat],
    x2: fa.EdgeField[ta.wpfloat],
    y1: fa.EdgeField[ta.wpfloat],
    y2: fa.EdgeField[ta.wpfloat],
    z1: fa.EdgeField[ta.wpfloat],
    z2: fa.EdgeField[ta.wpfloat],
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    """Compute cross product of cartesian vectors (x1, y1, z1) x (x2, y2, z2)"""
    x = y1 * z2 - z1 * y2
    y = z1 * x2 - x1 * z2
    z = x1 * y2 - y1 * x2
    return x, y, z


@gtx.field_operator
def norm2_on_edges(
    x: fa.EdgeField[ta.wpfloat], y: fa.EdgeField[ta.wpfloat], z: fa.EdgeField[ta.wpfloat]
) -> fa.EdgeField[ta.wpfloat]:
    """
    Compute 2 norm of a cartesian vector (x, y, z)
    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        norma

    """
    return sqrt(dot_product_on_edges(x1=x, x2=x, y1=y, y2=y, z1=z, z2=z))


@gtx.field_operator
def norm2_on_cells(
    x: fa.CellField[ta.wpfloat], y: fa.CellField[ta.wpfloat], z: fa.CellField[ta.wpfloat]
) -> fa.CellField[ta.wpfloat]:
    """
    Compute 2 norm of a cartesian vector (x, y, z)
    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        norma

    """
    return sqrt(dot_product_on_cells(x1=x, x2=x, y1=y, y2=y, z1=z, z2=z))


@gtx.field_operator
def norm2_on_vertices(
    x: fa.VertexField[ta.wpfloat], y: fa.VertexField[ta.wpfloat], z: fa.VertexField[ta.wpfloat]
) -> fa.VertexField[ta.wpfloat]:
    """
    Compute 2 norm of a cartesian vector (x, y, z)
    Args:
        x: x coordinate
        y: y coordinate
        z: z coordinate

    Returns:
        norma

    """
    return sqrt(dot_product_on_vertices(x1=x, x2=x, y1=y, y2=y, z1=z, z2=z))


@gtx.field_operator
def normalize_cartesian_vector_on_edges(
    v_x: fa.EdgeField[ta.wpfloat], v_y: fa.EdgeField[ta.wpfloat], v_z: fa.EdgeField[ta.wpfloat]
) -> tuple[fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat], fa.EdgeField[ta.wpfloat]]:
    """
    Normalize a cartesian vector.

    Args:
        v_x: x coordinate
        v_y: y coordinate
        v_z: z coordinate

    Returns:
        normalized vector

    """
    norm = norm2_on_edges(v_x, v_y, v_z)
    return v_x / norm, v_y / norm, v_z / norm
