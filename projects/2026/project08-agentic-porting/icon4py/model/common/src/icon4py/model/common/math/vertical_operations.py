# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

"""
Vertical level operations on unstructured grid fields.

Contains averaging and difference operations between adjacent vertical levels
on cell and edge fields.
"""

from gt4py import next as gtx
from gt4py.next.experimental import concat_where

from icon4py.model.common import dimension as dims, field_type_aliases as fa
from icon4py.model.common.dimension import KDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def average_level_plus1_on_cells(
    half_level_field: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Calculate the mean value of adjacent interface levels.

    Computes the average of two adjacent interface levels upwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[Dims[CellDim, dims.KDim], wpfloat]

    Returns: Field[Dims[CellDim, dims.KDim], wpfloat] full level field

    """
    return 0.5 * (half_level_field + half_level_field(KDim + 1))


@gtx.field_operator
def average_level_plus1_on_edges(
    half_level_field: fa.EdgeKField[wpfloat],
) -> fa.EdgeKField[wpfloat]:
    """
    Calculate the mean value of adjacent interface levels.

    Computes the average of two adjacent interface levels upwards over an edge field for storage
    in the corresponding full levels.
    Args:
        half_level_field: fa.EdgeKField[wpfloat]

    Returns: fa.EdgeKField[wpfloat] full level field

    """
    return 0.5 * (half_level_field + half_level_field(KDim + 1))


@gtx.field_operator
def difference_level_plus1_on_cells(
    half_level_field: fa.CellKField[wpfloat],
) -> fa.CellKField[wpfloat]:
    """
    Calculate the difference value of adjacent interface levels.

    Computes the difference of two adjacent interface levels upwards over a cell field for storage
    in the corresponding full levels.
    Args:
        half_level_field: Field[Dims[CellDim, dims.KDim], wpfloat]

    Returns: Field[Dims[CellDim, dims.KDim], wpfloat] full level field

    """
    return half_level_field - half_level_field(KDim + 1)


@gtx.field_operator
def with_boundaries_on_half_levels_on_cells(
    top: fa.CellKField[wpfloat],
    interior: fa.CellKField[wpfloat],
    bottom: fa.CellKField[wpfloat],
    nlev: gtx.int32,
) -> fa.CellKField[wpfloat]:
    """
    Assemble a half-level field: ``top`` at k==0, ``bottom`` at k==nlev, ``interior`` in between.

    Each branch is evaluated only on its own region, so vertical (``Koff``) shifts in the
    arguments need to be in bounds only within that region.
    """
    result = concat_where(
        (dims.KDim > 0) & (dims.KDim < nlev),
        interior,
        0.0,
    )
    result = concat_where(dims.KDim == 0, top, result)
    return concat_where(dims.KDim == nlev, bottom, result)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def average_two_vertical_levels_downwards_on_edges(  # noqa: PLR0917 [too-many-positional-arguments]
    input_field: fa.EdgeKField[wpfloat],
    average: fa.EdgeKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    average_level_plus1_on_edges(
        half_level_field=input_field,
        out=average,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def average_two_vertical_levels_downwards_on_cells(  # noqa: PLR0917 [too-many-positional-arguments]
    input_field: fa.CellKField[wpfloat],
    average: fa.CellKField[wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    average_level_plus1_on_cells(
        half_level_field=input_field,
        out=average,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
