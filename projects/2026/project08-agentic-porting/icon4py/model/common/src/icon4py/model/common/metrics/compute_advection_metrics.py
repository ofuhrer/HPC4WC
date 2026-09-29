# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
from __future__ import annotations

from gt4py import next as gtx

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta


@gtx.field_operator
def _compute_advection_deepatmo_fields(
    height_u: fa.KField[ta.wpfloat],
    height_l: fa.KField[ta.wpfloat],
    grid_sphere_radius: ta.wpfloat,
) -> tuple[fa.KField[ta.wpfloat], fa.KField[ta.wpfloat], fa.KField[ta.wpfloat]]:
    """
    Compute 'deepatmo_divh', 'deepatmo_divzL', 'deepatmo_divzU' from 'vct_a' and 'grid_sphere_radius'.

    # Input Fields:
    - height_u: expects vct_a[:nlev]
    - height_l: expects vct_a[1:nlev+1]

    # Output Fields:
    - deepatmo_divh
    - deepatmo_divzL
    - deepatmo_divzU
    """
    height = 0.5 * (height_l + height_u)
    radial_distance = height + grid_sphere_radius
    radial_distance_l = grid_sphere_radius + height_l
    radial_distance_u = grid_sphere_radius + height_u
    deepatmo_gradh = grid_sphere_radius / radial_distance
    deepatmo_divh = (
        deepatmo_gradh
        * 3.0
        / 4.0
        / (
            1.0
            - radial_distance_l * radial_distance_u / (radial_distance_l + radial_distance_u) ** 2
        )
    )
    deepatmo_divzL = 3.0 / (
        1.0 + radial_distance_u / radial_distance_l + (radial_distance_u / radial_distance_l) ** 2
    )
    deepatmo_divzU = 3.0 / (
        1.0 + radial_distance_l / radial_distance_u + (radial_distance_l / radial_distance_u) ** 2
    )
    return deepatmo_divh, deepatmo_divzL, deepatmo_divzU


@gtx.program
def compute_advection_deepatmo_fields(  # noqa: PLR0917 [too-many-positional-arguments]
    height_u: fa.KField[ta.wpfloat],
    height_l: fa.KField[ta.wpfloat],
    deepatmo_divh: fa.KField[ta.wpfloat],
    deepatmo_divzL: fa.KField[ta.wpfloat],
    deepatmo_divzU: fa.KField[ta.wpfloat],
    grid_sphere_radius: float,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Compute 'deepatmo_divh', 'deepatmo_divzL', 'deepatmo_divzU' from 'vct_a' and 'grid_sphere_radius'.

    # Input Fields:
    - height_u: expects vct_a[:nlev]
    - height_l: expects vct_a[1:nlev+1]

    # Inout Fields:
    - deepatmo_divh
    - deepatmo_divzL
    - deepatmo_divzU
    """
    _compute_advection_deepatmo_fields(
        height_u=height_u,
        height_l=height_l,
        grid_sphere_radius=grid_sphere_radius,
        out=(deepatmo_divh, deepatmo_divzL, deepatmo_divzU),
        domain={dims.KDim: (vertical_start, vertical_end)},
    )
