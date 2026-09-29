# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import abs, broadcast, maximum  # noqa: A004

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common.dimension import EdgeDim, KDim
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def _scale_k(field: fa.KField[float], factor: float) -> fa.KField[float]:
    return field * factor


@gtx.program
def scale_k(field: fa.KField[float], factor: float, scaled_field: fa.KField[float]) -> None:
    _scale_k(field, factor, out=scaled_field)


@gtx.field_operator
def _broadcast_zero_to_three_edge_kdim_fields_wp() -> tuple[
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
    fa.EdgeKField[wpfloat],
]:
    return (
        broadcast(wpfloat("0.0"), (EdgeDim, KDim)),
        broadcast(wpfloat("0.0"), (EdgeDim, KDim)),
        broadcast(wpfloat("0.0"), (EdgeDim, KDim)),
    )


@gtx.field_operator
def _calculate_reduced_fourth_order_divdamp_coeff_at_nest_boundary(
    fourth_order_divdamp_scaling_coeff: fa.KField[float],
    max_nudging_coefficient: float,
    dbl_eps: float,
) -> fa.KField[float]:
    return 0.75 / (max_nudging_coefficient + dbl_eps) * abs(fourth_order_divdamp_scaling_coeff)


@gtx.field_operator
def _calculate_fourth_order_divdamp_scaling_coeff(
    interpolated_fourth_order_divdamp_factor: fa.KField[float],
    divdamp_order: gtx.int32,
    mean_cell_area: float,
    second_order_divdamp_factor: float,
) -> fa.KField[float]:
    interpolated_fourth_order_divdamp_factor = (
        maximum(0.0, interpolated_fourth_order_divdamp_factor - 0.25 * second_order_divdamp_factor)
        if divdamp_order == 24
        else interpolated_fourth_order_divdamp_factor
    )
    return -interpolated_fourth_order_divdamp_factor * mean_cell_area**2


@gtx.field_operator
def _calculate_divdamp_fields(  # noqa: PLR0917 [too-many-positional-arguments]
    interpolated_fourth_order_divdamp_factor: fa.KField[float],
    divdamp_order: gtx.int32,
    mean_cell_area: float,
    second_order_divdamp_factor: float,
    max_nudging_coefficient: float,
    dbl_eps: float,
) -> tuple[fa.KField[float], fa.KField[float]]:
    fourth_order_divdamp_scaling_coeff = _calculate_fourth_order_divdamp_scaling_coeff(
        interpolated_fourth_order_divdamp_factor,
        divdamp_order,
        mean_cell_area,
        second_order_divdamp_factor,
    )
    reduced_fourth_order_divdamp_coeff_at_nest_boundary = (
        _calculate_reduced_fourth_order_divdamp_coeff_at_nest_boundary(
            fourth_order_divdamp_scaling_coeff, max_nudging_coefficient, dbl_eps
        )
    )
    return (fourth_order_divdamp_scaling_coeff, reduced_fourth_order_divdamp_coeff_at_nest_boundary)


@gtx.program
def calculate_divdamp_fields(  # noqa: PLR0917 [too-many-positional-arguments]
    interpolated_fourth_order_divdamp_factor: fa.KField[float],
    fourth_order_divdamp_scaling_coeff: fa.KField[float],
    reduced_fourth_order_divdamp_coeff_at_nest_boundary: fa.KField[float],
    divdamp_order: gtx.int32,
    mean_cell_area: float,
    second_order_divdamp_factor: float,
    max_nudging_coefficient: float,
    dbl_eps: float,
) -> None:
    _calculate_divdamp_fields(
        interpolated_fourth_order_divdamp_factor=interpolated_fourth_order_divdamp_factor,
        divdamp_order=divdamp_order,
        mean_cell_area=mean_cell_area,
        second_order_divdamp_factor=second_order_divdamp_factor,
        max_nudging_coefficient=max_nudging_coefficient,
        dbl_eps=dbl_eps,
        out=(
            fourth_order_divdamp_scaling_coeff,
            reduced_fourth_order_divdamp_coeff_at_nest_boundary,
        ),
    )


@gtx.field_operator
def _compute_rayleigh_damping_factor(
    rayleigh_w: fa.KField[float], dtime: float
) -> fa.KField[float]:
    return 1.0 / (1.0 + dtime * rayleigh_w)


@gtx.program
def compute_rayleigh_damping_factor(
    rayleigh_w: fa.KField[float], dtime: float, rayleigh_damping_factor: fa.KField[float]
) -> None:
    _compute_rayleigh_damping_factor(rayleigh_w, dtime, out=rayleigh_damping_factor)
