# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import gt4py.next as gtx
from gt4py.next import astype, max_over, maximum, min_over, minimum

from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import C2E2C
from icon4py.model.common.type_alias import vpfloat, wpfloat


# TODO(dastrm): this stencil has no test


@gtx.field_operator
def _compute_monotone_horizontal_multiplicative_flux_factors_min_max(
    z_tracer_max: fa.CellKField[ta.vpfloat],
    z_tracer_min: fa.CellKField[ta.vpfloat],
    beta_fct: ta.wpfloat,
    r_beta_fct: ta.wpfloat,
) -> tuple[fa.CellKField[ta.vpfloat], fa.CellKField[ta.vpfloat]]:
    z_max = astype(beta_fct, vpfloat) * maximum(
        max_over(z_tracer_max(C2E2C), axis=dims.C2E2CDim), z_tracer_max
    )
    z_min = astype(r_beta_fct, vpfloat) * minimum(
        min_over(z_tracer_min(C2E2C), axis=dims.C2E2CDim), z_tracer_min
    )
    return z_max, z_min


@gtx.field_operator
def _compute_monotone_horizontal_multiplicative_flux_factors_p_m(
    z_mflx_anti_in: fa.CellKField[ta.vpfloat],
    z_mflx_anti_out: fa.CellKField[ta.vpfloat],
    z_tracer_new_low: fa.CellKField[ta.wpfloat],
    z_max: fa.CellKField[ta.vpfloat],
    z_min: fa.CellKField[ta.vpfloat],
    dbl_eps: ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    r_p = (astype(z_max, wpfloat) - z_tracer_new_low) / (astype(z_mflx_anti_in, wpfloat) + dbl_eps)
    r_m = (z_tracer_new_low - astype(z_min, wpfloat)) / (astype(z_mflx_anti_out, wpfloat) + dbl_eps)

    return r_p, r_m


@gtx.field_operator
def _compute_monotone_horizontal_multiplicative_flux_factors(
    z_tracer_max: fa.CellKField[ta.vpfloat],
    z_tracer_min: fa.CellKField[ta.vpfloat],
    z_mflx_anti_in: fa.CellKField[ta.vpfloat],
    z_mflx_anti_out: fa.CellKField[ta.vpfloat],
    z_tracer_new_low: fa.CellKField[ta.wpfloat],
    beta_fct: ta.wpfloat,
    r_beta_fct: ta.wpfloat,
    dbl_eps: ta.wpfloat,
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    z_max, z_min = _compute_monotone_horizontal_multiplicative_flux_factors_min_max(
        z_tracer_max=z_tracer_max,
        z_tracer_min=z_tracer_min,
        beta_fct=beta_fct,
        r_beta_fct=r_beta_fct,
    )

    r_p, r_m = _compute_monotone_horizontal_multiplicative_flux_factors_p_m(
        z_mflx_anti_in=z_mflx_anti_in,
        z_mflx_anti_out=z_mflx_anti_out,
        z_tracer_new_low=z_tracer_new_low,
        z_max=z_max,
        z_min=z_min,
        dbl_eps=dbl_eps,
    )
    return r_p, r_m


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_monotone_horizontal_multiplicative_flux_factors(
    z_tracer_max: fa.CellKField[ta.vpfloat],
    z_tracer_min: fa.CellKField[ta.vpfloat],
    z_mflx_anti_in: fa.CellKField[ta.vpfloat],
    z_mflx_anti_out: fa.CellKField[ta.vpfloat],
    z_tracer_new_low: fa.CellKField[ta.wpfloat],
    r_p: fa.CellKField[ta.wpfloat],
    r_m: fa.CellKField[ta.wpfloat],
    beta_fct: ta.wpfloat,
    r_beta_fct: ta.wpfloat,
    dbl_eps: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    _compute_monotone_horizontal_multiplicative_flux_factors(
        z_tracer_max=z_tracer_max,
        z_tracer_min=z_tracer_min,
        z_mflx_anti_in=z_mflx_anti_in,
        z_mflx_anti_out=z_mflx_anti_out,
        z_tracer_new_low=z_tracer_new_low,
        beta_fct=beta_fct,
        r_beta_fct=r_beta_fct,
        dbl_eps=dbl_eps,
        out=(r_p, r_m),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
