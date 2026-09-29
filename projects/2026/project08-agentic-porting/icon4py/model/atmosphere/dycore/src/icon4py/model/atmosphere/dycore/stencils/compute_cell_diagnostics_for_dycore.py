# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# This file is free software: you can redistribute it and/or modify it under
# the terms of the GNU General Public License as published by the
# Free Software Foundation, either version 3 of the License, or any later
# version. See the LICENSE.txt file at the top-level directory of this
# distribution for a copy of the license or check <https://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later


import gt4py.next as gtx
from gt4py.next import astype, broadcast, maximum
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.dycore.dycore_states import HorizontalPressureDiscretizationType
from icon4py.model.atmosphere.dycore.stencils.compute_perturbation_of_rho_and_theta import (
    _compute_perturbation_of_rho_and_theta,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_temporally_exner_pressure import (
    _extrapolate_temporally_exner_pressure,
)
from icon4py.model.atmosphere.dycore.stencils.interpolate_to_surface import _interpolate_to_surface
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import KDim
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels_vp import (
    _interpolate_cell_field_to_half_levels_vp,
)
from icon4py.model.common.interpolation.stencils.interpolate_cell_field_to_half_levels_wp import (
    _interpolate_cell_field_to_half_levels_wp,
)
from icon4py.model.common.math.derivative import _compute_first_vertical_derivative_at_cells
from icon4py.model.common.type_alias import vpfloat, wpfloat


@gtx.field_operator
def _calculate_nonhydro_buoy_at_cells_on_half_levels(
    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    ddqz_z_half: fa.CellKField[ta.wpfloat],
    perturbed_theta_v_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
) -> fa.CellKField[ta.wpfloat]:
    return exner_w_explicit_weight_parameter * theta_v_at_cells_on_half_levels * (
        perturbed_exner_at_cells_on_model_levels(KDim - 1)
        - perturbed_exner_at_cells_on_model_levels
    ) / ddqz_z_half + astype(
        perturbed_theta_v_at_cells_on_half_levels * ddz_of_reference_exner_at_cells_on_half_levels,
        wpfloat,
    )


@gtx.field_operator
def _compute_perturbed_quantities_and_interpolation(
    current_rho: fa.CellKField[ta.wpfloat],
    reference_rho_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    reference_theta_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    nonhydro_buoy_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[ta.vpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    igradp_method: gtx.int32,
    nflatlev: gtx.int32,
) -> tuple[
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
]:
    exner_at_cells_on_half_levels = (
        concat_where(
            (maximum(1, nflatlev) <= dims.KDim),
            _interpolate_cell_field_to_half_levels_vp(
                wgtfac_c=wgtfac_c, interpolant=temporal_extrapolation_of_perturbed_exner
            ),
            exner_at_cells_on_half_levels,
        )
        if igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO
        else exner_at_cells_on_half_levels
    )

    (
        perturbed_rho_at_cells_on_model_levels,
        perturbed_theta_v_at_cells_on_model_levels,
    ) = _compute_perturbation_of_rho_and_theta(
        current_rho,
        reference_rho_at_cells_on_model_levels,
        current_theta_v,
        reference_theta_at_cells_on_model_levels,
    )

    rho_at_cells_on_half_levels = concat_where(
        dims.KDim >= 1,
        _interpolate_cell_field_to_half_levels_wp(wgtfac_c, current_rho),
        rho_at_cells_on_half_levels,
    )

    wgtfac_c_wp = astype(wgtfac_c, wpfloat)

    perturbed_theta_v_at_cells_on_half_levels = concat_where(
        dims.KDim >= 1,
        _interpolate_cell_field_to_half_levels_vp(
            wgtfac_c=wgtfac_c, interpolant=perturbed_theta_v_at_cells_on_model_levels
        ),
        broadcast(0.0, (dims.CellDim, dims.KDim)),
    )

    theta_v_at_cells_on_half_levels = concat_where(
        dims.KDim >= 1,
        _interpolate_cell_field_to_half_levels_wp(
            wgtfac_c=wgtfac_c_wp, interpolant=current_theta_v
        ),
        theta_v_at_cells_on_half_levels,
    )

    ddqz_z_half_wp = astype(ddqz_z_half, wpfloat)

    nonhydro_buoy_at_cells_on_half_levels = concat_where(
        dims.KDim >= 1,
        _calculate_nonhydro_buoy_at_cells_on_half_levels(
            exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
            theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
            perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
            ddqz_z_half=ddqz_z_half_wp,
            perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
            ddz_of_reference_exner_at_cells_on_half_levels=ddz_of_reference_exner_at_cells_on_half_levels,
        ),
        nonhydro_buoy_at_cells_on_half_levels,
    )

    return (
        perturbed_rho_at_cells_on_model_levels,
        perturbed_theta_v_at_cells_on_model_levels,
        perturbed_exner_at_cells_on_model_levels,
        rho_at_cells_on_half_levels,
        exner_at_cells_on_half_levels,
        perturbed_theta_v_at_cells_on_half_levels,
        theta_v_at_cells_on_half_levels,
        nonhydro_buoy_at_cells_on_half_levels,
    )


@gtx.field_operator
def _surface_computations(
    wgtfacq_c: fa.CellKField[ta.wpfloat],
    exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[vpfloat],
    igradp_method: gtx.int32,
) -> fa.CellKField[ta.vpfloat]:
    exner_at_cells_on_half_levels = (
        _interpolate_to_surface(
            wgtfacq_c=wgtfacq_c, interpolant=temporal_extrapolation_of_perturbed_exner
        )
        if igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO
        else exner_at_cells_on_half_levels
    )

    return exner_at_cells_on_half_levels


@gtx.field_operator
def _compute_first_and_second_vertical_derivative_of_exner(
    exner_at_cells_on_half_levels: fa.CellKField[vpfloat],
    inv_ddqz_z_full: fa.CellKField[vpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[vpfloat],
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[vpfloat],
    perturbed_theta_v_at_cells_on_half_levels: fa.CellKField[vpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[vpfloat],
    perturbed_theta_v_at_cells_on_model_levels: fa.CellKField[vpfloat],
    igradp_method: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
) -> tuple[
    fa.CellKField[vpfloat],
    fa.CellKField[vpfloat],
]:
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
        concat_where(
            (nflatlev <= dims.KDim),
            _compute_first_vertical_derivative_at_cells(
                exner_at_cells_on_half_levels, inv_ddqz_z_full
            ),
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        )
        if igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO
        else ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels
    )

    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels = (
        concat_where(
            (nflat_gradp <= dims.KDim),
            -vpfloat("0.5")
            * (
                (
                    perturbed_theta_v_at_cells_on_half_levels
                    - perturbed_theta_v_at_cells_on_half_levels(KDim + 1)
                )
                * d2dexdz2_fac1_mc
                + perturbed_theta_v_at_cells_on_model_levels * d2dexdz2_fac2_mc
            ),
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        )
        if igradp_method == HorizontalPressureDiscretizationType.TAYLOR_HYDRO
        else d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels
    )

    return (
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
    )


@gtx.field_operator
def _set_theta_v_and_exner_on_surface_level(
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[vpfloat],
    wgtfacq_c: fa.CellKField[vpfloat],
    perturbed_theta_v_at_cells_on_model_levels: fa.CellKField[vpfloat],
    reference_theta_at_cells_on_half_levels: fa.CellKField[vpfloat],
) -> tuple[fa.CellKField[vpfloat], fa.CellKField[wpfloat], fa.CellKField[vpfloat]]:
    perturbed_theta_v_at_cells_on_half_levels = _interpolate_to_surface(
        wgtfacq_c=wgtfacq_c, interpolant=perturbed_theta_v_at_cells_on_model_levels
    )
    theta_v_at_cells_on_half_levels = (
        reference_theta_at_cells_on_half_levels + perturbed_theta_v_at_cells_on_half_levels
    )

    exner_at_cells_on_half_levels = _interpolate_to_surface(
        wgtfacq_c=wgtfacq_c, interpolant=temporal_extrapolation_of_perturbed_exner
    )

    return (
        perturbed_theta_v_at_cells_on_half_levels,
        astype(theta_v_at_cells_on_half_levels, wpfloat),
        exner_at_cells_on_half_levels,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_perturbed_quantities_and_interpolation(
    temporal_extrapolation_of_perturbed_exner: fa.CellKField[ta.vpfloat],
    ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: fa.CellKField[ta.vpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    perturbed_rho_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    perturbed_theta_v_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    perturbed_theta_v_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    reference_rho_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    reference_theta_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    reference_theta_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    wgtfacq_c: fa.CellKField[ta.vpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat],
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    nonhydro_buoy_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    time_extrapolation_parameter_for_exner: fa.CellKField[ta.vpfloat],
    current_exner: fa.CellKField[ta.wpfloat],
    reference_exner_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    inv_ddqz_z_full: fa.CellKField[ta.wpfloat],
    d2dexdz2_fac1_mc: fa.CellKField[ta.vpfloat],
    d2dexdz2_fac2_mc: fa.CellKField[ta.vpfloat],
    igradp_method: gtx.int32,
    nflatlev: gtx.int32,
    nflat_gradp: gtx.int32,
    start_cell_lateral_boundary: gtx.int32,
    start_cell_lateral_boundary_level_3: gtx.int32,
    start_cell_halo_level_2: gtx.int32,
    end_cell_halo: gtx.int32,
    end_cell_halo_level_2: gtx.int32,
    model_top: gtx.int32,
    surface_level: gtx.int32,
) -> None:
    """
    Formerly known as fused_solve_nonhydro_stencil_1_to_13_predictor.

    This program calculates the first and second vertical derivatives of the exner function, and also computes the perturbed
    air density and vitural potential temperature on half and model levels.

    Args:
        - temporal_extrapolation_of_perturbed_exner: temporal extrapolation of perturbed exner function (actual exner function minus reference exner function) at cells on model levels
        - ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: vertical gradient of temporal extrapolation of perturbed exner function at cells on model levels [m-1]
        - d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels: second vertical gradient of temporal extrapolation of perturbed exner function at cells on model levels [m-2]
        - perturbed_exner_at_cells_on_model_levels: perturbed exner function
        - exner_at_cells_on_half_levels: exner function
        - perturbed_rho_at_cells_on_model_levels: perturbed air density (actual density minus reference density) [kg m-3]
        - perturbed_theta_v_at_cells_on_model_levels: perturbed virtual potential temperature (actual virtual potential temperature minus reference virtual potential temperature) [K]
        - rho_at_cells_on_half_levels: air density [kg m-3]
        - perturbed_theta_v_at_cells_on_half_levels: perturbed virtual potential temperature (actual virtual potential temperature minus reference virtual potential temperature) [kg m-3]
        - theta_v_at_cells_on_half_levels: virtual potential temperature [K]
        - current_rho: virtual potential temperature at current substep [K]
        - reference_rho_at_cells_on_model_levels: reference air density [kg m-3]
        - current_theta_v: vertical potential temperature at current substep [K]
        - reference_theta_at_cells_on_model_levels: reference virtual potential temperature [K]
        - reference_theta_at_cells_on_half_levels: reference virtual potential temperature [K]
        - wgtfacq_c: metrics field (weights for interpolation)
        - wgtfac_c: metrics field
        - exner_w_explicit_weight_parameter: explicitness weight for exner and w in the vertically implicit dycore solver
        - ddz_of_reference_exner_at_cells_on_half_levels: vertical gradient of reference exner function [m-1]
        - ddqz_z_half: vertical spacing pn half levels (distance between the height of cell centers at k at k-1)  [m]
        - nonhydro_buoy_at_cells_on_half_levels: pressure buoyancy acceleration [m s-2]
        - time_extrapolation_parameter_for_exner: time extrapolation parameter for exner function
        - current_exner: exner function at current substep
        - reference_exner_at_cells_on_model_levels: reference exner function
        - inv_ddqz_z_full: inverse vertical spacing on full levels (distance between the height of interface at k+1/2 and k-1/2)
        - d2dexdz2_fac1_mc: precomputed factor for second vertical derivatives of exner function for model cell centers
        - d2dexdz2_fac2_mc: precomputed factor for second vertical derivatives of exner function for model cell centers
        - igradp_method: option for pressure gradient computation (see HorizontalPressureDiscretizationType)
        - nflatlev: starting vertical index of flat levels
        - nflat_gradp: starting vertical index when neighboring cell centers lie within the thickness of the layer
        - start_cell_lateral_boundary: start index of the first lateral boundary level zone for cells
        - start_cell_lateral_boundary_level_3: start index of the 3rd lateral boundary level zone for cells
        - start_cell_halo_level_2: start index of the 2nd halo level zone for cells
        - end_cell_halo: end index of the last halo level zone for cells
        - end_cell_halo_level_2: end index of the second halo level zone for cells
        - model_top: start index of the vertical domain
        - surface_level: end index of the vertical domain

    Returns:
        - temporal_extrapolation_of_perturbed_exner
        - perturbed_exner_at_cells_on_model_levels
        - exner_at_cells_on_half_levels
        - perturbed_rho_at_cells_on_model_levels
        - perturbed_theta_v_at_cells_on_model_levels
        - rho_at_cells_on_half_levels
        - perturbed_theta_v_at_cells_on_half_levels
        - theta_v_at_cells_on_half_levels
        - nonhydro_buoy_at_cells_on_half_levels
        - ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_level
        - d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels
    """

    _extrapolate_temporally_exner_pressure(
        exner_exfac=time_extrapolation_parameter_for_exner,
        exner=current_exner,
        exner_ref_mc=reference_exner_at_cells_on_model_levels,
        exner_pr=perturbed_exner_at_cells_on_model_levels,
        out=(temporal_extrapolation_of_perturbed_exner, perturbed_exner_at_cells_on_model_levels),
        domain={
            dims.CellDim: (start_cell_lateral_boundary_level_3, end_cell_halo),
            dims.KDim: (model_top, surface_level - 1),
        },
    )

    _surface_computations(
        wgtfacq_c=wgtfacq_c,
        exner_at_cells_on_half_levels=exner_at_cells_on_half_levels,
        temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
        igradp_method=igradp_method,
        out=exner_at_cells_on_half_levels,
        domain={
            dims.CellDim: (start_cell_lateral_boundary_level_3, end_cell_halo),
            dims.KDim: (surface_level - 1, surface_level),
        },
    )

    _compute_perturbed_quantities_and_interpolation(
        current_rho=current_rho,
        reference_rho_at_cells_on_model_levels=reference_rho_at_cells_on_model_levels,
        current_theta_v=current_theta_v,
        reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
        wgtfac_c=wgtfac_c,
        exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        ddz_of_reference_exner_at_cells_on_half_levels=ddz_of_reference_exner_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        nonhydro_buoy_at_cells_on_half_levels=nonhydro_buoy_at_cells_on_half_levels,
        rho_at_cells_on_half_levels=rho_at_cells_on_half_levels,
        exner_at_cells_on_half_levels=exner_at_cells_on_half_levels,
        temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
        theta_v_at_cells_on_half_levels=theta_v_at_cells_on_half_levels,
        igradp_method=igradp_method,
        nflatlev=nflatlev,
        out=(
            perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels,
            perturbed_exner_at_cells_on_model_levels,
            rho_at_cells_on_half_levels,
            exner_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels,
            nonhydro_buoy_at_cells_on_half_levels,
        ),
        domain={
            dims.CellDim: (start_cell_lateral_boundary_level_3, end_cell_halo),
            dims.KDim: (model_top, surface_level - 1),
        },
    )

    _set_theta_v_and_exner_on_surface_level(
        temporal_extrapolation_of_perturbed_exner=temporal_extrapolation_of_perturbed_exner,
        wgtfacq_c=wgtfacq_c,
        perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
        reference_theta_at_cells_on_half_levels=reference_theta_at_cells_on_half_levels,
        out=(
            perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels,
            exner_at_cells_on_half_levels,
        ),
        domain={
            dims.CellDim: (start_cell_lateral_boundary_level_3, end_cell_halo),
            dims.KDim: (surface_level - 1, surface_level),
        },
    )

    _compute_first_and_second_vertical_derivative_of_exner(
        exner_at_cells_on_half_levels=exner_at_cells_on_half_levels,
        inv_ddqz_z_full=inv_ddqz_z_full,
        ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels=d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        perturbed_theta_v_at_cells_on_half_levels=perturbed_theta_v_at_cells_on_half_levels,
        d2dexdz2_fac1_mc=d2dexdz2_fac1_mc,
        d2dexdz2_fac2_mc=d2dexdz2_fac2_mc,
        perturbed_theta_v_at_cells_on_model_levels=perturbed_theta_v_at_cells_on_model_levels,
        igradp_method=igradp_method,
        nflatlev=nflatlev,
        nflat_gradp=nflat_gradp,
        out=(
            ddz_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
            d2dz2_of_temporal_extrapolation_of_perturbed_exner_on_model_levels,
        ),
        domain={
            dims.CellDim: (start_cell_lateral_boundary_level_3, end_cell_halo),
            dims.KDim: (model_top, surface_level - 1),
        },
    )

    _compute_perturbation_of_rho_and_theta(
        rho=current_rho,
        rho_ref_mc=reference_rho_at_cells_on_model_levels,
        theta_v=current_theta_v,
        theta_ref_mc=reference_theta_at_cells_on_model_levels,
        out=(
            perturbed_rho_at_cells_on_model_levels,
            perturbed_theta_v_at_cells_on_model_levels,
        ),
        domain={
            dims.CellDim: (start_cell_halo_level_2, end_cell_halo_level_2),
            dims.KDim: (model_top, surface_level - 1),
        },
    )


@gtx.field_operator
def _compute_interpolation_and_nonhydro_buoy(
    w: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    reference_theta_at_cells_on_model_levels: fa.CellKField[ta.vpfloat],
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat],
    dtime: ta.wpfloat,
    rhotheta_explicit_weight_parameter: ta.wpfloat,
    rhotheta_implicit_weight_parameter: ta.wpfloat,
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.vpfloat],
]:
    (
        contravariant_correction_at_cells_on_half_levels_wp,
        wgtfac_c_wp,
        reference_theta_at_cells_on_model_levels_wp,
        ddqz_z_half_wp,
    ) = astype(
        (
            contravariant_correction_at_cells_on_half_levels,
            wgtfac_c,
            reference_theta_at_cells_on_model_levels,
            ddqz_z_half,
        ),
        wpfloat,
    )

    back_trajectory_w_at_cells_on_half_levels = (
        -(w - contravariant_correction_at_cells_on_half_levels_wp)
        * dtime
        * wpfloat("0.5")
        / ddqz_z_half_wp
    )

    time_averaged_rho = (
        rhotheta_explicit_weight_parameter * current_rho
        + rhotheta_implicit_weight_parameter * next_rho
    )
    time_averaged_rho_kup = rhotheta_explicit_weight_parameter * current_rho(
        KDim - 1
    ) + rhotheta_implicit_weight_parameter * next_rho(KDim - 1)

    time_averaged_theta_v = (
        rhotheta_explicit_weight_parameter * current_theta_v
        + rhotheta_implicit_weight_parameter * next_theta_v
    )
    time_averaged_theta_v_kup = rhotheta_explicit_weight_parameter * current_theta_v(
        KDim - 1
    ) + rhotheta_implicit_weight_parameter * next_theta_v(KDim - 1)

    rho_at_cells_on_half_levels = (
        wgtfac_c_wp * time_averaged_rho
        + (wpfloat("1.0") - wgtfac_c_wp) * time_averaged_rho_kup
        + back_trajectory_w_at_cells_on_half_levels * (time_averaged_rho_kup - time_averaged_rho)
    )

    time_averaged_perturbed_theta_v_kup = (
        time_averaged_theta_v_kup - reference_theta_at_cells_on_model_levels_wp(KDim - 1)
    )
    time_averaged_perturbed_theta_v = (
        time_averaged_theta_v - reference_theta_at_cells_on_model_levels_wp
    )

    time_averaged_perturbed_theta_v_vp, time_averaged_perturbed_theta_v_kup_vp = astype(
        (time_averaged_perturbed_theta_v, time_averaged_perturbed_theta_v_kup), vpfloat
    )
    perturbed_theta_v_at_cells_on_half_levels = (
        wgtfac_c * time_averaged_perturbed_theta_v_vp
        + (vpfloat("1.0") - wgtfac_c) * time_averaged_perturbed_theta_v_kup_vp
    )

    theta_v_at_cells_on_half_levels = (
        wgtfac_c_wp * time_averaged_theta_v
        + (wpfloat("1.0") - wgtfac_c_wp) * time_averaged_theta_v_kup
        + back_trajectory_w_at_cells_on_half_levels
        * (time_averaged_theta_v_kup - time_averaged_theta_v)
    )
    nonhydro_buoy_at_cells_on_half_levels = (
        exner_w_explicit_weight_parameter
        * theta_v_at_cells_on_half_levels
        * (
            perturbed_exner_at_cells_on_model_levels(KDim - 1)
            - perturbed_exner_at_cells_on_model_levels
        )
        / astype(ddqz_z_half, wpfloat)
        + astype(
            perturbed_theta_v_at_cells_on_half_levels
            * ddz_of_reference_exner_at_cells_on_half_levels,
            wpfloat,
        )
    )
    return (
        rho_at_cells_on_half_levels,
        perturbed_theta_v_at_cells_on_half_levels,
        theta_v_at_cells_on_half_levels,
        astype(nonhydro_buoy_at_cells_on_half_levels, vpfloat),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_interpolation_and_nonhydro_buoy(
    rho_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    perturbed_theta_v_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    theta_v_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    nonhydro_buoy_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    w: fa.CellKField[ta.wpfloat],
    contravariant_correction_at_cells_on_half_levels: fa.CellKField[ta.wpfloat],
    current_rho: fa.CellKField[ta.wpfloat],
    next_rho: fa.CellKField[ta.wpfloat],
    current_theta_v: fa.CellKField[ta.wpfloat],
    next_theta_v: fa.CellKField[ta.wpfloat],
    perturbed_exner_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    reference_theta_at_cells_on_model_levels: fa.CellKField[ta.wpfloat],
    ddz_of_reference_exner_at_cells_on_half_levels: fa.CellKField[ta.vpfloat],
    ddqz_z_half: fa.CellKField[ta.vpfloat],
    wgtfac_c: fa.CellKField[ta.vpfloat],
    exner_w_explicit_weight_parameter: fa.CellField[ta.wpfloat],
    dtime: ta.wpfloat,
    rhotheta_explicit_weight_parameter: ta.wpfloat,
    rhotheta_implicit_weight_parameter: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    Formerly known as fused_solve_nonhydro_stencil_1_to_13_corrector.

    This program calculates the air density, virtual potential temperature and perturberd virtual potential temperature on half levels,
    and also computes the pressure buoyancy acceleration.

    Args:
        - rho_at_cells_on_half_levels: air density at cells on half levels [kg m-3]
        - perturbed_theta_v_at_cells_on_half_levels: perturbed virtual potential temperature (actual virtual potential temperature minus reference virtual potential temperature) at cells on half levels [kg m-3]
        - theta_v_at_cells_on_half_levels: virtual potential temperature at cells on half levels [K]
        - nonhydro_buoy_at_cells_on_half_levels: pressure buoyancy acceleration at cells on half levels [m s-2]
        - w: vertical wind at cell centers [m s-1]
        - contravariant_correction_at_cells_on_half_levels: contravariant metric correction at cells on half levels
        - current_rho: air density at current substep [K]
        - next_rho: air density at next substep [K]
        - current_theta_v: virtual potential temperature at current substep [K]
        - next_theta_v: virtual potential temperature at next substep [K]
        - perturbed_exner_at_cells_on_model_levels: perturbed exner function at model levels
        - reference_theta_at_cells_on_model_levels: reference virtual potential temperature at cells on model levels [K]
        - ddz_of_reference_exner_at_cells_on_half_levels: vertical gradient of reference exner function at cells on half levels [m-1]
        - ddqz_z_half: vertical spacing on half levels (distance between the height of cell centers at k at k-1)  [m]
        - wgtfac_c: metrics field
        - exner_w_explicit_weight_parameter: explicitness weight for exner and w in the vertically implicit dycore solver
        - dtime: time step
        - rhotheta_explicit_weight_parameter: explicitness weight of density and virtual potential temperature
        - rhotheta_implicit_weight_parameter: implicitness weight of density and virtual potential temperature
        - horizontal_start: start index of the horizontal domain
        - horizontal_end: end index of the horizontal domain
        - vertical_start: start index of the vertical domain
        - vertical_end: end index of the vertical domain

    Returns:
        - rho_at_cells_on_half_levels
        - perturbed_theta_v_at_cells_on_half_levels
        - theta_v_at_cells_on_half_levels
        - nonhydro_buoy_at_cells_on_half_levels
    """
    _compute_interpolation_and_nonhydro_buoy(
        w=w,
        contravariant_correction_at_cells_on_half_levels=contravariant_correction_at_cells_on_half_levels,
        current_rho=current_rho,
        next_rho=next_rho,
        current_theta_v=current_theta_v,
        next_theta_v=next_theta_v,
        perturbed_exner_at_cells_on_model_levels=perturbed_exner_at_cells_on_model_levels,
        reference_theta_at_cells_on_model_levels=reference_theta_at_cells_on_model_levels,
        ddz_of_reference_exner_at_cells_on_half_levels=ddz_of_reference_exner_at_cells_on_half_levels,
        ddqz_z_half=ddqz_z_half,
        wgtfac_c=wgtfac_c,
        exner_w_explicit_weight_parameter=exner_w_explicit_weight_parameter,
        dtime=dtime,
        rhotheta_explicit_weight_parameter=rhotheta_explicit_weight_parameter,
        rhotheta_implicit_weight_parameter=rhotheta_implicit_weight_parameter,
        out=(
            rho_at_cells_on_half_levels,
            perturbed_theta_v_at_cells_on_half_levels,
            theta_v_at_cells_on_half_levels,
            nonhydro_buoy_at_cells_on_half_levels,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
