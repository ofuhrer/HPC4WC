# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import astype, neighbor_sum
from gt4py.next.experimental import concat_where

from icon4py.model.atmosphere.dycore.stencils.accumulate_prep_adv_fields import (
    _accumulate_prep_adv_fields,
)
from icon4py.model.atmosphere.dycore.stencils.compute_contravariant_correction import (
    _compute_contravariant_correction,
)
from icon4py.model.atmosphere.dycore.stencils.compute_diagnostics_from_normal_wind import (
    _compute_horizontal_kinetic_energy,
    _interpolate_to_half_levels,
)
from icon4py.model.atmosphere.dycore.stencils.compute_mass_flux import (
    _compute_mass_and_temperature_flux,
)
from icon4py.model.atmosphere.dycore.stencils.compute_tangential_wind import (
    _compute_tangential_wind,
)
from icon4py.model.atmosphere.dycore.stencils.extrapolate_at_top import _extrapolate_at_top
from icon4py.model.atmosphere.dycore.stencils.spatially_average_flux_or_velocity import (
    _spatially_average_flux_or_velocity,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.dimension import E2C2EO, E2C2EODim
from icon4py.model.common.type_alias import vpfloat


@gtx.field_operator
def _compute_horizontal_velocity_quantities_and_fluxes(
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    nflatlev: gtx.int32,
) -> tuple[
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
    fa.EdgeKField[ta.vpfloat],
]:
    spatially_averaged_vn = _spatially_average_flux_or_velocity(
        e_flx_avg=e_flx_avg, flux_or_velocity=vn
    )
    horizontal_gradient_of_normal_wind_divergence = astype(
        neighbor_sum(geofac_grdiv * vn(E2C2EO), axis=E2C2EODim), vpfloat
    )
    tangential_wind = _compute_tangential_wind(vn=vn, rbf_vec_coeff_e=rbf_vec_coeff_e)

    (
        mass_flux_at_edges_on_model_levels,
        theta_v_flux_at_edges_on_model_levels,
    ) = _compute_mass_and_temperature_flux(
        rho_at_edges_on_model_levels,
        spatially_averaged_vn,
        ddqz_z_full_e,
        theta_v_at_edges_on_model_levels,
    )

    horizontal_kinetic_energy_at_edges_on_model_levels = _compute_horizontal_kinetic_energy(
        vn, tangential_wind
    )

    vn_on_half_levels = _interpolate_to_half_levels(wgtfac_e, vn)
    tangential_wind_on_half_levels = _interpolate_to_half_levels(wgtfac_e, tangential_wind)

    contravariant_correction_at_edges_on_model_levels = concat_where(
        nflatlev <= dims.KDim,
        _compute_contravariant_correction(vn, ddxn_z_full, ddxt_z_full, tangential_wind),
        contravariant_correction_at_edges_on_model_levels,
    )

    return (
        spatially_averaged_vn,
        horizontal_gradient_of_normal_wind_divergence,
        tangential_wind,
        mass_flux_at_edges_on_model_levels,
        theta_v_flux_at_edges_on_model_levels,
        vn_on_half_levels,
        tangential_wind_on_half_levels,
        horizontal_kinetic_energy_at_edges_on_model_levels,
        contravariant_correction_at_edges_on_model_levels,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_horizontal_velocity_quantities_and_fluxes(
    spatially_averaged_vn: fa.EdgeKField[ta.wpfloat],
    horizontal_gradient_of_normal_wind_divergence: fa.EdgeKField[ta.vpfloat],
    tangential_wind: fa.EdgeKField[ta.vpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    tangential_wind_on_half_levels: fa.EdgeKField[ta.vpfloat],
    vn_on_half_levels: fa.EdgeKField[ta.vpfloat],
    horizontal_kinetic_energy_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    contravariant_correction_at_edges_on_model_levels: fa.EdgeKField[ta.vpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    geofac_grdiv: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    rbf_vec_coeff_e: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EDim], ta.wpfloat],
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    ddxn_z_full: fa.EdgeKField[ta.vpfloat],
    ddxt_z_full: fa.EdgeKField[ta.vpfloat],
    wgtfac_e: fa.EdgeKField[ta.vpfloat],
    wgtfacq_e: fa.EdgeKField[ta.vpfloat],
    nflatlev: gtx.int32,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    This program computes a variety of diagnostic quantities and fluxes
    related to horizontal wind including
    tangential wind, mass and virtual potential temperature fluxes, horizontal
    kinetic energy, and divergence-related terms. It also performs extrapolation
    of wind quantities to half levels near the model top.
    Args:
        - spatially_averaged_vn: spatial average of the normal wind at edges on model levels [m s⁻¹]
        - horizontal_gradient_of_normal_wind_divergence: horizontal gradient of the divergence of normal wind [s⁻²]
        - tangential_wind: tangential component of the horizontal wind [m s⁻¹]
        - mass_flux_at_edges_on_model_levels: mass flux at edges (ϼ * vn) on model levels [kg m⁻² s⁻¹]
        - theta_v_flux_at_edges_on_model_levels: virtual potential temperature flux (ϼ * θ_v * vn) on model levels [K kg m⁻² s⁻¹]
        - tangential_wind_on_half_levels: tangential wind at vertical half levels [m s⁻¹]
        - vn_on_half_levels: normal wind at vertical half levels [m s⁻¹]
        - horizontal_kinetic_energy_at_edges_on_model_levels: kinetic energy per unit mass at edges on model levels [m² s⁻²]
        - contravariant_correction_at_edges_on_model_levels: contravariant correction term at edges on model levels [m s⁻¹]
        - vn: normal wind component at edges on model levels [m s⁻¹]
        - e_flx_avg: average energy flux across edge-to-cell transitions
        - geofac_grdiv: geometric factor for gradient/divergence at edges
        - rbf_vec_coeff_e: radial basis function coefficients for edge vector interpolation
        - rho_at_edges_on_model_levels: air density at edges on model levels [kg m⁻³]
        - theta_v_at_edges_on_model_levels: virtual potential temperature at edges on model levels [K]
        - ddqz_z_full_e: vertical derivative of qz field at edges [1/m]
        - ddxn_z_full: zonal horizontal derivative of scalar field at edges [1/m]
        - ddxt_z_full: meridional horizontal derivative of scalar field at edges [1/m]
        - wgtfac_e: metrics field
        - wgtfacq_e: metrics field
        - nflatlev: number of flat vertical levels near the model top
        - horizontal_start: start index of the horizontal domain
        - horizontal_end: end index of the horizontal domain
        - vertical_start: start index of the vertical domain
        - vertical_end: end index of the vertical domain
    Returns:
        - spatially_averaged_vn
        - horizontal_gradient_of_normal_wind_divergence
        - tangential_wind
        - mass_flux_at_edges_on_model_levels
        - theta_v_flux_at_edges_on_model_levels
        - vn_on_half_levels
        - tangential_wind_on_half_levels
        - horizontal_kinetic_energy_at_edges_on_model_levels
        - contravariant_correction_at_edges_on_model_levels
    """

    _compute_horizontal_velocity_quantities_and_fluxes(
        contravariant_correction_at_edges_on_model_levels=contravariant_correction_at_edges_on_model_levels,
        vn=vn,
        e_flx_avg=e_flx_avg,
        geofac_grdiv=geofac_grdiv,
        rbf_vec_coeff_e=rbf_vec_coeff_e,
        rho_at_edges_on_model_levels=rho_at_edges_on_model_levels,
        theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
        ddqz_z_full_e=ddqz_z_full_e,
        ddxn_z_full=ddxn_z_full,
        ddxt_z_full=ddxt_z_full,
        wgtfac_e=wgtfac_e,
        nflatlev=nflatlev,
        out=(
            spatially_averaged_vn,
            horizontal_gradient_of_normal_wind_divergence,
            tangential_wind,
            mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels,
            vn_on_half_levels,
            tangential_wind_on_half_levels,
            horizontal_kinetic_energy_at_edges_on_model_levels,
            contravariant_correction_at_edges_on_model_levels,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end - 1),
        },
    )

    _extrapolate_at_top(
        wgtfacq_e,
        vn,
        out=vn_on_half_levels,
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_end - 1, vertical_end),
        },
    )


@gtx.field_operator
def _compute_averaged_vn_and_fluxes(
    substep_and_spatially_averaged_vn: fa.EdgeKField[ta.wpfloat],
    substep_averaged_mass_flux: fa.EdgeKField[ta.wpfloat],
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    prepare_advection: bool,
    at_first_substep: bool,
    r_nsubsteps: ta.wpfloat,
) -> tuple[
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
    fa.EdgeKField[ta.wpfloat],
]:
    spatially_averaged_vn = _spatially_average_flux_or_velocity(e_flx_avg, vn)

    (
        mass_flux_at_edges_on_model_levels,
        theta_v_flux_at_edges_on_model_levels,
    ) = _compute_mass_and_temperature_flux(
        rho_at_edges_on_model_levels,
        spatially_averaged_vn,
        ddqz_z_full_e,
        theta_v_at_edges_on_model_levels,
    )

    if prepare_advection:
        substep_and_spatially_averaged_vn, substep_averaged_mass_flux = (
            (r_nsubsteps * spatially_averaged_vn, r_nsubsteps * mass_flux_at_edges_on_model_levels)
            if at_first_substep
            else _accumulate_prep_adv_fields(
                spatially_averaged_vn,
                mass_flux_at_edges_on_model_levels,
                substep_and_spatially_averaged_vn,
                substep_averaged_mass_flux,
                r_nsubsteps,
            )
        )

    return (
        spatially_averaged_vn,
        mass_flux_at_edges_on_model_levels,
        theta_v_flux_at_edges_on_model_levels,
        substep_and_spatially_averaged_vn,
        substep_averaged_mass_flux,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_averaged_vn_and_fluxes(
    spatially_averaged_vn: fa.EdgeKField[ta.wpfloat],
    mass_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    theta_v_flux_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    substep_and_spatially_averaged_vn: fa.EdgeKField[ta.wpfloat],
    substep_averaged_mass_flux: fa.EdgeKField[ta.wpfloat],
    e_flx_avg: gtx.Field[gtx.Dims[dims.EdgeDim, dims.E2C2EODim], ta.wpfloat],
    vn: fa.EdgeKField[ta.wpfloat],
    rho_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    ddqz_z_full_e: fa.EdgeKField[ta.vpfloat],
    theta_v_at_edges_on_model_levels: fa.EdgeKField[ta.wpfloat],
    prepare_advection: bool,
    at_first_substep: bool,
    r_nsubsteps: ta.wpfloat,
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
) -> None:
    """
    This program performs:
    - Averaging the normal wind component across substeps
    - Calculating mass flux and theta_v flux on edges
    - Optional initialization of tracer advection preparation
    depending on current substep and solver configuration.
    Args:
        - spatially_averaged_vn: temporally averaged normal wind at edges [m s⁻¹]
        - mass_flux_at_edges_on_model_levels: temporally averaged mass flux at edges [kg m⁻² s⁻¹]
        - theta_v_flux_at_edges_on_model_levels: temporally averaged θ_v flux at edges [K kg m⁻² s⁻¹]
        - substep_and_spatially_averaged_vn: combined substep and spatially averaged vn [m s⁻¹]
        - substep_averaged_mass_flux: substep-averaged mass flux field [kg m⁻² s⁻¹]
        - e_flx_avg: average energy flux
        - vn: normal wind component at edges [m s⁻¹]
        - rho_at_edges_on_model_levels: air density at edges on model levels [kg m⁻³]
        - ddqz_z_full_e: vertical derivative of qz at edges [1/m]
        - theta_v_at_edges_on_model_levels: virtual potential temperature at edges [K]
        - prepare_advection: whether to prepare fields for tracer advection (True if in preparation phase)
        - at_first_substep: True if currently at the first substep of the time integration
        - r_nsubsteps: reciprocal of the total number of substeps (1 / N)
        - horizontal_start: start index of the horizontal domain
        - horizontal_end: end index of the horizontal domain
        - vertical_start: start index of the vertical domain
        - vertical_end: end index of the vertical domain
    Returns:
        - spatially_averaged_vn
        - mass_flux_at_edges_on_model_levels
        - theta_v_flux_at_edges_on_model_levels
        - substep_and_spatially_averaged_vn
        - substep_averaged_mass_flux
    """

    _compute_averaged_vn_and_fluxes(
        substep_and_spatially_averaged_vn=substep_and_spatially_averaged_vn,
        substep_averaged_mass_flux=substep_averaged_mass_flux,
        e_flx_avg=e_flx_avg,
        vn=vn,
        rho_at_edges_on_model_levels=rho_at_edges_on_model_levels,
        ddqz_z_full_e=ddqz_z_full_e,
        theta_v_at_edges_on_model_levels=theta_v_at_edges_on_model_levels,
        prepare_advection=prepare_advection,
        at_first_substep=at_first_substep,
        r_nsubsteps=r_nsubsteps,
        out=(
            spatially_averaged_vn,
            mass_flux_at_edges_on_model_levels,
            theta_v_flux_at_edges_on_model_levels,
            substep_and_spatially_averaged_vn,
            substep_averaged_mass_flux,
        ),
        domain={
            dims.EdgeDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
