# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import sys

import gt4py.next as gtx
from gt4py.next import broadcast, exp, log, maximum, minimum, where

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics.microphysics_constants import (
    MicrophysicsConstants,
)
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics.stencils.microphysical_processes import (
    autoconversion_and_rain_accretion,
    collision_and_ice_deposition_in_cold_ice_clouds,
    compute_cooper_inp_concentration,
    compute_snow_interception_and_collision_parameters,
    deposition_nucleation_at_low_temperature_or_in_clouds,
    evaporation_and_freezing_in_subsaturated_air,
    freezing_in_clouds,
    melting,
    reduced_deposition_in_clouds,
    riming_in_clouds,
    sat_pres_ice,
    sat_pres_water_scalar,
    snow_and_graupel_depositional_growth_in_cold_ice_clouds,
)
from icon4py.model.common import dimension as dims, field_type_aliases as fa, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


# TODO (Chia Rui): The limit has to be manually set to a huge value for a big scan operator. Remove it when neccesary.
sys.setrecursionlimit(350000)


@gtx.scan_operator(
    axis=dims.KDim,
    forward=True,
    init=(
        wpfloat("0.0"),  # temperature tendency
        wpfloat("0.0"),  # qv tendency
        wpfloat("0.0"),  # qc tendency
        wpfloat("0.0"),  # qi tendency
        wpfloat("0.0"),  # qr tendency
        wpfloat("0.0"),  # qs tendency
        wpfloat("0.0"),  # qg tendency
        wpfloat("0.0"),  # qv
        wpfloat("0.0"),  # qc
        wpfloat("0.0"),  # qi
        wpfloat("0.0"),  # qr
        wpfloat("0.0"),  # qs
        wpfloat("0.0"),  # qg
        wpfloat("0.0"),  # rhoqrv
        wpfloat("0.0"),  # rhoqsv
        wpfloat("0.0"),  # rhoqgv
        wpfloat("0.0"),  # rhoqiv
        wpfloat("0.0"),  # newv_r
        wpfloat("0.0"),  # newv_s
        wpfloat("0.0"),  # newv_g
        wpfloat("0.0"),  # newv_i
        wpfloat("0.0"),  # cloud top distance
        wpfloat("0.0"),  # density
        wpfloat("0.0"),  # density factor
        wpfloat("0.0"),  # density factor for ice
        wpfloat("0.0"),  # snow intercept parameter
        wpfloat("0.0"),  # saturation pressure
        gtx.int32(0),  # k level
    ),
)
def _icon_graupel_scan(  # noqa: PLR0912, PLR0915
    state_kup: tuple[
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        ta.wpfloat,
        gtx.int32,
    ],
    ground_level: gtx.int32,  # k bottom level
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    use_constant_latent_heat: bool,
    ice_stickeff_min: ta.wpfloat,
    snow2graupel_riming_coeff: ta.wpfloat,
    power_law_coeff_for_ice_mean_fall_speed: ta.wpfloat,
    exponent_for_density_factor_in_ice_sedimentation: ta.wpfloat,
    power_law_coeff_for_snow_fall_speed: ta.wpfloat,
    precomputed_riming_coef: ta.wpfloat,
    precomputed_agg_coef: ta.wpfloat,
    precomputed_snow_sed_coef: ta.wpfloat,
    power_law_exponent_for_rain_mean_fall_speed: ta.wpfloat,
    power_law_coeff_for_rain_mean_fall_speed: ta.wpfloat,
    precomputed_evaporation_alpha_exp_coeff: ta.wpfloat,
    precomputed_evaporation_alpha_coeff: ta.wpfloat,
    precomputed_evaporation_beta_exp_coeff: ta.wpfloat,
    precomputed_evaporation_beta_coeff: ta.wpfloat,
    power_law_exponent_for_rain_mean_fall_speed_ln1o2: ta.wpfloat,
    power_law_exponent_for_ice_mean_fall_speed_ln1o2: ta.wpfloat,
    power_law_exponent_for_graupel_mean_fall_speed_ln1o2: ta.wpfloat,
    dtime: ta.wpfloat,
    dz: ta.wpfloat,
    temperature: ta.wpfloat,
    pressure: ta.wpfloat,
    rho: ta.wpfloat,
    qv: ta.wpfloat,
    qc: ta.wpfloat,
    qi: ta.wpfloat,
    qr: ta.wpfloat,
    qs: ta.wpfloat,
    qg: ta.wpfloat,
    qnc: ta.wpfloat,
):
    """
    This is the ICON graupel scheme. The structure of the code can be split into several steps as follow:
        1. initialize tracer at k-1 level, and some pre-computed coefficients including the snow intercept parameter for later uses.
        2. compute sedimentation fluxes and update rain, snow, and graupel mass at the current k level.
        3. compute pre-computed coefficients after update from sedimentation fluxes to include implicitness of the graupel scheme.
        4. compute all transfer rates.
        5. check if tracers go below 0.
        6. update all tendencies.

    Below is a list of the so-called transfered rates for all microphyiscal processes in ICON graupel scheme:
        ice_net_deposition_rate_v2i         :   vapor   -> ice,     ice vapor deposition
        ice_net_sublimation_rate_v2i        :   vapor   -> ice,     ice vapor sublimation
        ice_deposition_rate_v2i             :   vapor   -> ice,     ice vapor net deposition
        snow_deposition_rate_v2s            :   vapor   -> snow,    snow vapor deposition
        graupel_deposition_rate_v2g         :   vapor   -> graupel, graupel vapor deposition
        ice_nucleation_rate_v2i             :   vapor   -> ice,     ice nucleation
        rain_deposition_rate_v2r            :   vapor   -> rain,    rain condensation on melting snow/graupel
        cloud_autoconversion_rate_c2r       :   cloud   -> rain,    cloud autoconversion into rain
        cloud_freezing_rate_c2i             :   cloud   -> ice,     cloud freezing
        rain_cloud_collision_rate_c2r       :   cloud   -> rain,    rain-cloud accretion
        rain_shedding_rate_c2r              :   cloud   -> rain,    rain shedding from riming above freezing
        snow_riming_rate_c2s                :   cloud   -> snow,    snow riming
        graupel_riming_rate_c2g             :   cloud   -> graupel, graupel riming
        ice_melting_rate_i2c                :   ice     -> cloud,   ice melting
        rain_ice_2graupel_ice_loss_rate_i2g :   ice     -> graupel, ice loss in rain-ice accretion
        ice_dep_autoconversion_rate_i2s     :   ice     -> snow,    ice vapor depositional autoconversion into snow
        snow_ice_collision_rate_i2s         :   ice     -> snow,    snow-ice aggregation
        graupel_ice_collision_rate_i2g      :   ice     -> graupel, graupel-ice aggregation
        ice_autoconverson_rate_i2s          :   ice     -> snow,    ice autoconversion into snow
        rain_ice_2graupel_rain_loss_rate_r2g:   rain    -> graupel, rain loss in rain-ice accretion
        rain_freezing_rate_r2g              :   rain    -> graupel, rain freezing
        rain_evaporation_rate_r2v           :   rain    -> vapor,   rain evaporation
        snow_melting_rate_s2r               :   snow    -> rain,    snow melting
        snow_autoconversion_rate_s2g        :   snow    -> graupel, snow autoconversion into graupel
        graupel_melting_rate_g2r            :   graupel -> rain,    graupel melting
    """

    (
        _,
        qv_tendency_kup,
        qc_tendency_kup,
        qi_tendency_kup,
        qr_tendency_kup,
        qs_tendency_kup,
        qg_tendency_kup,
        qv_old_kup,
        qc_old_kup,
        qi_old_kup,
        qr_old_kup,
        qs_old_kup,
        qg_old_kup,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        dist_cldtop_kup,
        rho_kup,
        crho1o2_kup,
        crhofac_qi_kup,
        snow_sed0_kup,
        qvsw_kup,
        k_lev,
    ) = state_kup

    # ------------------------------------------------------------------------------
    #  Section 1: Precomputed coefficients
    # ------------------------------------------------------------------------------

    qv_kup = qv_old_kup + qv_tendency_kup * dtime
    qc_kup = qc_old_kup + qc_tendency_kup * dtime
    qi_kup = qi_old_kup + qi_tendency_kup * dtime
    qr_kup = qr_old_kup + qr_tendency_kup * dtime
    qs_kup = qs_old_kup + qs_tendency_kup * dtime
    qg_kup = qg_old_kup + qg_tendency_kup * dtime

    is_surface = True if k_lev == ground_level else False  # noqa: SIM210

    # TODO (Chia Rui): duplicated function for computing latent heat. Saturation adjustment also uses the same function. Move to a common place.
    lhv = (
        PhysicsConstants.lh_vaporise
        if use_constant_latent_heat
        else PhysicsConstants.lh_vaporise
        + (MicrophysicsConstants.CP_V - PhysicsConstants.cpl)
        * (temperature - PhysicsConstants.tmelt)
        - PhysicsConstants.rv * temperature
    )
    lhs = (
        PhysicsConstants.lh_sublimate
        if use_constant_latent_heat
        else PhysicsConstants.lh_sublimate
        + (MicrophysicsConstants.CP_V - PhysicsConstants.cpi)
        * (temperature - PhysicsConstants.tmelt)
        - PhysicsConstants.rv * temperature
    )

    # for density correction of fall speeds
    chlp = log(MicrophysicsConstants.REF_AIR_DENSITY / rho)
    crho1o2 = exp(chlp / wpfloat("2.0"))
    crhofac_qi = exp(chlp * exponent_for_density_factor_in_ice_sedimentation)

    cdtdh = wpfloat("0.5") * dtime / dz
    cscmax = qc / dtime
    cnin = compute_cooper_inp_concentration(temperature)
    cmi = minimum(rho * qi / cnin, MicrophysicsConstants.ICE_MAX_MASS)
    cmi = maximum(MicrophysicsConstants.ICE_INITIAL_MASS, cmi)

    qvsw = sat_pres_water_scalar(temperature) / (rho * PhysicsConstants.rv * temperature)
    qvsi = sat_pres_ice(temperature) / (rho * PhysicsConstants.rv * temperature)

    rhoqr = qr * rho
    rhoqs = qs * rho
    rhoqg = qg * rho
    rhoqi = qi * rho

    rhoqrv_new_kup = qr_kup * rho_kup * vnew_r
    rhoqsv_new_kup = qs_kup * rho_kup * vnew_s
    rhoqgv_new_kup = qg_kup * rho_kup * vnew_g
    rhoqiv_new_kup = qi_kup * rho_kup * vnew_i

    if rhoqrv_new_kup <= MicrophysicsConstants.QMIN:
        rhoqrv_new_kup = wpfloat("0.0")
    if rhoqsv_new_kup <= MicrophysicsConstants.QMIN:
        rhoqsv_new_kup = wpfloat("0.0")
    if rhoqgv_new_kup <= MicrophysicsConstants.QMIN:
        rhoqgv_new_kup = wpfloat("0.0")
    if rhoqiv_new_kup <= MicrophysicsConstants.QMIN:
        rhoqiv_new_kup = wpfloat("0.0")

    rhoqr_intermediate = rhoqr / cdtdh + rhoqrv_new_kup + rhoqrv_old_kup
    rhoqs_intermediate = rhoqs / cdtdh + rhoqsv_new_kup + rhoqsv_old_kup
    rhoqg_intermediate = rhoqg / cdtdh + rhoqgv_new_kup + rhoqgv_old_kup
    rhoqi_intermediate = rhoqi / cdtdh + rhoqiv_new_kup + rhoqiv_old_kup

    rain_exists = True if (rhoqr > MicrophysicsConstants.QMIN) else False  # noqa: SIM210
    snow_exists = True if (rhoqs > MicrophysicsConstants.QMIN) else False  # noqa: SIM210
    graupel_exists = True if (rhoqg > MicrophysicsConstants.QMIN) else False  # noqa: SIM210
    ice_exists = True if (rhoqi > MicrophysicsConstants.QMIN) else False  # noqa: SIM210

    n0s, snow_sed0, crim, cagg, cbsdep = compute_snow_interception_and_collision_parameters(
        temperature,
        rho,
        qs,
        precomputed_riming_coef,
        precomputed_agg_coef,
        precomputed_snow_sed_coef,
        power_law_coeff_for_snow_fall_speed,
        snow_exists,
        snow_intercept_option,
    )

    # ------------------------------------------------------------------------------
    #  Section 2: Sedimentation fluxes
    # ------------------------------------------------------------------------------

    if k_lev > 0:
        vnew_s = (
            snow_sed0_kup
            * exp(MicrophysicsConstants.CCSWXP * log((qs_kup + qs) * wpfloat("0.5") * rho_kup))
            * crho1o2_kup
            if qs_kup + qs > MicrophysicsConstants.QMIN
            else wpfloat("0.0")
        )
        vnew_r = (
            power_law_coeff_for_rain_mean_fall_speed
            * exp(
                power_law_exponent_for_rain_mean_fall_speed
                * log((qr_kup + qr) * wpfloat("0.5") * rho_kup)
            )
            * crho1o2_kup
            if qr_kup + qr > MicrophysicsConstants.QMIN
            else wpfloat("0.0")
        )
        vnew_g = (
            MicrophysicsConstants.POWER_LAW_COEFF_FOR_GRAUPEL_MEAN_FALL_SPEED
            * exp(
                MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_GRAUPEL_MEAN_FALL_SPEED
                * log((qg_kup + qg) * wpfloat("0.5") * rho_kup)
            )
            * crho1o2_kup
            if qg_kup + qg > MicrophysicsConstants.QMIN
            else wpfloat("0.0")
        )
        vnew_i = (
            power_law_coeff_for_ice_mean_fall_speed
            * exp(
                MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_ICE_MEAN_FALL_SPEED
                * log((qi_kup + qi) * wpfloat("0.5") * rho_kup)
            )
            * crhofac_qi_kup
            if qi_kup + qi > MicrophysicsConstants.QMIN
            else wpfloat("0.0")
        )

    if snow_exists:
        terminal_velocity = snow_sed0 * exp(MicrophysicsConstants.CCSWXP * log(rhoqs)) * crho1o2
        # Prevent terminal fall speed of snow from being zero at the surface level
        if is_surface:
            terminal_velocity = maximum(
                terminal_velocity, MicrophysicsConstants.MINIMUM_SNOW_FALL_SPEED
            )

        rhoqsv = rhoqs * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_s == wpfloat("0.0"):
            vnew_s = terminal_velocity * MicrophysicsConstants.CCSWXP_LN1O2

    else:
        rhoqsv = wpfloat("0.0")

    if rain_exists:
        terminal_velocity = (
            power_law_coeff_for_rain_mean_fall_speed
            * exp(power_law_exponent_for_rain_mean_fall_speed * log(rhoqr))
            * crho1o2
        )
        # Prevent terminal fall speed of rain from being zero at the surface level
        if is_surface:
            terminal_velocity = maximum(
                terminal_velocity, MicrophysicsConstants.MINIMUM_RAIN_FALL_SPEED
            )

        rhoqrv = rhoqr * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_r == wpfloat("0.0"):
            vnew_r = terminal_velocity * power_law_exponent_for_rain_mean_fall_speed_ln1o2

    else:
        rhoqrv = wpfloat("0.0")

    if graupel_exists:
        terminal_velocity = (
            MicrophysicsConstants.POWER_LAW_COEFF_FOR_GRAUPEL_MEAN_FALL_SPEED
            * exp(MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_GRAUPEL_MEAN_FALL_SPEED * log(rhoqg))
            * crho1o2
        )
        # Prevent terminal fall speed of graupel from being zero at the surface level
        if is_surface:
            terminal_velocity = maximum(
                terminal_velocity, MicrophysicsConstants.MINIMUM_GRAUPEL_FALL_SPEED
            )

        rhoqgv = rhoqg * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_g == wpfloat("0.0"):
            vnew_g = terminal_velocity * power_law_exponent_for_graupel_mean_fall_speed_ln1o2

    else:
        rhoqgv = wpfloat("0.0")

    if ice_exists:
        terminal_velocity = (
            power_law_coeff_for_ice_mean_fall_speed
            * exp(MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_ICE_MEAN_FALL_SPEED * log(rhoqi))
            * crhofac_qi
        )

        rhoqiv = rhoqi * terminal_velocity

        # because we are at the model top, simply multiply by a factor of (0.5)^(V_intg_exp)
        if vnew_i == wpfloat("0.0"):
            vnew_i = terminal_velocity * power_law_exponent_for_ice_mean_fall_speed_ln1o2

    else:
        rhoqiv = wpfloat("0.0")

    # Prevent terminal fall speeds of precip hydrometeors from being zero at the surface level
    if is_surface:
        vnew_s = maximum(vnew_s, MicrophysicsConstants.MINIMUM_SNOW_FALL_SPEED)
        vnew_r = maximum(vnew_r, MicrophysicsConstants.MINIMUM_RAIN_FALL_SPEED)
        vnew_g = maximum(vnew_g, MicrophysicsConstants.MINIMUM_GRAUPEL_FALL_SPEED)

    # derive the intermediate density of hydrometeors, Eq. 5.21:
    # limit the precipitation flux at this k level such that mixing ratio won't go below zero
    rhoqrv = minimum(rhoqrv, rhoqr_intermediate)
    rhoqsv = minimum(rhoqsv, rhoqs_intermediate)
    rhoqgv = minimum(rhoqgv, maximum(wpfloat("0.0"), rhoqg_intermediate))
    rhoqiv = minimum(rhoqiv, rhoqi_intermediate)

    rhoqr_intermediate = cdtdh * (rhoqr_intermediate - rhoqrv)
    rhoqs_intermediate = cdtdh * (rhoqs_intermediate - rhoqsv)
    rhoqg_intermediate = cdtdh * (rhoqg_intermediate - rhoqgv)
    rhoqi_intermediate = cdtdh * (rhoqi_intermediate - rhoqiv)

    cimr = wpfloat("1.0") / (wpfloat("1.0") + vnew_r * cdtdh)
    cims = wpfloat("1.0") / (wpfloat("1.0") + vnew_s * cdtdh)
    cimg = wpfloat("1.0") / (wpfloat("1.0") + vnew_g * cdtdh)
    cimi = wpfloat("1.0") / (wpfloat("1.0") + vnew_i * cdtdh)

    # intermediate values
    rhoqr = rhoqr_intermediate * cimr
    rhoqs = rhoqs_intermediate * cims
    rhoqg = rhoqg_intermediate * cimg

    # ------------------------------------------------------------------------------
    #  Section 3: Precomputed coefficients after sedimentation for implicitness
    # ------------------------------------------------------------------------------

    rain_exists = True if (rhoqr > MicrophysicsConstants.QMIN) else False  # noqa: SIM210
    snow_exists = True if (rhoqs > MicrophysicsConstants.QMIN) else False  # noqa: SIM210
    graupel_exists = True if (rhoqg > MicrophysicsConstants.QMIN) else False  # noqa: SIM210
    ice_exists = True if (qi > MicrophysicsConstants.QMIN) else False  # noqa: SIM210
    cloud_exists = True if (qc > MicrophysicsConstants.QMIN) else False  # noqa: SIM210

    if rain_exists:
        clnrhoqr = log(rhoqr)
        csrmax = (
            rhoqr_intermediate / rho / dtime
        )  # GZ: shifting this computation ahead of the IF condition changes results!
        celn7o8qrk = (
            exp(wpfloat("7.0") / wpfloat("8.0") * clnrhoqr)
            if qi + qc > MicrophysicsConstants.QMIN
            else wpfloat("0.0")
        )
        celn7o4qrk = (
            exp(wpfloat("7.0") / wpfloat("4.0") * clnrhoqr)
            if temperature < MicrophysicsConstants.THRESHOLD_FREEZE_TEMPERATURE
            else wpfloat("0.0")
        )  # FR new
        celn13o8qrk = (
            exp(wpfloat("13.0") / wpfloat("8.0") * clnrhoqr) if ice_exists else wpfloat("0.0")
        )

    else:
        csrmax = wpfloat("0.0")
        celn7o8qrk = wpfloat("0.0")
        celn7o4qrk = wpfloat("0.0")
        celn13o8qrk = wpfloat("0.0")

    # ** GZ: the following computation differs substantially from the corresponding code in cloudice **
    if snow_exists:
        clnrhoqs = log(rhoqs)
        cssmax = (
            rhoqs_intermediate / rho / dtime
        )  # GZ: shifting this computation ahead of the IF condition changes results#
        if qi + qc > MicrophysicsConstants.QMIN:
            celn3o4qsk = exp(wpfloat("3.0") / wpfloat("4.0") * clnrhoqs)
        else:
            celn3o4qsk = wpfloat("0.0")
        celn8qsk = exp(wpfloat("0.8") * clnrhoqs)
    else:
        cssmax = wpfloat("0.0")
        celn3o4qsk = wpfloat("0.0")
        celn8qsk = wpfloat("0.0")

    if graupel_exists:
        clnrhoqg = log(rhoqg)
        csgmax = rhoqg_intermediate / rho / dtime
        if qi + qc > MicrophysicsConstants.QMIN:
            celnrimexp_g = exp(MicrophysicsConstants.GRAUPEL_RIMEXP * clnrhoqg)
        else:
            celnrimexp_g = wpfloat("0.0")
        celn6qgk = exp(wpfloat("0.6") * clnrhoqg)
    else:
        csgmax = wpfloat("0.0")
        celnrimexp_g = wpfloat("0.0")
        celn6qgk = wpfloat("0.0")

    if ice_exists | snow_exists:
        cdvtp = MicrophysicsConstants.CCDVTP * exp(wpfloat("1.94") * log(temperature)) / pressure
        chi = MicrophysicsConstants.CCSHI1 * cdvtp * rho * qvsi / (temperature * temperature)
        chlp = cdvtp / (wpfloat("1.0") + chi)
        cidep = MicrophysicsConstants.CCIDEP * chlp

        if snow_exists:
            cslam = exp(
                MicrophysicsConstants.CCSLXP * log(MicrophysicsConstants.CCSLAM * n0s / rhoqs)
            )
            cslam = minimum(cslam, wpfloat("1.0e15"))
            csdep = wpfloat("4.0") * n0s * chlp
        else:
            cslam = wpfloat("1.0e10")
            csdep = wpfloat("3.367e-2")
    else:
        cidep = wpfloat("1.3e-5")
        cslam = wpfloat("1.0e10")
        csdep = wpfloat("3.367e-2")

    # ------------------------------------------------------------------------------
    #  Section 4: Transfer rates
    # ------------------------------------------------------------------------------

    ice_nucleation_rate_v2i = deposition_nucleation_at_low_temperature_or_in_clouds(
        temperature, rho, qv, qi, qvsi, cnin, dtime, cloud_exists
    )

    (
        cloud_autoconversion_rate_c2r,
        rain_cloud_collision_rate_c2r,
    ) = autoconversion_and_rain_accretion(
        temperature, qc, qr, qnc, celn7o8qrk, cloud_exists, liquid_autoconversion_option
    )

    cloud_freezing_rate_c2i, rain_freezing_rate_r2g_in_clouds = freezing_in_clouds(
        temperature, qc, qr, cscmax, csrmax, celn7o4qrk, cloud_exists, rain_exists
    )

    (
        snow_riming_rate_c2s,
        graupel_riming_rate_c2g,
        rain_shedding_rate_c2r,
        snow_autoconversion_rate_s2g,
    ) = riming_in_clouds(
        temperature,
        qc,
        crim,
        cslam,
        celnrimexp_g,
        celn3o4qsk,
        snow2graupel_riming_coeff,
        cloud_exists,
        snow_exists,
    )

    dist_cldtop, reduce_dep = reduced_deposition_in_clouds(
        temperature,
        qv_kup,
        qc_kup,
        qi_kup,
        qs_kup,
        qg_kup,
        qvsw_kup,
        dz,
        dist_cldtop_kup,
        k_lev,
        is_surface,
        cloud_exists,
    )

    (
        snow_ice_collision_rate_i2s,
        graupel_ice_collision_rate_i2g,
        ice_autoconverson_rate_i2s,
        ice_deposition_rate_v2i,
        rain_ice_2graupel_ice_loss_rate_i2g,
        rain_ice_2graupel_rain_loss_rate_r2g,
        ice_dep_autoconversion_rate_i2s,
        ice_net_deposition_rate_v2i,
        ice_net_sublimation_rate_v2i,
    ) = collision_and_ice_deposition_in_cold_ice_clouds(
        temperature,
        rho,
        qv,
        qi,
        qs,
        qvsi,
        rhoqi_intermediate,
        dtime,
        cslam,
        cidep,
        cagg,
        cmi,
        ice_stickeff_min,
        reduce_dep,
        celnrimexp_g,
        celn7o8qrk,
        celn13o8qrk,
        ice_exists,
    )

    (
        snow_deposition_rate_v2s_in_cold_clouds,
        graupel_deposition_rate_v2g_in_cold_clouds,
    ) = snow_and_graupel_depositional_growth_in_cold_ice_clouds(
        temperature,
        pressure,
        qv,
        qs,
        qvsi,
        dtime,
        ice_net_deposition_rate_v2i,
        cslam,
        cbsdep,
        csdep,
        reduce_dep,
        celn6qgk,
        ice_exists,
        snow_exists,
        graupel_exists,
    )

    (
        ice_melting_rate_i2c,
        snow_melting_rate_s2r,
        graupel_melting_rate_g2r,
        snow_deposition_rate_v2s_in_melting_condition,
        graupel_deposition_rate_v2g_in_melting_condition,
        rain_deposition_rate_v2r,
    ) = melting(
        temperature,
        pressure,
        rho,
        qv,
        qvsw,
        rhoqi_intermediate,
        dtime,
        cssmax,
        csgmax,
        celn8qsk,
        celn6qgk,
        ice_exists,
        snow_exists,
        graupel_exists,
    )

    (
        rain_evaporation_rate_r2v,
        rain_freezing_rate_r2g,
    ) = evaporation_and_freezing_in_subsaturated_air(
        temperature,
        qv,
        qc,
        qvsw,
        rhoqr,
        dtime,
        rain_freezing_rate_r2g_in_clouds,
        csrmax,
        precomputed_evaporation_alpha_exp_coeff,
        precomputed_evaporation_alpha_coeff,
        precomputed_evaporation_beta_exp_coeff,
        precomputed_evaporation_beta_coeff,
        celn7o4qrk,
        rain_exists,
    )

    # ------------------------------------------------------------------------------
    #  Section 5: Check for negative mass
    # ------------------------------------------------------------------------------

    snow_deposition_rate_v2s = (
        snow_deposition_rate_v2s_in_cold_clouds + snow_deposition_rate_v2s_in_melting_condition
    )
    graupel_deposition_rate_v2g = (
        graupel_deposition_rate_v2g_in_cold_clouds
        + graupel_deposition_rate_v2g_in_melting_condition
    )

    # finalizing transfer rates in clouds and calculate depositional growth reduction
    if cloud_exists & (temperature > MicrophysicsConstants.HOMOGENEOUS_FREEZE_TEMPERATURE):
        # Check for maximum depletion of cloud water and adjust the
        # transfer rates accordingly
        csum = (
            cloud_autoconversion_rate_c2r
            + rain_cloud_collision_rate_c2r
            + snow_riming_rate_c2s
            + graupel_riming_rate_c2g
            + rain_shedding_rate_c2r
        )
        ccorr = cscmax / maximum(cscmax, csum)
        cloud_autoconversion_rate_c2r = ccorr * cloud_autoconversion_rate_c2r
        rain_cloud_collision_rate_c2r = ccorr * rain_cloud_collision_rate_c2r
        snow_riming_rate_c2s = ccorr * snow_riming_rate_c2s
        graupel_riming_rate_c2g = ccorr * graupel_riming_rate_c2g
        rain_shedding_rate_c2r = ccorr * rain_shedding_rate_c2r
        snow_autoconversion_rate_s2g = minimum(
            snow_autoconversion_rate_s2g, snow_riming_rate_c2s + cssmax
        )

    if ice_exists | snow_exists | graupel_exists:
        if temperature <= PhysicsConstants.tmelt:  # cold case
            qvsidiff = qv - qvsi
            csimax = rhoqi_intermediate / rho / dtime

            # Check for maximal depletion of cloud ice
            # No check is done for depositional autoconversion (sdau) because
            # this is a always a fraction of the gain rate due to
            # deposition (i.e the sum of this rates is always positive)
            csum = (
                ice_autoconverson_rate_i2s
                + snow_ice_collision_rate_i2s
                + graupel_ice_collision_rate_i2g
                + rain_ice_2graupel_ice_loss_rate_i2g
                + ice_net_sublimation_rate_v2i
            )
            ccorr = csimax / maximum(csimax, csum) if csimax > wpfloat("0.0") else wpfloat("0.0")
            ice_deposition_rate_v2i = (
                ice_net_deposition_rate_v2i - ccorr * ice_net_sublimation_rate_v2i
            )
            ice_autoconverson_rate_i2s = ccorr * ice_autoconverson_rate_i2s
            snow_ice_collision_rate_i2s = ccorr * snow_ice_collision_rate_i2s
            graupel_ice_collision_rate_i2g = ccorr * graupel_ice_collision_rate_i2g
            rain_ice_2graupel_ice_loss_rate_i2g = ccorr * rain_ice_2graupel_ice_loss_rate_i2g
            if qvsidiff < wpfloat("0.0"):
                snow_deposition_rate_v2s = maximum(snow_deposition_rate_v2s, -cssmax)
                graupel_deposition_rate_v2g = maximum(graupel_deposition_rate_v2g, -csgmax)

    csum = rain_evaporation_rate_r2v + rain_freezing_rate_r2g + rain_ice_2graupel_rain_loss_rate_r2g
    ccorr = csrmax / maximum(csrmax, csum) if csum > wpfloat("0.0") else wpfloat("1.0")
    rain_evaporation_rate_r2v = ccorr * rain_evaporation_rate_r2v
    rain_freezing_rate_r2g = ccorr * rain_freezing_rate_r2g
    rain_ice_2graupel_rain_loss_rate_r2g = ccorr * rain_ice_2graupel_rain_loss_rate_r2g

    # limit snow depletion in order to avoid negative values of qs
    ccorr = wpfloat("1.0")
    if snow_deposition_rate_v2s <= wpfloat("0.0"):
        csum = snow_melting_rate_s2r + snow_autoconversion_rate_s2g - snow_deposition_rate_v2s
        if csum > wpfloat("0.0"):
            ccorr = cssmax / maximum(cssmax, csum)
        snow_melting_rate_s2r = ccorr * snow_melting_rate_s2r
        snow_autoconversion_rate_s2g = ccorr * snow_autoconversion_rate_s2g
        snow_deposition_rate_v2s = ccorr * snow_deposition_rate_v2s
    else:
        csum = snow_melting_rate_s2r + snow_autoconversion_rate_s2g
        if csum > wpfloat("0.0"):
            ccorr = cssmax / maximum(cssmax, csum)
        snow_melting_rate_s2r = ccorr * snow_melting_rate_s2r
        snow_autoconversion_rate_s2g = ccorr * snow_autoconversion_rate_s2g

    # ------------------------------------------------------------------------------
    #  Section 6: Update tendencies
    # ------------------------------------------------------------------------------

    cqvt = (
        rain_evaporation_rate_r2v
        - ice_deposition_rate_v2i
        - snow_deposition_rate_v2s
        - graupel_deposition_rate_v2g
        - ice_nucleation_rate_v2i
        - rain_deposition_rate_v2r
    )
    cqct = (
        ice_melting_rate_i2c
        - cloud_autoconversion_rate_c2r
        - cloud_freezing_rate_c2i
        - rain_cloud_collision_rate_c2r
        - rain_shedding_rate_c2r
        - snow_riming_rate_c2s
        - graupel_riming_rate_c2g
    )
    cqit = (
        ice_nucleation_rate_v2i
        + cloud_freezing_rate_c2i
        - ice_melting_rate_i2c
        - rain_ice_2graupel_ice_loss_rate_i2g
        + ice_deposition_rate_v2i
        - ice_dep_autoconversion_rate_i2s
        - snow_ice_collision_rate_i2s
        - graupel_ice_collision_rate_i2g
        - ice_autoconverson_rate_i2s
    )
    cqrt = (
        cloud_autoconversion_rate_c2r
        + rain_shedding_rate_c2r
        + rain_cloud_collision_rate_c2r
        + snow_melting_rate_s2r
        + graupel_melting_rate_g2r
        - rain_evaporation_rate_r2v
        - rain_ice_2graupel_rain_loss_rate_r2g
        - rain_freezing_rate_r2g
        + rain_deposition_rate_v2r
    )
    cqst = (
        ice_autoconverson_rate_i2s
        + ice_dep_autoconversion_rate_i2s
        - snow_melting_rate_s2r
        + snow_riming_rate_c2s
        + snow_deposition_rate_v2s
        + snow_ice_collision_rate_i2s
        - snow_autoconversion_rate_s2g
    )
    cqgt = (
        graupel_ice_collision_rate_i2g
        - graupel_melting_rate_g2r
        + rain_ice_2graupel_ice_loss_rate_i2g
        + rain_ice_2graupel_rain_loss_rate_r2g
        + graupel_deposition_rate_v2g
        + rain_freezing_rate_r2g
        + graupel_riming_rate_c2g
        + snow_autoconversion_rate_s2g
    )

    # l_cv (is_isochoric) is removed in icon4py. So, heat_cap_r (reciprocal of heat capacity of dry air) = MicrophysicsConstants.RCVD (at constant volume)
    temperature_tendency = MicrophysicsConstants.RCVD * (
        lhv * (cqct + cqrt) + lhs * (cqit + cqst + cqgt)
    )
    qi_tendency = maximum((rhoqi_intermediate / rho * cimi - qi) / dtime + cqit * cimi, -qi / dtime)
    qr_tendency = maximum((rhoqr_intermediate / rho * cimr - qr) / dtime + cqrt * cimr, -qr / dtime)
    qs_tendency = maximum((rhoqs_intermediate / rho * cims - qs) / dtime + cqst * cims, -qs / dtime)
    qg_tendency = maximum((rhoqg_intermediate / rho * cimg - qg) / dtime + cqgt * cimg, -qg / dtime)
    qc_tendency = maximum(cqct, -qc / dtime)
    qv_tendency = maximum(cqvt, -qv / dtime)

    k_lev = k_lev + 1

    return (
        temperature_tendency,
        qv_tendency,
        qc_tendency,
        qi_tendency,
        qr_tendency,
        qs_tendency,
        qg_tendency,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        rhoqrv,
        rhoqsv,
        rhoqgv,
        rhoqiv,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        dist_cldtop,
        rho,
        crho1o2,
        crhofac_qi,
        snow_sed0,
        qvsw,
        k_lev,
    )


@gtx.field_operator
def _icon_graupel(
    ground_level: gtx.int32,
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    use_constant_latent_heat: bool,
    ice_stickeff_min: ta.wpfloat,
    snow2graupel_riming_coeff: ta.wpfloat,
    power_law_coeff_for_ice_mean_fall_speed: ta.wpfloat,
    exponent_for_density_factor_in_ice_sedimentation: ta.wpfloat,
    power_law_coeff_for_snow_fall_speed: ta.wpfloat,
    precomputed_riming_coef: ta.wpfloat,
    precomputed_agg_coef: ta.wpfloat,
    precomputed_snow_sed_coef: ta.wpfloat,
    power_law_exponent_for_rain_mean_fall_speed: ta.wpfloat,
    power_law_coeff_for_rain_mean_fall_speed: ta.wpfloat,
    precomputed_evaporation_alpha_exp_coeff: ta.wpfloat,
    precomputed_evaporation_alpha_coeff: ta.wpfloat,
    precomputed_evaporation_beta_exp_coeff: ta.wpfloat,
    precomputed_evaporation_beta_coeff: ta.wpfloat,
    power_law_exponent_for_rain_mean_fall_speed_ln1o2: ta.wpfloat,
    power_law_exponent_for_ice_mean_fall_speed_ln1o2: ta.wpfloat,
    power_law_exponent_for_graupel_mean_fall_speed_ln1o2: ta.wpfloat,
    dtime: ta.wpfloat,
    dz: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    qnc: fa.CellField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    (
        temperature_tendency,
        qv_tendency,
        qc_tendency,
        qi_tendency,
        qr_tendency,
        qs_tendency,
        qg_tendency,
        _,
        _,
        _,
        _,
        _,
        _,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        _,
        _,
        _,
        _,
        _,
        _,
        _,
    ) = _icon_graupel_scan(
        ground_level,
        liquid_autoconversion_option,
        snow_intercept_option,
        use_constant_latent_heat,
        ice_stickeff_min,
        snow2graupel_riming_coeff,
        power_law_coeff_for_ice_mean_fall_speed,
        exponent_for_density_factor_in_ice_sedimentation,
        power_law_coeff_for_snow_fall_speed,
        precomputed_riming_coef,
        precomputed_agg_coef,
        precomputed_snow_sed_coef,
        power_law_exponent_for_rain_mean_fall_speed,
        power_law_coeff_for_rain_mean_fall_speed,
        precomputed_evaporation_alpha_exp_coeff,
        precomputed_evaporation_alpha_coeff,
        precomputed_evaporation_beta_exp_coeff,
        precomputed_evaporation_beta_coeff,
        power_law_exponent_for_rain_mean_fall_speed_ln1o2,
        power_law_exponent_for_ice_mean_fall_speed_ln1o2,
        power_law_exponent_for_graupel_mean_fall_speed_ln1o2,
        dtime,
        dz,
        temperature,
        pressure,
        rho,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        qnc,
    )

    return (
        temperature_tendency,
        qv_tendency,
        qc_tendency,
        qi_tendency,
        qr_tendency,
        qs_tendency,
        qg_tendency,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def icon_graupel(
    ground_level: gtx.int32,
    liquid_autoconversion_option: gtx.int32,
    snow_intercept_option: gtx.int32,
    use_constant_latent_heat: bool,
    ice_stickeff_min: ta.wpfloat,
    snow2graupel_riming_coeff: ta.wpfloat,
    power_law_coeff_for_ice_mean_fall_speed: ta.wpfloat,
    exponent_for_density_factor_in_ice_sedimentation: ta.wpfloat,
    power_law_coeff_for_snow_fall_speed: ta.wpfloat,
    precomputed_riming_coef: ta.wpfloat,
    precomputed_agg_coef: ta.wpfloat,
    precomputed_snow_sed_coef: ta.wpfloat,
    power_law_exponent_for_rain_mean_fall_speed: ta.wpfloat,
    power_law_coeff_for_rain_mean_fall_speed: ta.wpfloat,
    precomputed_evaporation_alpha_exp_coeff: ta.wpfloat,
    precomputed_evaporation_alpha_coeff: ta.wpfloat,
    precomputed_evaporation_beta_exp_coeff: ta.wpfloat,
    precomputed_evaporation_beta_coeff: ta.wpfloat,
    power_law_exponent_for_rain_mean_fall_speed_ln1o2: ta.wpfloat,
    power_law_exponent_for_ice_mean_fall_speed_ln1o2: ta.wpfloat,
    power_law_exponent_for_graupel_mean_fall_speed_ln1o2: ta.wpfloat,
    dtime: ta.wpfloat,
    dz: fa.CellKField[ta.wpfloat],
    temperature: fa.CellKField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    rho: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    qnc: fa.CellField[ta.wpfloat],
    temperature_tendency: fa.CellKField[ta.wpfloat],
    qv_tendency: fa.CellKField[ta.wpfloat],
    qc_tendency: fa.CellKField[ta.wpfloat],
    qi_tendency: fa.CellKField[ta.wpfloat],
    qr_tendency: fa.CellKField[ta.wpfloat],
    qs_tendency: fa.CellKField[ta.wpfloat],
    qg_tendency: fa.CellKField[ta.wpfloat],
    rhoqrv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqsv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqgv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqiv_old_kup: fa.CellKField[ta.wpfloat],
    vnew_r: fa.CellKField[ta.wpfloat],
    vnew_s: fa.CellKField[ta.wpfloat],
    vnew_g: fa.CellKField[ta.wpfloat],
    vnew_i: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _icon_graupel(
        ground_level,
        liquid_autoconversion_option,
        snow_intercept_option,
        use_constant_latent_heat,
        ice_stickeff_min,
        snow2graupel_riming_coeff,
        power_law_coeff_for_ice_mean_fall_speed,
        exponent_for_density_factor_in_ice_sedimentation,
        power_law_coeff_for_snow_fall_speed,
        precomputed_riming_coef,
        precomputed_agg_coef,
        precomputed_snow_sed_coef,
        power_law_exponent_for_rain_mean_fall_speed,
        power_law_coeff_for_rain_mean_fall_speed,
        precomputed_evaporation_alpha_exp_coeff,
        precomputed_evaporation_alpha_coeff,
        precomputed_evaporation_beta_exp_coeff,
        precomputed_evaporation_beta_coeff,
        power_law_exponent_for_rain_mean_fall_speed_ln1o2,
        power_law_exponent_for_ice_mean_fall_speed_ln1o2,
        power_law_exponent_for_graupel_mean_fall_speed_ln1o2,
        dtime,
        dz,
        temperature,
        pressure,
        rho,
        qv,
        qc,
        qi,
        qr,
        qs,
        qg,
        qnc,
        out=(
            temperature_tendency,
            qv_tendency,
            qc_tendency,
            qi_tendency,
            qr_tendency,
            qs_tendency,
            qg_tendency,
            rhoqrv_old_kup,
            rhoqsv_old_kup,
            rhoqgv_old_kup,
            rhoqiv_old_kup,
            vnew_r,
            vnew_s,
            vnew_g,
            vnew_i,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


@gtx.field_operator
def _icon_graupel_flux_at_ground(
    do_latent_heat_nudging: bool,
    dtime: ta.wpfloat,
    rho: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr_tendency: fa.CellKField[ta.wpfloat],
    qs_tendency: fa.CellKField[ta.wpfloat],
    qg_tendency: fa.CellKField[ta.wpfloat],
    qi_tendency: fa.CellKField[ta.wpfloat],
    rhoqrv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqsv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqgv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqiv_old_kup: fa.CellKField[ta.wpfloat],
    vnew_r: fa.CellKField[ta.wpfloat],
    vnew_s: fa.CellKField[ta.wpfloat],
    vnew_g: fa.CellKField[ta.wpfloat],
    vnew_i: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    rain_flux = wpfloat("0.5") * ((qr + qr_tendency * dtime) * rho * vnew_r + rhoqrv_old_kup)
    snow_flux = wpfloat("0.5") * ((qs + qs_tendency * dtime) * rho * vnew_s + rhoqsv_old_kup)
    graupel_flux = wpfloat("0.5") * ((qg + qg_tendency * dtime) * rho * vnew_g + rhoqgv_old_kup)
    ice_flux = wpfloat("0.5") * ((qi + qi_tendency * dtime) * rho * vnew_i + rhoqiv_old_kup)
    zero = broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))
    # for the latent heat nudging
    total_flux = rain_flux + snow_flux + graupel_flux if do_latent_heat_nudging else zero
    return rain_flux, snow_flux, graupel_flux, ice_flux, total_flux


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def icon_graupel_flux_at_ground(
    do_latent_heat_nudging: bool,
    dtime: ta.wpfloat,
    rho: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr_tendency: fa.CellKField[ta.wpfloat],
    qs_tendency: fa.CellKField[ta.wpfloat],
    qg_tendency: fa.CellKField[ta.wpfloat],
    qi_tendency: fa.CellKField[ta.wpfloat],
    rhoqrv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqsv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqgv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqiv_old_kup: fa.CellKField[ta.wpfloat],
    vnew_r: fa.CellKField[ta.wpfloat],
    vnew_s: fa.CellKField[ta.wpfloat],
    vnew_g: fa.CellKField[ta.wpfloat],
    vnew_i: fa.CellKField[ta.wpfloat],
    rain_precipitation_flux: fa.CellKField[ta.wpfloat],
    snow_precipitation_flux: fa.CellKField[ta.wpfloat],
    graupel_precipitation_flux: fa.CellKField[ta.wpfloat],
    ice_precipitation_flux: fa.CellKField[ta.wpfloat],
    total_precipitation_flux: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    ground_level: gtx.int32,
    model_num_levels: gtx.int32,
):
    _icon_graupel_flux_at_ground(
        do_latent_heat_nudging,
        dtime,
        rho,
        qr,
        qs,
        qg,
        qi,
        qr_tendency,
        qs_tendency,
        qg_tendency,
        qi_tendency,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        out=(
            rain_precipitation_flux,
            snow_precipitation_flux,
            graupel_precipitation_flux,
            ice_precipitation_flux,
            total_precipitation_flux,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (ground_level, model_num_levels),
        },
    )


@gtx.field_operator
def _icon_graupel_flux_above_ground(
    do_latent_heat_nudging: bool,
    dtime: ta.wpfloat,
    rho: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr_tendency: fa.CellKField[ta.wpfloat],
    qs_tendency: fa.CellKField[ta.wpfloat],
    qg_tendency: fa.CellKField[ta.wpfloat],
    qi_tendency: fa.CellKField[ta.wpfloat],
    rhoqrv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqsv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqgv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqiv_old_kup: fa.CellKField[ta.wpfloat],
    vnew_r: fa.CellKField[ta.wpfloat],
    vnew_s: fa.CellKField[ta.wpfloat],
    vnew_g: fa.CellKField[ta.wpfloat],
    vnew_i: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    zero = broadcast(wpfloat("0.0"), (dims.CellDim, dims.KDim))

    rain_flux_ = (qr + qr_tendency * dtime) * rho * vnew_r
    snow_flux_ = (qs + qs_tendency * dtime) * rho * vnew_s
    graupel_flux_ = (qg + qg_tendency * dtime) * rho * vnew_g
    ice_flux_ = (qi + qi_tendency * dtime) * rho * vnew_i

    rain_flux_new = where(rain_flux_ <= MicrophysicsConstants.QMIN, zero, rain_flux_)
    snow_flux_new = where(snow_flux_ <= MicrophysicsConstants.QMIN, zero, snow_flux_)
    graupel_flux_new = where(graupel_flux_ <= MicrophysicsConstants.QMIN, zero, graupel_flux_)
    ice_flux_new = where(ice_flux_ <= MicrophysicsConstants.QMIN, zero, ice_flux_)

    rain_flux = wpfloat("0.5") * (rain_flux_new + rhoqrv_old_kup)
    snow_flux = wpfloat("0.5") * (snow_flux_new + rhoqsv_old_kup)
    graupel_flux = wpfloat("0.5") * (graupel_flux_new + rhoqgv_old_kup)
    ice_flux = wpfloat("0.5") * (ice_flux_new + rhoqiv_old_kup)
    total_flux = rain_flux + snow_flux + graupel_flux + ice_flux if do_latent_heat_nudging else zero

    return rain_flux, snow_flux, graupel_flux, ice_flux, total_flux


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def icon_graupel_flux_above_ground(
    do_latent_heat_nudging: bool,
    dtime: ta.wpfloat,
    rho: fa.CellKField[ta.wpfloat],
    qr: fa.CellKField[ta.wpfloat],
    qs: fa.CellKField[ta.wpfloat],
    qg: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    qr_tendency: fa.CellKField[ta.wpfloat],
    qs_tendency: fa.CellKField[ta.wpfloat],
    qg_tendency: fa.CellKField[ta.wpfloat],
    qi_tendency: fa.CellKField[ta.wpfloat],
    rhoqrv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqsv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqgv_old_kup: fa.CellKField[ta.wpfloat],
    rhoqiv_old_kup: fa.CellKField[ta.wpfloat],
    vnew_r: fa.CellKField[ta.wpfloat],
    vnew_s: fa.CellKField[ta.wpfloat],
    vnew_g: fa.CellKField[ta.wpfloat],
    vnew_i: fa.CellKField[ta.wpfloat],
    rain_precipitation_flux: fa.CellKField[ta.wpfloat],
    snow_precipitation_flux: fa.CellKField[ta.wpfloat],
    graupel_precipitation_flux: fa.CellKField[ta.wpfloat],
    ice_precipitation_flux: fa.CellKField[ta.wpfloat],
    total_precipitation_flux: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    model_top: gtx.int32,
    ground_level: gtx.int32,
) -> None:
    _icon_graupel_flux_above_ground(
        do_latent_heat_nudging,
        dtime,
        rho,
        qr,
        qs,
        qg,
        qi,
        qr_tendency,
        qs_tendency,
        qg_tendency,
        qi_tendency,
        rhoqrv_old_kup,
        rhoqsv_old_kup,
        rhoqgv_old_kup,
        rhoqiv_old_kup,
        vnew_r,
        vnew_s,
        vnew_g,
        vnew_i,
        out=(
            rain_precipitation_flux,
            snow_precipitation_flux,
            graupel_precipitation_flux,
            ice_precipitation_flux,
            total_precipitation_flux,
        ),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (model_top, ground_level),
        },
    )
