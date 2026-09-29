# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause
import gt4py.next as gtx
from gt4py.next import exp, log, maximum, minimum, sqrt

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics.microphysics_constants import (
    MicrophysicsConstants,
)
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics.microphysics_options import (
    LiquidAutoConversionType,
    SnowInterceptParameterization,
)
from icon4py.model.common import field_type_aliases as fa, type_alias as ta
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.type_alias import wpfloat


@gtx.field_operator
def compute_cooper_inp_concentration(temperature: ta.wpfloat) -> ta.wpfloat:
    cnin = 5.0 * exp(0.304 * (PhysicsConstants.tmelt - temperature))
    cnin = minimum(cnin, MicrophysicsConstants.NIMAX_THOM)
    return cnin


@gtx.field_operator
def compute_snow_interception_and_collision_parameters(
    temperature: ta.wpfloat,
    rho: ta.wpfloat,
    qs: ta.wpfloat,
    precomputed_riming_coef: ta.wpfloat,
    precomputed_agg_coef: ta.wpfloat,
    precomputed_snow_sed_coef: ta.wpfloat,
    power_law_coeff_for_snow_fall_speed: ta.wpfloat,
    snow_exists: bool,
    snow_intercept_option: gtx.int32,
) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat]:
    """
    Compute the intercept parameter, N0, of the snow exponential size distribution.

    First method: Explained in paragraphs at pages 2008 and 2009 in Field et al. (2005). N0s_23 = (M_2)^4 / (M_3)^3, M_2 = Gamma(3) N0s / lamda^3, M_2 = Gamma(4) N0s / lamda^4, so N0s_23 = 2/27 N0s. And N0s_23 = 5.65E5 exp(-0.107Tc)
    Second method: Eq. 5.160 in the COSMO microphysics documentation, originally in Table 2 in Field et al. (2005).

    Args:
        temperature: air temperature [K]
        rho: air density [kg/m3]
        qs: specific snow content [kg/kg]
        precomputed_riming_coef: parameter for snow riming with clouds
        precomputed_agg_coef: parameter for ice aggregation (becomes snow)
        precomputed_snow_sed_coef: parameter for snow sedimentation
        power_law_coeff_for_snow_fall_speed: power law coefficient in snow v-D relationship
        snow_exists: snow grid cell
        snow_intercept_option: estimation method for snow intercept parameter
    Returns:
        n0s: snow size distribution intercept parameter
        snow_sed0: integration factor for snow sedimendation
        crim: riming parameter
        cagg: aggregation parameter
        cbsdep: deposition parameter
    """
    if snow_exists:
        if snow_intercept_option == SnowInterceptParameterization.FIELD_BEST_FIT_ESTIMATION:
            # Calculate n0s using the temperature-dependent
            # formula of Field et al. (2005)
            local_tc = temperature - PhysicsConstants.tmelt
            local_tc = minimum(local_tc, wpfloat("0.0"))
            local_tc = maximum(local_tc, wpfloat("-40.0"))
            n0s = MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_N0S1 * exp(
                MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_N0S2 * local_tc
            )
            n0s = minimum(n0s, wpfloat("1.0e9"))
            n0s = maximum(n0s, wpfloat("1.0e6"))

        elif snow_intercept_option == SnowInterceptParameterization.FIELD_GENERAL_MOMENT_ESTIMATION:
            # Calculate n0s using the temperature-dependent moment
            # relations of Field et al. (2005)
            local_tc = temperature - PhysicsConstants.tmelt
            local_tc = minimum(local_tc, wpfloat("0.0"))
            local_tc = maximum(local_tc, wpfloat("-40.0"))

            local_nnr = wpfloat("3.0")
            local_hlp = (
                MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMA1
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMA2 * local_tc
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMA3 * local_nnr
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMA4 * local_tc * local_nnr
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMA5 * local_tc**2.0
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMA6 * local_nnr**2.0
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMA7 * local_tc**2.0 * local_nnr
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMA8 * local_tc * local_nnr**2.0
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMA9 * local_tc**3.0
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMA10 * local_nnr**3.0
            )
            local_alf = exp(local_hlp * log(wpfloat("10.0")))
            local_bet = (
                MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMB1
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMB2 * local_tc
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMB3 * local_nnr
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMB4 * local_tc * local_nnr
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMB5 * local_tc**2.0
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMB6 * local_nnr**2.0
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMB7 * local_tc**2.0 * local_nnr
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMB8 * local_tc * local_nnr**2.0
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMB9 * local_tc**3.0
                + MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_MMB10 * local_nnr**3.0
            )

            # Here is the exponent bms=2.0 hardwired# not ideal# (Uli Blahak)
            local_m2s = (
                qs * rho / MicrophysicsConstants.POWER_LAW_COEFF_FOR_SNOW_MD_RELATION
            )  # UB rho added as bugfix
            local_m3s = local_alf * exp(local_bet * log(local_m2s))

            local_hlp = MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_N0S1 * exp(
                MicrophysicsConstants.SNOW_INTERCEPT_PARAMETER_N0S2 * local_tc
            )
            n0s = wpfloat("13.50") * local_m2s * (local_m2s / local_m3s) ** 3.0
            n0s = maximum(n0s, wpfloat("0.5") * local_hlp)
            n0s = minimum(n0s, wpfloat("1.0e2") * local_hlp)
            n0s = minimum(n0s, wpfloat("1.0e9"))
            n0s = maximum(n0s, wpfloat("1.0e6"))

        else:
            n0s = MicrophysicsConstants.SNOW_DEFAULT_INTERCEPT_PARAM

        # compute integration factor for terminal velocity
        snow_sed0 = precomputed_snow_sed_coef * exp(MicrophysicsConstants.CCSVXP * log(n0s))
        # compute constants for riming, aggregation, and deposition processes for snow
        crim = precomputed_riming_coef * n0s
        cagg = precomputed_agg_coef * n0s
        cbsdep = MicrophysicsConstants.CCSDEP * sqrt(power_law_coeff_for_snow_fall_speed)
    else:
        n0s = MicrophysicsConstants.SNOW_DEFAULT_INTERCEPT_PARAM
        snow_sed0 = wpfloat("0.0")
        crim = wpfloat("0.0")
        cagg = wpfloat("0.0")
        cbsdep = wpfloat("0.0")

    return n0s, snow_sed0, crim, cagg, cbsdep


@gtx.field_operator
def deposition_nucleation_at_low_temperature_or_in_clouds(
    temperature: ta.wpfloat,
    rho: ta.wpfloat,
    qv: ta.wpfloat,
    qi: ta.wpfloat,
    qvsi: ta.wpfloat,
    cnin: ta.wpfloat,
    dtime: ta.wpfloat,
    cloud_exists: bool,
) -> ta.wpfloat:
    """
    Heterogeneous deposition nucleation for low temperatures below a threshold or in clouds.
    When in clouds, we require water saturation for this process (i.e. the existence of cloud water) to exist.
    In this case, heterogeneous deposition nucleation is assumed to occur only when no cloud ice is present and the temperature is below a nucleation threshold.

    ice nucleation rate = ice_initial_mass Ni / rho / dt, Eq. 5.101 in the COSMO microphysics documentation
    ice_initial_mass is the initial ice crystal mass

    Args:
        temperature: air temperature [K]
        rho: air density [kg/m3]
        qv: specific humidity [kg/kg]
        qvsi: saturated vapor mixing ratio over ice
        cnin: number concentration of ice nucleating particles [/m3]
        dtime: time step [s]
        cloud_exists: cloud grid cell
    Returns:
        Deposition nucleation rate
    """
    ice_nucleation_rate_v2i = (
        MicrophysicsConstants.ICE_INITIAL_MASS / rho * cnin / dtime
        if (cloud_exists & (temperature <= wpfloat("267.15")) & (qi <= MicrophysicsConstants.QMIN))
        | (
            (temperature < MicrophysicsConstants.HETEROGENEOUS_FREEZE_TEMPERATURE)
            & (qv > wpfloat("8.0e-6"))
            & (qi <= wpfloat("0.0"))
            & (qv > qvsi)
        )
        else wpfloat("0.0")
    )
    return ice_nucleation_rate_v2i


@gtx.field_operator
def autoconversion_and_rain_accretion(
    temperature: ta.wpfloat,
    qc: ta.wpfloat,
    qr: ta.wpfloat,
    qnc: ta.wpfloat,
    celn7o8qrk: ta.wpfloat,
    cloud_exists: bool,
    liquid_autoconversion_option: gtx.int32,
) -> tuple[ta.wpfloat, ta.wpfloat]:
    """
    Compute the rate of cloud-to-rain autoconversion and the mass of cloud accreted by rain.
    Method 1: liquid_autoconversion_option = LiquidAutoConversionType.KESSLER, Kessler (1969)
    Method 2: liquid_autoconversion_option = LiquidAutoConversionType.SEIFERT_BEHENG, Seifert and beheng (2001)

    Args:
        temperature: air temperature [K]
        qc: specific cloud content [kg/kg]
        qr: specific rain content [kg/kg]
        qnc: number concentration of CCN [/m3]
        celn7o8qrk: constant (refer to equation or documentation in the docstring above)
        cloud_exists: cloud grid cell
        liquid_autoconversion_option: liquid auto conversion mode
    Returns:
        cloud-to-rain autoconversionn rate, rain-cloud accretion rate
    """
    if cloud_exists & (temperature > MicrophysicsConstants.HOMOGENEOUS_FREEZE_TEMPERATURE):
        if liquid_autoconversion_option == LiquidAutoConversionType.KESSLER:
            # Kessler(1969) autoconversion rate
            cloud_autoconversion_rate_c2r = (
                MicrophysicsConstants.KESSLER_CLOUD2RAIN_AUTOCONVERSION_COEFF_FOR_CLOUD
                * maximum(qc - MicrophysicsConstants.QC0, wpfloat("0.0"))
            )
            rain_cloud_collision_rate_c2r = (
                MicrophysicsConstants.KESSLER_CLOUD2RAIN_AUTOCONVERSION_COEFF_FOR_RAIN
                * qc
                * celn7o8qrk
            )

        elif liquid_autoconversion_option == LiquidAutoConversionType.SEIFERT_BEHENG:
            # Seifert and Beheng (2001) autoconversion rate
            local_const = (
                MicrophysicsConstants.KCAU
                / (wpfloat("20.0") * MicrophysicsConstants.XSTAR)
                * (MicrophysicsConstants.CNUE + wpfloat("2.0"))
                * (MicrophysicsConstants.CNUE + wpfloat("4.0"))
                / (MicrophysicsConstants.CNUE + wpfloat("1.0")) ** 2.0
            )

            # with constant cloud droplet number concentration qnc
            if qc > wpfloat("1.0e-6"):
                local_tau = minimum(wpfloat("1.0") - qc / (qc + qr), wpfloat("0.9"))
                local_tau = maximum(local_tau, wpfloat("1.0e-30"))
                local_hlp = exp(MicrophysicsConstants.KPHI2 * log(local_tau))
                local_phi = (
                    MicrophysicsConstants.KPHI1 * local_hlp * (wpfloat("1.0") - local_hlp) ** 3.0
                )
                cloud_autoconversion_rate_c2r = (
                    local_const
                    * qc
                    * qc
                    * qc
                    * qc
                    / (qnc * qnc)
                    * (wpfloat("1.0") + local_phi / (wpfloat("1.0") - local_tau) ** 2.0)
                )
                local_phi = (local_tau / (local_tau + MicrophysicsConstants.KPHI3)) ** 4.0
                rain_cloud_collision_rate_c2r = MicrophysicsConstants.KCAC * qc * qr * local_phi
            else:
                cloud_autoconversion_rate_c2r = wpfloat("0.0")
                rain_cloud_collision_rate_c2r = wpfloat("0.0")
        else:
            cloud_autoconversion_rate_c2r = wpfloat("0.0")
            rain_cloud_collision_rate_c2r = wpfloat("0.0")
    else:
        cloud_autoconversion_rate_c2r = wpfloat("0.0")
        rain_cloud_collision_rate_c2r = wpfloat("0.0")

    return cloud_autoconversion_rate_c2r, rain_cloud_collision_rate_c2r


@gtx.field_operator
def freezing_in_clouds(
    temperature: ta.wpfloat,
    qc: ta.wpfloat,
    qr: ta.wpfloat,
    cscmax: ta.wpfloat,
    csrmax: ta.wpfloat,
    celn7o4qrk: ta.wpfloat,
    cloud_exists: bool,
    rain_exists: bool,
) -> tuple[ta.wpfloat, ta.wpfloat]:
    """
    Compute the freezing rate of cloud and rain in clouds if there is cloud water and the temperature is above homogeneuous freezing temperature.
    Cloud is frozen to ice. Rain is frozen to graupel.
    (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)

    Method 1: rain_freezing_option = 1, Eq. 5.168
    Method 2 (ABANDONED in icon4py): rain_freezing_option = 2, Eq. 5.126, an approximation of combined immersion Eq. 5.80 and contact freezing Eq. 5.83.
    Args:
        temperature: air temperature [K]
        qc: specific cloud content [kg/kg]
        qr: specific rain content [kg/kg]
        cscmax: maximum specific cloud content [kg/kg]
        csrmax maximum specific rain content [kg/kg]
        celn7o4qrk: constant (refer to equation or documentation in the docstring above)
        cloud_exists: cloud grid cell
        rain_exists: rain grid cell
    Returns:
        cloud freezing rate, rain freezing rate
    """
    if cloud_exists:
        if temperature > MicrophysicsConstants.HOMOGENEOUS_FREEZE_TEMPERATURE:
            # Calculation of in-cloud rainwater freezing
            if (
                rain_exists
                & (temperature < MicrophysicsConstants.THRESHOLD_FREEZE_TEMPERATURE)
                & (qr > wpfloat("0.1") * qc)
            ):
                rain_freezing_rate_r2g_in_clouds = (
                    MicrophysicsConstants.COEFF_RAIN_FREEZE1
                    * (
                        exp(
                            MicrophysicsConstants.COEFF_RAIN_FREEZE2
                            * (MicrophysicsConstants.THRESHOLD_FREEZE_TEMPERATURE - temperature)
                        )
                        - wpfloat("1.0")
                    )
                    * celn7o4qrk
                )
            else:
                rain_freezing_rate_r2g_in_clouds = wpfloat("0.0")
            cloud_freezing_rate_c2i = wpfloat("0.0")
        else:
            # tg <= tg: ! hom. freezing of cloud and rain water
            cloud_freezing_rate_c2i = cscmax
            rain_freezing_rate_r2g_in_clouds = csrmax
    else:
        cloud_freezing_rate_c2i = wpfloat("0.0")
        rain_freezing_rate_r2g_in_clouds = wpfloat("0.0")

    return cloud_freezing_rate_c2i, rain_freezing_rate_r2g_in_clouds


@gtx.field_operator
def riming_in_clouds(
    temperature: ta.wpfloat,
    qc: ta.wpfloat,
    crim: ta.wpfloat,
    cslam: ta.wpfloat,
    celnrimexp_g: ta.wpfloat,
    celn3o4qsk: ta.wpfloat,
    snow2graupel_riming_coeff: ta.wpfloat,
    cloud_exists: bool,
    snow_exists: bool,
) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat]:
    """
    Compute the rate of riming by snow and graupel in clouds if there is cloud water and the temperature is above homogeneuous freezing temperature.
    (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)

    riming or accretion rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
    m_dot = pi/4 D^2 E(D) v(D) rho qc (Eq. 5.67)

    snow: f = N0 exp(-lamda D), E is constant, m(D) = alpha D^beta, v(D) = v0 D^b
        snow riming = pi/4 qc N0 E v0 Gamma(b+3) / lamda^(b+3), lamda = (alpha N0 Gamma(beta+1) / rhoqs)^(beta+1)

    graupel:
        graupel riming = 4.43 qc rhoqg^0.94878 (Eq 5.152)
        snow to graupel conversion = 0.5 qc rhoqs^0.75 (above Eq 5.132)

    rain shedding is on if temperature is above zero degree celcius. In this case, riming tendencies are converted to rain shedding.

    Args:
        temperature: air temperature [K]
        qc: specific cloud content [kg/kg]
        crim: constant (refer to equation or documentation in the docstring above)
        cslam: constant (refer to equation or documentation in the docstring above)
        celnrimexp_g: constant (refer to equation or documentation in the docstring above)
        celn3o4qsk: constant (refer to equation or documentation in the docstring above)
        cloud_exists: cloud grid cell
        snow_exists: snow grid cell
    Returns:
        snow-cloud riming rate, graupel-cloud riming rate, rain shed by snow-cloud and graupel-cloud riming rates, snow-graupel autoconversion rate
    """
    if cloud_exists & (temperature > MicrophysicsConstants.HOMOGENEOUS_FREEZE_TEMPERATURE):
        if snow_exists:
            snow_riming_rate_c2s = crim * qc * exp(MicrophysicsConstants.CCSAXP * log(cslam))
        else:
            snow_riming_rate_c2s = wpfloat("0.0")

        graupel_riming_rate_c2g = MicrophysicsConstants.CRIM_G * qc * celnrimexp_g

        if temperature >= PhysicsConstants.tmelt:
            rain_shedding_rate_c2r = snow_riming_rate_c2s + graupel_riming_rate_c2g
            snow_riming_rate_c2s = wpfloat("0.0")
            graupel_riming_rate_c2g = wpfloat("0.0")
            snow_autoconversion_rate_s2g = wpfloat("0.0")
        else:
            if qc >= MicrophysicsConstants.QC0:
                snow_autoconversion_rate_s2g = snow2graupel_riming_coeff * qc * celn3o4qsk
            else:
                snow_autoconversion_rate_s2g = wpfloat("0.0")
            rain_shedding_rate_c2r = wpfloat("0.0")
    else:
        snow_riming_rate_c2s = wpfloat("0.0")
        graupel_riming_rate_c2g = wpfloat("0.0")
        rain_shedding_rate_c2r = wpfloat("0.0")
        snow_autoconversion_rate_s2g = wpfloat("0.0")

    return (
        snow_riming_rate_c2s,
        graupel_riming_rate_c2g,
        rain_shedding_rate_c2r,
        snow_autoconversion_rate_s2g,
    )


@gtx.field_operator
def reduced_deposition_in_clouds(
    temperature: ta.wpfloat,
    qv_kup: ta.wpfloat,
    qc_kup: ta.wpfloat,
    qi_kup: ta.wpfloat,
    qs_kup: ta.wpfloat,
    qg_kup: ta.wpfloat,
    qvsw_kup: ta.wpfloat,
    dz: ta.wpfloat,
    dist_cldtop_kup: ta.wpfloat,
    k_lev: gtx.int32,
    is_surface: bool,
    cloud_exists: bool,
) -> tuple[ta.wpfloat, ta.wpfloat]:
    """
    Artificially reduce the deposition rate in clouds.

    Args:
        temperature: air temperature [K]
        qv_kup: specific humidity at k-1 level [kg/kg]
        qc_kup: specific cloud content at k-1 level [kg/kg]
        qi_kup: specific ice content at k-1 level [kg/kg]
        qs_kup: specific snow content at k-1 level [kg/kg]
        qg_kup: specific graupel content at k-1 level [kg/kg]
        qvsw_kup: saturated vapor mixing ratio at k-1 level
        dz: vertical grid spacing
        dist_cldtop_kup: vertical distance to cloud top
        k_lev: current vertical level index
        is_surface: True if the current k level is at the bottom
        cloud_exists: cloud grid cell
    Returns:
        vertical distance to cloud top, reduced factor for ice deposition
    """
    if cloud_exists:
        if (k_lev > 0) & (not is_surface):
            cqcgk_1 = qi_kup + qs_kup + qg_kup

            # distance from cloud top
            if (qv_kup + qc_kup < qvsw_kup) & (cqcgk_1 < MicrophysicsConstants.QMIN):
                # upper cloud layer
                dist_cldtop = wpfloat("0.0")  # reset distance to upper cloud layer
            else:
                dist_cldtop = dist_cldtop_kup + dz
        else:
            dist_cldtop = dist_cldtop_kup

        if (k_lev > 0) & (not is_surface):
            # finalizing transfer rates in clouds and calculate depositional growth reduction
            cnin = compute_cooper_inp_concentration(temperature)
            cfnuc = minimum(cnin / MicrophysicsConstants.NIMIX, wpfloat("1.0"))

            # with asymptotic behaviour dz -> 0 (xxx)
            #        reduce_dep = MIN(fnuc + (1.0_wp-fnuc)*(reduce_dep_ref + &
            #                             dist_cldtop(iv)/dist_cldtop_ref + &
            #                             (1.0_wp-reduce_dep_ref)*(zdh/dist_cldtop_ref)**4), 1.0_wp)

            # without asymptotic behaviour dz -> 0
            reduce_dep = cfnuc + (wpfloat("1.0") - cfnuc) * (
                MicrophysicsConstants.REDUCE_DEP_REF
                + dist_cldtop / MicrophysicsConstants.DIST_CLDTOP_REF
            )
            reduce_dep = minimum(reduce_dep, wpfloat("1.0"))
        else:
            reduce_dep = wpfloat("1.0")
    else:
        dist_cldtop = dist_cldtop_kup
        reduce_dep = wpfloat("1.0")

    return dist_cldtop, reduce_dep


@gtx.field_operator
def collision_and_ice_deposition_in_cold_ice_clouds(
    temperature: ta.wpfloat,
    rho: ta.wpfloat,
    qv: ta.wpfloat,
    qi: ta.wpfloat,
    qs: ta.wpfloat,
    qvsi: ta.wpfloat,
    rhoqi_intermediate: ta.wpfloat,
    dtime: ta.wpfloat,
    cslam: ta.wpfloat,
    cidep: ta.wpfloat,
    cagg: ta.wpfloat,
    cmi: ta.wpfloat,
    ice_stickeff_min: ta.wpfloat,
    reduce_dep: ta.wpfloat,
    celnrimexp_g: ta.wpfloat,
    celn7o8qrk: ta.wpfloat,
    celn13o8qrk: ta.wpfloat,
    ice_exists: bool,
) -> tuple[
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
    ta.wpfloat,
]:
    """
    Compute (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)
        1. the aggregation of snow and graupel in ice clouds when temperature is below zero degree celcius.
            aggregation rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
            m_dot = pi/4 D^2 E(T) v(D) rho qi (Eq. 5.67)

            snow: f = N0 exp(-lamda D), E changes with temperature, m(D) = alpha D^beta, v(D) = v0 D^b
                snow aggregation = pi/4 qi N0 E(T) v0 Gamma(b+3) / lamda^(b+3), lamda = (alpha N0 Gamma(beta+1) / rhoqs)^(beta+1)

            graupel:
                graupel aggregation = 2.46 qc rhoqg^0.94878 (Eq 5.154)
        2. the autoconversion of ice crystals in ice clouds when temperature is below zero degree celcius and depositional growth.
            iceAutoconversion = max(0.001 (qi - qi0), 0) Eq. 5.133

            ice: f = Ni delta(D-Di), mi = ai Di^3 = rho qi / Nin, v(D) = 0
            ice deposition or sublimation rate = c_dep Ni mi (qv - qvsi), Eq. 5.104

        3. the ice loss and rain loss due to accretion of rain in ice clouds when temperature is below zero degree celcius.

            riming rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
            m_dot(ice loss) = pi/4 D^2 E(T) v(D) rho qi (Eq. 5.67)
            m_dot(rain loss) = pi/4 D^5 E(T) v(D) rho qi (Eq. 5.67)

            rain: f = N0 D^mu exp(-lamda D), E is a constant, m(D) = alpha D^beta, v(D) = v0 D^b, b = 0.5 (Eq. 5.57)
            ice: uniform size=Di and mass=mi

            ice loss = pi/4 qi N0 E v0 Gamma(b+3) / lamda^(b+3), lamda = (alpha N0 Gamma(beta+1) / rhoqs)^(beta+1)

            rain loss = pi/4 qi N0 E v0 Gamma(b+3) / lamda^(b+3), lamda = (alpha N0 Gamma(beta+1) / rhoqs)^(beta+1)

        E(T) = max( 0.02, min( exp(0.09(T - T_0 )), 1.0), C_se (T - T_se )), Eq. 5.163
        Another E(T) = max(0.2, min(exp(0.09(T - T 0 )), 1.0)), Eq. 5.162, based on Lin et al. 1983 is ABANDONNED.

    Args:
        temperature: air temperature [K]
        rho: air density [kg/m3]
        qv: specific humidity [kg/kg]
        qi: specific ice content [kg/kg]
        qs: specific snow content [kg/kg]
        qvsi: saturated vapor mixing ratio over ice
        rhoqi_intermediate: ice mass with sedimendation flux from above [kg/m3]
        dtime: time step [s]
        cslam: constant (refer to equation or documentation in the docstring above)
        cidep: constant (refer to equation or documentation in the docstring above)
        cagg: constant (refer to equation or documentation in the docstring above)
        cmi: constant (refer to equation or documentation in the docstring above)
        ice_stickeff_min: constant (refer to equation or documentation in the docstring above)
        reduce_dep: ice deposition reduced factor
        celnrimexp_g: constant (refer to equation or documentation in the docstring above)
        celn7o8qrk: constant (refer to equation or documentation in the docstring above)
        celn13o8qrk: constant (refer to equation or documentation in the docstring above)
        ice_exists: ice grid cell
    Returns:
        snow-ice accretion rate,
        graupel-ice accretion rate,
        ice-to-snow autoconversion rate,
        depositional growth rate of ice,
        rain-ice accretion ice loss rate,
        rain-ice accretion rain loss rate,
        ice-to-snow deposition autoconversion rate,
        net depositional growth rate of ice,
        net sublimation growth rate of ice
    """
    if (temperature <= PhysicsConstants.tmelt) & ice_exists:
        # Change in sticking efficiency needed in case of cloud ice sedimentation
        # (based on Guenther Zaengls work)
        local_eff = minimum(
            exp(wpfloat("0.09") * (temperature - PhysicsConstants.tmelt)),
            wpfloat("1.0"),
        )
        local_eff = maximum(local_eff, ice_stickeff_min)
        local_eff = maximum(
            local_eff,
            MicrophysicsConstants.ICE_STICKING_EFF_FACTOR
            * (temperature - MicrophysicsConstants.TMIN_ICEAUTOCONV),
        )

        local_nid = rho * qi / cmi
        local_lnlogmi = log(cmi)

        local_qvsidiff = qv - qvsi
        local_svmax = local_qvsidiff / dtime

        snow_ice_collision_rate_i2s = (
            local_eff * qi * cagg * exp(MicrophysicsConstants.CCSAXP * log(cslam))
        )
        graupel_ice_collision_rate_i2g = (
            local_eff * qi * MicrophysicsConstants.CAGG_G * celnrimexp_g
        )
        ice_autoconverson_rate_i2s = (
            local_eff
            * MicrophysicsConstants.CIAU
            * maximum(qi - MicrophysicsConstants.QI0, wpfloat("0.0"))
        )

        rain_ice_2graupel_ice_loss_rate_i2g = MicrophysicsConstants.CICRI * qi * celn7o8qrk
        if qs > wpfloat("1.0e-7"):
            rain_ice_2graupel_rain_loss_rate_r2g = (
                MicrophysicsConstants.CRCRI * (qi / cmi) * celn13o8qrk
            )
        else:
            rain_ice_2graupel_rain_loss_rate_r2g = wpfloat("0.0")

        local_icetotaldeposition = (
            cidep * local_nid * exp(wpfloat("0.33") * local_lnlogmi) * local_qvsidiff
        )
        ice_deposition_rate_v2i = local_icetotaldeposition

        # for sedimenting quantities the maximum
        # allowed depletion is determined by the predictor value.
        local_simax = rhoqi_intermediate / rho / dtime

        if local_icetotaldeposition > wpfloat("0.0"):
            local_icetotaldeposition = (
                local_icetotaldeposition * reduce_dep
            )  # FR new: depositional growth reduction
            ice_net_deposition_rate_v2i = minimum(local_icetotaldeposition, local_svmax)
            ice_net_sublimation_rate_v2i = wpfloat("0.0")
        elif local_icetotaldeposition < wpfloat("0.0"):
            ice_net_deposition_rate_v2i = wpfloat("0.0")
            ice_net_sublimation_rate_v2i = maximum(local_icetotaldeposition, local_svmax)
            ice_net_sublimation_rate_v2i = -maximum(ice_net_sublimation_rate_v2i, -local_simax)
        else:
            ice_net_deposition_rate_v2i = wpfloat("0.0")
            ice_net_sublimation_rate_v2i = wpfloat("0.0")

        local_lnlogmi = log(MicrophysicsConstants.MSMIN / cmi)
        local_ztau = wpfloat("1.5") * (exp(wpfloat("0.66") * local_lnlogmi) - wpfloat("1.0"))
        ice_dep_autoconversion_rate_i2s = ice_net_deposition_rate_v2i / local_ztau
    else:
        snow_ice_collision_rate_i2s = wpfloat("0.0")
        graupel_ice_collision_rate_i2g = wpfloat("0.0")
        ice_autoconverson_rate_i2s = wpfloat("0.0")
        ice_deposition_rate_v2i = wpfloat("0.0")
        rain_ice_2graupel_ice_loss_rate_i2g = wpfloat("0.0")
        rain_ice_2graupel_rain_loss_rate_r2g = wpfloat("0.0")
        ice_dep_autoconversion_rate_i2s = wpfloat("0.0")
        ice_net_deposition_rate_v2i = wpfloat("0.0")
        ice_net_sublimation_rate_v2i = wpfloat("0.0")

    return (
        snow_ice_collision_rate_i2s,
        graupel_ice_collision_rate_i2g,
        ice_autoconverson_rate_i2s,
        ice_deposition_rate_v2i,
        rain_ice_2graupel_ice_loss_rate_i2g,
        rain_ice_2graupel_rain_loss_rate_r2g,
        ice_dep_autoconversion_rate_i2s,
        ice_net_deposition_rate_v2i,
        ice_net_sublimation_rate_v2i,
    )


@gtx.field_operator
def snow_and_graupel_depositional_growth_in_cold_ice_clouds(
    temperature: ta.wpfloat,
    pressure: ta.wpfloat,
    qv: ta.wpfloat,
    qs: ta.wpfloat,
    qvsi: ta.wpfloat,
    dtime: ta.wpfloat,
    ice_net_deposition_rate_v2i: ta.wpfloat,
    cslam: ta.wpfloat,
    cbsdep: ta.wpfloat,
    csdep: ta.wpfloat,
    reduce_dep: ta.wpfloat,
    celn6qgk: ta.wpfloat,
    ice_exists: bool,
    snow_exists: bool,
    graupel_exists: bool,
) -> tuple[ta.wpfloat, ta.wpfloat]:
    """
    Compute the vapor deposition of ice crystals and snow in ice clouds when temperature is below zero degree celcius.
    (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)

    deposition rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
    m_dot = 4 pi C(D) G F(v,D) d (qv - qvsi),
    G = 1/(1+Hw) and d are functions of environment
    F = 1 + 0.26 sqrt(D v(D)/eta/2) is ventilation factor
    C(D) = C0 D is capacitance (C0 = D/2 for a sphere, D/pi for a circular disk)

    snow resulted from fast ice deposition = ice deposition rate / time_scale, Eq. 5.108

    snow: f = N0 exp(-lamda D), v = v0 D^b
    snow deposition rate = Eq. 5.118 (wrong?) , derived from Eq. 5.71 and 5.72
       = 4 G d (qv - qvsi) N0 (1 + 0.26 sqrt(v0/eta/2) Gamma((5+b)/2)) / lamda^((1+b)/2) 1/lamda^(2)

    graupel deposition = Eq. 5.140

    Args:
        temperature: air temperature [K]
        pressure: air pressure [Pa]
        qv: specific humidity [kg/kg]
        qs: specific snow content [kg/kg]
        qvsi: saturated vapor mixing ratio over ice
        dtime: time step [s]
        ice_net_deposition_rate_v2i: ice deposition transfer rate
        cslam: constant (refer to equation or documentation in the docstring above)
        cbsdep: constant (refer to equation or documentation in the docstring above)
        csdep: constant (refer to equation or documentation in the docstring above)
        reduce_dep: ice deposition reduced factor
        celn6qgk: constant (refer to equation or documentation in the docstring above)
        ice_exists: ice grid cell
        snow_exists: snow grid cell
        graupel_exists: graupel grid cell
    Returns:
        depositional growth rate of snow in cold clouds, depositional growth rate of graupel in cold clouds
    """
    if ice_exists | snow_exists | graupel_exists:
        if temperature <= PhysicsConstants.tmelt:
            local_qvsidiff = qv - qvsi
            local_svmax = local_qvsidiff / dtime

            local_xfac = wpfloat("1.0") + cbsdep * exp(MicrophysicsConstants.CCSDXP * log(cslam))
            snow_deposition_rate_v2s_in_cold_clouds = (
                csdep * local_xfac * local_qvsidiff / (cslam + PhysicsConstants.eps) ** 2.0
            )
            # FR new: depositional growth reduction
            if snow_deposition_rate_v2s_in_cold_clouds > wpfloat("0.0"):
                snow_deposition_rate_v2s_in_cold_clouds = (
                    snow_deposition_rate_v2s_in_cold_clouds * reduce_dep
                )

            # GZ: This limitation, which was missing in the original graupel scheme,
            # is crucial for numerical stability in the tropics!
            if snow_deposition_rate_v2s_in_cold_clouds > wpfloat("0.0"):
                snow_deposition_rate_v2s_in_cold_clouds = minimum(
                    snow_deposition_rate_v2s_in_cold_clouds,
                    local_svmax - ice_net_deposition_rate_v2i,
                )
            # Suppress depositional growth of snow if the existing amount is too small for a
            # a meaningful distiction between cloud ice and snow
            if qs <= wpfloat("1.0e-7"):
                snow_deposition_rate_v2s_in_cold_clouds = minimum(
                    snow_deposition_rate_v2s_in_cold_clouds, wpfloat("0.0")
                )
            # ** GZ: this numerical fit should be replaced with a physically more meaningful formulation **
            graupel_deposition_rate_v2g_in_cold_clouds = (
                (
                    wpfloat("0.398561")
                    - wpfloat("0.00152398") * temperature
                    + wpfloat("2554.99") / pressure
                    + wpfloat("2.6531e-7") * pressure
                )
                * local_qvsidiff
                * celn6qgk
            )
        else:
            snow_deposition_rate_v2s_in_cold_clouds = wpfloat("0.0")
            graupel_deposition_rate_v2g_in_cold_clouds = wpfloat("0.0")
    else:
        snow_deposition_rate_v2s_in_cold_clouds = wpfloat("0.0")
        graupel_deposition_rate_v2g_in_cold_clouds = wpfloat("0.0")

    return snow_deposition_rate_v2s_in_cold_clouds, graupel_deposition_rate_v2g_in_cold_clouds


@gtx.field_operator
def melting(
    temperature: ta.wpfloat,
    pressure: ta.wpfloat,
    rho: ta.wpfloat,
    qv: ta.wpfloat,
    qvsw: ta.wpfloat,
    rhoqi_intermediate: ta.wpfloat,
    dtime: ta.wpfloat,
    cssmax: ta.wpfloat,
    csgmax: ta.wpfloat,
    celn8qsk: ta.wpfloat,
    celn6qgk: ta.wpfloat,
    ice_exists: bool,
    snow_exists: bool,
    graupel_exists: bool,
) -> tuple[ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat, ta.wpfloat]:
    """
    Compute the vapor deposition of ice crystals, snow, and graupel in ice clouds when temperature is above zero degree celcius.
    When the air is supersubsaturated over both ice and water, depositional growth of snow and graupel is converted to growth of rain.
    (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)

    Ice crystals completely melt when temperature is above zero.
    For snow and graupel, follow Eqs. 5.141 - 5.146

    Args:
        temperature: air temperature [K]
        pressure: air pressure [Pa]
        rho: air density [kg/m3]
        qv: specific humidity [kg/kg]
        qvsw: saturated vapor mixing ratio
        rhoqi_intermediate: ice mass with sedimendation flux from above [kg/m3]
        dtime: time step [s]
        cssmax: maximum specific snow content
        csgmax: maximum specific graupel content
        celn8qsk: constant (refer to equation or documentation in the docstring above)
        celn6qgk: constant (refer to equation or documentation in the docstring above)
        ice_exists: ice grid cell
        snow_exists: snow grid cell
        graupel_exists: graupel grid cell
    Returns:
        melting rate of ice,
        melting rate of snow,
        melting rate of graupel,
        depositional growth rate of snow in melting condition,
        depositional growth rate of graupel in melting condition,
        growth rate of rain in melting condition
    """
    if ice_exists | snow_exists | graupel_exists:
        if temperature > PhysicsConstants.tmelt:
            # cloud ice melts instantaneously
            ice_melting_rate_i2c = rhoqi_intermediate / rho / dtime

            local_qvsw0 = MicrophysicsConstants.PVSW0 / (
                rho * PhysicsConstants.rv * PhysicsConstants.tmelt
            )
            local_qvsw0diff = qv - local_qvsw0

            # ** GZ: several numerical fits in this section should be replaced with physically more meaningful formulations **
            if temperature > PhysicsConstants.tmelt - MicrophysicsConstants.TCRIT * local_qvsw0diff:
                # calculate melting rate
                local_x1 = (
                    temperature
                    - PhysicsConstants.tmelt
                    + MicrophysicsConstants.ASMEL * local_qvsw0diff
                )
                snow_melting_rate_s2r = (
                    (wpfloat("79.6863") / pressure + wpfloat("0.612654e-3")) * local_x1 * celn8qsk
                )
                snow_melting_rate_s2r = minimum(snow_melting_rate_s2r, cssmax)
                graupel_melting_rate_g2r = (
                    (wpfloat("12.31698") / pressure + wpfloat("7.39441e-05")) * local_x1 * celn6qgk
                )
                graupel_melting_rate_g2r = minimum(graupel_melting_rate_g2r, csgmax)
                # deposition + melting, ice particle temperature: t0
                # calculation without howell-factor!
                snow_deposition_rate_v2s_in_melting_condition = (
                    (wpfloat("31282.3") / pressure + wpfloat("0.241897"))
                    * local_qvsw0diff
                    * celn8qsk
                )
                graupel_deposition_rate_v2g_in_melting_condition = (
                    (wpfloat("0.153907") - pressure * wpfloat("7.86703e-07"))
                    * local_qvsw0diff
                    * celn6qgk
                )
                if local_qvsw0diff < wpfloat("0.0"):
                    # melting + evaporation of snow/graupel
                    snow_deposition_rate_v2s_in_melting_condition = maximum(
                        -cssmax, snow_deposition_rate_v2s_in_melting_condition
                    )
                    graupel_deposition_rate_v2g_in_melting_condition = maximum(
                        -csgmax, graupel_deposition_rate_v2g_in_melting_condition
                    )
                    # melt water evaporates
                    snow_melting_rate_s2r = (
                        snow_melting_rate_s2r + snow_deposition_rate_v2s_in_melting_condition
                    )
                    graupel_melting_rate_g2r = (
                        graupel_melting_rate_g2r + graupel_deposition_rate_v2g_in_melting_condition
                    )
                    snow_melting_rate_s2r = maximum(snow_melting_rate_s2r, wpfloat("0.0"))
                    graupel_melting_rate_g2r = maximum(graupel_melting_rate_g2r, wpfloat("0.0"))
                    rain_deposition_rate_v2r = wpfloat("0.0")
                else:
                    # deposition on snow/graupel is interpreted as increase in rain water ( qv --> qr, sconr), therefore,  sconr=(zssdep+zsgdep)
                    rain_deposition_rate_v2r = (
                        snow_deposition_rate_v2s_in_melting_condition
                        + graupel_deposition_rate_v2g_in_melting_condition
                    )
                    snow_deposition_rate_v2s_in_melting_condition = wpfloat("0.0")
                    graupel_deposition_rate_v2g_in_melting_condition = wpfloat("0.0")
            else:
                snow_melting_rate_s2r = wpfloat("0.0")
                graupel_melting_rate_g2r = wpfloat("0.0")
                rain_deposition_rate_v2r = wpfloat("0.0")
                # if t<t_crit, no melting, only evaporation of snow/graupel
                local_qvsidiff = qv - qvsw
                snow_deposition_rate_v2s_in_melting_condition = (
                    (wpfloat("0.28003") - pressure * wpfloat("0.146293e-6"))
                    * local_qvsidiff
                    * celn8qsk
                )
                graupel_deposition_rate_v2g_in_melting_condition = (
                    (wpfloat("0.0418521") - pressure * wpfloat("4.7524e-8"))
                    * local_qvsidiff
                    * celn6qgk
                )
                snow_deposition_rate_v2s_in_melting_condition = maximum(
                    -cssmax, snow_deposition_rate_v2s_in_melting_condition
                )
                graupel_deposition_rate_v2g_in_melting_condition = maximum(
                    -csgmax, graupel_deposition_rate_v2g_in_melting_condition
                )
        else:
            ice_melting_rate_i2c = wpfloat("0.0")
            snow_melting_rate_s2r = wpfloat("0.0")
            graupel_melting_rate_g2r = wpfloat("0.0")
            rain_deposition_rate_v2r = wpfloat("0.0")
            snow_deposition_rate_v2s_in_melting_condition = wpfloat("0.0")
            graupel_deposition_rate_v2g_in_melting_condition = wpfloat("0.0")
    else:
        ice_melting_rate_i2c = wpfloat("0.0")
        snow_melting_rate_s2r = wpfloat("0.0")
        graupel_melting_rate_g2r = wpfloat("0.0")
        rain_deposition_rate_v2r = wpfloat("0.0")
        snow_deposition_rate_v2s_in_melting_condition = wpfloat("0.0")
        graupel_deposition_rate_v2g_in_melting_condition = wpfloat("0.0")

    return (
        ice_melting_rate_i2c,
        snow_melting_rate_s2r,
        graupel_melting_rate_g2r,
        snow_deposition_rate_v2s_in_melting_condition,
        graupel_deposition_rate_v2g_in_melting_condition,
        rain_deposition_rate_v2r,
    )


@gtx.field_operator
def evaporation_and_freezing_in_subsaturated_air(
    temperature: ta.wpfloat,
    qv: ta.wpfloat,
    qc: ta.wpfloat,
    qvsw: ta.wpfloat,
    rhoqr: ta.wpfloat,
    dtime: ta.wpfloat,
    rain_freezing_rate_r2g_in_clouds: ta.wpfloat,
    csrmax: ta.wpfloat,
    precomputed_evaporation_alpha_exp_coeff: ta.wpfloat,
    precomputed_evaporation_alpha_coeff: ta.wpfloat,
    precomputed_evaporation_beta_exp_coeff: ta.wpfloat,
    precomputed_evaporation_beta_coeff: ta.wpfloat,
    celn7o4qrk: ta.wpfloat,
    rain_exists: bool,
) -> tuple[ta.wpfloat, ta.wpfloat]:
    """
    Compute the evaporation rate of rain in subsaturated condition.
    (Please refer to the COSMO microphysics documentation via the link given in the docstring of SingleMomentSixClassIconGraupelConfig for all the equations)

    deposition rate = 1/rho intg_0^inf m_dot f dD (Eq. 5.64)
    m_dot = 4 pi C(D) G F(v,D) d (qv - qvsw),
    G = 1/(1+Hw) and d are functions of environment
    F = 1 + 0.26 sqrt(D v(D)/eta/2) is ventilation factor
    C(D) = C0 D is capacitance (C0 = D/2 for a sphere, D/pi for a circular disk)

    snow resulted from fast ice deposition = ice deposition rate / time_scale, Eq. 5.108

    rain: gamma distribution f = N0 D^(mu) exp(-lamda D), m = alpha D^beta, v = v0 D^b
        V = v0 Gamma(b + beta + mu + 1) 1 / (alpha N0 Gamma(beta + mu + 1) )^(b/(beta + mu + 1)) (rho q)^(b/(beta + mu + 1)) rho_factor
        rain evaporation rate = Eq. 5.117 (wrong?) , derived from Eq. 5.71 and 5.72
                              = 2 pi G d (qv - qvsw) N0 (Gamma(2+mu) + 0.26 sqrt(v0/eta/2) Gamma((5+b+mu)/2)) / lamda^((1+b)/2) 1/lamda^(2+mu)
        lamda = (alpha N0 Gamma(beta+mu+1) / rhoq )^(1/(beta+mu+1))

        rain freezing rate =
           Method 1: rain_freezing_option = 1, Eq. 5.168
           Method 2 (ABANDONED in icon4py): rain_freezing_option = 2, Eq. 5.126, an approximation of combined immersion Eq. 5.80 and contact freezing Eq. 5.83

    Args:
        temperature: air temperature [K]
        qv: specific humidity [kg/kg]
        qc: specific cloud content [kg/kg]
        qvsw: saturated vapor mixing ratio
        rhoqr: rain mass [kg/m3]
        dtime: time step [s]
        rain_freezing_rate_r2g_in_clouds: rain freezing transfer rate in clouds
        csrmax: maximum specific rain content
        precomputed_evaporation_alpha_exp_coeff: constant (refer to equation or documentation in the docstring above)
        precomputed_evaporation_alpha_coeff: constant (refer to equation or documentation in the docstring above)
        precomputed_evaporation_beta_exp_coeff: constant (refer to equation or documentation in the docstring above)
        precomputed_evaporation_beta_coeff: constant (refer to equation or documentation in the docstring above)
        celn7o4qrk: constant (refer to equation or documentation in the docstring above)
        rain_exists: rain grid cell
    Returns:
        evaporation rate of rain, freezing rate of rain
    """
    rain_freezing_rate_r2g = rain_freezing_rate_r2g_in_clouds
    if rain_exists & (qv + qc <= qvsw):
        local_lnqr = log(rhoqr)
        local_x1 = wpfloat("1.0") + precomputed_evaporation_beta_coeff * exp(
            precomputed_evaporation_beta_exp_coeff * local_lnqr
        )
        # Limit evaporation rate in order to avoid overshoots towards supersaturation, the pre-factor approximates (esat(T_wb)-e)/(esat(T)-e) at temperatures between 0 degC and 30 degC
        local_temp_c = temperature - PhysicsConstants.tmelt
        local_maxevap = (
            (
                wpfloat("0.61")
                - wpfloat("0.0163") * local_temp_c
                + wpfloat("1.111e-4") * local_temp_c**2.0
            )
            * (qvsw - qv)
            / dtime
        )
        rain_evaporation_rate_r2v = (
            precomputed_evaporation_alpha_coeff
            * local_x1
            * (qvsw - qv)
            * exp(precomputed_evaporation_alpha_exp_coeff * local_lnqr)
        )
        rain_evaporation_rate_r2v = minimum(rain_evaporation_rate_r2v, local_maxevap)

        if temperature > MicrophysicsConstants.HOMOGENEOUS_FREEZE_TEMPERATURE:
            # Calculation of below-cloud rainwater freezing
            if temperature < MicrophysicsConstants.THRESHOLD_FREEZE_TEMPERATURE:
                # FR new: reduced rain freezing rate
                rain_freezing_rate_r2g = (
                    MicrophysicsConstants.COEFF_RAIN_FREEZE1
                    * (
                        exp(
                            MicrophysicsConstants.COEFF_RAIN_FREEZE2
                            * (MicrophysicsConstants.THRESHOLD_FREEZE_TEMPERATURE - temperature)
                        )
                        - wpfloat("1.0")
                    )
                    * celn7o4qrk
                )
        else:  # Hom. freezing of rain water
            rain_freezing_rate_r2g = csrmax
    else:
        rain_evaporation_rate_r2v = wpfloat("0.0")

    return rain_evaporation_rate_r2v, rain_freezing_rate_r2g


@gtx.field_operator
def sat_pres_water_scalar(temperature: ta.wpfloat) -> ta.wpfloat:
    """
    Compute saturation water vapour pressure by the Tetens formula.
        psat = p0 exp( aw (T-T0)/(T-bw)) )  [Tetens formula]

    Args:
        temperature: temperature [K]
    Returns:
        saturation water vapour pressure.
    """
    return MicrophysicsConstants.TETENS_P0 * exp(
        MicrophysicsConstants.TETENS_AW
        * (temperature - PhysicsConstants.tmelt)
        / (temperature - MicrophysicsConstants.TETENS_BW)
    )


@gtx.field_operator
def sat_pres_water(temperature: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    """
    Compute saturation water vapour pressure by the Tetens formula.
        psat = p0 exp( aw (T-T0)/(T-bw)) )  [Tetens formula]

    Args:
        temperature: temperature [K]
    Returns:
        saturation water vapour pressure.
    """
    return MicrophysicsConstants.TETENS_P0 * exp(
        MicrophysicsConstants.TETENS_AW
        * (temperature - PhysicsConstants.tmelt)
        / (temperature - MicrophysicsConstants.TETENS_BW)
    )


@gtx.field_operator
def sat_pres_ice(temperature: ta.wpfloat) -> ta.wpfloat:
    return MicrophysicsConstants.TETENS_P0 * exp(
        MicrophysicsConstants.TETENS_AI
        * (temperature - PhysicsConstants.tmelt)
        / (temperature - MicrophysicsConstants.TETENS_BI)
    )


@gtx.field_operator
def latent_heat_vaporization(
    temperature: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the latent heat of vaporisation with Kirchoff's relations (users can refer to Pruppacher and Klett textbook).
        dL/dT ~= cpv - cpw + v dp/dT
        L ~= (cpv - cpw) (T - T0) - Rv T

    Args:
        temperature: temperature [K]
    Returns:
        latent heat of vaporization.
    """
    return (
        PhysicsConstants.lh_vaporise
        + (1850.0 - PhysicsConstants.cpl) * (temperature - PhysicsConstants.tmelt)
        - PhysicsConstants.rv * temperature
    )


@gtx.field_operator
def qsat_rho(
    temperature: fa.CellKField[ta.wpfloat], rho: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute specific humidity at water saturation (with respect to flat surface).
        qsat = Rd/Rv psat/(p - psat) ~= Rd/Rv psat/p = 1/Rv psat/(rho T)
    Tetens formula is used for saturation water pressure (psat).
        psat = p0 exp( aw (T-T0)/(T-bw)) )  [Tetens formula]

    Args:
        temperature: temperature [K]
        rho: total air density (including hydrometeors) [kg m-3]
    Returns:
        specific humidity at water saturation.
    """
    return sat_pres_water(temperature) / (rho * PhysicsConstants.rv * temperature)


@gtx.field_operator
def dqsatdT_rho(
    temperature: fa.CellKField[ta.wpfloat], zqsat: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    """
    Compute the partical derivative of the specific humidity at water saturation (qsat) with respect to the temperature at
    constant total density. qsat is approximated as
        qsat = Rd/Rv psat/(p - psat) ~= Rd/Rv psat/p = 1/Rv psat/(rho T)
    Tetens formula is used for saturation water pressure (psat).
        psat = p0 exp( aw (T-T0)/(T-bw)) )  [Tetens formula]
    FInally, the derivative with respect to temperature is
        dpsat/dT = psat (T0-bw)/(T-bw)^2
        dqsat/dT = 1/Rv psat/(rho T) (T0-bw)/(T-bw)^2 - 1/Rv psat/(rho T^2) = qsat ((T0-bw)/(T-bw)^2 - 1/T)

    Args:
        temperature: temperature [K]
        zqsat: saturated water mixing ratio
    Returns:
        partial derivative of the specific humidity at water saturation.
    """
    beta = (
        MicrophysicsConstants.TETENS_DER / (temperature - MicrophysicsConstants.TETENS_BW) ** 2
        - 1.0 / temperature
    )
    return beta * zqsat
