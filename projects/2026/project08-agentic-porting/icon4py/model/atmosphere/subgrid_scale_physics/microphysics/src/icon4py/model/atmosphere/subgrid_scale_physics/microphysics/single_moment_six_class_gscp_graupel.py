# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import dataclasses
import math
from typing import TYPE_CHECKING, Any

import gt4py.next as gtx

from icon4py.model.atmosphere.subgrid_scale_physics.microphysics import (
    microphysics_options as mphys_options,
)
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics.microphysics_constants import (
    MicrophysicsConstants,
)
from icon4py.model.atmosphere.subgrid_scale_physics.microphysics.stencils import graupel_stencils
from icon4py.model.common import (
    dimension as dims,
    field_type_aliases as fa,
    model_options,
    type_alias as ta,
)
from icon4py.model.common.constants import PhysicsConstants
from icon4py.model.common.grid import horizontal as h_grid
from icon4py.model.common.utils import data_allocation as data_alloc, fortran_config


if TYPE_CHECKING:
    import gt4py.next.typing as gtx_typing

    from icon4py.model.common.grid import icon as icon_grid, vertical as v_grid


@dataclasses.dataclass(frozen=True)
class SingleMomentSixClassIconGraupelConfig:
    """
    Contains necessary parameters to configure icon graupel microphysics scheme.

    Encapsulates namelist parameters.
    Values should be read from configuration.
    Default values are taken from the defaults in the corresponding ICON Fortran namelist files.

    lsedi_ice (option for whether ice sedimendation is performed), lstickeff (option for different empirical formulae of sticking efficiency), and lred_depgrow (option for reduced depositional growth) are removed because they are set to True in the original code gscp_graupel.f90.

    lpres_pri is removed because it is a duplicated option for lsedi_ice.

    ldiag_ttend, and ldiag_qtend are removed because default outputs in icon4py physics granules include tendencies.

    l_cv is removed. This option is to swtich between whether the microphysical processes are isochoric or isobaric. Originally defined as  as input to the graupel in ICON. It is hardcoded to True in ICON.
    The reason for its existence and isobaric may have a positive effect on reducing sound waves. In case it needs to be restored, is_isochoric is a better name.

    The COSMO microphysics documentation "A Description of the Nonhydrostatic Regional COSMO-Model Part II Physical Parameterizations" can be downloaded via this link: https://www.cosmo-model.org/content/model/cosmo/coreDocumentation/cosmo_physics_6.00.pdf
    """

    #: liquid auto conversion mode. Originally defined as isnow_n0temp (PARAMETER) in gscp_data.f90 in ICON. I keep it because I think the choice depends on resolution.
    liquid_autoconversion_option: mphys_options.LiquidAutoConversionType = (
        mphys_options.LiquidAutoConversionType.SEIFERT_BEHENG
    )
    #: snow size distribution interception parameter. Originally defined as isnow_n0temp (PARAMETER) in gscp_data.f90 in ICON. I keep it because I think the choice depends on resolution.
    snow_intercept_option: mphys_options.SnowInterceptParameterization = (
        mphys_options.SnowInterceptParameterization.FIELD_GENERAL_MOMENT_ESTIMATION
    )
    #: Do latent heat nudging. Originally defined as dass_lhn in mo_run_config.f90 in ICON.
    do_latent_heat_nudging: bool = False
    #: Whether a fixed latent heat capacities are used for water. Originally defined as ithermo_water in mo_nwp_tuning_config.f90 in ICON (0 means True).
    use_constant_latent_heat: bool = True
    #: First parameter in RHS of eq. 5.163 in the COSMO microphysics documentation for the sticking efficiency when lstickeff = True (repricated in icon4py because it is always True in ICON). Originally defined as tune_zceff_min in mo_tuning_nwp_config.f90 in ICON.
    ice_stickeff_min: ta.wpfloat = 0.075
    #: Power law coefficient in v-qi ice terminal velocity-mixing ratio relationship, see eq. 5.169 in the COSMO microphysics documentation. Originally defined as tune_zvz0i in mo_tuning_nwp_config.f90 in ICON.
    power_law_coeff_for_ice_mean_fall_speed: ta.wpfloat = 1.25
    #: Exponent of the density factor in ice terminal velocity equation to account for density (air thermodynamic state) change. Originally defined as tune_icesedi_exp in mo_tuning_nwp_config.f90 in ICON.
    exponent_for_density_factor_in_ice_sedimentation: ta.wpfloat = 0.33
    #: Power law coefficient in v-D snow terminal velocity-Diameter relationship, see eqs. 5.57 (depricated after COSMO 3.0) and unnumbered eq. (v = 25 D^0.5) below eq. 5.159 in the COSMO microphysics documentation. Originally defined as tune_v0snow in mo_tuning_nwp_config.f90 in ICON.
    power_law_coeff_for_snow_fall_speed: ta.wpfloat = 20.0
    #: mu exponential factor in gamma distribution of rain particles. Originally defined as mu_rain in mo_nwp_tuning_config.f90 in ICON.
    rain_mu: ta.wpfloat = 0.0
    #: Interception parameter in gamma distribution of rain particles. Originally defined as rain_n0_factor in mo_nwp_tuning_config.f90 in ICON.
    rain_n0: ta.wpfloat = 1.0
    #: coefficient for snow-graupel conversion by riming. Originally defined as csg in mo_nwp_tuning_config.f90 in ICON.
    snow2graupel_riming_coeff: ta.wpfloat = 0.5

    @classmethod
    def from_fortran_dict(
        cls, atmo_dict: dict[str, Any], **overrides: Any
    ) -> SingleMomentSixClassIconGraupelConfig:
        run_nml = atmo_dict["run_nml"]

        nwp_phy_nml = atmo_dict["nwp_phy_nml"]
        nwp_tuning_nml = atmo_dict["nwp_tuning_nml"]
        return cls(
            do_latent_heat_nudging=run_nml["ldass_lhn"],
            use_constant_latent_heat=fortran_config.list_to_value(nwp_phy_nml["ithermo_water"])
            == 0,
            ice_stickeff_min=nwp_tuning_nml["tune_zceff_min"],
            power_law_coeff_for_ice_mean_fall_speed=nwp_tuning_nml["tune_zvz0i"],
            exponent_for_density_factor_in_ice_sedimentation=nwp_tuning_nml["tune_icesedi_exp"],
            power_law_coeff_for_snow_fall_speed=nwp_tuning_nml["tune_v0snow"],
            rain_mu=nwp_phy_nml["mu_rain"],
            rain_n0=nwp_phy_nml["rain_n0_factor"],
            snow2graupel_riming_coeff=nwp_tuning_nml["tune_zcsg"],
            **overrides,
        )


@dataclasses.dataclass
class MetricStateIconGraupel:
    ddqz_z_full: fa.CellKField[ta.wpfloat]


class SingleMomentSixClassIconGraupel:
    def __init__(
        self,
        *,
        graupel_config: SingleMomentSixClassIconGraupelConfig,
        grid: icon_grid.IconGrid,
        metric_state: MetricStateIconGraupel,
        vertical_params: v_grid.VerticalGrid,
        backend: gtx_typing.Backend | None,
    ):
        self.config = graupel_config
        self._initialize_configurable_parameters()
        self._grid = grid
        self._metric_state = metric_state
        self.vertical_params = vertical_params
        self._backend = backend

        self._initialize_local_fields()
        self._determine_horizontal_domains()
        self._initialize_gt4py_programs()

    def _initialize_configurable_parameters(self):
        precomputed_riming_coef: ta.wpfloat = (
            0.25
            * math.pi
            * MicrophysicsConstants.SNOW_CLOUD_COLLECTION_EFF
            * self.config.power_law_coeff_for_snow_fall_speed
            * math.gamma(MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_SNOW_FALL_SPEED + 3.0)
        )
        precomputed_agg_coef: ta.wpfloat = (
            0.25
            * math.pi
            * self.config.power_law_coeff_for_snow_fall_speed
            * math.gamma(MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_SNOW_FALL_SPEED + 3.0)
        )
        _ccsvxp = -(
            MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_SNOW_FALL_SPEED
            / (MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_SNOW_MD_RELATION + 1.0)
            + 1.0
        )
        precomputed_snow_sed_coef: ta.wpfloat = (
            MicrophysicsConstants.POWER_LAW_COEFF_FOR_SNOW_MD_RELATION
            * self.config.power_law_coeff_for_snow_fall_speed
            * math.gamma(
                MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_SNOW_MD_RELATION
                + MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_SNOW_FALL_SPEED
                + 1.0
            )
            * (
                MicrophysicsConstants.POWER_LAW_COEFF_FOR_SNOW_MD_RELATION
                * math.gamma(MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_SNOW_MD_RELATION + 1.0)
            )
            ** _ccsvxp
        )
        _n0r: ta.wpfloat = (
            8.0e6 * math.exp(3.2 * self.config.rain_mu) * 0.01 ** (-self.config.rain_mu)
        )  # empirical relation adapted from Ulbrich (1983)
        _n0r: ta.wpfloat = _n0r * self.config.rain_n0  # apply tuning factor to rain_n0 variable
        _ar: ta.wpfloat = (
            math.pi
            * PhysicsConstants.water_density
            / 6.0
            * _n0r
            * math.gamma(self.config.rain_mu + 4.0)
        )  # pre-factor

        power_law_exponent_for_rain_mean_fall_speed: ta.wpfloat = 0.5 / (self.config.rain_mu + 4.0)
        power_law_coeff_for_rain_mean_fall_speed: ta.wpfloat = (
            130.0
            * math.gamma(self.config.rain_mu + 4.5)
            / math.gamma(self.config.rain_mu + 4.0)
            * _ar ** (-power_law_exponent_for_rain_mean_fall_speed)
        )

        precomputed_evaporation_alpha_exp_coeff: ta.wpfloat = (self.config.rain_mu + 2.0) / (
            self.config.rain_mu + 4.0
        )
        precomputed_evaporation_alpha_coeff: ta.wpfloat = (
            2.0
            * math.pi
            * MicrophysicsConstants.DIFFUSION_COEFF_FOR_WATER_VAPOR
            / MicrophysicsConstants.HOWELL_FACTOR
            * _n0r
            * _ar ** (-precomputed_evaporation_alpha_exp_coeff)
            * math.gamma(self.config.rain_mu + 2.0)
        )
        precomputed_evaporation_beta_exp_coeff: ta.wpfloat = (2.0 * self.config.rain_mu + 5.5) / (
            2.0 * self.config.rain_mu + 8.0
        ) - precomputed_evaporation_alpha_exp_coeff
        precomputed_evaporation_beta_coeff: ta.wpfloat = (
            0.26
            * math.sqrt(
                MicrophysicsConstants.REF_AIR_DENSITY
                * 130.0
                / MicrophysicsConstants.AIR_KINEMATIC_VISCOSITY
            )
            * _ar ** (-precomputed_evaporation_beta_exp_coeff)
            * math.gamma((2.0 * self.config.rain_mu + 5.5) / 2.0)
            / math.gamma(self.config.rain_mu + 2.0)
        )

        # Precomputations for optimization
        power_law_exponent_for_rain_mean_fall_speed_ln1o2: ta.wpfloat = math.exp(
            power_law_exponent_for_rain_mean_fall_speed * math.log(0.5)
        )
        power_law_exponent_for_ice_mean_fall_speed_ln1o2: ta.wpfloat = math.exp(
            MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_ICE_MEAN_FALL_SPEED * math.log(0.5)
        )
        power_law_exponent_for_graupel_mean_fall_speed_ln1o2: ta.wpfloat = math.exp(
            MicrophysicsConstants.POWER_LAW_EXPONENT_FOR_GRAUPEL_MEAN_FALL_SPEED * math.log(0.5)
        )

        self._ice_collision_precomputed_coef = (
            precomputed_riming_coef,
            precomputed_agg_coef,
            precomputed_snow_sed_coef,
        )
        self._rain_precomputed_coef = (
            power_law_exponent_for_rain_mean_fall_speed,
            power_law_coeff_for_rain_mean_fall_speed,
            precomputed_evaporation_alpha_exp_coeff,
            precomputed_evaporation_alpha_coeff,
            precomputed_evaporation_beta_exp_coeff,
            precomputed_evaporation_beta_coeff,
        )
        self._sed_dens_factor_coef = (
            power_law_exponent_for_rain_mean_fall_speed_ln1o2,
            power_law_exponent_for_ice_mean_fall_speed_ln1o2,
            power_law_exponent_for_graupel_mean_fall_speed_ln1o2,
        )

    def _initialize_local_fields(self):
        self.rhoqrv_old_kup = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        self.rhoqsv_old_kup = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        self.rhoqgv_old_kup = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        self.rhoqiv_old_kup = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        self.vnew_r = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        self.vnew_s = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        self.vnew_g = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        self.vnew_i = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        self.rain_precipitation_flux = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        self.snow_precipitation_flux = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        self.graupel_precipitation_flux = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        self.ice_precipitation_flux = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )
        self.total_precipitation_flux = data_alloc.zero_field(
            self._grid, dims.CellDim, dims.KDim, dtype=ta.wpfloat, allocator=self._backend
        )

    def _determine_horizontal_domains(self):
        cell_domain = h_grid.domain(dims.CellDim)
        self._start_cell_nudging = self._grid.start_index(cell_domain(h_grid.Zone.NUDGING))
        self._end_cell_local = self._grid.end_index(cell_domain(h_grid.Zone.LOCAL))

    def _initialize_gt4py_programs(self):
        self._icon_graupel = model_options.setup_program(
            backend=self._backend,
            program=graupel_stencils.icon_graupel,
            constant_args={
                "ground_level": gtx.int32(self._grid.num_levels - 1),
                "liquid_autoconversion_option": self.config.liquid_autoconversion_option,
                "snow_intercept_option": self.config.snow_intercept_option,
                "use_constant_latent_heat": self.config.use_constant_latent_heat,
                "ice_stickeff_min": self.config.ice_stickeff_min,
                "snow2graupel_riming_coeff": self.config.snow2graupel_riming_coeff,
                "power_law_coeff_for_ice_mean_fall_speed": self.config.power_law_coeff_for_ice_mean_fall_speed,
                "exponent_for_density_factor_in_ice_sedimentation": self.config.exponent_for_density_factor_in_ice_sedimentation,
                "power_law_coeff_for_snow_fall_speed": self.config.power_law_coeff_for_snow_fall_speed,
                "precomputed_riming_coef": self._ice_collision_precomputed_coef[0],
                "precomputed_agg_coef": self._ice_collision_precomputed_coef[1],
                "precomputed_snow_sed_coef": self._ice_collision_precomputed_coef[2],
                "power_law_exponent_for_rain_mean_fall_speed": self._rain_precomputed_coef[0],
                "power_law_coeff_for_rain_mean_fall_speed": self._rain_precomputed_coef[1],
                "precomputed_evaporation_alpha_exp_coeff": self._rain_precomputed_coef[2],
                "precomputed_evaporation_alpha_coeff": self._rain_precomputed_coef[3],
                "precomputed_evaporation_beta_exp_coeff": self._rain_precomputed_coef[4],
                "precomputed_evaporation_beta_coeff": self._rain_precomputed_coef[5],
                "power_law_exponent_for_rain_mean_fall_speed_ln1o2": self._sed_dens_factor_coef[0],
                "power_law_exponent_for_ice_mean_fall_speed_ln1o2": self._sed_dens_factor_coef[1],
                "power_law_exponent_for_graupel_mean_fall_speed_ln1o2": self._sed_dens_factor_coef[
                    2
                ],
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "vertical_start": self.vertical_params.kstart_moist,
                "vertical_end": self._grid.num_levels,
            },
        )

        self._icon_graupel_flux_above_ground = model_options.setup_program(
            backend=self._backend,
            program=graupel_stencils.icon_graupel_flux_above_ground,
            constant_args={
                "do_latent_heat_nudging": self.config.do_latent_heat_nudging,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "model_top": gtx.int32(0),
                "ground_level": gtx.int32(self._grid.num_levels - 1),
            },
        )

        self._icon_graupel_flux_at_ground = model_options.setup_program(
            backend=self._backend,
            program=graupel_stencils.icon_graupel_flux_at_ground,
            constant_args={
                "do_latent_heat_nudging": self.config.do_latent_heat_nudging,
            },
            horizontal_sizes={
                "horizontal_start": self._start_cell_nudging,
                "horizontal_end": self._end_cell_local,
            },
            vertical_sizes={
                "ground_level": gtx.int32(self._grid.num_levels - 1),
                "model_num_levels": self._grid.num_levels,
            },
        )

    def run(
        self,
        *,
        dtime: ta.wpfloat,
        rho: fa.CellKField[ta.wpfloat],
        temperature: fa.CellKField[ta.wpfloat],
        pressure: fa.CellKField[ta.wpfloat],
        qv: fa.CellKField[ta.wpfloat],
        qc: fa.CellKField[ta.wpfloat],
        qr: fa.CellKField[ta.wpfloat],
        qi: fa.CellKField[ta.wpfloat],
        qs: fa.CellKField[ta.wpfloat],
        qg: fa.CellKField[ta.wpfloat],
        qnc: fa.CellField[ta.wpfloat],
        temperature_tendency: fa.CellKField[ta.wpfloat],
        qv_tendency: fa.CellKField[ta.wpfloat],
        qc_tendency: fa.CellKField[ta.wpfloat],
        qr_tendency: fa.CellKField[ta.wpfloat],
        qi_tendency: fa.CellKField[ta.wpfloat],
        qs_tendency: fa.CellKField[ta.wpfloat],
        qg_tendency: fa.CellKField[ta.wpfloat],
    ):
        """
        Run the ICON single-moment graupel (nwp) microphysics. Precipitation flux is also computed.
        Args:
            dtime: microphysics time step [s]
            temperature: air temperature [K]
            pressure: air pressure [Pa]
            qv: specific humidity [kg/kg]
            qc: specific cloud content [kg/kg]
            qr: specific rain content [kg/kg]
            qi: specific ice content [kg/kg]
            qs: specific snow content [kg/kg]
            qg: specific graupel content [kg/kg]
            qnc: cloud number concentration [/m3]
        Returns:
            temperature_tendency: air temperature tendency [K/s]
            qv_tendency: specific humidity tendency [kg/kg/s]
            qc_tendency: specific cloud content tendency [kg/kg/s]
            qr_tendency: specific rain content tendency [kg/kg/s]
            qi_tendency: specific ice content tendency [kg/kg/s]
            qs_tendency: specific snow content tendency [kg/kg/s]
            qg_tendency: specific graupel content tendency [kg/kg/s]
        """
        self._icon_graupel(
            dtime=dtime,
            dz=self._metric_state.ddqz_z_full,
            temperature=temperature,
            pressure=pressure,
            rho=rho,
            qv=qv,
            qc=qc,
            qi=qi,
            qr=qr,
            qs=qs,
            qg=qg,
            qnc=qnc,
            temperature_tendency=temperature_tendency,
            qv_tendency=qv_tendency,
            qc_tendency=qc_tendency,
            qi_tendency=qi_tendency,
            qr_tendency=qr_tendency,
            qs_tendency=qs_tendency,
            qg_tendency=qg_tendency,
            rhoqrv_old_kup=self.rhoqrv_old_kup,
            rhoqsv_old_kup=self.rhoqsv_old_kup,
            rhoqgv_old_kup=self.rhoqgv_old_kup,
            rhoqiv_old_kup=self.rhoqiv_old_kup,
            vnew_r=self.vnew_r,
            vnew_s=self.vnew_s,
            vnew_g=self.vnew_g,
            vnew_i=self.vnew_i,
        )

        self._icon_graupel_flux_above_ground(
            dtime=dtime,
            rho=rho,
            qr=qr,
            qs=qs,
            qg=qg,
            qi=qi,
            qr_tendency=qr_tendency,
            qs_tendency=qs_tendency,
            qg_tendency=qg_tendency,
            qi_tendency=qi_tendency,
            rhoqrv_old_kup=self.rhoqrv_old_kup,
            rhoqsv_old_kup=self.rhoqsv_old_kup,
            rhoqgv_old_kup=self.rhoqgv_old_kup,
            rhoqiv_old_kup=self.rhoqiv_old_kup,
            vnew_r=self.vnew_r,
            vnew_s=self.vnew_s,
            vnew_g=self.vnew_g,
            vnew_i=self.vnew_i,
            rain_precipitation_flux=self.rain_precipitation_flux,
            snow_precipitation_flux=self.snow_precipitation_flux,
            graupel_precipitation_flux=self.graupel_precipitation_flux,
            ice_precipitation_flux=self.ice_precipitation_flux,
            total_precipitation_flux=self.total_precipitation_flux,
        )

        self._icon_graupel_flux_at_ground(
            dtime=dtime,
            rho=rho,
            qr=qr,
            qs=qs,
            qg=qg,
            qi=qi,
            qr_tendency=qr_tendency,
            qs_tendency=qs_tendency,
            qg_tendency=qg_tendency,
            qi_tendency=qi_tendency,
            rhoqrv_old_kup=self.rhoqrv_old_kup,
            rhoqsv_old_kup=self.rhoqsv_old_kup,
            rhoqgv_old_kup=self.rhoqgv_old_kup,
            rhoqiv_old_kup=self.rhoqiv_old_kup,
            vnew_r=self.vnew_r,
            vnew_s=self.vnew_s,
            vnew_g=self.vnew_g,
            vnew_i=self.vnew_i,
            rain_precipitation_flux=self.rain_precipitation_flux,
            snow_precipitation_flux=self.snow_precipitation_flux,
            graupel_precipitation_flux=self.graupel_precipitation_flux,
            ice_precipitation_flux=self.ice_precipitation_flux,
            total_precipitation_flux=self.total_precipitation_flux,
        )
