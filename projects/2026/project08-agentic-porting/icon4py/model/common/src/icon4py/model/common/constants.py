# ICON4Py - ICON inspired code in Python and GT4Py
#
# Copyright (c) 2022-2024, ETH Zurich and MeteoSwiss
# All rights reserved.
#
# Please, refer to the LICENSE file in the root directory.
# SPDX-License-Identifier: BSD-3-Clause

import enum
import sys
from typing import Final

from icon4py.model.common import type_alias as ta


#: Gas constant for dry air [J/K/kg], called 'rd' in ICON (mo_physical_constants.f90),
#: see https://glossary.ametsoc.org/wiki/Gas_constant.
GAS_CONSTANT_DRY_AIR: Final[ta.wpfloat] = 287.04
RD: Final[ta.wpfloat] = GAS_CONSTANT_DRY_AIR

#: Specific heat capacity of dry air at constant pressure [J/K/kg]
SPECIFIC_HEAT_CAPACITY_PRESSURE_DRY_AIR: Final[ta.wpfloat] = 1004.64
CPD = SPECIFIC_HEAT_CAPACITY_PRESSURE_DRY_AIR

#: [J/K/kg] specific heat capacity at constant volume
SPECIFIC_HEAT_CAPACITY_VOLUME_DRY_AIR: Final[ta.wpfloat] = CPD - RD
CVD: Final[ta.wpfloat] = SPECIFIC_HEAT_CAPACITY_VOLUME_DRY_AIR
CVD_O_RD: Final[ta.wpfloat] = CVD / RD
RD_O_CPD: Final[ta.wpfloat] = RD / CPD
CPD_O_RD: Final[ta.wpfloat] = CPD / RD
RD_O_CVD: Final[ta.wpfloat] = RD / CVD

#: Gas constant for water vapor [J/K/kg], rv in ICON.
GAS_CONSTANT_WATER_VAPOR: Final[ta.wpfloat] = 461.51
RV: Final[ta.wpfloat] = GAS_CONSTANT_WATER_VAPOR

#: Specific heat capacity of water vapor at constant pressure [J/K/kg]
SPECIFIC_HEAT_CAPACITY_PRESSURE_WATER_VAPOR: Final[ta.wpfloat] = 1869.46
CPV = SPECIFIC_HEAT_CAPACITY_PRESSURE_WATER_VAPOR

#: Specific heat capacity of water vapor at constant volume [J/K/kg]
SPECIFIC_HEAT_CAPACITY_VOLUME_WATER_VAPOR: Final[ta.wpfloat] = CPV - RV
CVV = SPECIFIC_HEAT_CAPACITY_VOLUME_WATER_VAPOR

#: cp_dry_air / cp_liquid_water - 1
_RCPL: Final[ta.wpfloat] = 3.1733

#: Specific heat capacity of liquid water [J/K/kg]. Originally expressed as clw in ICON.
SPECIFIC_HEAT_CAPACITY_LIQUID_WATER: Final[ta.wpfloat] = (_RCPL + 1.0) * CPD
CPL = SPECIFIC_HEAT_CAPACITY_LIQUID_WATER

#: density of liquid water. Originally expressed as rhow in ICON. [kg/m3]
WATER_DENSITY: Final[ta.wpfloat] = 1.000e3

#: specific heat capacity of ice. Originally expressed as ci in ICON. [J/K/kg]
SPECIFIC_HEAT_CAPACITY_ICE: Final[ta.wpfloat] = 2108.0

#: Melting temperature of ice/snow [K]. Originally expressed as tmelt in ICON.
MELTING_TEMPERATURE: Final[ta.wpfloat] = 273.15

#: Latent heat of vaporisation for water [J/kg]. Originally expressed as alv in ICON.
LATENT_HEAT_FOR_VAPORISATION: Final[ta.wpfloat] = 2.5008e6

#: Latent heat of sublimation for water [J/kg]. Originally expressed as als in ICON.
LATENT_HEAT_FOR_SUBLIMATION: Final[ta.wpfloat] = 2.8345e6

#: Latent heat of fusion for water [J/kg]. Originally expressed as alf in ICON.
LATENT_HEAT_FOR_FUSION: Final[ta.wpfloat] = (
    LATENT_HEAT_FOR_SUBLIMATION - LATENT_HEAT_FOR_VAPORISATION
)

#: Triple point of water at 611hPa [K]
WATER_TRIPLE_POINT_TEMPERATURE: Final[ta.wpfloat] = 273.16

# Tetens formula constants for the saturation vapour pressure, called c1es, c3les,
# c4les, c3ies and c4ies in ICON (mo_lookup_tables_constants.f90).
# e_sat = TETENS_P0 * exp(A * (T - tmelt) / (T - B)), with the *_WATER coefficients
# over liquid water and the *_ICE coefficients over ice.
TETENS_P0: Final[ta.wpfloat] = 610.78
TETENS_A_WATER: Final[ta.wpfloat] = 17.269
TETENS_B_WATER: Final[ta.wpfloat] = 35.86
TETENS_A_ICE: Final[ta.wpfloat] = 21.875
TETENS_B_ICE: Final[ta.wpfloat] = 7.66

# Minimum temperature for saturation-over-ice calculations [K]. Used to clamp T
# in the Tetens ice branch (mo_thdyn_functions.f90).
MINIMUM_TEMPERATURE_ICE_SATURATION: Final[ta.wpfloat] = 180.0

# Reference pressure for the APE/JW relative-humidity profile [Pa].
RELATIVE_HUMIDITY_REFERENCE_PRESSURE: Final[ta.wpfloat] = 200000.0

# Pressure threshold below which the stratospheric specific-humidity cap applies [Pa].
STRATOSPHERE_PRESSURE_THRESHOLD: Final[ta.wpfloat] = 10000.0

# Stratospheric specific-humidity cap [kg/kg].
STRATOSPHERIC_QV_CAP: Final[ta.wpfloat] = 5.0e-6

#: RV/RD - 1, tvmpc1 in ICON.
RV_O_RD_MINUS_1: Final[ta.wpfloat] = GAS_CONSTANT_WATER_VAPOR / GAS_CONSTANT_DRY_AIR - 1.0
TVMPC1: Final[ta.wpfloat] = RV_O_RD_MINUS_1

#: Av. gravitational acceleration [m/s^2]
GRAVITATIONAL_ACCELERATION: Final[ta.wpfloat] = 9.80665
GRAV: Final[ta.wpfloat] = GRAVITATIONAL_ACCELERATION
GRAV_O_RD: Final[ta.wpfloat] = GRAV / RD
GRAV_O_CPD: Final[ta.wpfloat] = GRAV / CPD

#: reference pressure for Exner function [Pa]
REFERENCE_PRESSURE: Final[ta.wpfloat] = 100000.0
P0REF: Final[ta.wpfloat] = REFERENCE_PRESSURE
RD_O_P0REF: Final[ta.wpfloat] = RD / P0REF

#: sea level pressure [Pa]
SEA_LEVEL_PRESSURE: Final[ta.wpfloat] = 101325.0
P0SL_BG: Final[ta.wpfloat] = SEA_LEVEL_PRESSURE

#: average earth radius in [m]
EARTH_RADIUS: Final[ta.wpfloat] = 6.371229e6

#: Earth angular velocity [rad/s]
EARTH_ANGULAR_VELOCITY: Final[ta.wpfloat] = 7.29212e-5

#: sea level temperature for reference atmosphere [K]
SEA_LEVEL_TEMPERATURE: Final[ta.wpfloat] = 288.15
T0SL_BG: Final[ta.wpfloat] = SEA_LEVEL_TEMPERATURE

#: difference between sea level temperature and asymptotic stratospheric temperature
DELTA_TEMPERATURE: Final[ta.wpfloat] = 75.0
DEL_T_BG: Final[ta.wpfloat] = DELTA_TEMPERATURE

#: height scale for reference atmosphere [m], defined  in mo_vertical_grid
#: scale height [m]
HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE = 10000.0
_H_SCAL_BG: Final[ta.wpfloat] = HEIGHT_SCALE_FOR_REFERENCE_ATMOSPHERE

# Math constants
DBL_EPS = sys.float_info.epsilon  # EPSILON(1._wp)

# Implementation constants
#: default dynamics to physics time step ratio
DEFAULT_DYNAMICS_TO_PHYSICS_TIMESTEP_RATIO: Final[float] = 5.0


class PhysicsConstants(ta.wpfloat, enum.Enum):
    """
    Constants used in gt4py stencils.
    """

    rd = GAS_CONSTANT_DRY_AIR
    rv = GAS_CONSTANT_WATER_VAPOR
    rv_o_rd_minus_1 = RV_O_RD_MINUS_1
    cvd = SPECIFIC_HEAT_CAPACITY_VOLUME_DRY_AIR
    cpd = SPECIFIC_HEAT_CAPACITY_PRESSURE_DRY_AIR
    cpv = SPECIFIC_HEAT_CAPACITY_PRESSURE_WATER_VAPOR
    cvv = SPECIFIC_HEAT_CAPACITY_VOLUME_WATER_VAPOR
    cpl = SPECIFIC_HEAT_CAPACITY_LIQUID_WATER
    cpi = SPECIFIC_HEAT_CAPACITY_ICE
    water_density = WATER_DENSITY
    tmelt = MELTING_TEMPERATURE
    water_triple_point_temperature = WATER_TRIPLE_POINT_TEMPERATURE
    lh_vaporise = LATENT_HEAT_FOR_VAPORISATION
    lh_sublimate = LATENT_HEAT_FOR_SUBLIMATION
    lh_fusion = LATENT_HEAT_FOR_FUSION
    rd_o_cpd = RD_O_CPD
    rd_o_cvd = RD_O_CVD
    cpd_o_rd = CPD_O_RD
    cvd_o_rd = CVD_O_RD
    rd_o_p0ref = RD_O_P0REF
    grav_o_cpd = GRAV_O_CPD
    grav_o_rd = GRAV_O_RD
    p0ref = REFERENCE_PRESSURE
    eps = DBL_EPS


class RayleighType(int, enum.Enum):
    #: classical Rayleigh damping, which makes use of a reference state.
    CLASSIC = 1
    #: Klemp (2008) type Rayleigh damping
    KLEMP = 2
