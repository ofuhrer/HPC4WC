"""Saturation vapor pressure and vapor diffusivity, shared by several processes.

Tetens formula (mo_satad.f90::sat_pres_water/sat_pres_ice, ipsat=1, using the
Tetens constants from mo_lookup_tables_constants.f90) and
mo_2mom_mcrph_processes.f90::diffusivity (line 475-483).
"""

import enum

import gt4py.next as gtx
from gt4py.next import exp, log

from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common import type_alias as ta


class _SaturationConst(ta.wpfloat, enum.Enum):
    """See unit_conversion.py's ``_UnitConversionConst`` docstring for why these must be
    enum members rather than plain module floats."""

    T_MELT = 273.15
    TETENS_P0 = 610.78  # c1es
    TETENS_AW = 17.269  # c3les
    TETENS_BW = 35.86  # c4les
    TETENS_AI = 21.875  # c3ies
    TETENS_BI = 7.66  # c4ies


@gtx.field_operator
def e_ws(temperature: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    """Saturation vapor pressure over liquid water. mo_satad.f90:458-472."""
    return _SaturationConst.TETENS_P0 * exp(
        _SaturationConst.TETENS_AW * (temperature - _SaturationConst.T_MELT) / (temperature - _SaturationConst.TETENS_BW)
    )


@gtx.field_operator
def e_es(temperature: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    """Saturation vapor pressure over ice. mo_satad.f90:476-490."""
    return _SaturationConst.TETENS_P0 * exp(
        _SaturationConst.TETENS_AI * (temperature - _SaturationConst.T_MELT) / (temperature - _SaturationConst.TETENS_BI)
    )


@gtx.field_operator
def diffusivity(
    temperature: fa.CellKField[ta.wpfloat], pressure: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    """Molecular diffusivity of water vapor in air. mo_2mom_mcrph_processes.f90:475-483."""
    return 8.7602e-5 * exp(1.81 * log(temperature)) / pressure
