"""cloud_freeze and ice_melting.

Mirrors mo_2mom_mcrph_processes.f90::cloud_freeze (lines 486-558) and
::ice_melting (lines 1515-1560). Both operate on densities (see
porting_plan.md's "Physical units" section) -- callers must run these after
the mixing-ratio -> density conversion, not on raw CSV columns.
"""

import enum
import math

import gt4py.next as gtx
from gt4py.next import exp, maximum, minimum, where

from icon4py.model.common import dimension as dims
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common import type_alias as ta

from minimal_mcrph.stencils.particle_helpers import particle_meanmass


class _ProcessesConst(ta.wpfloat, enum.Enum):
    """See unit_conversion.py's ``_ProcessesConst`` docstring: these must be enum
    members, not plain module floats, to compile under gtfn_cpu."""

    T_3 = 273.15  # tmelt
    RHO_W = 1000.0  # rhoh2o
    LOG_10 = math.log(10.0)  # Fortran's log_10 = LOG(10.0); EXP(x*LOG_10) == 10**x


# -- cloud_freeze (mo_2mom_mcrph_processes.f90:486-558) --
# Homogeneous freezing of cloud droplets. Only ever exercises the T_c<-30
# polynomial branch of j_hom in practice: the Fortran's inner
# "IF (T_c > -30.0)" check for the *other* j_hom formula is unreachable given
# the outer "T_c < -30.0" gate already holds at that point -- kept here anyway
# as a where() that will simply never select, for literal correspondence with
# the source in case the outer threshold ever changes.
# The nuc_c_typ==0 special-case branch (constant drop number) is NOT
# implemented -- this scheme's nuc_c_typ=8, so it never fires in the Fortran
# either.


@gtx.field_operator
def cloud_freeze(
    dt: ta.wpfloat,
    cloud_coeffs_c_z: ta.wpfloat,
    cloud_x_max: ta.wpfloat,
    cloud_x_min: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    cloud_q: fa.CellKField[ta.wpfloat],
    cloud_n: fa.CellKField[ta.wpfloat],
    ice_q: fa.CellKField[ta.wpfloat],
    ice_n: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    t_c = temperature - _ProcessesConst.T_3
    active = (temperature < _ProcessesConst.T_3) & (cloud_q > 0.0) & (t_c < -30.0)
    instant = active & (t_c < -50.0)
    gradual = active & (t_c >= -50.0)

    x_c = particle_meanmass(cloud_q, cloud_n, cloud_x_min, cloud_x_max)

    poly = (
        -243.4
        - 14.75 * t_c
        - 0.307 * t_c * t_c
        - 0.00287 * t_c * t_c * t_c
        - 0.0000102 * t_c * t_c * t_c * t_c
    )
    j_hom = 1.0e6 / _ProcessesConst.RHO_W * exp(poly * _ProcessesConst.LOG_10)

    fr_n_gradual = minimum(j_hom * cloud_q * dt, cloud_n)
    fr_q_gradual = minimum(j_hom * cloud_q * x_c * dt * cloud_coeffs_c_z, cloud_q)

    fr_q = where(instant, cloud_q, where(gradual, fr_q_gradual, 0.0))
    fr_n = where(instant, cloud_n, where(gradual, fr_n_gradual, 0.0))

    fr_n_to_ice = maximum(fr_n, fr_q / cloud_x_max)

    return (
        cloud_q - fr_q,
        cloud_n - fr_n,
        ice_q + fr_q,
        ice_n + fr_n_to_ice,
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def cloud_freeze_program(
    dt: ta.wpfloat,
    cloud_coeffs_c_z: ta.wpfloat,
    cloud_x_max: ta.wpfloat,
    cloud_x_min: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    cloud_q: fa.CellKField[ta.wpfloat],
    cloud_n: fa.CellKField[ta.wpfloat],
    ice_q: fa.CellKField[ta.wpfloat],
    ice_n: fa.CellKField[ta.wpfloat],
    out_cloud_q: fa.CellKField[ta.wpfloat],
    out_cloud_n: fa.CellKField[ta.wpfloat],
    out_ice_q: fa.CellKField[ta.wpfloat],
    out_ice_n: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    cloud_freeze(
        dt,
        cloud_coeffs_c_z,
        cloud_x_max,
        cloud_x_min,
        temperature,
        cloud_q,
        cloud_n,
        ice_q,
        ice_n,
        out=(out_cloud_q, out_cloud_n, out_ice_q, out_ice_n),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# -- ice_melting (mo_2mom_mcrph_processes.f90:1515-1560) --
# Complete melt within one step, routed to rain or cloud depending on the
# melting ice's mean mass vs. cloud.x_max.


@gtx.field_operator
def ice_melting(
    cloud_x_max: ta.wpfloat,
    ice_x_min: ta.wpfloat,
    ice_x_max: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    ice_q: fa.CellKField[ta.wpfloat],
    ice_n: fa.CellKField[ta.wpfloat],
    cloud_q: fa.CellKField[ta.wpfloat],
    cloud_n: fa.CellKField[ta.wpfloat],
    rain_q: fa.CellKField[ta.wpfloat],
    rain_n: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    melting = (temperature > _ProcessesConst.T_3) & (ice_q > 0.0)
    x_i = particle_meanmass(ice_q, ice_n, ice_x_min, ice_x_max)
    to_rain = melting & (x_i > cloud_x_max)
    to_cloud = melting & (x_i <= cloud_x_max)

    return (
        where(melting, 0.0, ice_q),
        where(melting, 0.0, ice_n),
        where(to_cloud, cloud_q + ice_q, cloud_q),
        where(to_cloud, cloud_n + ice_n, cloud_n),
        where(to_rain, rain_q + ice_q, rain_q),
        where(to_rain, rain_n + ice_n, rain_n),
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_melting_program(
    cloud_x_max: ta.wpfloat,
    ice_x_min: ta.wpfloat,
    ice_x_max: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    ice_q: fa.CellKField[ta.wpfloat],
    ice_n: fa.CellKField[ta.wpfloat],
    cloud_q: fa.CellKField[ta.wpfloat],
    cloud_n: fa.CellKField[ta.wpfloat],
    rain_q: fa.CellKField[ta.wpfloat],
    rain_n: fa.CellKField[ta.wpfloat],
    out_ice_q: fa.CellKField[ta.wpfloat],
    out_ice_n: fa.CellKField[ta.wpfloat],
    out_cloud_q: fa.CellKField[ta.wpfloat],
    out_cloud_n: fa.CellKField[ta.wpfloat],
    out_rain_q: fa.CellKField[ta.wpfloat],
    out_rain_n: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    ice_melting(
        cloud_x_max,
        ice_x_min,
        ice_x_max,
        temperature,
        ice_q,
        ice_n,
        cloud_q,
        cloud_n,
        rain_q,
        rain_n,
        out=(out_ice_q, out_ice_n, out_cloud_q, out_cloud_n, out_rain_q, out_rain_n),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
