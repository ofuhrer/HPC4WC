"""Mixing-ratio <-> density conversion and the latent-heat temperature update.

Mirrors ``mo_2mom_prepare.f90`` (``prepare_twomoment``/``post_twomoment``) and the
latent-heat block in ``mo_2mom_mcrph_driver.f90::two_moment_mcrph`` (lines
392-450). See porting_plan.md's "Physical units" and "Latent-heat temperature
update" sections for the derivation -- in particular, why this scheme uses
temperature-*dependent* latent heat (``lconstant_lh=.FALSE.`` for the default
config), not the simpler constant-latent-heat formula.

All process routines in ``mo_2mom_mcrph_processes.f90`` operate on densities
(``q*rho``, ``n*rho``), not the CSV's mixing ratios -- these stencils are the
explicit boundary between the two, not an implementation detail to fold into a
process stencil.
"""

import enum

import gt4py.next as gtx
from gt4py.next import exp, log, maximum

from icon4py.model.common import dimension as dims
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common import type_alias as ta

from minimal_mcrph import constants as const


class _UnitConversionConst(ta.wpfloat, enum.Enum):
    """Constants referenced from inside @gtx.field_operator/@gtx.program bodies.

    Plain module-level Python floats work fine there under the embedded
    backend, but are NOT resolved under a compiled backend (gtfn_cpu):
    compiling this module's first draft with plain floats failed with
    ``EveValueError: Symbols {...} not found`` for every constant referenced
    inside a decorated body, even ones just forwarded as a call argument to a
    nested field_operator. This enum pattern (matching icon4py's own
    ``MicrophysicsConstants(ta.wpfloat, enum.Enum)``) is the one that actually
    compiles under gtfn_cpu -- confirmed by testing, not assumed. See
    port_log.md, 2026-07-23.
    """

    # rho_vel/rho_vel_c/rho0: mo_2mom_mcrph_driver.f90:92-94
    RHO_VEL = 0.4
    RHO_VEL_C = 0.2
    RHO0 = 1.225

    # cv_d = cpd - rd (mo_physical_constants.f90:112); l_cv=True is always
    # passed from nwp_gscp_interface.f90, so z_heat_cap_r = 1/cv_d, never 1/cpd.
    CV_D = const.CPD - const.R_DRY_AIR

    T_MELT = const.T_MELT
    R_D_VAPOR = const.R_D_VAPOR
    L_SUBLIMATION = const.L_SUBLIMATION
    L_VAPORIZATION = const.L_VAPORIZATION

    # latent_heat_sublimation/melting constants (mo_satad.f90:44-53,84,87,612-669)
    CP_V = 1850.0  # specific heat of water vapor
    CI = 2108.0  # specific heat of ice
    CLW = (3.1733 + 1.0) * const.CPD  # specific heat of liquid water, rcpl=3.1733


# -- mixing ratio <-> density (mo_2mom_prepare.f90:47-65, 229-249) --


@gtx.field_operator
def _multiply(
    a: fa.CellKField[ta.wpfloat], b: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    return a * b


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def multiply_field(
    a: fa.CellKField[ta.wpfloat],
    b: fa.CellKField[ta.wpfloat],
    out: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _multiply(
        a,
        b,
        out=out,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


def convert_fields(
    fields, factor, *, horizontal_start, horizontal_end, vertical_start, vertical_end, backend=None
):
    """In-place elementwise multiply of each field in ``fields`` by ``factor``.

    Used for both directions: pass ``rho`` to convert mixing ratios/specific
    concentrations to densities (``prepare_twomoment``), or ``rho_r`` (=1/rho)
    to convert back (``post_twomoment``). Safe to write each field back into
    itself here -- purely pointwise, no neighbor or vertical-offset access
    (see porting_plan.md's "Temp-Field Reuse Pattern" note).

    Only pass the field list this scheme actually converts: qv, qc, qnc, qr,
    qnr, qi, qni, qs, qns, qg, qng, qh, qnh, ninact, nccn, ninpot. Do NOT
    include ninagi/ssat/qgl/qhl -- their Fortran IF-guards
    (luse_agi/lexpl_supersat/lprogmelt) are all False for this config, so they
    are not converted in the reference, and multiplying them here would
    introduce a spurious rho factor.
    """
    program = multiply_field.with_backend(backend) if backend is not None else multiply_field
    for field in fields:
        program(
            field,
            factor,
            field,
            horizontal_start,
            horizontal_end,
            vertical_start,
            vertical_end,
            offset_provider={},
        )


# -- density-correction factors for terminal fall velocity --
# (mo_2mom_mcrph_driver.f90:355-365; consumed by vapor_dep_relaxation via each
# hydrometeor's rho_v, see porting_plan.md)


@gtx.field_operator
def _rhocorr(rho: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    return exp(-_UnitConversionConst.RHO_VEL * log(maximum(rho, 1.0e-6) / _UnitConversionConst.RHO0))


@gtx.field_operator
def _rhocld(rho: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    return exp(-_UnitConversionConst.RHO_VEL_C * log(maximum(rho, 1.0e-6) / _UnitConversionConst.RHO0))


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def compute_density_corrections(
    rho: fa.CellKField[ta.wpfloat],
    rhocorr: fa.CellKField[ta.wpfloat],
    rhocld: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _rhocorr(
        rho,
        out=rhocorr,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
    _rhocld(
        rho,
        out=rhocld,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# -- negative-mixing-ratio clip (mo_2mom_mcrph_driver.f90:312-318) --


@gtx.field_operator
def _clip_negative(q: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    return maximum(q, 0.0)


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def clip_negative(
    q: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _clip_negative(
        q,
        out=q,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )


# -- latent-heat temperature update (mo_2mom_mcrph_driver.f90:392-450) --


@gtx.field_operator
def _latent_heat_sublimation(temperature: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    """mo_satad.f90:634-650."""
    return (
        _UnitConversionConst.L_SUBLIMATION
        + (_UnitConversionConst.CP_V - _UnitConversionConst.CI) * (temperature - _UnitConversionConst.T_MELT)
        - _UnitConversionConst.R_D_VAPOR * temperature
    )


@gtx.field_operator
def _latent_heat_melting(temperature: fa.CellKField[ta.wpfloat]) -> fa.CellKField[ta.wpfloat]:
    """mo_satad.f90:652-669."""
    return (
        _UnitConversionConst.L_VAPORIZATION
        - _UnitConversionConst.L_SUBLIMATION
        + (_UnitConversionConst.CI - _UnitConversionConst.CLW) * (temperature - _UnitConversionConst.T_MELT)
    )


@gtx.field_operator
def _update_temperature(
    temperature: fa.CellKField[ta.wpfloat],
    rho_r: fa.CellKField[ta.wpfloat],
    qv_old: fa.CellKField[ta.wpfloat],
    qv_new: fa.CellKField[ta.wpfloat],
    q_liq_old: fa.CellKField[ta.wpfloat],
    q_liq_new: fa.CellKField[ta.wpfloat],
) -> fa.CellKField[ta.wpfloat]:
    conv_ice = _latent_heat_sublimation(temperature) / _UnitConversionConst.CV_D
    conv_liq = _latent_heat_melting(temperature) / _UnitConversionConst.CV_D
    return temperature - conv_ice * rho_r * (qv_new - qv_old) + conv_liq * rho_r * (
        q_liq_new - q_liq_old
    )


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def update_temperature(
    temperature: fa.CellKField[ta.wpfloat],
    rho_r: fa.CellKField[ta.wpfloat],
    qv_old: fa.CellKField[ta.wpfloat],
    qv_new: fa.CellKField[ta.wpfloat],
    q_liq_old: fa.CellKField[ta.wpfloat],
    q_liq_new: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    _update_temperature(
        temperature,
        rho_r,
        qv_old,
        qv_new,
        q_liq_old,
        q_liq_new,
        out=temperature,
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
