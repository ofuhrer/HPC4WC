"""Driver-level housekeeping that runs *after* post_twomoment, in mixing-ratio
space (mo_2mom_mcrph_driver.f90:489-522). Not part of clouds_twomoment or any
of the 5 retained processes -- easy to miss for exactly that reason (an
earlier pass at this port did; see port_log.md, 2026-07-23 Phase 4 entry).

- Resets `nccn` up to a height-dependent background profile at cloud-free
  points (`qc <= q_crit`), then floors it at 35e6.
- Relaxes `ninact` toward zero where `qi == 0`.
- Relaxes `ninpot` toward a height-dependent background profile
  (unconditional relaxation, not gated on any q).
"""

import enum

import gt4py.next as gtx
from gt4py.next import exp, maximum, where

from icon4py.model.common import dimension as dims
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common import type_alias as ta


class _PostHousekeepingConst(ta.wpfloat, enum.Enum):
    """See unit_conversion.py's ``_UnitConversionConst`` docstring for why
    these must be enum members rather than plain module floats."""

    Q_CRIT = 1.0e-9  # mo_2mom_mcrph_processes.f90:235 (the active value)
    TAU_INACT = 600.0  # mo_2mom_mcrph_driver.f90:218
    TAU_INPOT = 1800.0  # mo_2mom_mcrph_driver.f90:219
    NCCN_FLOOR = 35.0e6  # mo_2mom_mcrph_driver.f90:519


@gtx.field_operator
def post_housekeeping(
    ccn_ncn0: ta.wpfloat,
    ccn_z0: ta.wpfloat,
    ccn_z1e: ta.wpfloat,
    in_n0: ta.wpfloat,
    in_z0: ta.wpfloat,
    in_z1e: ta.wpfloat,
    dt: ta.wpfloat,
    zf: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    nccn: fa.CellKField[ta.wpfloat],
    ninact: fa.CellKField[ta.wpfloat],
    ninpot: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    cloud_free = qc <= _PostHousekeepingConst.Q_CRIT
    nccn_background = where(
        zf > ccn_z0, ccn_ncn0 * exp((ccn_z0 - zf) / ccn_z1e), ccn_ncn0
    )
    nccn_reset = where(cloud_free, maximum(nccn, nccn_background), nccn)
    new_nccn = maximum(nccn_reset, _PostHousekeepingConst.NCCN_FLOOR)

    new_ninact = where(
        qi == 0.0,
        ninact - ninact * (1.0 / _PostHousekeepingConst.TAU_INACT) * dt,
        ninact,
    )

    in_background = where(zf > in_z0, in_n0 * exp((in_z0 - zf) / in_z1e), in_n0)
    new_ninpot = ninpot - (ninpot - in_background) * (1.0 / _PostHousekeepingConst.TAU_INPOT) * dt

    return new_nccn, new_ninact, new_ninpot


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def post_housekeeping_program(
    ccn_ncn0: ta.wpfloat,
    ccn_z0: ta.wpfloat,
    ccn_z1e: ta.wpfloat,
    in_n0: ta.wpfloat,
    in_z0: ta.wpfloat,
    in_z1e: ta.wpfloat,
    dt: ta.wpfloat,
    zf: fa.CellKField[ta.wpfloat],
    qc: fa.CellKField[ta.wpfloat],
    qi: fa.CellKField[ta.wpfloat],
    nccn: fa.CellKField[ta.wpfloat],
    ninact: fa.CellKField[ta.wpfloat],
    ninpot: fa.CellKField[ta.wpfloat],
    out_nccn: fa.CellKField[ta.wpfloat],
    out_ninact: fa.CellKField[ta.wpfloat],
    out_ninpot: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    post_housekeeping(
        ccn_ncn0, ccn_z0, ccn_z1e, in_n0, in_z0, in_z1e, dt,
        zf, qc, qi, nccn, ninact, ninpot,
        out=(out_nccn, out_ninact, out_ninpot),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )  # fmt: skip
