"""GT4Py port of ICON's saturation-adjustment kernel (``mo_satad.f90``).

This is an independent re-implementation of the ``satad_v_3D`` / ``satad_v_3D_gpu``
subroutines from ``satad_only/fortran/mo_satad.f90`` in the GT4Py declarative
(``gt4py.next``) DSL.

Physics
-------
Saturation adjustment relaxes each grid point to liquid/vapour equilibrium **at
constant total density**, moving water between vapour (``qv``) and cloud water
(``qc``) and adjusting temperature (``T``) by the associated latent heat. Every
grid point is treated independently -- there is no vertical coupling -- so the
whole kernel is elementwise (no ``scan_operator`` is needed).

Per grid point the Fortran does:

1. ``qw = qv + qc``  (total adjustable water, conserved).
2. ``lwdocvd = L_v(T) / cvd``  (computed **once** from the input T).
3. ``Ttest = T - lwdocvd*qc``;  ``qtest = qsat_rho(Ttest, rho)``.
4. If ``qw <= qtest``  (branch A): all cloud evaporates and the air is still
   sub-saturated -> ``qv = qw``, ``qc = 0``, ``T = Ttest``  (no iteration).
5. Else  (branch B): Newton-iterate ``twork`` from ``T`` until
   ``|dtwork| <= tol`` or ``maxiter`` steps, then
   ``T = twork``, ``qwa = qsat_rho(T, rho)``,
   ``qc = max(qc + qv - qwa, zqwmin)``, ``qv = qwa``.

Only the active Tetens saturation formula (``ipsat == 1``) is ported; the
Murphy-Koop branch (``ipsat == 2``) is dead code in the source and is not
reproduced here.

DSL notes
---------
* GT4Py field operators allow no ``if``/``while``/``for`` on fields, so:
    - the two branches become ``where(...)`` selections, and
    - the Newton loop is **manually unrolled** ``MAXITER`` (= 10) times, each
      step masked so that already-converged / non-iterating points freeze.
  This reproduces the ``satad_v_3D_gpu`` structure exactly and is numerically
  identical to the ``satad_v_3D`` ``while``-loop.
* All arithmetic is float64, matching the Fortran working precision (``wp``).

Interfaces
----------
* ``satad``            -- the ``@program`` operating on ``(Cell, K)`` fields.
* ``satad_numpy``      -- a thin wrapper: numpy arrays in, numpy arrays out
                          (handles a 1-D column or a 2-D ``(ncells, nlev)`` block).

See ``README.md`` in this directory for how to run and test.
"""

from __future__ import annotations

import numpy as np

import gt4py.next as gtx
from gt4py.next import Dims, Field, float64, where, exp, maximum


# ---------------------------------------------------------------------------
# Dimensions
# ---------------------------------------------------------------------------
CellDim = gtx.Dimension("Cell")
KDim = gtx.Dimension("K")

wpfloat = float64  # ICON working precision (real64)


# ---------------------------------------------------------------------------
# Physical constants (transcribed verbatim from the ICON Fortran modules;
# values in mo_physical_constants.f90 / mo_lookup_tables_constants.f90).
# ---------------------------------------------------------------------------
RD = 287.04          # [J/K/kg] gas constant, dry air              (mo_physical_constants: rd)
RV = 461.51          # [J/K/kg] gas constant, water vapour         (rv)
CPD = 1004.64        # [J/K/kg] c_p, dry air                        (cpd)
CVD = CPD - RD       # [J/K/kg] c_v, dry air  (= 717.60)            (cvd = cpd - rd)
TMELT = 273.15       # [K]      melting temperature                 (tmelt; = b3)
ALV = 2.5008e6       # [J/kg]   latent heat of vaporisation         (alv; = lwd)

# NOTE: satad uses a LOCAL parameter cp_v = 1850.0 for the latent-heat formula,
# which is deliberately *not* the physical-constants cpv = 1869.46.
CP_V = 1850.0        # [J/K/kg] c_p water vapour (satad-local)      (mo_satad: cp_v)
RCPL = 3.1733        # cp_d/cp_l - 1                                (mo_physical_constants: rcpl)
CLW = (RCPL + 1.0) * CPD  # [J/K/kg] specific heat of liquid water (clw; = cl)

# Tetens saturation-vapour-pressure constants (ipsat == 1).
C1ES = 610.78                    # (mo_lookup_tables_constants: c1es;  = b1)
C3LES = 17.269                   # (c3les; = b2w)
C4LES = 35.86                    # (c4les; = b4w)
C5LES = C3LES * (TMELT - C4LES)  # (c5les; = b234w)

ZQWMIN = 1.0e-20     # minimum adjusted cloud water                (mo_satad: zqwmin)

# Iteration controls. maxiter is fixed at the value used by the reference
# driver (column_driver.f90). Because GT4Py forbids data-dependent loops, the
# Newton iteration below is unrolled exactly this many times.
MAXITER = 10
DEFAULT_TOL = 1.0e-3  # [K] default temperature tolerance (column_driver.f90)


# ---------------------------------------------------------------------------
# Thermodynamic helper field operators (Tetens / ipsat == 1 path only)
# ---------------------------------------------------------------------------
@gtx.field_operator
def latent_heat_vaporization(
    t: Field[Dims[CellDim, KDim], wpfloat],
) -> Field[Dims[CellDim, KDim], wpfloat]:
    """L_v(T) [J/kg] = alv + (cp_v - clw)*(T - tmelt) - rv*T  (internal energy form)."""
    return ALV + (CP_V - CLW) * (t - TMELT) - RV * t


@gtx.field_operator
def sat_pres_water(
    t: Field[Dims[CellDim, KDim], wpfloat],
) -> Field[Dims[CellDim, KDim], wpfloat]:
    """Saturation vapour pressure over water [Pa], Tetens formula."""
    return C1ES * exp(C3LES * (t - TMELT) / (t - C4LES))


@gtx.field_operator
def qsat_rho(
    t: Field[Dims[CellDim, KDim], wpfloat],
    rho: Field[Dims[CellDim, KDim], wpfloat],
) -> Field[Dims[CellDim, KDim], wpfloat]:
    """Specific humidity at water saturation at constant total density [kg/kg]."""
    return sat_pres_water(t) / (rho * RV * t)


@gtx.field_operator
def dqsatdT_rho(
    qs: Field[Dims[CellDim, KDim], wpfloat],
    t: Field[Dims[CellDim, KDim], wpfloat],
) -> Field[Dims[CellDim, KDim], wpfloat]:
    """d(qsat)/dT at constant total density [1/K]  (ipsat == 1)."""
    return (C5LES / (t - C4LES) ** 2 - 1.0 / t) * qs


@gtx.field_operator
def _newton_update(
    tw: Field[Dims[CellDim, KDim], wpfloat],
    te: Field[Dims[CellDim, KDim], wpfloat],
    qve: Field[Dims[CellDim, KDim], wpfloat],
    lwdocvd: Field[Dims[CellDim, KDim], wpfloat],
    rho: Field[Dims[CellDim, KDim], wpfloat],
) -> Field[Dims[CellDim, KDim], wpfloat]:
    """One raw (unmasked) Newton step for the working temperature ``tw``.

    Solves f(tw) = tw - te + lwdocvd*(qsat(tw) - qve) = 0 for the equilibrium
    temperature. ``te``, ``qve`` and ``lwdocvd`` are the *fixed* original-state
    quantities (they do not change across iterations).
    """
    qwd = qsat_rho(tw, rho)
    dqwd = dqsatdT_rho(qwd, tw)
    ft = tw - te + lwdocvd * (qwd - qve)
    dft = 1.0 + lwdocvd * dqwd
    return tw - ft / dft


@gtx.field_operator
def _satad(
    te: Field[Dims[CellDim, KDim], wpfloat],
    qve: Field[Dims[CellDim, KDim], wpfloat],
    qce: Field[Dims[CellDim, KDim], wpfloat],
    rhotot: Field[Dims[CellDim, KDim], wpfloat],
    tol: wpfloat,
) -> tuple[
    Field[Dims[CellDim, KDim], wpfloat],
    Field[Dims[CellDim, KDim], wpfloat],
    Field[Dims[CellDim, KDim], wpfloat],
]:
    """Saturation adjustment for a whole (Cell, K) field. Returns (te, qve, qce)."""
    # --- pre-checks (all from the *input* state) ---------------------------
    qw = qve + qce
    lwdocvd = latent_heat_vaporization(te) / CVD
    ttest = te - lwdocvd * qce
    qtest = qsat_rho(ttest, rhotot)
    needs_iter = qw > qtest  # True -> Newton branch (B); False -> direct branch (A)

    # --- Newton iteration, manually unrolled MAXITER (=10) times -----------
    # Start twork = te; the Fortran seeds tworkold = te + 10 so the first step
    # always fires where needs_iter. Each subsequent step is guarded by the
    # convergence test |twork - tworkold| > tol, written here in the equivalent
    # squared form (twork-tworkold)^2 > tol^2 (both sides non-negative) because
    # the GT4Py `abs` builtin is not accepted inside this comparison. Converged
    # points therefore freeze, exactly as in satad_v_3D_gpu.
    tw_prev = te
    tw = where(needs_iter, _newton_update(te, te, qve, lwdocvd, rhotot), te)  # step 1

    active = needs_iter & ((tw - tw_prev) * (tw - tw_prev) > tol * tol)  # step 2
    tw_prev = tw
    tw = where(active, _newton_update(tw, te, qve, lwdocvd, rhotot), tw)

    active = needs_iter & ((tw - tw_prev) * (tw - tw_prev) > tol * tol)  # step 3
    tw_prev = tw
    tw = where(active, _newton_update(tw, te, qve, lwdocvd, rhotot), tw)

    active = needs_iter & ((tw - tw_prev) * (tw - tw_prev) > tol * tol)  # step 4
    tw_prev = tw
    tw = where(active, _newton_update(tw, te, qve, lwdocvd, rhotot), tw)

    active = needs_iter & ((tw - tw_prev) * (tw - tw_prev) > tol * tol)  # step 5
    tw_prev = tw
    tw = where(active, _newton_update(tw, te, qve, lwdocvd, rhotot), tw)

    active = needs_iter & ((tw - tw_prev) * (tw - tw_prev) > tol * tol)  # step 6
    tw_prev = tw
    tw = where(active, _newton_update(tw, te, qve, lwdocvd, rhotot), tw)

    active = needs_iter & ((tw - tw_prev) * (tw - tw_prev) > tol * tol)  # step 7
    tw_prev = tw
    tw = where(active, _newton_update(tw, te, qve, lwdocvd, rhotot), tw)

    active = needs_iter & ((tw - tw_prev) * (tw - tw_prev) > tol * tol)  # step 8
    tw_prev = tw
    tw = where(active, _newton_update(tw, te, qve, lwdocvd, rhotot), tw)

    active = needs_iter & ((tw - tw_prev) * (tw - tw_prev) > tol * tol)  # step 9
    tw_prev = tw
    tw = where(active, _newton_update(tw, te, qve, lwdocvd, rhotot), tw)

    active = needs_iter & ((tw - tw_prev) * (tw - tw_prev) > tol * tol)  # step 10
    tw_prev = tw
    tw = where(active, _newton_update(tw, te, qve, lwdocvd, rhotot), tw)

    # --- closure of the Newton branch (B) ----------------------------------
    qwa = qsat_rho(tw, rhotot)
    te_b = tw
    qce_b = maximum(qce + qve - qwa, ZQWMIN)
    qve_b = qwa

    # --- select between branch B (needs_iter) and branch A (direct) --------
    te_out = where(needs_iter, te_b, ttest)
    qve_out = where(needs_iter, qve_b, qw)
    qce_out = where(needs_iter, qce_b, 0.0)
    return te_out, qve_out, qce_out


@gtx.program
def satad(
    te: Field[Dims[CellDim, KDim], wpfloat],
    qve: Field[Dims[CellDim, KDim], wpfloat],
    qce: Field[Dims[CellDim, KDim], wpfloat],
    rhotot: Field[Dims[CellDim, KDim], wpfloat],
    tol: wpfloat,
):
    """In-place saturation adjustment: writes results back into te, qve, qce.

    Parameters
    ----------
    te      : temperature            [K]      (in/out)
    qve     : specific vapour        [kg/kg]  (in/out)
    qce     : specific cloud water   [kg/kg]  (in/out)
    rhotot  : total density          [kg/m^3] (in)
    tol     : temperature tolerance  [K]      (scalar; e.g. 1.0e-3)

    ``maxiter`` is fixed at MAXITER (= 10), matching the reference driver.
    """
    _satad(te, qve, qce, rhotot, tol, out=(te, qve, qce))


# ---------------------------------------------------------------------------
# Convenience numpy wrapper (no GT4Py knowledge required to call)
# ---------------------------------------------------------------------------
def satad_numpy(
    rho: np.ndarray,
    tk: np.ndarray,
    qv: np.ndarray,
    qc: np.ndarray,
    tol: float = DEFAULT_TOL,
    backend=None,
):
    """Run saturation adjustment on plain numpy arrays.

    Accepts either 1-D arrays of shape ``(nlev,)`` (a single column) or 2-D
    arrays of shape ``(ncells, nlev)``. Returns adjusted ``(tk, qv, qc)`` as
    numpy arrays with the same shape as the input. ``rho`` is unchanged.

    ``backend=None`` uses GT4Py embedded (numpy) execution -- no compiler
    toolchain required. Pass e.g. ``gtx.gtfn_cpu`` for a compiled backend.
    """
    rho_a = np.asarray(rho, dtype=np.float64)
    tk_a = np.asarray(tk, dtype=np.float64)
    qv_a = np.asarray(qv, dtype=np.float64)
    qc_a = np.asarray(qc, dtype=np.float64)

    squeeze = False
    if rho_a.ndim == 1:
        rho_a = rho_a[np.newaxis, :]
        tk_a = tk_a[np.newaxis, :]
        qv_a = qv_a[np.newaxis, :]
        qc_a = qc_a[np.newaxis, :]
        squeeze = True

    ncells, nlev = tk_a.shape

    def mk(arr):
        return gtx.as_field([CellDim, KDim], arr.copy(), allocator=backend)

    te_f = mk(tk_a)
    qve_f = mk(qv_a)
    qce_f = mk(qc_a)
    rho_f = mk(rho_a)

    satad(te_f, qve_f, qce_f, rho_f, float(tol), offset_provider={})

    tk_out = te_f.asnumpy()
    qv_out = qve_f.asnumpy()
    qc_out = qce_f.asnumpy()

    if squeeze:
        tk_out, qv_out, qc_out = tk_out[0], qv_out[0], qc_out[0]
    return tk_out, qv_out, qc_out
