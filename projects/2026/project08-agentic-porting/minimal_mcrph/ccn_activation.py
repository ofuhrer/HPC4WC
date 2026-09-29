"""ccn_activation_sk_4d: Segal & Khain (2006) CCN activation via a 4D lookup table.

**Standing decision (see porting_plan.md and CLAUDE.md): this is plain NumPy,
not a GT4Py stencil.** The interpolation indexes a 4D table using indices
computed at runtime from field values -- a data-dependent gather that doesn't
fit GT4Py's elementwise/neighbor-offset stencil model. This module is a
one-time table build (mirroring `get_otab`+`equi_table`,
mo_2mom_mcrph_processes.f90:1812-2058) plus a per-level NumPy loop (mirroring
`ccn_activation_sk_4d`'s body, lines 1641-1809) that a driver calls once per
timestep as an ordinary Python function, handing off plain NumPy arrays
between the compiled GT4Py stencils on either side.

**Simplification, not an approximation:** the Fortran subroutine has a branch
for computing `Ncn` from a height-dependent profile (`z0_nccn`/`z1e_nccn`),
reachable only when the optional `n_cn` argument is *absent*. This scheme's
`clouds_twomoment` always calls with `n_cn` present (`lprogccn =
PRESENT(nccn) = True`, since `column_driver.f90` always supplies the CSV's
`nccn` column) -- so that whole branch is dead code here and isn't
implemented; `Ncn` is always read directly from the prognostic `n_cn` array.

**Known assumption about an out-of-bounds Fortran array read:** the gate uses
`atmo%w(k+1)` (the "lower cell face" vertical velocity) unclamped, unlike the
otherwise-identical `kp1_fl = MIN(k+1, SIZE(atmo%rho))` clamp used for the
gradient check just above it in the same condition. Since `atmo%w` is sized
`nlev` (confirmed in `mo_nwp_gscp_interface.f90`), `atmo%w(k+1)` for the last
level `k=nlev` reads one element past the end of the array -- undefined
behavior in Fortran, not a value this port can faithfully reproduce (there is
no well-defined "correct" value to match). Implemented as "no data past the
array end -> gate closed" (`wcb=0` at the last level) -- the same category of
decision as `ice_nucleation_het_inas`'s uninitialized `ndiag_mask` handling.
Inconsequential for validating against `example/fields.csv`/
`example/output_fields.csv`: `atmo%w` is exactly 0.0 at every real level in
that column, so this boundary case is never observably different from the
reference regardless of which assumption is made here.
"""

import dataclasses
from pathlib import Path

import numpy as np

_DATA_FILE = Path(__file__).parent / "data" / "ccn_otab.npz"

# equi_table's target equidistant-grid sizes (mo_2mom_mcrph_main.f90:1686,
# CALL equi_table(nr2,nlsigs,nncn,nwcb) inside ccn_activation_sk_4d's
# no-argument init call)
_NR2, _NLSIGS, _NNCN, _NWCB = 3, 5, 129, 11

_NUC_EPS = 1.0e-20


@dataclasses.dataclass(frozen=True)
class CcnTable:
    """The equidistant 4D lookup table ("tab" in the Fortran)."""

    x1: np.ndarray
    x2: np.ndarray
    x3: np.ndarray
    x4: np.ndarray
    odx1: float
    odx2: float
    odx3: float
    odx4: float
    ltable: np.ndarray  # shape (n1, n2, n3, n4)


def _bracket(value: float, axis: np.ndarray) -> int:
    """Lower-bracket index search matching equi_table's exact linear scan
    (mo_2mom_mcrph_processes.f90:1987-2025): first ii with axis[ii] <= value
    <= axis[ii+1], 1 (0-based: 0) if none found."""
    for ii in range(len(axis) - 1):
        if axis[ii] <= value <= axis[ii + 1]:
            return ii
    return 0


def build_ccn_table() -> CcnTable:
    """Port of equi_table: tetra-linear interpolation of the small
    non-equidistant `otab` (loaded from data/ccn_otab.npz, see
    scripts/extract_ccn_otab.py for its provenance) onto an equidistant
    3x5x129x11 grid. Computed once at driver start-up, matching Fortran's
    one-time `init_2mom_scheme_once` call."""
    data = np.load(_DATA_FILE)
    ox1, ox2, ox3, ox4 = data["x1"], data["x2"], data["x3"], data["x4"]
    ol = data["ltable"]

    dx1 = (ox1[-1] - ox1[0]) / (_NR2 - 1)
    dx2 = (ox2[-1] - ox2[0]) / (_NLSIGS - 1)
    dx3 = (ox3[-1] - ox3[0]) / (_NNCN - 1)
    dx4 = (ox4[-1] - ox4[0]) / (_NWCB - 1)

    x1 = ox1[0] + np.arange(_NR2) * dx1
    x2 = ox2[0] + np.arange(_NLSIGS) * dx2
    x3 = ox3[0] + np.arange(_NNCN) * dx3
    x4 = ox4[0] + np.arange(_NWCB) * dx4

    iuv = [_bracket(x1[i], ox1) for i in range(_NR2)]
    juv = [_bracket(x2[j], ox2) for j in range(_NLSIGS)]
    kuv = [_bracket(x3[k], ox3) for k in range(_NNCN)]
    luv = [_bracket(x4[l], ox4) for l in range(_NWCB)]

    ltable = np.zeros((_NR2, _NLSIGS, _NNCN, _NWCB))
    for l in range(_NWCB):
        lu = luv[l]
        odx4_o = 1.0 / (ox4[lu + 1] - ox4[lu])
        for k in range(_NNCN):
            ku = kuv[k]
            odx3_o = 1.0 / (ox3[ku + 1] - ox3[ku])
            for j in range(_NLSIGS):
                ju = juv[j]
                odx2_o = 1.0 / (ox2[ju + 1] - ox2[ju])
                for i in range(_NR2):
                    iu = iuv[i]
                    odx1_o = 1.0 / (ox1[iu + 1] - ox1[iu])
                    hilf1 = ol[iu : iu + 2, ju : ju + 2, ku : ku + 2, lu : lu + 2]
                    hilf2 = hilf1[0] + (hilf1[1] - hilf1[0]) * odx1_o * (x1[i] - ox1[iu])
                    hilf3 = hilf2[0] + (hilf2[1] - hilf2[0]) * odx2_o * (x2[j] - ox2[ju])
                    hilf4 = hilf3[0] + (hilf3[1] - hilf3[0]) * odx3_o * (x3[k] - ox3[ku])
                    ltable[i, j, k, l] = hilf4[0] + (hilf4[1] - hilf4[0]) * odx4_o * (
                        x4[l] - ox4[lu]
                    )

    return CcnTable(
        x1=x1, x2=x2, x3=x3, x4=x4,
        odx1=1.0 / dx1, odx2=1.0 / dx2, odx3=1.0 / dx3, odx4=1.0 / dx4,
        ltable=ltable,
    )  # fmt: skip


def ccn_activation_sk_4d(
    *,
    rho: np.ndarray,
    w: np.ndarray,
    cloud_q: np.ndarray,
    cloud_n: np.ndarray,
    qv: np.ndarray,
    n_cn: np.ndarray,
    ccn_ncn0: float,
    ccn_wcb_min: float,
    ccn_r2: float,
    ccn_lsigs: float,
    ccn_etas: float,
    cloud_x_min: float,
    table: CcnTable,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """One column, one timestep. All arguments/returns are density-space
    (mo_2mom_mcrph_processes.f90:1641-1809, n_cn-present branch only -- see
    module docstring)."""
    nlev = rho.shape[0]
    new_cloud_q = cloud_q.copy()
    new_cloud_n = np.minimum(cloud_n, ccn_ncn0)  # hard upper cap, every level, line 1719
    new_qv = qv.copy()
    new_n_cn = n_cn.copy()

    for k in range(nlev):
        kp1 = min(k + 1, nlev - 1)
        q_c = new_cloud_q[k]
        gradient_ok = (k == kp1) or (q_c / rho[k] > new_cloud_q[kp1] / rho[kp1])

        if k + 1 < nlev:
            w_lower = w[k + 1]
        else:
            w_lower = None  # see module docstring: no data past array end

        if q_c > _NUC_EPS and gradient_ok and w_lower is not None and w_lower > 0.0:
            wcb = w_lower
        else:
            wcb = 0.0

        if wcb <= 0.0:
            continue

        n_c = new_cloud_n[k]
        wcb = max(wcb, ccn_wcb_min)
        ncn = new_n_cn[k]

        r2_loc = min(max(ccn_r2, table.x1[0]), table.x1[-1])
        iu = min(int(np.floor((r2_loc - table.x1[0]) * table.odx1)), _NR2 - 2)
        lsigs_loc = min(max(ccn_lsigs, table.x2[0]), table.x2[-1])
        ju = min(int(np.floor((lsigs_loc - table.x2[0]) * table.odx2)), _NLSIGS - 2)
        ncn_loc = min(max(ncn, table.x3[0]), table.x3[-1])
        ku = min(int(np.floor((ncn_loc - table.x3[0]) * table.odx3)), _NNCN - 2)
        wcb_loc = min(max(wcb, table.x4[0]), table.x4[-1])
        lu = min(int(np.floor((wcb_loc - table.x4[0]) * table.odx4)), _NWCB - 2)

        hilf1 = table.ltable[iu : iu + 2, ju : ju + 2, ku : ku + 2, lu : lu + 2]
        hilf2 = hilf1[0] + (hilf1[1] - hilf1[0]) * table.odx1 * (r2_loc - table.x1[iu])
        hilf3 = hilf2[0] + (hilf2[1] - hilf2[0]) * table.odx2 * (lsigs_loc - table.x2[ju])
        hilf4 = hilf3[0] + (hilf3[1] - hilf3[0]) * table.odx3 * (ncn_loc - table.x3[ku])
        nccn = hilf4[0] + (hilf4[1] - hilf4[0]) * table.odx4 * (wcb_loc - table.x4[lu])
        nccn = min(nccn, ccn_ncn0)

        nuc_n = max(ccn_etas * nccn - n_c, 0.0)
        nuc_q = min(nuc_n * cloud_x_min, new_qv[k])
        nuc_n = nuc_q / cloud_x_min

        new_cloud_n[k] += nuc_n
        new_cloud_q[k] += nuc_q
        new_qv[k] -= nuc_q
        new_n_cn[k] -= min(ncn, nuc_n)

    return new_cloud_q, new_cloud_n, new_qv, new_n_cn
