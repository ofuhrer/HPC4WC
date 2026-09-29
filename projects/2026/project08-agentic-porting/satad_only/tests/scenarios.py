"""Input columns for the satad consistency tests.

Each scenario is a named atmospheric column ``(rho, tk, qv, qc)``. Together they
exercise both branches of the saturation adjustment -- the direct evaporation
branch (A) and the Newton condensation loop (B) -- plus the near-zero ``qc``
regime that genuinely needs a small absolute tolerance.
"""

from __future__ import annotations

import numpy as np

from satad_only import EXAMPLE_DIR

FIELDS_CSV = EXAMPLE_DIR / "fields.csv"


def _bundled():
    """The canonical 10-level sample column shipped in example/fields.csv."""
    data = np.loadtxt(FIELDS_CSV, delimiter=",", skiprows=1)
    return data[:, 0], data[:, 1], data[:, 2], data[:, 3]


def _dry_unchanged():
    # warm, very dry, no cloud -> far sub-saturated, passes through unchanged
    rho = np.array([1.0, 0.9, 1.1])
    tk = np.array([300.0, 295.0, 305.0])
    qv = np.array([1.0e-4, 5.0e-5, 2.0e-4])
    qc = np.array([0.0, 0.0, 0.0])
    return rho, tk, qv, qc


def _evaporation_A():
    # cloud water in sub-saturated air -> branch A: cloud fully evaporates, cools
    rho = np.array([1.0, 1.05, 0.95])
    tk = np.array([290.0, 288.0, 292.0])
    qv = np.array([1.0e-3, 2.0e-3, 5.0e-4])
    qc = np.array([1.0e-3, 5.0e-4, 2.0e-3])
    return rho, tk, qv, qc


def _supersaturation_B():
    # cold, vapour-rich, some cloud -> branch B: Newton condensation, warms
    rho = np.array([1.0, 1.1, 0.9])
    tk = np.array([275.0, 270.0, 278.0])
    qv = np.array([1.0e-2, 8.0e-3, 1.2e-2])
    qc = np.array([1.0e-4, 2.0e-4, 0.0])
    return rho, tk, qv, qc


def _near_zero_qc():
    # Points sitting *just* above saturation (qv = qsat*(1+delta), tiny delta),
    # so only a sliver condenses and qc_out lands near zero -- the regime where
    # rtol alone is unusable and atol carries the comparison. The qv values below
    # were tuned (delta 1e-9..1e-5) and VERIFIED to yield the qc_out shown; the
    # last row is a dry branch-A point giving qc_out == 0.0 exactly.
    #   (rho,  tk,     qv)                    -> qc_out
    rho = np.array([1.00, 0.95, 1.10, 0.80, 1.00])
    tk = np.array([273.0, 280.0, 265.0, 255.0, 300.0])
    qv = np.array([
        4.79509392965094330e-03,   # qc_out ~ 2.3e-12
        8.07699673057062431e-03,   # qc_out ~ 3.0e-10
        2.45648009757710962e-03,   # qc_out ~ 1.5e-09
        1.55207816279949039e-03,   # qc_out ~ 1.1e-08
        1.00000000000000008e-05,   # qc_out = 0.0 (dry, branch A)
    ])
    qc = np.zeros(5)
    return rho, tk, qv, qc


def _mixed_random():
    # fixed-seed multi-level column spanning both branches, for breadth
    rng = np.random.default_rng(20260724)
    n = 40
    rho = rng.uniform(0.4, 1.3, n)
    tk = rng.uniform(230.0, 305.0, n)
    qv = rng.uniform(1.0e-5, 1.5e-2, n)
    qc = rng.uniform(0.0, 2.0e-3, n)
    return rho, tk, qv, qc


SCENARIOS = {
    "bundled": _bundled(),
    "dry_unchanged": _dry_unchanged(),
    "evaporation_A": _evaporation_A(),
    "supersaturation_B": _supersaturation_B(),
    "near_zero_qc": _near_zero_qc(),
    "mixed_random": _mixed_random(),
}
