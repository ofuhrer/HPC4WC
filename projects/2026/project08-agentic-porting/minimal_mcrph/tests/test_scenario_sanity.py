"""Check that each scenario column is the physical regime its name claims.

A consistency test compares the port against the Fortran whatever the input is, so it
passes just as happily on a column that exercises nothing. These checks are what make the
scenario names load-bearing: if ``deep_cold`` drifted subsaturated over ice, or ``warm``
crept above water saturation, the branches those scenarios exist to cover would quietly
stop being covered and every consistency test would still be green.

The thermodynamics used here (:mod:`mcrph_common.thermo`, Murphy-Koop) is deliberately
*not* the scheme's own (Tetens). The two agree to ~1%, which is ample to establish
"ice-supersaturated by 35%" and useless as a numerical oracle -- which is the correct
division of labour: this file describes the inputs, the Fortran binary judges the outputs.

Saturation values here are *ratios*: 1.0 is exactly saturated.
"""

from __future__ import annotations

import numpy as np
import pytest

from mcrph_common import thermo
from minimal_mcrph import particles as p
from minimal_mcrph.tests.scenarios import SCENARIOS

_NAMES = sorted(SCENARIOS)

# (q, n, particle) triples whose mean mass q/n must land inside [x_min, x_max]. A column
# that violates this is not wrong so much as wasted: prepare_twomoment's size clip would
# rewrite n before any process saw it, so the scenario would silently test the clip
# instead of the physics it was written for.
_SPECIES = (
    ("qc", "qnc", p.CLOUD),
    ("qr", "qnr", p.RAIN),
    ("qi", "qni", p.ICE),
    ("qs", "qns", p.SNOW),
    ("qg", "qng", p.GRAUPEL),
    ("qh", "qnh", p.HAIL),
)


@pytest.mark.parametrize("name", _NAMES)
def test_column_is_physical(name):
    """Basic well-formedness that every scenario must satisfy."""
    fields, hhl, dt = SCENARIOS[name]
    nlev = len(fields["rho"])

    assert hhl.shape[0] == nlev + 1, "hhl must have nlev+1 half levels"
    assert np.all(np.diff(hhl) < 0.0), "hhl must decrease with index (ICON convention)"
    assert dt > 0.0

    for field, values in fields.items():
        assert np.all(np.isfinite(values)), f"{field} has non-finite entries"
    assert np.all(fields["rho"] > 0.0)
    assert np.all(fields["pres"] > 0.0)
    assert np.all(fields["tk"] > 150.0) and np.all(fields["tk"] < 350.0)
    assert np.all(fields["qv"] > 0.0)

    for q_name, n_name, particle in _SPECIES:
        q, n = fields[q_name], fields[n_name]
        present = q > 0.0
        if not present.any():
            continue
        assert np.all(n[present] > 0.0), f"{q_name} > 0 with {n_name} == 0"
        x = q[present] / n[present]
        assert np.all(x >= particle.x_min) and np.all(x <= particle.x_max), (
            f"{name}: mean mass {q_name}/{n_name} outside "
            f"[{particle.x_min:.3g}, {particle.x_max:.3g}] -- the size clip would "
            f"overwrite {n_name} before any process ran"
        )


def _ratios(fields):
    return (
        thermo.saturation_ratio_water(fields["qv"], fields["pres"], fields["tk"]),
        thermo.saturation_ratio_ice(fields["qv"], fields["pres"], fields["tk"]),
    )


# The lowest level of every constructed scenario is deliberately dried out by
# `scenarios._dry_bottom_level` to stay clear of the reference's out-of-bounds read (see
# `test_scenarios_avoid_the_out_of_bounds_ccn_read`). Regime checks therefore exclude it
# -- explicitly, so that the exception stays visible rather than being absorbed into a
# vaguer assertion that would also stop catching real drift.
_BODY = slice(None, -1)


def test_warm_is_warm_and_subsaturated():
    """`warm` must shut the ice gates and run satad's evaporation branch."""
    fields, _, _ = SCENARIOS["warm"]
    s_w, _ = _ratios(fields)
    assert np.all(fields["tk"] > 273.15), "a sub-freezing level would open the ice gates"
    assert np.all(s_w < 1.0), "subsaturation is the point: satad must evaporate, not condense"
    assert np.all(fields["qc"][_BODY] > 0.0), "needs cloud water for the evaporation branch to act on"
    assert np.all(fields["qi"] == 0.0) and np.all(fields["qs"] == 0.0)


def test_deep_cold_is_ice_supersaturated():
    """`deep_cold` must drive depositional growth above the dried bottom level."""
    fields, _, _ = SCENARIOS["deep_cold"]
    _, s_i = _ratios(fields)
    body = s_i[_BODY]
    assert np.all(fields["tk"] < 273.15), "deposition is gated on T < T_3"
    assert np.all(body > 1.05), (
        f"needs clear ice supersaturation, got Si in {body.min():.3f}..{body.max():.3f}"
    )
    assert np.all(fields["qi"] > 0.0), "vapor_deposition_generic is a no-op where q == 0"
    assert np.all(fields["w"] > 0.0), "homogeneous nucleation needs an updraft"


def test_updraft_opens_the_ccn_gate():
    """`updraft` must satisfy every term of the CCN activation gate somewhere.

    The gate needs cloud water present, a positive velocity at the *lower* cell face
    (``w(k+1)``), and cloud water decreasing downward in specific terms. Checking all
    three together is the point -- satisfying two of them activates nothing.
    """
    fields, _, _ = SCENARIOS["updraft"]
    qc, rho, w = fields["qc"], fields["rho"], fields["w"]
    nlev = len(rho)

    open_levels = [
        k
        for k in range(nlev - 1)
        if qc[k] > 1.0e-20 and w[k + 1] > 0.0 and qc[k] / rho[k] > qc[k + 1] / rho[k + 1]
    ]
    assert open_levels, (
        "no level satisfies the CCN activation gate -- this scenario exists precisely "
        "to reach the branch the bundled column (w == 0 everywhere) cannot"
    )


def test_mixed_species_seeds_all_four_depositing_species():
    """`mixed_species` must give snow, graupel and hail something to deposit onto."""
    fields, _, _ = SCENARIOS["mixed_species"]
    _, s_i = _ratios(fields)
    for q_name in ("qi", "qs", "qg", "qh"):
        active = (fields[q_name] > 0.0) & (fields["tk"] < 273.15) & (s_i > 1.0)
        assert active.any(), (
            f"{q_name} is never present at a sub-freezing, ice-supersaturated level, so "
            f"its branch of vapor_dep_relaxation stays inert"
        )


def test_scenarios_avoid_the_out_of_bounds_ccn_read():
    """No scenario may put cloud water on the lowest level.

    ``ccn_activation_sk_4d`` decides the gate on ``atmo%w(k+1)`` without clamping the
    index, while the gradient test beside it clamps with
    ``kp1_fl = MIN(k+1, SIZE(atmo%rho))`` (mo_2mom_mcrph_processes.f90:1725-1728). Since
    ``w`` is sized ``nlev``, the lowest level reads one element past the end of the array
    and the outcome is whatever sits in adjacent memory.

    That read is guarded behind ``cloud%q(k) > nuc_eps``, so a cloud-free bottom level
    never reaches it. Keeping every scenario on that side of the guard is what makes this
    suite reproducible: when three constructed columns did carry cloud water down to the
    surface, the Fortran activated CCN there off the out-of-bounds value and the port did
    not, a 100% disagreement in ``qnc`` with no correct answer to pick.

    This is a real defect in the reference worth reporting upstream. It is not something
    a port can or should reproduce, and not something to encode one accidental outcome of.
    """
    for name in _NAMES:
        fields, _, _ = SCENARIOS[name]
        assert fields["qc"][-1] == 0.0, (
            f"{name}: cloud water on the lowest level makes the result depend on an "
            f"out-of-bounds read of atmo%w in the reference -- see this test's docstring"
        )


def test_mixed_random_spans_both_deposition_directions():
    """`mixed_random` must reach sublimation, which no other scenario does."""
    fields, _, _ = SCENARIOS["mixed_random"]
    _, s_i = _ratios(fields)
    cold = fields["tk"] < 273.15
    assert (s_i[cold] > 1.0).any(), "expected some ice-supersaturated levels"
    assert (s_i[cold] < 1.0).any(), (
        "expected some ice-subsaturated levels -- this is the only scenario exercising "
        "the qvsidiff < 0 limiter and the reduce_sublimation number reduction"
    )
