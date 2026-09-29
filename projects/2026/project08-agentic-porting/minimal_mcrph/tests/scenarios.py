"""Input columns for the minimal_mcrph consistency tests.

Each scenario is a named ``(fields, hhl, dt)``: a full set of the 23 CSV columns the
Fortran driver reads, the nlev+1 half-level heights, and the timestep.

WHY MORE THAN ONE COLUMN
------------------------
The bundled ``example/fields.csv`` is a single profile with two properties that between
them switch off most of the scheme:

* ``w == 0`` at every level, so CCN activation's nucleation branch and KHL06 homogeneous
  freezing never fire -- only their "gate closed, no-op" paths were ever tested;
* ``qs = qns = qg = qng = qh = qnh == 0`` at every level, so three of the four species in
  ``vapor_dep_relaxation`` are inert.

A port could get all of that wrong and still pass. The scenarios below open those paths.

HOW THE COLUMNS ARE BUILT
-------------------------
Humidity is derived from a target saturation ratio via :mod:`mcrph_common.thermo` rather
than typed in as a number that looks about right, so a scenario means what its name says.
``tests/test_scenario_sanity.py`` asserts the regime independently; the ``S``/``Si``
values quoted in each function's comments were measured with it. Note that those are
saturation *ratios* -- 1.0 is exactly saturated, 0.98 is 2% subsaturated.

The thermodynamics used for construction (Murphy-Koop) is deliberately not the scheme's
own (Tetens); it describes the column, it never validates a result.
"""

from __future__ import annotations

import numpy as np

from mcrph_common import thermo
from mcrph_common.csv_io import read_columns
from minimal_mcrph import EXAMPLE_DIR
from minimal_mcrph.csv_io import COLUMNS, read_hhl_csv

_DT = 30.0  # column_driver.f90's default dt

# Note on the column set: `_blank` fills every name in `COLUMNS`, including ssat, ninagi
# and qrsflux, which the driver reads but this configuration never touches (they need
# lexpl_supersat / luse_agi / ldass_lhn, all False here). They stay zero throughout.


def _blank(nlev):
    return {name: np.zeros(nlev) for name in COLUMNS}


def _dry_bottom_level(fields):
    """Leave the lowest level cloud-free and clearly subsaturated.

    Not cosmetic -- it keeps the scenarios out of undefined behaviour in the reference.
    ``ccn_activation_sk_4d`` reads the vertical velocity at the lower cell face as
    ``atmo%w(k+1)``, with no clamp, while the gradient test on the very same line uses
    ``kp1_fl = MIN(k+1, SIZE(atmo%rho))`` (mo_2mom_mcrph_processes.f90:1725-1728). ``w``
    is sized ``nlev``, so at ``k = nlev`` that reads one element past the end of the
    array and the gate is decided by whatever happens to sit in the adjacent memory.

    The read is only reached when the lowest level has cloud water at that point in the
    pipeline (the ``cloud%q(k) > nuc_eps`` term is evaluated first), which is why the
    bundled column never trips it: its lowest level is cloud-free. Constructed columns
    with cloud water all the way down do trip it, and did -- the Fortran activated CCN at
    the bottom level of `warm`, `deep_cold` and `mixed_random` off that out-of-bounds
    value while the port, which treats "no data past the array end" as a closed gate,
    did not.

    There is no right answer to match here: the value is whatever the allocator left
    there, and could differ with another compiler, another array size, or another day.
    So the scenarios stay out of that regime rather than encoding one accidental
    outcome. The behaviour itself is documented and asserted in
    ``test_scenario_sanity.py::test_scenarios_avoid_the_out_of_bounds_ccn_read``.
    """
    fields["qc"][-1] = 0.0
    fields["qnc"][-1] = 0.0
    fields["qv"][-1] = thermo.qv_from_saturation_ratio(
        0.80, fields["pres"][-1], fields["tk"][-1]
    )
    return fields


def _hhl(nlev, top=12000.0, surface=0.0):
    """Half-level heights, model top first (ICON convention: height falls with index)."""
    return np.linspace(top, surface, nlev + 1)


def _bundled():
    """The shipped 13-level column: the reference case the port was developed against."""
    fields = read_columns(EXAMPLE_DIR / "fields.csv")
    hhl = read_hhl_csv(EXAMPLE_DIR / "hhl.csv")
    return fields, hhl, _DT


def _updraft():
    """Bundled column plus a real updraft, opening the two branches it cannot reach.

    CCN activation reads the vertical velocity at the *lower* cell face (``w(k+1)``) and
    additionally requires cloud water present and decreasing downward in specific terms
    (``qc/rho`` at k above ``qc/rho`` at k+1). The bundled column has cloud water at
    levels 0-3, so putting ``w > 0`` just below those satisfies the face-velocity term,
    and the existing qc profile already falls off below level 3.

    Measured: S = 0.885-1.059, Si = 1.014-1.742 over the sub-freezing levels.
    """
    fields, hhl, dt = _bundled()
    fields = {k: v.copy() for k, v in fields.items()}
    fields["w"][1:6] = np.array([0.8, 1.2, 2.0, 1.5, 0.5])
    return fields, hhl, dt


def _mixed_species():
    """Bundled column seeded with snow, graupel and hail.

    ``vapor_dep_relaxation`` handles four species with identical structure but different
    coefficients; with the bundled column only the ice branch does anything, so a
    coefficient mixed up between species -- exactly the kind of error the per-particle
    constants invite -- would go unseen. Number concentrations are chosen to give mean
    masses ``q/n`` inside each species' [x_min, x_max], so the size clips do not simply
    overwrite them.

    Measured: Si = 1.014-1.742 at the seeded levels, so all four species deposit rather
    than sublimate.
    """
    fields, hhl, dt = _bundled()
    fields = {k: v.copy() for k, v in fields.items()}
    cold = fields["tk"] < 273.15
    fields["qs"][cold] = 1.0e-5
    fields["qns"][cold] = 1.0e-5 / 1.0e-8  # x = 1e-8 kg, inside snow [1e-10, 2e-5]
    fields["qg"][cold] = 5.0e-5
    fields["qng"][cold] = 5.0e-5 / 1.0e-6  # x = 1e-6 kg, inside graupel [4.19e-9, 5.3e-4]
    fields["qh"][cold] = 2.0e-5
    fields["qnh"][cold] = 2.0e-5 / 1.0e-5  # x = 1e-5 kg, inside hail [2.6e-9, 5e-3]
    return fields, hhl, dt


def _warm():
    """Warm, ice-free, slightly subsaturated with some cloud water.

    Every level is above T_3, so the deposition and nucleation gates are shut and the
    column isolates saturation adjustment plus ice melting. Built at S = 0.97, i.e.
    genuinely subsaturated -- the point of the case is that satad's direct-evaporation
    branch runs, which it would not if the numbers drifted above saturation.

    Measured: S = 0.970 at every level except the lowest, which `_dry_bottom_level` sets
    to 0.800; T = 275-295 K.
    """
    nlev = 10
    f = _blank(nlev)
    f["tk"] = np.linspace(275.0, 295.0, nlev)
    f["pres"] = np.linspace(6.0e4, 1.0e5, nlev)
    f["rho"] = thermo.rho_air(f["tk"], f["pres"])
    f["qv"] = thermo.qv_from_saturation_ratio(0.97, f["pres"], f["tk"])
    f["qc"] = np.full(nlev, 3.0e-4)
    f["qnc"] = f["qc"] / 1.0e-10  # x = 1e-10 kg, inside cloud [4.2e-15, 2.6e-10]
    f["nccn"] = np.full(nlev, 1.5e9)
    f["ninpot"] = np.full(nlev, 1.0e5)
    return _dry_bottom_level(f), _hhl(nlev, top=4000.0), _DT


def _deep_cold():
    """Cold and strongly ice-supersaturated, with ice already present and an updraft.

    This is the depositional-growth case: Si well above 1 means the relaxation runs in
    its growth direction at every level, and the updraft additionally opens the
    homogeneous-freezing branch. Humidity is set from a target ratio over *ice*, since
    water saturation says nothing useful about whether ice will grow.

    Measured: Si = 1.350 at every level by construction except the lowest (0.909, which
    `_dry_bottom_level` dries out); S = 0.800-1.143, T = 225-260 K. S exceeding 1 at the
    warmer levels is intended, not a slip: at 260 K the air can be supersaturated with
    respect to supercooled water while still being far colder than T_3, so satad
    condenses cloud water there and cloud_freeze then has something to act on --
    deposition, homogeneous freezing and droplet freezing all run in one column.
    """
    nlev = 10
    f = _blank(nlev)
    f["tk"] = np.linspace(225.0, 260.0, nlev)
    f["pres"] = np.linspace(2.0e4, 5.0e4, nlev)
    f["rho"] = thermo.rho_air(f["tk"], f["pres"])
    f["qv"] = thermo.qv_from_saturation_ratio_ice(1.35, f["pres"], f["tk"])
    f["w"] = np.full(nlev, 1.5)
    f["qi"] = np.full(nlev, 2.0e-6)
    f["qni"] = f["qi"] / 1.0e-11  # x = 1e-11 kg, inside ice [1e-12, 1e-5]
    f["nccn"] = np.full(nlev, 1.5e9)
    f["ninpot"] = np.full(nlev, 1.0e5)
    return _dry_bottom_level(f), _hhl(nlev, top=12000.0, surface=6000.0), _DT


def _mixed_random():
    """Fixed-seed column spanning both phases, for breadth rather than a named regime.

    Deliberately not built from a target saturation ratio: the job here is to sweep
    combinations no hand-designed profile would think to include, including levels that
    are subsaturated, supersaturated, glaciated and warm within one column. The sanity
    test therefore only checks that it is physical (positive, finite, mean masses in
    range), not that it sits in a particular regime.

    Measured: S = 0.713-1.122, Si = 0.846-1.436 over the sub-freezing levels, T =
    234-299 K. Si straddling 1 is the valuable part -- it is the only scenario that
    drives vapor_dep_relaxation in its *sublimation* direction, where the ``qvsidiff <
    0`` limiter and the ``reduce_sublimation`` number-reduction branch come into play.
    """
    rng = np.random.default_rng(20260817)
    nlev = 20
    f = _blank(nlev)
    f["tk"] = rng.uniform(230.0, 300.0, nlev)
    f["pres"] = np.linspace(2.5e4, 1.0e5, nlev)
    f["rho"] = thermo.rho_air(f["tk"], f["pres"])
    f["qv"] = thermo.qv_from_saturation_ratio(
        rng.uniform(0.7, 1.15, nlev), f["pres"], f["tk"]
    )
    f["w"] = rng.uniform(-0.5, 2.0, nlev)
    f["qc"] = rng.uniform(0.0, 5.0e-4, nlev)
    f["qnc"] = f["qc"] / 1.0e-10
    f["qi"] = rng.uniform(0.0, 5.0e-6, nlev)
    f["qni"] = f["qi"] / 1.0e-11
    f["qs"] = rng.uniform(0.0, 2.0e-5, nlev)
    f["qns"] = f["qs"] / 1.0e-8
    f["nccn"] = np.full(nlev, 1.5e9)
    f["ninpot"] = np.full(nlev, 1.0e5)
    return _dry_bottom_level(f), _hhl(nlev), _DT


SCENARIOS = {
    "bundled": _bundled(),
    "updraft": _updraft(),
    "mixed_species": _mixed_species(),
    "warm": _warm(),
    "deep_cold": _deep_cold(),
    "mixed_random": _mixed_random(),
}
