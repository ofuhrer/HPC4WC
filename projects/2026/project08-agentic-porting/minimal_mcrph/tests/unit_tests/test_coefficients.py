"""particles.py's coefficient math against the Fortran's own values, at full precision.

The Fortran computes these once in ``init_2mom_scheme_once``; ``mo_stage_dump.f90`` writes
them to ``coeffs.csv`` at 17 digits when the driver is given a dump directory, so this
test regenerates its reference on every run.

WHY THE PRECISION MATTERS HERE
------------------------------
This test used to assert ``rel=1e-6`` against values pasted from a temporary ``WRITE`` --
``init_2mom_sedi_vel``'s three coefficients are printed by the unmodified Fortran, but
only through ``D14.7`` (7 digits), and ``setup_particle_coeffs``' ``a_f``/``b_f``/``c_i``/
``c_z`` were printed nowhere at all.

That tolerance was three orders of magnitude too loose to see what was actually wrong.
The particle constants in ``mo_2mom_mcrph_main.f90`` are written as plain decimal literals
(``0.333333``, not ``0.333333_wp``), so Fortran holds them at *single* precision, while
the port had transcribed the same decimal strings as float64. Every gamma-derived
coefficient was off by 1e-8 to 4e-7 -- comfortably inside ``rel=1e-6``, and large enough
to shift every depositional-growth increment by ~5e-7. See ``particles.py::_r4``.

So the tolerance below is not a formality. At 1e-13 it is the check that would have
caught that bug on the day it was written.
"""

from __future__ import annotations

import pytest

from minimal_mcrph import particles as p
from minimal_mcrph.tests.conftest import compare
from minimal_mcrph.tests.fortran_runner import run_fortran_coeffs

# Gamma-function evaluations differ between math.gamma and gfortran's GAMMA by a few ULP,
# which is the only difference that should remain once the constants themselves match.
RTOL = 1.0e-13

_PARTICLES = {
    "ice": p.ICE,
    "snow": p.SNOW,
    "graupel": p.GRAUPEL,
    "hail": p.HAIL,
    "cloud": p.CLOUD,
}

# init_2mom_sedi_vel's coefficients exist only on the particle_sphere types; cloud_coeffs
# is a particle_cloud_coeffs and has none, so the dump writes zeros for it.
_HAS_SEDI_COEFFS = ("ice", "snow", "graupel", "hail")


@pytest.fixture(scope="module")
def fortran_coeffs():
    return run_fortran_coeffs()


@pytest.mark.parametrize("name", sorted(_PARTICLES))
def test_setup_particle_coeffs_matches_fortran(name, fortran_coeffs):
    got = p.setup_particle_coeffs(_PARTICLES[name])
    ref = fortran_coeffs[name]
    for field in ("a_f", "b_f", "c_i", "c_z"):
        compare(f"[{name}] {field}", [getattr(got, field)], [ref[field]], rtol=RTOL)


@pytest.mark.parametrize("name", _HAS_SEDI_COEFFS)
def test_init_2mom_sedi_vel_matches_fortran(name, fortran_coeffs):
    """Unused by the retained processes, but it shares the gamma-function math with
    ``setup_particle_coeffs`` -- so it is a free second opinion on that math, over a
    different combination of the same particle constants."""
    got = p.init_2mom_sedi_vel(_PARTICLES[name])
    ref = fortran_coeffs[name]
    for field in ("coeff_alfa_n", "coeff_alfa_q", "coeff_lambda"):
        compare(f"[{name}] {field}", [getattr(got, field)], [ref[field]], rtol=RTOL)


@pytest.mark.parametrize("name", sorted(_PARTICLES))
def test_particle_constants_carry_fortran_literal_precision(name):
    """The single-precision literals must stay single-precision.

    A direct guard on the bug above, independent of the coefficient comparisons. It would
    be very easy for someone tidying `particles.py` to read ``_r4(0.333333)`` as clutter
    and "simplify" it to ``0.333333``, which reintroduces a 4.3e-8 error in a form that
    looks like a cleanup. Asserting the stored value is representable in float32 makes
    that regression fail immediately and legibly.

    Fields written in ``d`` notation in the Fortran (x_min, x_max, and some prefactors)
    are genuinely double and are deliberately not checked here.
    """
    import numpy as np

    particle = _PARTICLES[name]
    for field in ("nu", "mu", "b_geo", "b_vel", "a_ven", "b_ven", "cap"):
        value = getattr(particle, field)
        assert float(np.float32(value)) == value, (
            f"{name}.{field} = {value!r} is not a float32-representable value. The "
            f"Fortran literal for it carries no `_wp` suffix, so it must be wrapped in "
            f"particles._r4 -- see that function's docstring."
        )
