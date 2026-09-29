"""Independent thermodynamics, for describing what a test column *is*.

WHAT THIS IS FOR
----------------
A scenario named "deep_cold" is worthless if the numbers in it turn out to be
subsaturated over ice, and "warm" is worthless if a level quietly sits above water
saturation -- the test still passes and still covers nothing. These functions let a
scenario's claimed regime be constructed and then asserted, instead of hand-tuned and
hoped for.

WHAT THIS IS NOT FOR
--------------------
**This is not a reference implementation and must never be used as one.** The schemes
under test compute saturation vapour pressure with the Tetens formula
(``b1*EXP(b2i*(T-b3)/(T-b4i))``, mo_satad.f90); the fits here are Murphy-Koop. The two
disagree by ~1% -- ample for confirming "this level is ice-supersaturated by roughly
20%", nowhere near enough to check a ported result. The only oracle for results is the
compiled Fortran binary, via :mod:`mcrph_common.fortran_runner`.

That distinction is the whole reason this file carries such a long header: it is the one
place in the tree where a second implementation of the physics could plausibly be
mistaken for a reference.

PROVENANCE
----------
Hoisted from ``parcelmodel/utils.py`` and ``parcelmodel/constants.py`` in the separate
``Supersaturation`` checkout (a parcel model, not part of this repo). Copied here rather
than imported because that checkout lives outside this tree and cannot go into
``pyproject.toml`` without breaking ``uv sync`` on another machine. Parametrisations are
Murphy and Koop (2005) as given in Lohmann and Mahrt (2025), pp. 118-119.

Units: temperatures K, pressures Pa, mixing ratios kg/kg, densities kg/m3.
"""

from __future__ import annotations

import numpy as np

# Constants belonging to the *independent* reference, not to the schemes under test.
# They differ from the ICON values in each variant's own constants module -- Rv here is
# 461.5 against ICON's 461.51, Ra 287.1 against 287.04. That is deliberate: these come
# from the parcel model and must NOT be "corrected" to match, or the check stops being
# independent. Nothing in this file may be imported by scheme code.
RV = 461.5  # gas constant, water vapour [J/kg/K]
RA = 287.1  # gas constant, dry air [J/kg/K]
M_W = 0.01802  # molar mass of water [kg/mol]
M_A = 0.02897  # molar mass of dry air [kg/mol]
T_0 = 273.15  # [K]


def e_sat_water(temperature):
    """Saturation vapour pressure over liquid water [Pa]. Murphy-Koop; valid 123-332 K."""
    t = np.asarray(temperature, dtype=np.float64)
    return np.exp(
        54.842763
        - 6763.22 / t
        - 4.21 * np.log(t)
        + 0.000367 * t
        + np.tanh(0.0415 * (t - 218.8))
        * (53.878 - 1331.22 / t - 9.44523 * np.log(t) + 0.014025 * t)
    )


def e_sat_ice(temperature):
    """Saturation vapour pressure over ice [Pa]. Murphy-Koop; valid 110-273.16 K."""
    t = np.asarray(temperature, dtype=np.float64)
    return np.exp(9.550426 - 5723.265 / t + 3.53068 * np.log(t) - 0.00728332 * t)


def vapour_pressure(qv, pressure):
    """Vapour pressure [Pa] from mixing ratio [kg/kg] and pressure [Pa]."""
    return np.asarray(pressure, dtype=np.float64) * (M_A / M_W) * np.asarray(qv, dtype=np.float64)


def saturation_ratio_water(qv, pressure, temperature):
    """Saturation **ratio** over water: 1.0 is exactly saturated.

    Named "ratio", not "supersaturation", on purpose. The upstream parcel-model helpers
    are called ``S``/``Si`` and documented as supersaturation but return ``e/Ew`` -- so
    "slightly subsaturated" is 0.98, not -0.02. Reading that backwards silently inverts
    the meaning of a scenario, which is exactly the failure this module exists to stop.
    """
    return vapour_pressure(qv, pressure) / e_sat_water(temperature)


def saturation_ratio_ice(qv, pressure, temperature):
    """Saturation **ratio** over ice: 1.0 is exactly saturated. See the note above."""
    return vapour_pressure(qv, pressure) / e_sat_ice(temperature)


def qv_from_saturation_ratio(ratio, pressure, temperature):
    """Mixing ratio [kg/kg] giving a target saturation ratio over water.

    Inverse of :func:`saturation_ratio_water`. Build scenario columns with this rather
    than tuning ``qv`` by hand: asking for "2% subsaturated" and getting it is a great
    deal more robust than picking a number and checking afterwards what it turned out
    to mean.
    """
    ratio = np.asarray(ratio, dtype=np.float64)
    return ratio * e_sat_water(temperature) * M_W / (np.asarray(pressure, dtype=np.float64) * M_A)


def qv_from_saturation_ratio_ice(ratio, pressure, temperature):
    """Mixing ratio [kg/kg] giving a target saturation ratio over *ice*.

    The ice-phase counterpart of :func:`qv_from_saturation_ratio`, for columns whose
    point is depositional growth -- there the regime that matters is supersaturation
    over ice, and water saturation says little about whether ice will grow.
    """
    ratio = np.asarray(ratio, dtype=np.float64)
    return ratio * e_sat_ice(temperature) * M_W / (np.asarray(pressure, dtype=np.float64) * M_A)


def rho_vapour_sat_water(temperature):
    """Vapour density at water saturation [kg/m3], from the ideal gas law."""
    return e_sat_water(temperature) / (RV * np.asarray(temperature, dtype=np.float64))


def rho_vapour_sat_ice(temperature):
    """Vapour density at ice saturation [kg/m3].

    Comparable to the density-space state after ``prepare_twomoment`` and to the
    scheme's own ``s_si = qv*R_v*T/e_es - 1``, so it is the natural quantity for
    checking that a column really will grow ice.
    """
    return e_sat_ice(temperature) / (RV * np.asarray(temperature, dtype=np.float64))


def rho_air(temperature, pressure):
    """Density of dry air [kg/m3] from the ideal gas law."""
    return np.asarray(pressure, dtype=np.float64) / (
        RA * np.asarray(temperature, dtype=np.float64)
    )
