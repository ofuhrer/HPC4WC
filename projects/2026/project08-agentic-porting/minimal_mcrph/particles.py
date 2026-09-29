"""Particle-type constants and their derived coefficients.

Static constants are transcribed verbatim from the ``PARAMETER`` particle
definitions in ``mo_2mom_mcrph_main.f90`` (the exact named instances actually
selected by ``init_2mom_scheme`` for the default column-driver config -- see
``porting_plan.md``'s "Default Configuration" section for why these six and not
one of the other predefined variants in that file).

Derived coefficients mirror ``setup_particle_coeffs``/``init_2mom_sedi_vel`` in
``mo_2mom_mcrph_processes.f90`` (lines 405-483, 1503-1512): pure functions of the
static constants above, computed once, with no vertical-field dependence.
"""

import math
from dataclasses import dataclass

import numpy as np

from minimal_mcrph import constants as const


def _r4(value: float) -> float:
    """Round to IEEE single precision and widen back to double.

    The particle constants in ``mo_2mom_mcrph_main.f90`` are written as plain decimal
    literals (``0.333333``, not ``0.333333_wp``), which Fortran types as *default real*
    -- single precision -- before widening them into the ``REAL(wp)`` components of the
    PARAMETER. Transcribing the same decimal string as a Python float therefore yields a
    different number: ``0.333333`` is ``3.33332999999999990e-01`` in float64 but
    ``3.33332985639572144e-01`` as the Fortran actually holds it, a relative difference
    of 4.3e-8.

    That is not a rounding curiosity to wave away. These constants feed the
    gamma-function coefficient setup (``a_f``, ``b_f``, ``c_z``) and the per-level
    diameter/velocity power laws, so the discrepancy propagates into every depositional
    growth increment. It was the whole cause of the ~5e-7 disagreement in ``qi`` that the
    old ``rtol=1e-6`` tests were tuned loosely enough to accept; with this applied, the
    coefficients match the Fortran to ~1e-16 instead of ~1e-7.

    Literals written in ``d`` notation (``1.00d-05``, ``2.77d+01``) *are* double in the
    Fortran and must NOT be passed through here -- hence the per-field distinction below
    rather than a blanket conversion. If ICON ever gives these literals a ``_wp`` suffix,
    the corresponding ``_r4`` calls here have to go with it.
    """
    return float(np.float32(value))


@dataclass(frozen=True)
class ParticleConfig:
    """Static per-hydrometeor constants (Fortran ``TYPE(particle)``)."""

    name: str
    nu: float  # 1st shape parameter of the size distribution
    mu: float  # 2nd shape parameter
    x_max: float  # max mean particle mass [kg]
    x_min: float  # min mean particle mass [kg]
    a_geo: float  # diameter-mass relation prefactor: D = a_geo * x**b_geo
    b_geo: float  # diameter-mass relation exponent
    a_vel: float  # fall-speed power-law prefactor: v = a_vel * x**b_vel
    b_vel: float  # fall-speed power-law exponent
    a_ven: float  # 1st ventilation coefficient (PK, S.541)
    b_ven: float  # 2nd ventilation coefficient (PK, S.541)
    cap: float  # capacity coefficient
    vsedi_max: float  # max bulk sedimentation velocity [m/s]
    vsedi_min: float  # min bulk sedimentation velocity [m/s]


@dataclass(frozen=True)
class FrozenParticleConfig(ParticleConfig):
    """Adds the ``particle_frozen`` fields (unused by the retained processes for
    any hydrometeor except via ``particle_meanmass``'s x_min/x_max, but kept for
    fidelity to the Fortran type and in case a later process needs them)."""

    ecoll_c: float = 0.0  # max collision efficiency with cloud droplets
    d_crit_c: float = 0.0  # D-threshold for cloud riming
    q_crit_c: float = 0.0  # q-threshold for cloud riming
    s_vel: float = 0.0  # dispersion of fall velocity for collection kernel


# -- Named instances actually selected by init_2mom_scheme for igscp=5 --
# (mo_2mom_mcrph_main.f90:663-721; every per-species cfg_2mom override is the
# -999.99 sentinel in cfg_2mom_default, so these are used completely unmodified)
#
# Each field below mirrors its Fortran literal's *kind*, not just its decimal digits:
# `_r4(...)` where the Fortran writes a plain decimal (default real -> single
# precision), a bare Python float where it writes `d` notation (double). See `_r4`'s
# docstring -- getting this wrong shifts the gamma-derived coefficients by ~1e-7, which
# is large enough to move every depositional-growth increment. The `# d` / `# R4`
# comments record which form the Fortran actually uses, so the mapping stays checkable
# against the source without re-reading it.

CLOUD = ParticleConfig(
    name="cloud_nue1mue1",  # mo_2mom_mcrph_main.f90:298-315
    nu=_r4(1.000000),
    mu=_r4(1.000000),
    x_max=2.60e-10,  # 2.60d-10
    x_min=4.20e-15,  # 4.20d-15
    a_geo=1.24e-01,  # 1.24d-01
    b_geo=_r4(0.333333),
    a_vel=3.75e05,  # 3.75d+05
    b_vel=_r4(0.666667),
    a_ven=_r4(0.780000),
    b_ven=_r4(0.308000),
    cap=_r4(2.00),
    vsedi_max=_r4(1.0),
    vsedi_min=_r4(0.0),
)

RAIN = ParticleConfig(
    name="rainSBB",  # mo_2mom_mcrph_main.f90:436-453
    nu=_r4(1.000000),
    mu=_r4(0.333333),
    x_max=6.50e-05,  # 6.50d-05
    x_min=2.60e-10,  # 2.60d-10
    a_geo=1.24e-01,  # 1.24d-01
    b_geo=_r4(0.333333),
    a_vel=_r4(114.0137),
    b_vel=_r4(0.234370),
    a_ven=_r4(0.780000),
    b_ven=_r4(0.308000),
    cap=_r4(2.000000),
    vsedi_max=2.000e01,  # 2.000d+1
    vsedi_min=_r4(0.1),
)

ICE = FrozenParticleConfig(
    name="ice_cosmo5",  # mo_2mom_mcrph_main.f90:317-340
    nu=_r4(0.000000),
    mu=_r4(0.333333),
    x_max=1.00e-05,  # 1.00d-05
    x_min=1.00e-12,  # 1.00d-12
    a_geo=_r4(0.835000),
    b_geo=_r4(0.390000),
    a_vel=2.77e01,  # 2.77d+01
    b_vel=_r4(0.215790),
    a_ven=_r4(0.780000),
    b_ven=_r4(0.308000),
    cap=_r4(3.0),
    vsedi_max=_r4(3.0),
    vsedi_min=_r4(0.0),
    ecoll_c=_r4(0.80),
    d_crit_c=150.0e-6,  # 150.0d-6
    q_crit_c=1.000e-5,  # 1.000d-5
    s_vel=_r4(0.25),
)

SNOW = FrozenParticleConfig(
    name="snowSBB",  # mo_2mom_mcrph_main.f90:367-390
    nu=_r4(0.000000),
    mu=_r4(0.500000),
    x_max=2.00e-05,  # 2.00d-05
    x_min=1.00e-10,  # 1.00d-10
    a_geo=_r4(5.130000),
    b_geo=_r4(0.500000),
    a_vel=_r4(400.0000),
    b_vel=_r4(0.350000),
    a_ven=_r4(0.780000),
    b_ven=_r4(0.308000),
    cap=_r4(3.00),
    vsedi_max=_r4(3.0),
    vsedi_min=_r4(0.1),
    ecoll_c=_r4(0.80),
    d_crit_c=150.0e-6,  # 150.0d-6
    q_crit_c=1.000e-5,  # 1.000d-5
    s_vel=_r4(0.25),
)

GRAUPEL = FrozenParticleConfig(
    name="graupelhail_cosmo5",  # mo_2mom_mcrph_main.f90:170-193 (particle_frozen branch, igscp != 7)
    nu=_r4(1.000000),
    mu=_r4(0.333333),
    x_max=5.30e-04,  # 5.30d-04
    x_min=4.19e-09,  # 4.19d-09
    a_geo=1.42e-01,  # 1.42d-01
    b_geo=_r4(0.314000),
    a_vel=_r4(100.0),
    b_vel=_r4(0.34),
    a_ven=_r4(0.780000),
    b_ven=_r4(0.308000),
    cap=_r4(2.00),
    vsedi_max=_r4(80.0),
    vsedi_min=_r4(0.10),
    ecoll_c=_r4(1.0),
    d_crit_c=100.0e-6,  # 100.0d-6
    q_crit_c=1.000e-6,  # 1.000d-6
    s_vel=_r4(0.0),
)

HAIL = FrozenParticleConfig(
    name="hail_cosmo5",  # mo_2mom_mcrph_main.f90:225-248 (particle_frozen branch, igscp != 7)
    nu=_r4(1.000000),
    mu=_r4(0.333333),
    x_max=5.00e-03,  # 5.00d-03
    x_min=2.60e-9,  # 2.60d-9
    a_geo=_r4(0.1366),
    b_geo=_r4(0.333333),
    a_vel=_r4(39.3),
    b_vel=_r4(0.166667),
    a_ven=_r4(0.780000),
    b_ven=_r4(0.308000),
    cap=_r4(2.00),
    vsedi_max=_r4(30.0),
    vsedi_min=_r4(0.1),
    ecoll_c=_r4(1.0),
    d_crit_c=100.0e-6,  # 100.0d-6
    q_crit_c=1.000e-6,  # 1.000d-6
    s_vel=_r4(0.0),
)


# -- Particle helper functions (mo_2mom_mcrph_processes.f90:364-402) --


def particle_meanmass(p: ParticleConfig, q: float, n: float) -> float:
    """Eq. (94) of SB2006, with limiters."""
    eps = 1e-20
    return min(max(q / (n + eps), p.x_min), p.x_max)


def particle_diameter(p: ParticleConfig, x: float) -> float:
    """Eq. (32) of SB2006: D = a_geo * x**b_geo."""
    return p.a_geo * math.exp(p.b_geo * math.log(x))


def particle_velocity(p: ParticleConfig, x: float) -> float:
    """Eq. (33) of SB2006: v = a_vel * x**b_vel."""
    return p.a_vel * math.exp(p.b_vel * math.log(x))


def vent_coeff_a(p: ParticleConfig, n: int) -> float:
    """Eq. (88) of SB2006."""
    g = math.gamma
    return (
        p.a_ven
        * g((p.nu + n + p.b_geo) / p.mu)
        / g((p.nu + 1.0) / p.mu)
        * (g((p.nu + 1.0) / p.mu) / g((p.nu + 2.0) / p.mu)) ** (p.b_geo + n - 1.0)
    )


def vent_coeff_b(p: ParticleConfig, n: int) -> float:
    """Eq. (89) of SB2006."""
    m_f = 0.500  # PK, S.541 -- do not change
    g = math.gamma
    exponent = (p.nu + n + (m_f + 1.0) * p.b_geo + m_f * p.b_vel) / p.mu
    return (
        p.b_ven
        * g(exponent)
        / g((p.nu + 1.0) / p.mu)
        * (g((p.nu + 1.0) / p.mu) / g((p.nu + 2.0) / p.mu))
        ** ((m_f + 1.0) * p.b_geo + m_f * p.b_vel + n - 1.0)
    )


def moment_gamma(p: ParticleConfig, n: int) -> float:
    """Eq. (82) of SB2006: complete mass moment of the particle size distribution."""
    g = math.gamma
    return g((n + p.nu + 1.0) / p.mu) / g((p.nu + 1.0) / p.mu) * (
        g((p.nu + 1.0) / p.mu) / g((p.nu + 2.0) / p.mu)
    ) ** n


# -- Derived run-time coefficients --


@dataclass(frozen=True)
class ParticleCoeffs:
    """Ventilation/diffusion coefficients (Fortran ``TYPE(particle_coeffs)``:
    ``a_f, b_f, c_i, c_z`` -- not ``a_vel``/``b_vel``, those live on
    ``ParticleConfig`` above, not here)."""

    a_f: float
    b_f: float
    c_i: float
    c_z: float


def setup_particle_coeffs(p: ParticleConfig) -> ParticleCoeffs:
    """mo_2mom_mcrph_processes.f90:1503-1512."""
    return ParticleCoeffs(
        c_i=1.0 / p.cap,
        a_f=vent_coeff_a(p, 1),
        b_f=vent_coeff_b(p, 1) * const.N_SC**const.N_F / math.sqrt(const.NU_L),
        c_z=moment_gamma(p, 2),
    )


@dataclass(frozen=True)
class SedimentationCoeffs:
    """mo_2mom_mcrph_processes.f90:451-467 (``init_2mom_sedi_vel``).

    NOT used by any of the 5 retained processes (grep-confirmed: only written
    here and printed in an isprint debug block) -- they only ever fed
    sedimentation, which this scheme doesn't have. Ported anyway because it's
    the one coefficient set Fortran already prints via ``isprint`` with zero
    source modification, making it a free validation point for the gamma-function
    math shared with ``setup_particle_coeffs`` above.
    """

    coeff_alfa_n: float
    coeff_alfa_q: float
    coeff_lambda: float


def init_2mom_sedi_vel(p: ParticleConfig) -> SedimentationCoeffs:
    g = math.gamma
    return SedimentationCoeffs(
        coeff_alfa_n=p.a_vel * g((p.nu + p.b_vel + 1.0) / p.mu) / g((p.nu + 1.0) / p.mu),
        coeff_alfa_q=p.a_vel * g((p.nu + p.b_vel + 2.0) / p.mu) / g((p.nu + 2.0) / p.mu),
        coeff_lambda=g((p.nu + 1.0) / p.mu) / g((p.nu + 2.0) / p.mu),
    )
