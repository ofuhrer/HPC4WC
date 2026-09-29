"""Scheme configuration for the default column-driver run (igscp=5, ice_type=1).

See porting_plan.md's "Default Configuration" section for the full cloud_type
derivation this is transcribed from. Do not hand-wave nuc_i_typ/nuc_c_typ values
again without re-deriving them from mo_2mom_mcrph_driver.f90's cloud_type
arithmetic -- an earlier draft of this port got nuc_i_typ wrong (claimed 6/Phillips,
is actually 1/INAS) by trusting a plausible-looking but unverified value.
"""

from dataclasses import dataclass

# column_driver.f90 defaults, unchanged by any cfg_2mom override (cfg_2mom_default
# has -999.99/-1 sentinels for everything these depend on):
#   ccn_type   = ccn_type_gscp5           = 8
#   cloud_type = 2003 + 10*8 + 100*1      = 2183
#   nuc_c_typ  = MOD(cloud_type/10, 10)   = 8   -> ccn_activation_sk_4d
#   nuc_i_typ  = MOD(cloud_type/100, 10)  = 1   -> ice_nucleation_het_inas
NUC_C_TYP = 8
NUC_I_TYP = 1

LUSE_AGI = False
LEXPL_SUPERSAT = False
IICEPHASE = 1  # cfg_2mom_default: mixed-phase 2-moment: on

# ccn_wcb_min from cfg_2mom_default (mo_2mom_mcrph_config_default.f90:41)
CCN_WCB_MIN = 0.1


@dataclass(frozen=True)
class CCNCoeffs:
    """Fortran ``TYPE(aerosol_ccn)`` -- fields are Ncn0/Nmin/lsigs/R2/etas/wcb_min/
    z0/z1e, NOT a/b/c/d/e/f (that naming belongs to the unrelated hardcoded
    ccn_activation_hdcp2 polynomial-fit constants, not this struct)."""

    ncn0: float
    nmin: float
    lsigs: float
    r2: float
    etas: float
    wcb_min: float
    z0: float
    z1e: float


# ccn_type=8, cfg_2mom_default%tune_sbmccn=1.0 -> NOT < 1.0 -> continental branch
# (mo_2mom_mcrph_driver.f90:747-762). Do not use the maritime numbers from the same
# CASE(8) block (Ncn0=100e6, lsigs=0.4, etas=0.9) -- easy to grab the wrong branch.
CCN_COEFFS = CCNCoeffs(
    ncn0=1700.0e6,
    nmin=35.0e6,
    lsigs=0.2,
    r2=0.03,
    etas=0.7,
    wcb_min=CCN_WCB_MIN,
    z0=4000.0,  # mo_2mom_mcrph_driver.f90:723, unconditional
    z1e=2000.0,  # mo_2mom_mcrph_driver.f90:724, unconditional
)


@dataclass(frozen=True)
class INCoeffs:
    """Fortran ``TYPE(aerosol_in)``: N0/z0/z1e for the potential-IN background
    profile used by the post-`post_twomoment` relaxation step in
    ``two_moment_mcrph`` (mo_2mom_mcrph_driver.f90:507-516) -- NOT part of
    ``clouds_twomoment``/the 5 retained processes; this is driver-level
    housekeeping, easy to miss (an earlier pass at this port did: see
    port_log.md, 2026-07-23 Phase 4 entry)."""

    n0: float
    z0: float
    z1e: float


# mo_2mom_mcrph_driver.f90:798-800, unconditional (only depends on lprogin=True)
IN_COEFFS = INCoeffs(n0=200.0e6, z0=3000.0, z1e=1000.0)

# mo_2mom_mcrph_driver.f90:218-219 (relaxation timescales) and
# mo_2mom_mcrph_processes.f90:235 (q_crit, the *active* value -- not the
# commented-out 1e-7 just above it in the source)
TAU_INACT = 600.0
TAU_INPOT = 1800.0
Q_CRIT = 1.0e-9
