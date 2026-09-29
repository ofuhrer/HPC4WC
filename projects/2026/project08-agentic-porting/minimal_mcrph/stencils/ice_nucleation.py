"""ice_nucleation_homhet: heterogeneous (INAS) + homogeneous (KHL06) ice nucleation.

Mirrors mo_2mom_mcrph_processes.f90::ice_nucleation_homhet (lines 561-823),
which for this scheme's default config (nuc_i_typ=1, luse_agi=False) dispatches
to ice_nucleation_het_inas (lines 826-996, Ullrich et al. 2007) and then always
runs the homogeneous KHL06 block (lines 748-820) -- see porting_plan.md's
"Default Configuration" section for why nuc_i_typ=1, not Phillips/6.

**Simplification, not an approximation:** ice_nucleation_het_inas loops over 3
dust modes, but this scheme's use_prog_in=True (confirmed: clouds_twomoment's
lprogin = PRESENT(ninpot) = True, since column_driver.f90 always supplies
ninpot). In that branch, each mode iteration does
`inp(k) = n_inpot(k) + ndust*(...)` -- overwriting, not accumulating -- so only
the LAST mode (mode 3: ndust_background=1e2, ddust_background=0.6e-6,
sigdust_background=1.5) survives; modes 1-2's contributions are computed and
discarded. `ssw`'s value (which selects the immersion-vs-deposition branch)
doesn't depend on the mode index either, so the mode loop collapses to a
single mode-3 computation with no loss of fidelity for this config. Verified
against real Fortran output, not just derived on paper -- see port_log.md.

**Known assumption about uninitialized state:** `ndiag_mask`/`nuc_n_a` (which
gate n_inpot's depletion) are local arrays in the Fortran that are only
assigned where the INAS gate (`lnuc_k`) is true; the reference binary's
behavior elsewhere depends on whatever was on the stack, not a defined value.
This implementation treats the gate-closed case as `nuc_n_a=0`,
`ndiag_mask=False` (i.e. no depletion) -- the sane reading of the algorithm's
intent, matching this Makefile's default (no -finit-local-zero, no
optimization flags) gfortran build in practice for the test column. See
port_log.md.
"""

import enum
import math

import gt4py.next as gtx
import numpy as np
from gt4py.next import arctan, cos, exp, log, maximum, minimum, sqrt, where

from icon4py.model.common import dimension as dims
from icon4py.model.common import field_type_aliases as fa
from icon4py.model.common import type_alias as ta

from minimal_mcrph.stencils.particle_helpers import particle_meanmass
from minimal_mcrph.stencils.saturation import diffusivity, e_es, e_ws


class _IceNucleationConst(ta.wpfloat, enum.Enum):
    """See unit_conversion.py's ``_UnitConversionConst`` docstring for why these must be
    enum members rather than plain module floats."""

    PI = 3.14159265358979323846
    T_3 = 273.15

    R_D_VAPOR = 461.51  # "R_d" alias in the Fortran -- water vapor gas constant
    R_L = 287.04  # "R_l" alias -- dry air gas constant
    CP = 1004.64  # cpd
    GRAV = 9.80665
    K_B = 1.3806504e-23  # Boltzmann constant
    RHO_ICE = 916.7
    L_ED = 2.8345e6  # als, latent heat of sublimation

    NI_HET_MAX = 100.0e3
    NI_HOM_MAX = 5000.0e3
    SSINUC = 0.02  # ice supersaturation threshold for heterogeneous nucleation

    # het_icenuc_inas_depo's "param_dust" (Ullrich et al. 2007)
    PARAM_DUST_1 = 286.0
    PARAM_DUST_2 = 0.017
    PARAM_DUST_3 = 256.7
    PARAM_DUST_4 = 0.080
    PARAM_DUST_5 = 200.75

    # dust background mode 3 only -- see module docstring for why
    NDUST_BG_MODE3 = 1.0e2
    # ddust_background = (/0.2_wp, 0.4_wp, 0.6_wp/) * 1e-6 -- note the scale factor
    # carries no `_wp`, so it is a *default real* (single-precision) literal and the
    # Fortran's mode-3 value is 0.6_wp * float64(float32(1e-6)), not 0.6e-6. See
    # particles.py::_r4 for the same trap in the particle constants; here it moves
    # sdust (which goes as ddust^2) by ~5e-9 relative and shows up in every
    # ice-nucleation output.
    DDUST_BG_MODE3 = 0.6 * float(np.float32(1.0e-6))
    SIGDUST2_MODE3 = PI * math.exp(2.0 * math.log(1.5) ** 2)

    # KHL06 homogeneous-nucleation constants (mo_2mom_mcrph_processes.f90:606-612)
    R_0 = 0.25e-6  # aerosol particle radius prior to freezing
    ALPHA_D = 0.5  # deposition coefficient
    M_W = 18.01528e-3  # molecular mass of water [kg/mol]
    M_A = 28.96e-3  # molecular mass of air [kg/mol]
    N_AVO = 6.02214179e23  # Avogadro constant
    MA_W = M_W / N_AVO  # mass of one water molecule [kg]
    SVOL = MA_W / RHO_ICE  # specific volume of a water molecule in ice


@gtx.field_operator
def _het_icenuc_inas_depo(
    temperature: fa.CellKField[ta.wpfloat], ssi: fa.CellKField[ta.wpfloat]
) -> fa.CellKField[ta.wpfloat]:
    """mo_2mom_mcrph_processes.f90:999-1016, INAS deposition nucleation."""
    temp = minimum(maximum(temperature, 190.0), 260.0)
    acotan = _IceNucleationConst.PI / 2.0 - arctan(_IceNucleationConst.PARAM_DUST_4 * (temp - _IceNucleationConst.PARAM_DUST_5))
    angle = _IceNucleationConst.PARAM_DUST_2 * (temp - _IceNucleationConst.PARAM_DUST_3)
    inas = exp(
        _IceNucleationConst.PARAM_DUST_1 * exp(0.25 * log(minimum(ssi, 1.0))) * cos(angle * angle) * acotan / _IceNucleationConst.PI
    )
    return minimum(maximum(inas, 1.0e5), 1.0e15)


@gtx.field_operator
def _ice_nucleation_het_inas(
    ice_x_min: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    cloud_q: fa.CellKField[ta.wpfloat],
    ice_q: fa.CellKField[ta.wpfloat],
    ice_n: fa.CellKField[ta.wpfloat],
    n_inact: fa.CellKField[ta.wpfloat],
    n_inpot: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    ssi = qv * _IceNucleationConst.R_D_VAPOR * temperature / e_es(temperature) - 1.0
    active = (
        ((ssi > _IceNucleationConst.SSINUC) | (cloud_q > 1.0e-20))
        & (temperature < 265.0)
        & (temperature > 190.0)
    )

    ssw = qv * _IceNucleationConst.R_D_VAPOR * temperature / e_ws(temperature)
    immersion = (ssw > 0.99) & (temperature > 235.0)

    inas_imm = exp(151.548 - 0.521 * temperature)
    inas_dep = _het_icenuc_inas_depo(temperature, ssi)
    inas = where(immersion, inas_imm, inas_dep)

    sdust = _IceNucleationConst.SIGDUST2_MODE3 * _IceNucleationConst.DDUST_BG_MODE3 * _IceNucleationConst.DDUST_BG_MODE3
    inp = n_inpot + _IceNucleationConst.NDUST_BG_MODE3 * (
        1.0 - exp(-minimum(maximum(inas * sdust, 0.0), 30.0))
    )

    nhet = minimum(inp, _IceNucleationConst.NI_HET_MAX)
    nuc_n_raw = maximum(nhet - n_inact, 0.0)
    nuc_q = minimum(nuc_n_raw * ice_x_min, qv)
    nuc_n = nuc_q / ice_x_min

    depletes = active & (inp > 1.0e-12)

    return (
        ice_q + where(active, nuc_q, 0.0),
        ice_n + where(active, nuc_n, 0.0),
        qv - where(active, nuc_q, 0.0),
        n_inact + where(active, nuc_n, 0.0),
        where(depletes, maximum(n_inpot - nuc_n, 0.0), n_inpot),
    )


@gtx.field_operator
def _homogeneous_nucleation(
    ice_x_min: ta.wpfloat,
    ice_x_max: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    w: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    ice_q: fa.CellKField[ta.wpfloat],
    ice_n: fa.CellKField[ta.wpfloat],
) -> tuple[fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat], fa.CellKField[ta.wpfloat]]:
    """KHL06 homogeneous nucleation. mo_2mom_mcrph_processes.f90:748-820.

    Always active for this scheme (nuc_i_typ=1 is in Fortran's `CASE(1:9)` ->
    use_homnuc=.TRUE.), independent of the INAS branch above.
    """
    e_si = e_es(temperature)
    ssi = qv * _IceNucleationConst.R_D_VAPOR * temperature / e_si
    scr = 2.349 - temperature / 259.0
    gate = (ssi > scr) & (temperature < 235.0) & (ice_n < _IceNucleationConst.NI_HOM_MAX)

    x_i = particle_meanmass(ice_q, ice_n, ice_x_min, ice_x_max)
    r_i = exp((1.0 / 3.0) * log(x_i / (4.0 / 3.0 * _IceNucleationConst.PI * _IceNucleationConst.RHO_ICE)))

    v_th = sqrt(8.0 * _IceNucleationConst.K_B * temperature / (_IceNucleationConst.PI * _IceNucleationConst.MA_W))
    flux = _IceNucleationConst.ALPHA_D * v_th / 4.0
    n_sat = e_si / (_IceNucleationConst.K_B * temperature)

    acoeff1 = (_IceNucleationConst.L_ED * _IceNucleationConst.GRAV) / (_IceNucleationConst.CP * _IceNucleationConst.R_D_VAPOR * temperature * temperature) - (
        _IceNucleationConst.GRAV / (_IceNucleationConst.R_L * temperature)
    )
    acoeff2 = 1.0 / n_sat
    acoeff3 = (_IceNucleationConst.L_ED * _IceNucleationConst.L_ED * _IceNucleationConst.M_W * _IceNucleationConst.MA_W) / (
        _IceNucleationConst.CP * pressure * temperature * _IceNucleationConst.M_A
    )

    bcoeff1 = flux * _IceNucleationConst.SVOL * n_sat * (ssi - 1.0)
    bcoeff2 = flux / diffusivity(temperature, pressure)

    ri_dot = bcoeff1 / (1.0 + bcoeff2 * r_i)
    r_ik = (4.0 * _IceNucleationConst.PI) / _IceNucleationConst.SVOL * ice_n * r_i * r_i * ri_dot
    w_pre = maximum((acoeff2 + acoeff3 * ssi) / (acoeff1 * ssi) * r_ik, 0.0)

    nucleates = gate & (w > w_pre)

    cool = _IceNucleationConst.GRAV / _IceNucleationConst.CP * w
    ctau = temperature * (0.004 * temperature - 2.0) + 304.4
    tau = 1.0 / (ctau * cool)
    delta = bcoeff2 * _IceNucleationConst.R_0
    phi = acoeff1 * ssi / (acoeff2 + acoeff3 * ssi) * (w - w_pre)

    kappa = 2.0 * bcoeff1 * bcoeff2 * tau / ((1.0 + delta) * (1.0 + delta))
    sqrtkap = sqrt(kappa)
    ren = 3.0 * sqrtkap / (2.0 + sqrt(1.0 + 9.0 * kappa / _IceNucleationConst.PI))
    r_imfc = 4.0 * _IceNucleationConst.PI * bcoeff1 / (bcoeff2 * bcoeff2) / _IceNucleationConst.SVOL
    r_im = (
        r_imfc
        / (1.0 + delta)
        * (delta * delta - 1.0 + (1.0 + 0.5 * kappa * (1.0 + delta) * (1.0 + delta)) * ren / sqrtkap)
    )

    ni_hom = phi / r_im
    ri_0 = 1.0 + 0.5 * sqrtkap * ren
    ri_hom = (ri_0 * (1.0 + delta) - 1.0) / bcoeff2
    mi_hom = maximum(
        (4.0 / 3.0 * _IceNucleationConst.PI * _IceNucleationConst.RHO_ICE) * ni_hom * ri_hom * ri_hom * ri_hom, ice_x_min
    )

    nuc_n = maximum(minimum(ni_hom, _IceNucleationConst.NI_HOM_MAX), 0.0)
    nuc_q = minimum(nuc_n * mi_hom, qv)

    return (
        ice_q + where(nucleates, nuc_q, 0.0),
        ice_n + where(nucleates, nuc_n, 0.0),
        qv - where(nucleates, nuc_q, 0.0),
    )


@gtx.field_operator
def ice_nucleation_homhet(
    ice_x_min: ta.wpfloat,
    ice_x_max: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    w: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    cloud_q: fa.CellKField[ta.wpfloat],
    ice_q: fa.CellKField[ta.wpfloat],
    ice_n: fa.CellKField[ta.wpfloat],
    n_inact: fa.CellKField[ta.wpfloat],
    n_inpot: fa.CellKField[ta.wpfloat],
) -> tuple[
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
    fa.CellKField[ta.wpfloat],
]:
    """INAS heterogeneous nucleation, then homogeneous KHL06 -- in that order,
    reading the INAS step's updated ice_q/ice_n/qv, matching
    mo_2mom_mcrph_processes.f90:561-823's sequential structure."""
    ice_q_1, ice_n_1, qv_1, n_inact_1, n_inpot_1 = _ice_nucleation_het_inas(
        ice_x_min, temperature, qv, cloud_q, ice_q, ice_n, n_inact, n_inpot
    )
    ice_q_2, ice_n_2, qv_2 = _homogeneous_nucleation(
        ice_x_min, ice_x_max, temperature, pressure, w, qv_1, ice_q_1, ice_n_1
    )
    return ice_q_2, ice_n_2, qv_2, n_inact_1, n_inpot_1


@gtx.program(grid_type=gtx.GridType.UNSTRUCTURED)
def ice_nucleation_homhet_program(
    ice_x_min: ta.wpfloat,
    ice_x_max: ta.wpfloat,
    temperature: fa.CellKField[ta.wpfloat],
    pressure: fa.CellKField[ta.wpfloat],
    w: fa.CellKField[ta.wpfloat],
    qv: fa.CellKField[ta.wpfloat],
    cloud_q: fa.CellKField[ta.wpfloat],
    ice_q: fa.CellKField[ta.wpfloat],
    ice_n: fa.CellKField[ta.wpfloat],
    n_inact: fa.CellKField[ta.wpfloat],
    n_inpot: fa.CellKField[ta.wpfloat],
    out_ice_q: fa.CellKField[ta.wpfloat],
    out_ice_n: fa.CellKField[ta.wpfloat],
    out_qv: fa.CellKField[ta.wpfloat],
    out_n_inact: fa.CellKField[ta.wpfloat],
    out_n_inpot: fa.CellKField[ta.wpfloat],
    horizontal_start: gtx.int32,
    horizontal_end: gtx.int32,
    vertical_start: gtx.int32,
    vertical_end: gtx.int32,
):
    ice_nucleation_homhet(
        ice_x_min,
        ice_x_max,
        temperature,
        pressure,
        w,
        qv,
        cloud_q,
        ice_q,
        ice_n,
        n_inact,
        n_inpot,
        out=(out_ice_q, out_ice_n, out_qv, out_n_inact, out_n_inpot),
        domain={
            dims.CellDim: (horizontal_start, horizontal_end),
            dims.KDim: (vertical_start, vertical_end),
        },
    )
