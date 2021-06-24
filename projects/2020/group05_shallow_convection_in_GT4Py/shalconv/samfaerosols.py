from .physcons import con_g as g, qamin
from . import DTYPE_FLOAT, BACKEND
import gt4py as gt
from gt4py import gtscript
from __gtscript__ import PARALLEL, FORWARD, BACKWARD, computation, interval
import numpy as np

## Constants
epsil = 1e-22 # prevent division by zero
escav = 0.8   # wet scavenging efficiency

@gtscript.function
def set_qaero(qtr, kmax, index_k):
    qaero = max(qamin, qtr) if index_k <= kmax else 0.0
    return qaero

@gtscript.function
def set_xmbp(xmb, delp):
    xmbp = g * xmb / delp
    return xmbp

@gtscript.function
def set_ctro2(qaero, kmax, index_k):
    if index_k + 1 <= kmax:
        ctro2 = 0.5 * (qaero[0, 0, 0] + qaero[0, 0, 1])
    else:
        ctro2 = qaero # boundary already set in qaero
        #if index_k == kmax:
        #    ctro2 = qaero
        #else:
        #    ctro2 = 0.0
    return ctro2

@gtscript.function
def set_ecko2(ctro2, cnvflg, kb, index_k):
    if cnvflg and (index_k <= kb):
        ecko2 = ctro2
    else:
        ecko2 = 0.0
    return ecko2

@gtscript.stencil(backend = BACKEND)
def set_work_arrays(
    qtr: gtscript.Field[DTYPE_FLOAT],
    xmb: gtscript.Field[DTYPE_FLOAT],
    delp: gtscript.Field[DTYPE_FLOAT],
    kmax: gtscript.Field[int],
    kb: gtscript.Field[int],
    cnvflg: gtscript.Field[int],
    index_k: gtscript.Field[int],
    qaero: gtscript.Field[DTYPE_FLOAT],
    xmbp: gtscript.Field[DTYPE_FLOAT],
    ctro2: gtscript.Field[DTYPE_FLOAT],
    ecko2: gtscript.Field[DTYPE_FLOAT]
):
    with computation(PARALLEL), interval(...):
        qaero = set_qaero(qtr, kmax, index_k)
        xmbp = set_xmbp(xmb, delp)
    with computation(PARALLEL), interval(...):
        ctro2 = set_ctro2(qaero, kmax, index_k)
        ecko2 = set_ecko2(ctro2, cnvflg, kb, index_k)

@gtscript.stencil(backend = BACKEND)
def calc_ecko2_chem_c_dellae2(
    zi: gtscript.Field[DTYPE_FLOAT],
    xlamue: gtscript.Field[DTYPE_FLOAT],
    xlamud: gtscript.Field[DTYPE_FLOAT],
    ctro2: gtscript.Field[DTYPE_FLOAT],
    c0t: gtscript.Field[DTYPE_FLOAT],
    eta: gtscript.Field[DTYPE_FLOAT],
    xmbp: gtscript.Field[DTYPE_FLOAT],
    kb: gtscript.Field[int],
    ktcon: gtscript.Field[int],
    cnvflg: gtscript.Field[int],
    index_k: gtscript.Field[int],
    ecko2: gtscript.Field[DTYPE_FLOAT],
    chem_c: gtscript.Field[DTYPE_FLOAT],
    # chem_pw: gtscript.Field[DTYPE_FLOAT],
    dellae2: gtscript.Field[DTYPE_FLOAT],
    *,
    fscav: float
):
    with computation(FORWARD), interval(1, -1):
        if cnvflg and (index_k > kb) and (index_k < ktcon):
            dz   = zi[0, 0, 0] - zi[0, 0, -1]
            tem  = 0.5  * (xlamue[0, 0, 0] + xlamue[0, 0, -1]) * dz
            tem1 = 0.25 * (xlamud[0, 0, 0] + xlamud[0, 0,  0]) * dz
            factor = 1.0 + tem - tem1

            # if conserved (not scavenging) then
            ecko2 = ((1.0 - tem1) * ecko2[0, 0, -1] +
                     0.5 * tem * (ctro2[0, 0, 0] + ctro2[0, 0, -1])) / factor
            #    how much will be scavenged
            #    this choice was used in GF, and is also described in a
            #    successful implementation into CESM in GRL (Yu et al. 2019),
            #    it uses dimesnsionless scavenging coefficients (fscav),
            #    but includes henry coeffs with gas phase chemistry
            #    fraction fscav is going into liquid
            chem_c = escav * fscav * ecko2
            # of that part is going into rain out (chem_pw)
            tem2 = chem_c / (1.0 + c0t * dz)
            # chem_pw = c0t * dz * tem2 * eta # etah
            ecko2 = tem2 + ecko2 - chem_c
    with computation(PARALLEL), interval(0, -1):
        if index_k >= ktcon:
            ecko2 = ctro2
        if cnvflg and (index_k == ktcon):
            #for the subsidence term already is considered
            dellae2 = eta[0, 0, -1] * ecko2[0, 0, -1] * xmbp

@gtscript.stencil(backend = BACKEND)
def calc_dtime_max_arr(
    delp: gtscript.Field[DTYPE_FLOAT],
    ktcon: gtscript.Field[int],
    index_k: gtscript.Field[int],
    dtime_max_arr: gtscript.Field[DTYPE_FLOAT],
    *,
    delt: float
):
    with computation(FORWARD):
        with interval(0, 1):
            dtime_max_arr = delt
        with interval(1, -1):
            if index_k - 1 < ktcon:
                dtime_max_arr = min(dtime_max_arr[0, 0, -1], 0.5 * delp[0, 0, -1])
            else:
                dtime_max_arr = dtime_max_arr[0, 0, -1]

@gtscript.stencil(backend = BACKEND)
def calc_detrainment_entrainment(
    zi: gtscript.Field[DTYPE_FLOAT],
    xlamue: gtscript.Field[DTYPE_FLOAT],
    xlamud: gtscript.Field[DTYPE_FLOAT],
    ecko2: gtscript.Field[DTYPE_FLOAT],
    ctro2: gtscript.Field[DTYPE_FLOAT],
    eta: gtscript.Field[DTYPE_FLOAT],
    xmbp: gtscript.Field[DTYPE_FLOAT],
    kb: gtscript.Field[int],
    ktcon: gtscript.Field[int],
    cnvflg: gtscript.Field[int],
    index_k: gtscript.Field[int],
    dellae2: gtscript.Field[DTYPE_FLOAT]
):
    with computation(PARALLEL), interval(1, -1):
        if cnvflg and (index_k < ktcon):
            dz   = zi[0, 0, 0] - zi[0, 0, -1]
            aup  = 1.0 if index_k > kb else 0.0

            dv1q = 0.5 * (ecko2[0, 0, 0] + ecko2[0, 0, -1])
            dv2q = 0.5 * (ctro2[0, 0, 0] + ctro2[0, 0, -1])

            tem  = 0.5 * (xlamue[0, 0, 0] + xlamue[0, 0, -1])
            tem1 = 0.5 * (xlamud[0, 0, 0] + xlamud[0, 0,  0])

            dellae2 = dellae2 + (
                aup * tem1 * eta[0, 0, -1] * dv1q # detrainment from updraft
              - aup * tem  * eta[0, 0, -1] * dv2q # entrainment into up and downdraft
            ) * dz * xmbp

            if index_k == kb:
                dellae2 = dellae2 - eta * ctro2 * xmbp
    with computation(FORWARD), interval(0, 1):
        if cnvflg and (kb == 1):
            dellae2 = dellae2 - eta * ctro2 * xmbp

@gtscript.stencil(backend = BACKEND)
def calc_mass_flux(
    eta: gtscript.Field[DTYPE_FLOAT],
    xmb: gtscript.Field[DTYPE_FLOAT],
    qaero: gtscript.Field[DTYPE_FLOAT],
    delp: gtscript.Field[DTYPE_FLOAT],
    kb: gtscript.Field[int],
    ktcon: gtscript.Field[int],
    cnvflg: gtscript.Field[int],
    index_k: gtscript.Field[int],
    flx_lo: gtscript.Field[DTYPE_FLOAT],
    totlout: gtscript.Field[DTYPE_FLOAT],
    clipout: gtscript.Field[DTYPE_FLOAT],
    dellae2: gtscript.Field[DTYPE_FLOAT],
    *,
    dtime_max: float
):
    with computation(PARALLEL), interval(1, -1):
        if cnvflg and (index_k - 1 < ktcon):
            tem = 0.0 if index_k - 1 < kb else eta[0, 0, -1]
            # low-order flux, upstream
            qaero_val = qaero[0, 0, 0] if tem > 0.0 else qaero[0, 0, -1]
            flx_lo = - xmb * tem * qaero_val
    # make sure low-ord fluxes don't violate positive-definiteness
    with computation(PARALLEL), interval(0, -1):
        if cnvflg and (index_k <= ktcon):
            # time step / grid spacing
            dtovdz = g * dtime_max / abs(delp)
            # total flux out
            totlout = max(0.0, flx_lo[0, 0, 1]) - min(0.0, flx_lo[0, 0, 0])
            clipout = min(1.0, qaero / max(epsil, totlout) / (1.0001 * dtovdz))
    # recompute upstream mass fluxes
    with computation(PARALLEL), interval(1, -1):
        if cnvflg and index_k - 1 <= ktcon:
            tem = 0.0 if index_k - 1 < kb else eta[0, 0, -1]
            clipout_val = clipout[0, 0, 0] if tem > 0.0 else clipout[0, 0, -1]
            flx_lo = flx_lo * clipout_val
    # a positive-definite low-order (diffusive) solution for the subsidnce fluxes
    with computation(PARALLEL), interval(0, -1):
        if cnvflg and index_k <= ktcon:
            # time step / grid spacing
            dtovdz = g * dtime_max / abs(delp)
            dellae2 = dellae2 - (flx_lo[0, 0, 1] - flx_lo[0, 0, 0]) * dtovdz / dtime_max

@gtscript.stencil(backend = BACKEND)
def calc_final(
    dellae2: gtscript.Field[DTYPE_FLOAT],
    kmax: gtscript.Field[int],
    ktcon: gtscript.Field[int],
    cnvflg: gtscript.Field[int],
    index_k: gtscript.Field[int],
    qaero: gtscript.Field[DTYPE_FLOAT],
    *,
    delt: float
):
    # compute final aerosol concentrations
    with computation(PARALLEL), interval(...):
        if cnvflg and (index_k <= min(kmax, ktcon)):
            qaero = qaero + dellae2 * delt
            if qaero < 0.0:
                qaero = qamin

def samfshalcnv_aerosols(im, ix, km, itc, ntc, ntr, delt,
                         cnvflg, kb, kmax, kbcon, ktcon, fscav_np,
                         xmb, c0t, eta, zi, xlamue, xlamud, delp,
                         qtr_np, qaero_np):
    """
    Aerosol process in shallow convection
    :param im: horizontal loop extent
    :param ix: horizontal dimension (im <= ix)
    :param km: vertical layer dimension
    :param itc: number of aerosol tracers transported/scavenged by convection
    :param ntc: number of chemical tracers
    :param ntr: number of tracers for scale-aware mass flux schemes
    :param delt: physics time step
    :param cnvflg: (im) flag of convection
    :param kb: (im)
    :param kmax: (im)
    :param kbcon: (im)
    :param ktcon: (im)
    :param fscav_np: (ntc) numpy array of aerosol scavenging coefficients
    :param xmb: (im)
    :param c0t: (im,km) Cloud water parameters
    :param eta: (im,km)
    :param zi: (im,km) height
    :param xlamue: (im,km)
    :param xlamud: (im)
    :param delp: (ix,km) pressure?
    :param qtr_np: (ix,km,ntr+2) numpy array
    :param qaero_np: (im,km,ntc) numpy array
    """
    shape_2d = (1, ix, km)
    default_origin = (0, 0, 0)
    # Initialization
    xmbp = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ## Chemical transport variables (2D slices)
    qtr     = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    qaero   = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ctro2   = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ecko2   = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ecdo2   = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    dellae2 = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ## Additional variables for tracers for wet deposition (2D slices)
    chem_c  = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    # chem_pw = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    # wet_dep = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ## Additional variables for fct
    flx_lo  = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    totlout = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    clipout = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)
    ## Misc
    index_ijk_np = np.indices(shape_2d) 
    index_k = gt.storage.from_array(index_ijk_np[2] + 1, BACKEND, default_origin, shape_2d, dtype = int) # index STARTING FROM 1
    dtime_max_arr = gt.storage.empty(BACKEND, default_origin, shape_2d, dtype = DTYPE_FLOAT)

    # Begin
    ## Check if aerosols are present
    if (ntc <= 0) or (itc <= 0) or (ntr <= 0): return
    if (ntr < itc + ntc - 3): return

    calc_dtime_max_arr(delp, ktcon, index_k, dtime_max_arr, delt = delt)
    dtime_max_arr.synchronize()
    dtime_max = dtime_max_arr[0, :, -1].view(np.ndarray).min()

    ## Tracer loop
    for n in range(ntc):
        ## Initialize work variables
        chem_c[...]  = 0.0
        # chem_pw[...] = 0.0
        ctro2[...]   = 0.0
        dellae2[...] = 0.0
        ecdo2[...]   = 0.0
        ecko2[...]   = 0.0
        qaero[...]   = 0.0

        it = n + itc - 1
        qtr[...] = qtr_np[np.newaxis, :, :, it]
        set_work_arrays(qtr, xmb, delp, kmax, kb, cnvflg, 
                        index_k, qaero, xmbp, ctro2, ecko2)
        # do chemical tracers, first need to know how much reevaporates
        # aerosol re-evaporation is set to zero for now
        # calculate include mixing ratio (ecko2), how much goes into
        # rainwater to be rained out (chem_pw), and total scavenged,
        # if not reevaporated (pwav)
        fscav = fscav_np[n]
        calc_ecko2_chem_c_dellae2(zi, xlamue, xlamud, ctro2, c0t, eta, xmbp, kb,
                                   ktcon, cnvflg, index_k, ecko2, chem_c,
                                   dellae2, fscav = fscav)

        # initialize maximum allowed timestep for upstream difference approach
        # MOVED OUTSIDE OF TRACER LOOP

        calc_detrainment_entrainment(zi, xlamue, xlamud, ecko2, ctro2, eta,
                                     xmbp, kb, ktcon, cnvflg, index_k, dellae2)
        
        calc_mass_flux(eta, xmb, qaero, delp, kb, ktcon, cnvflg, index_k,
                       flx_lo, totlout, clipout, dellae2, dtime_max = dtime_max)

        calc_final(dellae2, kmax, ktcon, cnvflg, index_k, qaero, delt = delt)

        qaero.synchronize()
        qaero_np[:, :, n] = qaero.view(np.ndarray)[0, :, :]

