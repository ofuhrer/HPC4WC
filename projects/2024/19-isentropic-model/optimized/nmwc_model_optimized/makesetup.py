# -*- coding: utf-8 -*-
import numpy as np
import sys

from nmwc_model_optimized.meteo_utilities import rrmixv1
from nmwc_model_optimized.namelist import (
    bv00,
    cp,
    cpdr,
    dth,
    dx,
    exn00,
    g,
    idbg,
    imicrophys,
    imoist,
    ishear,
    k_shl,
    k_sht,
    nz,
    pref,
    th00,
    topomx,
    topowd,
    u00,
    u00_sh,
    z00,
)


def maketopo(topo, nxb):
    """ Topography definition.

    Parameters
    ----------
    topo : np.ndarray
        Array to be filled with the topography profile in [m].
    nxb : int
        Number of items in ``topo``.

    Returns
    -------
    np.ndarray :
        The input array ``topo`` filled with the topography profile in [m].
    """
    if idbg == 1:
        print("Topography ...\n")

    x = np.arange(0, nxb, dtype=np.float64)
    x0 = (nxb - 1) / 2.0 + 1
    x = (x + 1 - x0) * dx

    topomx2, topomx3, topomx4 = 1000, 500, 500
    x0_2, x0_3, x0_4 = -900000, 0, 1400000
    # Calculate Gaussian functions
    toponf1 = topomx * np.exp(-(x / float(topowd)) ** 2)
    toponf2 = topomx2 * np.exp(-((x - x0_2) / float(topowd) / 2) ** 2)
    toponf3 = topomx3 * np.exp(-((x - x0_3) / float(topowd)) ** 2)
    toponf4 = topomx4 * np.exp(-((x - x0_4) / float(topowd) / 2.5) ** 2)

    # Combine Gaussian terms
    toponf = toponf1 + toponf2 + toponf3 + toponf4

    # Calculate final topography profile
    topo[1:-1, 0] = toponf[1:-1] + 0.25 * (
        toponf[0:-2] - 2.0 * toponf[1:-1] + toponf[2:]
    )

    return topo


def makeprofile(
    sold,
    uold,
    qvold=None,
    qvnow=None,
    qcold=None,
    qcnow=None,
    qrold=None,
    qrnow=None,
    ncold=None,
    ncnow=None,
    nrold=None,
    nrnow=None,
):
    """ Initialize the solution.

    Make upstream profiles and initial conditions for isentropic density (sigma)
    and velocity (u).

    Returns
    -------
    th0 : np.ndarray
        1-d array representing the initial staggered vertical profile of
        potential temperature in [K].
    exn0 : np.ndarray
        1-d array representing the initial staggered vertical profile of
        Exner function in [J kg^-1 K^-1].
    prs0 : np.ndarray
        1-d array representing the initial staggered vertical profile of pressure
        in [Pa].
    z0 : np.ndarray
        1-d array collecting the initial geometric height of the interface
        vertical levels in [m].
    mtg0 : np.ndarray
        1-d array representing the initial vertical profile of Montgomery potential
        in [m^2 s^-2].
    s0 : np.ndarray
        1-d array representing the initial vertical profile of isentropic density
        in [kg m^-2 K^-1].
    u0 : np.ndarray
        1-d array representing the initial vertical profile of horizontal velocity
        in [m s^-1].
    sold : np.ndarray
        Isentropic density in [kg m^-2 K^-1] at the "previous" (i.e. zero-th) time level.
    snow : np.ndarray
        Isentropic density in [kg m^-2 K^-1] at the current (i.e. first) time level.
    uold : np.ndarray
        Horizontal velocity in [m s^-1] at the "previous" (i.e. zero-th) time level.
    qv0 : ``np.ndarray``, optional
        1-d array representing the initial vertical profile of mass fraction
        of water vapor in [g g^-1].
    qc0 : ``np.ndarray``, optional
        1-d array representing the initial vertical profile of mass fraction
        of cloud liquid water in [g g^-1].
    qr0 : ``np.ndarray``, optional
        1-d array representing the initial vertical profile of mass fraction
        of precipitation water in [g g^-1].
    qvold : ``np.ndarray``, optional
        Mass fraction of water vapor in [g g^-1] at the "previous" (i.e. zero-th)
        time level.
    qvnow : ``np.ndarray``, optional
        Mass fraction of water vapor in [g g^-1] at the current (i.e. first) time level.
    qcold : ``np.ndarray``, optional
        Mass fraction of cloud liquid water in [g g^-1] at the "previous" (i.e. zero-th)
        time level.
    qcnow : ``np.ndarray``, optional
        Mass fraction of cloud liquid water in [g g^-1] at the current (i.e. first)
        time level.
    qrold : ``np.ndarray``, optional
        Mass fraction of precipitation water in [g g^-1] at the "previous" (i.e. zero-th)
        time level.
    qrnow : ``np.ndarray``, optional
        Mass fraction of precipitation water in [g g^-1] at the current (i.e. first)
        time level.
    ncold : ``np.ndarray``, optional
        Number density of cloud liquid water in [g^-1] at the "previous" (i.e. zero-th)
        time level.
    ncnow : ``np.ndarray``, optional
        Number density of cloud liquid water in [g^-1] at the current (i.e. first)
        time level.
    nrold : ``np.ndarray``, optional
        Number density of precipitation water in [g^-1] at the "previous" (i.e. zero-th)
        time level.
    nrnow : ``np.ndarray``, optional
        Number density of precipitation water in [g^-1] at the current (i.e. first)
        time level.
    """
    # global dth
    if idbg == 1:
        print("Create initial profile ...\n")

    exn0 = np.zeros(nz + 1)
    z0 = np.zeros(nz + 1)
    mtg0 = np.zeros(nz)
    prs0 = np.zeros(nz + 1)
    exn0 = np.zeros(nz + 1)
    rh0 = np.zeros(nz)
    qv0 = np.zeros(nz)

    if imoist == 1:
        qc0 = np.zeros(nz)
        qr0 = np.zeros(nz)
    if imoist == 1 and imicrophys == 2:
        nc0 = np.zeros(nz)
        nr0 = np.zeros(nz)

    # Upstream profile for Brunt-Vaisalla frequency (unstaggered)
    # ------------------------------------------------------------
    bv0 = bv00 * np.ones(nz + 1)

    # Upstream profile of theta (staggered)
    # -----------------------------------------------------------
    th0 = th00 * np.ones(nz + 1) + dth * np.arange(0, nz + 1)

    # Upstream profile for Exner function and pressure (staggered)
    # -------------------------------------------------------------
    exn0[0] = exn00
    for k in range(1, nz + 1):
        exn0[k] = exn0[k - 1] - 16 * (g ** 2) * (th0[k] - th0[k - 1]) / (
            (bv0[k - 1] + bv0[k]) ** 2 * (th0[k - 1] + th0[k]) ** 2
        )

    prs0[:] = pref * (exn0[:] / cp) ** cpdr

    # Upstream profile for geometric height (staggered)
    # -------------------------------------------------------------
    z0[0] = z00
    for k in range(1, nz + 1):
        z0[k] = z0[k - 1] + 8 * g * (th0[k] - th0[k - 1]) / (
            (th0[k - 1] + th0[k]) * (bv0[k - 1] + bv0[k]) ** 2
        )

    # Upstream profile for Montgomery potential (unstaggered)
    # --------------------------------------------------------
    mtg0[0] = g * z0[0] + th00 * exn0[0] + dth * exn0[0] / 2.0
    for k in range(1, nz):
        mtg0[k] = mtg0[k - 1] + dth * exn0[k]

    # Upstream profile for isentropic density (unstaggered)
    # ------------------------------------------------------
    s0 = -1.0 / g * (prs0[1:] - prs0[0:-1]) / float(dth)

    # Upstream profile for velocity (unstaggered)
    # --------------------------------------------
    u0 = float(u00) * np.ones(nz)

    if ishear == 1:
        if idbg == 1:
            print("Using wind shear profile ...\n")
        # *** Exercise 3.3 Downslope windstorm ***
        # *** use indices k_shl, k_sht, and wind speeds u00_sh, u00
        #

        # # *** edit here ***
        k_low = np.arange(0, k_shl)
        k_between = np.arange(k_shl, k_sht)
        k_high = np.arange(k_sht, nz)

        u0[k_low] = u00_sh
        u0[k_between] = np.linspace(u00_sh, u00, len(np.arange(k_shl, k_sht)))
        u0[k_high] = u00
        #
        # *** Exercise 3.3 Downslope windstorm ***
    else:
        if idbg == 1:
            print("Using uniform wind profile ...\n")

    # Upstream profile for moisture (unstaggered)
    # -------------------------------------------

    if imoist == 1:
        # *** Exercise
        #  Initial Moisture profile ***
        # *** define new indices and create the profile ***
        # *** for rh0; then use function rrmixv1 to compute qv0 ***
        #

        # *** edit here ***
        kc = 12
        kw = 10
        k_between = np.arange(kc-kw, kc+kw)
        rh_max = 0.98
        rh0[k_between] = rh_max*np.cos(abs(k_between - kc)/kw*np.pi/2)**2

        k = np.arange(0, nz)
        qv0[k] = rrmixv1(0.5*(prs0[k]+prs0[k+1])/100, 0.5 *
                         (th0[k]/cp*exn0[k]+th0[k+1]/cp*exn0[k+1]), rh0[k], 2)

        #
        # *** Exercise 4.1 Initial Moisture profile ***

        # Upstream profile for number densities (unstaggered)
        # -------------------------------------------
        if imicrophys == 2:
            nc0 = np.zeros(nz)
            nr0 = np.zeros(nz)

    # Initial conditions for isentropic density (sigma), velocity u, and moisture qv
    # ---------------------------------------------------------------------

    sold = s0 * np.ones_like(sold, dtype=float)
    snow = s0 * np.ones_like(sold, dtype=float)
    mtg = mtg0 * np.ones_like(sold, dtype=float)
    mtgnew = mtg0 * np.ones_like(sold, dtype=float)
    uold = u0 * np.ones_like(uold, dtype=float)
    unow = u0 * np.ones_like(uold, dtype=float)

    if imoist == 1:
        # if imicrophys!=None:
        qvold = qv0 * np.ones_like(qvold, dtype=float)
        qvnow = qv0 * np.ones_like(qvold, dtype=float)

        # # Wave-like perturbation to check tracer advection
        # wave = np.sin(np.linspace(0, 2*np.pi, len(qvold)))**2
        # qvold *= wave[:, None]
        # qvnow *= wave[:, None]

        qcold = qc0 * np.ones_like(qcold, dtype=float)
        qcnow = qc0 * np.ones_like(qcold, dtype=float)
        qrold = qr0 * np.ones_like(qrold, dtype=float)
        qrnow = qr0 * np.ones_like(qrold, dtype=float)

        # droplet density for 2-moment scheme
        if imicrophys == 2:
            ncold = nc0 * np.ones_like(ncold, dtype=float)
            ncnow = nc0 * np.ones_like(ncold, dtype=float)
            nrold = nr0 * np.ones_like(nrold, dtype=float)
            nrnow = nr0 * np.ones_like(nrold, dtype=float)

    if imoist == 0:
        return th0, exn0, prs0, z0, mtg0, s0, u0, sold, snow, uold, unow, mtg, mtgnew
    else:
        if imicrophys == 0 or imicrophys == 1:
            return (
                th0,
                exn0,
                prs0,
                z0,
                mtg0,
                s0,
                u0,
                sold,
                snow,
                uold,
                unow,
                mtg,
                mtgnew,
                qv0,
                qc0,
                qr0,
                qvold,
                qvnow,
                qcold,
                qcnow,
                qrold,
                qrnow,
            )
        elif imicrophys == 2:
            return (
                th0,
                exn0,
                prs0,
                z0,
                mtg0,
                s0,
                u0,
                sold,
                snow,
                uold,
                unow,
                mtg,
                mtgnew,
                qv0,
                qc0,
                qr0,
                qvold,
                qvnow,
                qcold,
                qcnow,
                qrold,
                qrnow,
                ncold,
                ncnow,
                nrold,
                nrnow,
            )


# END OF MAKESETUP.PY
