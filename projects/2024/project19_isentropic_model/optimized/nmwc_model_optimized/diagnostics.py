# -*- coding: utf-8 -*-
import numpy as np

from nmwc_model_optimized.namelist import idbg, nz, g, dth, cp, pref, rdcp  # global variables


def diag_montgomery(prs, mtg, th0, topo, topofact):
    """ Diagnostic computation of the Montgomery potential.

    Calculate the Exner function and the Montgomery potential.
    Based on diag_montgomery.m from the full isentropic model in MATLAB, 2014.

    Parameters
    ----------
    prs : np.ndarray
        Vertically staggered pressure field in [Pa].
    mtg : np.ndarray
        Array to be filled with the Montgomery potential field in [m^2 s^-2].
    th0 : np.ndarray
        1-d array storing the values of potential temperature in [K] on the
        interface vertical levels.
    topo : np.ndarray
        Array collecting the terrain height in [m].
    topofact : float
        Multiplicative factor for the topography.

    Returns
    -------
    exn : np.ndarray
        The computed Exner function in [J kg^-1 K^-1].
    mtg : np.ndarray
        The computed Montgomery potential in [m^2 s^-2].

    Examples
    --------
    >>> nx = 100
    >>> nb = 3
    >>> nz = 60
    >>> prs = np.random.rand(nx + 2 * nb, nz+1)
    >>> mtg = np.zeros(nx + 2 * nb, nz)
    >>> th0 = np.linspace(340, 280, nz + 1)
    >>> topo = np.random.rand(nx + 2 * nb, 1)
    >>> exn, mtg = diag_montgomery(prs, mtg, th0, topo, 1.0)
    """
    if idbg == 1:
        print("Diagnostic step: Exner function and Montgomery potential ...\n")

    # *** Exercise 2.2 Diagnostic computation of Montgomery ***
    # *** Calculate Exner function and Montgomery potential ***
    #

    # Computation of Exner function
    # *** edit here ***
    exn = cp*(prs/pref)**rdcp
    # Add lower boundary condition at height mtg[:,0]
    # *** edit here ***
    mtg[:, 0] = topo.squeeze()*topofact*g + th0[0]*exn[:, 0] + dth/2*exn[:, 0]

    # Integration loop upwards
    # *** edit here ***
    for k in range(1, nz):
        mtg[:, k] = mtg[:, k-1] + dth*exn[:, k]
    #
    # *** Exercise 2.2 Diagnostic computation  ***

    return exn, mtg


def diag_pressure(prs0, prs, snew):
    """  Diagnostic computation of pressure.

    Diagnostic computation of pressure with upper boundary condition
    and integration downwards.
    Based on diag_pressure.m from the full isentropic model in MATLAB, 2014.

    Parameters
    ----------
    prs0 : np.ndarray
        1-d array storing the staggered vertical profile of pressure in [Pa].
    prs : np.ndarray
        Array to be filled with the vertically staggered pressure field in [Pa].
    snew : np.ndarray
        The isentropic density in [kg m^-2 K^-1] used to retrieve the pressure.

    Returns
    -------
    np.ndarray :
        The computed pressure field in [Pa].

    Examples
    --------
    >>> nx = 100
    >>> nb = 3
    >>> nz = 60
    >>> prs = np.random.rand(nx + 2 * nb, nz + 1)
    >>> prs0 = np.random.rand(nz + 1)
    >>> snew = np.random.rand(nx + 2 * nb, nz)
    >>> prs = diag_pressure(prs0, prs, snew)
    """
    if idbg == 1:
        print("Diagnostic step: Pressure ...\n")

    # *** Exercise 2.2 Diagnostic computation of pressure ***
    # *** Diagnostic computation of pressure ***
    # *** (upper boundary condition and integration downwards) ***
    #

    # Upper boundary condition
    # *** edit here ***
    prs[:, -1] = prs0[-1]
    # Integration loop downwards
    # *** edit here ***

    for k in range(0, nz)[::-1]:
        prs[:, k] = prs[:, k+1] + dth*snew[:, k]*g
    #
    # *** Exercise 2.2 Diagnostic computation of pressure ***

    return prs


def diag_height(prs, exn, zht, th0, topo, topofact):
    """ Diagnostic computation of the geometric height of the isentropic surfaces.

    Parameters
    ----------
    prs : np.ndarray
        Vertically staggered pressure field in [Pa].
    exn : np.ndarray
        Vertically staggered Exner function in [J kg^-1 K^-1].
    zht : np.ndarray
        Array to be filled with the vertically staggered geometric height in [m].
    th0 : np.ndarray
        1-d array storing the values of potential temperature in [K] on the
        interface vertical levels.
    topo : np.ndarray
        Array collecting the terrain height in [m].
    topofact : float
        Multiplicative factor for the topography.

    Returns
    -------
    zht : np.ndarray
        The computed geometric height in [m].
    """
    zht[:, 0] = topofact * topo[:, 0]
    for k in range(1, nz + 1):
        zht[:, k] = zht[:, k - 1] - rdcp * (
            th0[k - 1] * exn[:, k - 1] + th0[k] * exn[:, k]
        ) * (prs[:, k] - prs[:, k - 1]) / (g * (prs[:, k - 1] + prs[:, k]))
    return zht


def diag_density_and_temperature(s, exn, zht, th0):
    """ Diagnostic computation of the density and temperature of air.

    Parameters
    ----------
    s : np.ndarray
        Isentropic density in [J m^-2 K^-1].
    exn : np.ndarray
        Vertically staggered Exner function in [J kg^-1 K^-1].
    zht : np.ndarray
        Vertically staggered geometric height in [m].
    th0 : np.ndarray
        1-d array storing the values of potential temperature in [K] on the
        interface vertical levels.

    Returns
    -------
    rho : np.ndarray
        The computed air density in [kg m^-3].
    temp : np.ndarray
        The computed air temperature in [K].
    """
    k = np.arange(0, nz)
    th = th0[np.newaxis, :]

    rho = np.zeros_like(s)
    rho[:, k] = s * (th[:, k + 1] - th[:, k]) / (zht[:, k + 1] - zht[:, k])

    temp = np.zeros_like(s)
    temp[:, k] = 0.5 * (th[:, k] * exn[:, k] +
                        th[:, k + 1] * exn[:, k + 1]) / cp

    return rho, temp


# END OF DIAGNOSTICS.PY
