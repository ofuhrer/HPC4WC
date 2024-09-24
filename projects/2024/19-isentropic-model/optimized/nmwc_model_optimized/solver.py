# -*- coding: utf-8 -*-
"""


***************************************************************
* 2-dimensional isentropic model                              *
* Christoph Schaer, Spring 2000                               *
* Several extensions, Juerg Schmidli, 2005                    *
* Converted to matlab, David Masson, 2009                     *
* Subfunction structure, bugfixes, and Kessler scheme added   *
* Wolfgang Langhans, 2009/2010                                *
* 2 moment scheme, bug fixes, vectorizations, addition of     *
* improvements_n by Mathias Hauser, Deniz Ural,                 *
* Maintenance Lukas Papritz, 2012 / 2013                      *
* Maintenance David Leutwyler 2014                            *
* Port of dry model to python 2.7, Martina Beer, 2014         *
* Finish python v1.0, David Leutwyler, Marina Dütsch 2015     *
* Maintenance Roman Brogli 2017                               *
* Ported to Python3, maintenance, Christian Zeman 2018/2019   *
***************************************************************

TODO:
    - Move definitions in own fuction -> remove cirecular dependency
    - Then re-write output using import to get rid of massive if/else trees

 -----------------------------------------------------------
 -------------------- MAIN PROGRAM: SOLVER -----------------
 -----------------------------------------------------------

"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np  # Scientific computing with Python
from time import time as tm  # Benchmarking tools
import sys

# import model functions
from nmwc_model_optimized.makesetup import maketopo, makeprofile
from nmwc_model_optimized.boundary import periodic, relax
from nmwc_model_optimized.parallel import exchange_borders_2d, gather_1d, gather_2d
from nmwc_model_optimized.prognostics import (
    prog_isendens,
    prog_velocity,
    prog_moisture,
    prog_numdens,
)
from nmwc_model_optimized.diagnostics import diag_montgomery, diag_pressure, diag_height
from nmwc_model_optimized.diffusion import horizontal_diffusion
from nmwc_model_optimized.output import makeoutput, write_output
from nmwc_model_optimized.microphysics import kessler, seifert

# import global namelist variables
from nmwc_model_optimized.namelist import (
    imoist as imoist_n,
    imicrophys as imicrophys_n,
    irelax as irelax_n,
    idthdt as idthdt_n,
    idbg as idbg_n,
    iprtcfl as iprtcfl_n,
    nts as nts_n,
    dt as dt_n,
    iiniout as iiniout_n,
    nout as nout_n,
    iout as iout_n,
    dx as dx_n,
    nx as nx_n,
    nb as nb_n,
    nz as nz_n,
    nz1 as nz1_n,
    nab as nab_n,
    diff as diff_n,
    diffabs as diffabs_n,
    topotim as topotim_n,
    itime as itime_n,
)

from mpi4py import MPI

comm = MPI.COMM_WORLD
rank_p = comm.Get_rank()
rank_size = comm.Get_size()

assert nx_n % rank_size == 0, "Number of grid points_n must be compatible with rank size"


def initialize_gathered_variables(nout: int):
    # region Define zero-filled gathered variables

    # topography
    topo_g = np.zeros((nx_n + 2 * nb_n, 1))

    # height in z-coordinates
    zhtold_g = np.zeros((nx_n + 2 * nb_n, nz1_n))
    zhtnow_g = np.zeros_like(zhtold_g)
    Z_g = np.zeros((nout, nz1_n, nx_n))  # auxilary field for output

    # horizontal velocity
    uold_g = np.zeros((nx_n + 1 + 2 * nb_n, nz_n))
    unow_g = np.zeros_like(uold_g)
    unew_g = np.zeros_like(uold_g)
    U_g = np.zeros((nout, nz_n, nx_n))  # auxilary field for output

    # isentropic density
    sold_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    snow_g = np.zeros_like(sold_g)
    snew_g = np.zeros_like(sold_g)
    S_g = np.zeros((nout, nz_n, nx_n))  # auxilary field for output

    # Montgomery potential
    mtg_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    mtgnew_g = np.zeros_like(mtg_g)

    # Exner function
    exn_g = np.zeros((nx_n + 2 * nb_n, nz1_n))

    # pressure
    prs_g = np.zeros((nx_n + 2 * nb_n, nz1_n))

    # output time vector
    T_g = np.arange(1, nout + 1)

    # precipitation
    prec_g = np.zeros(nx_n + 2 * nb_n)
    PREC_g = np.zeros((nout, nx_n))  # auxiliary field for output

    # accumulated precipitation
    tot_prec_g = np.zeros(nx_n + 2 * nb_n)
    TOT_PREC_g = np.zeros((nout, nx_n))  # auxiliary field for output

    # specific humidity
    qvold_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    qvnow_g = np.zeros_like(qvold_g)
    qvnew_g = np.zeros_like(qvold_g)
    QV_g = np.zeros((nout, nz_n, nx_n))  # auxiliary field for output

    # specific cloud water content
    qcold_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    qcnow_g = np.zeros_like(qcold_g)
    qcnew_g = np.zeros_like(qcold_g)
    QC_g = np.zeros((nout, nz_n, nx_n))  # auxiliary field for output

    # specific rain water content
    qrold_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    qrnow_g = np.zeros_like(qrold_g)
    qrnew_g = np.zeros_like(qrold_g)
    QR_g = np.zeros((nout, nz_n, nx_n))  # auxiliary field for output

    # cloud droplet number density
    ncold_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    ncnow_g = np.zeros_like(ncold_g)
    ncnew_g = np.zeros_like(ncold_g)
    NC_g = np.zeros((nout, nz_n, nx_n))  # auxiliary field for output

    # rain-droplet number density
    nrold_g = np.zeros((nx_n + 2 * nb_n, nz_n))
    nrnow_g = np.zeros_like(nrold_g)
    nrnew_g = np.zeros_like(nrold_g)
    NR_g = np.zeros((nout, nz_n, nx_n))  # auxiliary field for output

    # latent heating
    dthetadt_g = np.zeros((nx_n + 2 * nb_n, nz1_n))
    DTHETADT_g = np.zeros((nout, nz_n, nx_n))  # auxiliary field for output

    # Define fields at lateral boundaries
    # 1 denotes the left boundary
    # 2 denotes the right boundary
    # ----------------------------------------------------------------------------
    # topography
    tbnd1_g = 0.0
    tbnd2_g = 0.0

    # isentropic density
    sbnd1_g = np.zeros(nz_n)
    sbnd2_g = np.zeros(nz_n)

    # horizontal velocity
    ubnd1_g = np.zeros(nz_n)
    ubnd2_g = np.zeros(nz_n)

    # specific humidity
    qvbnd1_g = np.zeros(nz_n)
    qvbnd2_g = np.zeros(nz_n)

    # specific cloud water content
    qcbnd1_g = np.zeros(nz_n)
    qcbnd2_g = np.zeros(nz_n)

    # specific rain water content
    qrbnd1_g = np.zeros(nz_n)
    qrbnd2_g = np.zeros(nz_n)

    # latent heating
    dthetadtbnd1_g = np.zeros(nz1_n)
    dthetadtbnd2_g = np.zeros(nz1_n)

    # cloud droplet number density
    ncbnd1_g = np.zeros(nz_n)
    ncbnd2_g = np.zeros(nz_n)

    # rain droplet number density
    nrbnd1_g = np.zeros(nz_n)
    nrbnd2_g = np.zeros(nz_n)

    # variables later set by `makeprofile`
    th0_g = np.empty((nz_n + 1))
    exn0_g = np.empty((nz_n + 1))
    prs0_g = np.empty((nz_n + 1))
    z0_g = np.empty((nz_n + 1))
    mtg0_g = np.empty((nz_n))
    s0_g = np.empty((nz_n + 1))
    qv0_g = np.empty((nz_n))
    qc0_g = np.empty((nz_n))
    qr0_g = np.empty((nz_n))
    tau_g = np.empty((nz_n))

    # endregion

    # region Set initial conditions
    if idbg_n == 1:
        print("Setting initial conditions ...\n")

    if imoist_n == 0:
        # Dry atmosphere
        th0_g, exn0_g, prs0_g, z0_g, mtg0_g, s0_g, u0_g, sold_g, snow_g, uold_g, unow_g, mtg_g, mtgnew_g = makeprofile(
            sold_g, uold_g, mtg_g, mtgnew_g
        )
    elif imicrophys_n == 0 or imicrophys_n == 1:
        # moist atmosphere with kessler scheme
        th0_g, exn0_g, prs0_g, z0_g, mtg0_g, s0_g, u0_g, sold_g, snow_g, uold_g, unow_g, mtg_g, mtgnew_g, qv0_g, qc0_g, qr0_g, qvold_g, qvnow_g, qcold_g, qcnow_g, qrold_g, qrnow_g = makeprofile(
            sold_g,
            uold_g,
            qvold=qvold_g,
            qvnow=qvnow_g,
            qcold=qcold_g,
            qcnow=qcnow_g,
            qrold=qrold_g,
            qrnow=qrnow_g,
        )
    elif imicrophys_n == 2:
        # moist atmosphere with 2-moment scheme
        th0_g, exn0_g, prs0_g, z0_g, mtg0_g, s0_g, u0_g, sold_g, snow_g, uold_g, unow_g, mtg_g, mtgnew_g, qv0_g, qc0_g, qr0_g, qvold_g, qvnow_g, qcold_g, qcnow_g, qrold_g, qrnow_g, ncold_g, ncnow_g, nrold_g, nrnow_g = makeprofile(
            sold_g,
            uold_g,
            qvold=qvold_g,
            qvnow=qvnow_g,
            qcold=qcold_g,
            qcnow=qcnow_g,
            qrold=qrold_g,
            qrnow=qrnow_g,
            ncold=ncold_g,
            ncnow=ncnow_g,
            nrold=nrold_g,
            nrnow=nrnow_g,
        )

    # endregion

    # region Save boundary values for the lateral boundary relaxation
    if irelax_n == 1:
        if idbg_n == 1:
            print("Saving initial lateral boundary values ...\n")

        sbnd1_g[:] = snow_g[0, :]
        sbnd2_g[:] = snow_g[-1, :]

        ubnd1_g[:] = unow_g[0, :]
        ubnd2_g[:] = unow_g[-1, :]

        if imoist_n == 1:
            qvbnd1_g[:] = qvnow_g[0, :]
            qvbnd2_g[:] = qvnow_g[-1, :]

            qcbnd1_g[:] = qcnow_g[0, :]
            qcbnd2_g[:] = qcnow_g[-1, :]

        if imicrophys_n != 0:
            qrbnd1_g[:] = qrnow_g[0, :]
            qrbnd2_g[:] = qrnow_g[-1, :]

        # 2-moment microphysics scheme
        if imicrophys_n == 2:
            ncbnd1_g[:] = ncnow_g[0, :]
            ncbnd2_g[:] = ncnow_g[-1, :]

            nrbnd1_g[:] = nrnow_g[0, :]
            nrbnd2_g[:] = nrnow_g[-1, :]

        if idthdt_n == 1:
            dthetadtbnd1_g[:] = dthetadt_g[0, :]
            dthetadtbnd2_g[:] = dthetadt_g[-1, :]

    # endregion

    # region Make topography
    # ----------------
    topo_g = maketopo(topo_g, nx_n + 2 * nb_n)

    # switch between boundary relaxation / periodic boundary conditions
    # ------------------------------------------------------------------
    if irelax_n == 1:  # boundary relaxation
        if idbg_n == 1:
            print("Relax topography ...\n")

        # save lateral boundary values of topography
        tbnd1_g = topo_g[0]
        tbnd2_g = topo_g[-1]

        # relax topography
        topo_g = relax(topo_g, nx_n, nb_n, tbnd1_g, tbnd2_g)
    else:
        if idbg_n == 1:
            print("Periodic topography ...\n")

        # make topography periodic
        topo_g = periodic(topo_g, nx_n, nb_n)

    # endregion

    # region Height-dependent settings

    # calculate geometric height (staggered)
    zhtnow_g = diag_height(
        prs0_g[np.newaxis, :], exn0_g[np.newaxis,
                                      :], zhtnow_g, th0_g, topo_g, 0.0
    )

    # Height-dependent diffusion coefficient
    # --------------------------------------
    tau_g = diff_n * np.ones(nz_n)

    # *** Exercise 3.1 height-dependent diffusion coefficient ***
    tau_g = diff_n + (diffabs_n - diff_n) * \
        np.sin(np.pi / 2 * (np.arange(nz_n) - nz_n + nab_n - 1) / nab_n)**2
    tau_g[0:nz_n-nab_n] = diff_n
    # *** Exercise 3.1 height-dependent diffusion coefficient ***

    # endregion

    # region Output initial fields
    its_out_g = -1  # output index
    if iiniout_n == 1 and imoist_n == 0:
        its_out_g, Z_g, U_g, S_g, T_g = makeoutput(
            unow_g, snow_g, zhtnow_g, its_out_g, 0, Z_g, U_g, S_g, T_g)
    elif iiniout_n == 1 and imoist_n == 1:
        if imicrophys_n == 0 or imicrophys_n == 1:
            if idthdt_n == 0:
                its_out_g, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g = makeoutput(
                    unow_g,
                    snow_g,
                    zhtnow_g,
                    its_out_g,
                    0,
                    Z_g,
                    U_g,
                    S_g,
                    T_g,
                    qvnow=qvnow_g,
                    qcnow=qcnow_g,
                    qrnow=qrnow_g,
                    tot_prec=tot_prec_g,
                    prec=prec_g,
                    QV=QV_g,
                    QC=QC_g,
                    QR=QR_g,
                    TOT_PREC=TOT_PREC_g,
                    PREC=PREC_g,
                )
            elif idthdt_n == 1:
                its_out_g, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g, DTHETADT_g = makeoutput(
                    unow_g,
                    snow_g,
                    zhtnow_g,
                    its_out_g,
                    0,
                    Z_g,
                    U_g,
                    S_g,
                    T_g,
                    qvnow=qvnow_g,
                    qcnow=qcnow_g,
                    qrnow=qrnow_g,
                    tot_prec=tot_prec_g,
                    prec=prec_g,
                    QV=QV_g,
                    QC=QC_g,
                    QR=QR_g,
                    TOT_PREC=TOT_PREC_g,
                    PREC=PREC_g,
                    dt_nhetadt_n=dthetadt_g,
                    dt_nHETAdt_n=DTHETADT_g,
                )
        elif imicrophys_n == 2:
            if idthdt_n == 0:
                its_out_g, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g, NC_g, NR_g = makeoutput(
                    unow_g,
                    snow_g,
                    zhtnow_g,
                    its_out_g,
                    0,
                    Z_g,
                    U_g,
                    S_g,
                    T_g,
                    qvnow=qvnow_g,
                    qcnow=qcnow_g,
                    qrnow=qrnow_g,
                    tot_prec=tot_prec_g,
                    prec=prec_g,
                    nrnow=nrnow_g,
                    ncnow=ncnow_g,
                    QV=QV_g,
                    QC=QC_g,
                    QR=QR_g,
                    TOT_PREC=TOT_PREC_g,
                    PREC=PREC_g,
                    NC=NC_g,
                    NR=NR_g,
                )
            elif idthdt_n == 1:
                its_out_g, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g, NC_g, NR_g, DTHETADT_g = makeoutput(
                    unow_g,
                    snow_g,
                    zhtnow_g,
                    its_out_g,
                    0,
                    Z_g,
                    U_g,
                    S_g,
                    T_g,
                    qvnow=qvnow_g,
                    qcnow=qcnow_g,
                    qrnow=qrnow_g,
                    tot_prec=tot_prec_g,
                    prec=prec_g,
                    nrnow=nrnow_g,
                    ncnow=ncnow_g,
                    QV=QV_g,
                    QC=QC_g,
                    QR=QR_g,
                    TOT_PREC=TOT_PREC_g,
                    PREC=PREC_g,
                    NC=NC_g,
                    NR=NR_g,
                    dt_nhetadt_n=dthetadt_g,
                    dt_nHETAdt_n=DTHETADT_g,
                )

    # endregion

    # region Return relevant variables
    return (
        sold_g, snow_g, snew_g, S_g,
        uold_g, unow_g, unew_g, U_g,
        qvold_g, qvnow_g, qvnew_g, QV_g,
        qcold_g, qcnow_g, qcnew_g, QC_g,
        qrold_g, qrnow_g, qrnew_g, QR_g,
        ncold_g, ncnow_g, ncnew_g, NC_g,
        nrold_g, nrnow_g, nrnew_g, NR_g,
        dthetadt_g, DTHETADT_g,
        mtg_g, tau_g, prs0_g, prs_g, T_g,
        prec_g, PREC_g, tot_prec_g, TOT_PREC_g,
        topo_g, zhtold_g, zhtnow_g, Z_g, th0_g,
        dthetadtbnd1_g, dthetadtbnd2_g,
        exn_g, qvbnd1_g, qvbnd2_g, qcbnd1_g, qcbnd2_g, qrbnd1_g, qrbnd2_g,
        ncbnd1_g, ncbnd2_g, nrbnd1_g, nrbnd2_g, sbnd1_g, sbnd2_g, ubnd1_g, ubnd2_g,
        its_out_g
    )
    # endregion


def run_optimized():
    # region Setup
    # Print the full precision
    np.set_printoptions(threshold=sys.maxsize)

    # Define physical fields
    nx_p = nx_n // rank_size
    borderless_slice = slice(nb_n, -nb_n)

    # increase number of output steps by 1 for initial profile
    nout = nout_n
    if iiniout_n == 1:
        nout += 1
    # endregion

    # region Declare process-specific variables
    sold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    snow_p = np.empty_like(sold_p)
    snew_p = np.empty_like(sold_p)
    uold_p = np.empty((nx_p + 1 + 2 * nb_n, nz_n))
    unow_p = np.empty_like(uold_p)
    qvold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    qvnow_p = np.empty_like(qvold_p)
    qvnew_p = np.empty_like(qvold_p)
    qcold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    qcnow_p = np.empty_like(qcold_p)
    qcnew_p = np.empty_like(qcold_p)
    qrold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    qrnow_p = np.empty_like(qrold_p)
    qrnew_p = np.empty_like(qrold_p)
    ncold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    ncnow_p = np.empty_like(ncold_p)
    ncnew_p = np.empty_like(ncold_p)
    nrold_p = np.empty((nx_p + 2 * nb_n, nz_n))
    nrnow_p = np.empty_like(nrold_p)
    nrnew_p = np.empty_like(nrold_p)
    mtg_p = np.empty((nx_p + 2 * nb_n, nz_n))
    tau_p = np.empty((nz_n))
    prs0_p = np.zeros((nz_n + 1))
    prs_p = np.empty((nx_p + 2 * nb_n, nz1_n))
    topo_p = np.empty((nx_p + 2 * nb_n, 1))
    zhtold_p = np.empty((nx_p + 2 * nb_n, nz1_n))
    zhtnow_p = np.empty((nx_p + 2 * nb_n, nz1_n))
    th0_p = np.empty((nz_n + 1))
    tot_prec_p = np.empty((nx_p + 2 * nb_n))
    prec_p = np.empty(nx_p + 2 * nb_n)
    exn_p = np.empty((nx_p + 2 * nb_n, nz1_n))

    if rank_p == rank_size - 1:
        qvbnd1_p = np.empty(nz_n)
        qvbnd2_p = np.empty(nz_n)
        qcbnd1_p = np.empty(nz_n)
        qcbnd2_p = np.empty(nz_n)
        qrbnd1_p = np.empty(nz_n)
        qrbnd2_p = np.empty(nz_n)
        ncbnd1_p = np.empty(nz_n)
        ncbnd2_p = np.empty(nz_n)
        nrbnd1_p = np.empty(nz_n)
        nrbnd2_p = np.empty(nz_n)
        sbnd1_p = np.empty(nz_n)
        sbnd2_p = np.empty(nz_n)
        ubnd1_p = np.empty(nz_n)
        ubnd2_p = np.empty(nz_n)

    # Define variable names of main process for ease of use
    sold_g, snow_g, snew_g, S_g = None, None, None, None
    uold_g, unow_g, unew_g, U_g = None, None, None, None
    qvold_g, qvnow_g, qvnew_g, QV_g = None, None, None, None
    qcold_g, qcnow_g, qcnew_g, QC_g = None, None, None, None
    qrold_g, qrnow_g, qrnew_g, QR_g = None, None, None, None
    ncold_g, ncnow_g, ncnew_g, NC_g = None, None, None, None
    nrold_g, nrnow_g, nrnew_g, NR_g = None, None, None, None
    dthetadt_g, DTHETADT_g = None, None
    mtg_g, tau_g, prs0_g, prs_g, T_g = None, None, None, None, None
    prec_g, PREC_g, tot_prec_g, TOT_PREC_g = None, None, None, None
    topo_g, zhtold_g, zhtnow_g, Z_g, th0_g = None, None, None, None, None
    dthetadtbnd1_g, dthetadtbnd2_g = None, None
    exn_g = None
    qvbnd1_g, qvbnd2_g, qcbnd1_g, qcbnd2_g, qrbnd1_g, qrbnd2_g  = None, None, None, None, None, None
    ncbnd1_g, ncbnd2_g, nrbnd1_g, nrbnd2_g, sbnd1_g, sbnd2_g, ubnd1_g, ubnd2_g = None, None, None, None, None, None, None, None
    its_out_g = None
    # endregion

    if rank_p == 0:
        # region Distribute slices to processes
        (
            sold_g, snow_g, snew_g, S_g,
            uold_g, unow_g, unew_g, U_g,
            qvold_g, qvnow_g, qvnew_g, QV_g,
            qcold_g, qcnow_g, qcnew_g, QC_g,
            qrold_g, qrnow_g, qrnew_g, QR_g,
            ncold_g, ncnow_g, ncnew_g, NC_g,
            nrold_g, nrnow_g, nrnew_g, NR_g,
            dthetadt_g, DTHETADT_g,
            mtg_g, tau_g, prs0_g, prs_g, T_g,
            prec_g, PREC_g, tot_prec_g, TOT_PREC_g,
            topo_g, zhtold_g, zhtnow_g, Z_g, th0_g, 
            dthetadtbnd1_g, dthetadtbnd2_g, 
            exn_g, qvbnd1_g, qvbnd2_g, qcbnd1_g, qcbnd2_g, qrbnd1_g, qrbnd2_g,
            ncbnd1_g, ncbnd2_g, nrbnd1_g, nrbnd2_g, sbnd1_g, sbnd2_g, ubnd1_g, ubnd2_g,
            its_out_g
        ) = initialize_gathered_variables(nout)

        # For each process, send relevant slices
        for i in range(1, rank_size):
            start_index = i * nx_p
            end_index = (i + 1) * nx_p + 2 * nb_n
            rank_slice = slice(start_index, end_index)
            rank_slice_staggered = slice(start_index, end_index + 1)

            comm.Send(sold_g[rank_slice, :], dest=i, tag=i * 1000 + 0)
            comm.Send(snow_g[rank_slice, :], dest=i, tag=i * 1000 + 1)
            comm.Send(snew_g[rank_slice, :], dest=i, tag=i * 1000 + 2)

            comm.Send(uold_g[rank_slice_staggered, :],
                      dest=i, tag=i * 1000 + 3)
            comm.Send(unow_g[rank_slice_staggered, :],
                      dest=i, tag=i * 1000 + 4)

            comm.Send(qvold_g[rank_slice, :], dest=i, tag=i * 1000 + 5)
            comm.Send(qvnow_g[rank_slice, :], dest=i, tag=i * 1000 + 6)
            comm.Send(qvnew_g[rank_slice, :], dest=i, tag=i * 1000 + 7)

            comm.Send(qcold_g[rank_slice, :], dest=i, tag=i * 1000 + 8)
            comm.Send(qcnow_g[rank_slice, :], dest=i, tag=i * 1000 + 9)
            comm.Send(qcnew_g[rank_slice, :], dest=i, tag=i * 1000 + 10)

            comm.Send(qrold_g[rank_slice, :], dest=i, tag=i * 1000 + 11)
            comm.Send(qrnow_g[rank_slice, :], dest=i, tag=i * 1000 + 12)
            comm.Send(qrnew_g[rank_slice, :], dest=i, tag=i * 1000 + 13)

            if imoist_n == 1 and imicrophys_n == 2:
                comm.Send(ncold_g[rank_slice, :], dest=i, tag=i * 1000 + 14)
                comm.Send(ncnow_g[rank_slice, :], dest=i, tag=i * 1000 + 15)
                comm.Send(ncnew_g[rank_slice, :], dest=i, tag=i * 1000 + 16)

                comm.Send(nrold_g[rank_slice, :], dest=i, tag=i * 1000 + 17)
                comm.Send(nrnow_g[rank_slice, :], dest=i, tag=i * 1000 + 18)
                comm.Send(nrnew_g[rank_slice, :], dest=i, tag=i * 1000 + 19)

            comm.Send(mtg_g[rank_slice, :], dest=i, tag=i * 1000 + 20)
            comm.Send(tau_g, dest=i, tag=i * 1000 + 21)
            comm.Send(prs0_g, dest=i, tag=i * 1000 + 22)
            comm.Send(prs_g[rank_slice, :], dest=i, tag=i * 1000 + 23)
            comm.Send(topo_g[rank_slice, :], dest=i, tag=i * 1000 + 24)
            comm.Send(zhtold_g[rank_slice, :], dest=i, tag=i * 1000 + 25)
            comm.Send(zhtnow_g[rank_slice, :], dest=i, tag=i * 1000 + 26)
            comm.Send(th0_g, dest=i, tag=i * 1000 + 27)
            comm.Send(tot_prec_g[rank_slice], dest=i, tag=i * 1000 + 28)
            comm.Send(prec_g[rank_slice], dest=i, tag=i * 1000 + 29)
            comm.Send(exn_p[rank_slice], dest=i, tag=i * 1000 + 30)
        
        comm.Send(qvbnd1_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 31)
        comm.Send(qvbnd2_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 32)
        comm.Send(qcbnd1_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 33)
        comm.Send(qcbnd2_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 34)
        comm.Send(qrbnd1_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 35)
        comm.Send(qrbnd2_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 36)
        comm.Send(ncbnd1_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 37)
        comm.Send(ncbnd2_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 38)
        comm.Send(nrbnd1_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 39)
        comm.Send(nrbnd2_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 40)
        comm.Send(sbnd1_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 41)
        comm.Send(sbnd2_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 42)
        comm.Send(ubnd1_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 43)
        comm.Send(ubnd2_g, dest=rank_size - 1, tag=(rank_size - 1) * 1000 + 44)
        # endregion

        # region Set process variables for process with rank 0
        start_index = 0
        end_index = nx_p + 2 * nb_n
        rank_slice = slice(start_index, end_index)
        rank_slice_staggered = slice(start_index, end_index + 1)

        sold_p = sold_g[rank_slice, :]
        snow_p = snow_g[rank_slice, :]
        snew_p = snew_g[rank_slice, :]

        uold_p = uold_g[rank_slice_staggered, :]
        unow_p = unow_g[rank_slice_staggered, :]

        qvold_p = qvold_g[rank_slice, :]
        qvnow_p = qvnow_g[rank_slice, :]
        qvnew_p = qvnew_g[rank_slice, :]

        qcold_p = qcold_g[rank_slice, :]
        qcnow_p = qcnow_g[rank_slice, :]
        qcnew_p = qcnew_g[rank_slice, :]

        qrold_p = qrold_g[rank_slice, :]
        qrnow_p = qrnow_g[rank_slice, :]
        qrnew_p = qrnew_g[rank_slice, :]

        if imoist_n == 1 and imicrophys_n == 2:
            ncold_p = ncold_g[rank_slice, :]
            ncnow_p = ncnow_g[rank_slice, :]
            ncnew_p = ncnew_g[rank_slice, :]

            nrold_p = nrold_g[rank_slice, :]
            nrnow_p = nrnow_g[rank_slice, :]
            nrnew_p = nrnew_g[rank_slice, :]

        mtg_p = mtg_g[rank_slice, :]
        tau_p = tau_g
        prs0_p = prs0_g
        prs_p = prs_g[rank_slice, :]
        topo_p = topo_g[rank_slice, :]
        zhtold_p = zhtold_g[rank_slice, :]
        zhtnow_p = zhtnow_g[rank_slice, :]
        th0_p = th0_g
        tot_prec_p = tot_prec_g[rank_slice]
        prec_p = prec_g[rank_slice]
        exn_p = exn_g[rank_slice]
        qvbnd1_p = qvbnd1_g
        qvbnd2_p = qvbnd2_g
        qcbnd1_p = qcbnd1_g
        qcbnd2_p = qcbnd2_g
        qrbnd1_p = qrbnd1_g
        qrbnd2_p = qrbnd2_g
        ncbnd1_p = ncbnd1_g
        ncbnd2_p = ncbnd2_g
        nrbnd1_p = nrbnd1_g
        nrbnd2_p = nrbnd2_g
        sbnd1_p = sbnd1_g
        sbnd2_p = sbnd2_g
        ubnd1_p = ubnd1_g
        ubnd2_p = ubnd2_g
        # endregion
    else:
        # region Receive process-specific variable values
        comm.Recv(sold_p, source=0, tag=rank_p * 1000 + 0)
        comm.Recv(snow_p, source=0, tag=rank_p * 1000 + 1)
        comm.Recv(snew_p, source=0, tag=rank_p * 1000 + 2)

        comm.Recv(uold_p, source=0, tag=rank_p * 1000 + 3)
        comm.Recv(unow_p, source=0, tag=rank_p * 1000 + 4)

        comm.Recv(qvold_p, source=0, tag=rank_p * 1000 + 5)
        comm.Recv(qvnow_p, source=0, tag=rank_p * 1000 + 6)
        comm.Recv(qvnew_p, source=0, tag=rank_p * 1000 + 7)

        comm.Recv(qcold_p, source=0, tag=rank_p * 1000 + 8)
        comm.Recv(qcnow_p, source=0, tag=rank_p * 1000 + 9)
        comm.Recv(qcnew_p, source=0, tag=rank_p * 1000 + 10)

        comm.Recv(qrold_p, source=0, tag=rank_p * 1000 + 11)
        comm.Recv(qrnow_p, source=0, tag=rank_p * 1000 + 12)
        comm.Recv(qrnew_p, source=0, tag=rank_p * 1000 + 13)

        if imoist_n == 1 and imicrophys_n == 2:
            comm.Recv(ncold_p, source=0, tag=rank_p * 1000 + 14)
            comm.Recv(ncnow_p, source=0, tag=rank_p * 1000 + 15)
            comm.Recv(ncnew_p, source=0, tag=rank_p * 1000 + 16)

            comm.Recv(nrold_p, source=0, tag=rank_p * 1000 + 17)
            comm.Recv(nrnow_p, source=0, tag=rank_p * 1000 + 18)
            comm.Recv(nrnew_p, source=0, tag=rank_p * 1000 + 19)

        comm.Recv(mtg_p, source=0, tag=rank_p * 1000 + 20)
        comm.Recv(tau_p, source=0, tag=rank_p * 1000 + 21)
        comm.Recv(prs0_p, source=0, tag=rank_p * 1000 + 22)
        comm.Recv(prs_p, source=0, tag=rank_p * 1000 + 23)
        comm.Recv(topo_p, source=0, tag=rank_p * 1000 + 24)

        comm.Recv(zhtold_p, source=0, tag=rank_p * 1000 + 25)
        comm.Recv(zhtnow_p, source=0, tag=rank_p * 1000 + 26)
        comm.Recv(th0_p, source=0, tag=rank_p * 1000 + 27)
        comm.Recv(tot_prec_p, source=0, tag=rank_p * 1000 + 28)
        comm.Recv(prec_p, source=0, tag=rank_p * 1000 + 29)
        comm.Recv(exn_p, source=0, tag=rank_p * 1000 + 30)

        if rank_p == rank_size - 1:
            comm.Recv(qvbnd1_p, source=0, tag=rank_p * 1000 + 31)
            comm.Recv(qvbnd2_p, source=0, tag=rank_p * 1000 + 32)
            comm.Recv(qcbnd1_p, source=0, tag=rank_p * 1000 + 33)
            comm.Recv(qcbnd2_p, source=0, tag=rank_p * 1000 + 34)
            comm.Recv(qrbnd1_p, source=0, tag=rank_p * 1000 + 35)
            comm.Recv(qrbnd2_p, source=0, tag=rank_p * 1000 + 36)
            comm.Recv(ncbnd1_p, source=0, tag=rank_p * 1000 + 37)
            comm.Recv(ncbnd2_p, source=0, tag=rank_p * 1000 + 38)
            comm.Recv(nrbnd1_p, source=0, tag=rank_p * 1000 + 39)
            comm.Recv(nrbnd2_p, source=0, tag=rank_p * 1000 + 40)
            comm.Recv(sbnd1_p, source=0, tag=rank_p * 1000 + 41)
            comm.Recv(sbnd2_p, source=0, tag=rank_p * 1000 + 42)
            comm.Recv(ubnd1_p, source=0, tag=rank_p * 1000 + 43)
            comm.Recv(ubnd2_p, source=0, tag=rank_p * 1000 + 44)

        # endregion

    # region Loop over all time steps
    if idbg_n == 1 and rank_p == 0:
        print("Starting time loop ...\n")

    t0_p = tm()
    for its_p in range(1, int(nts_n + 1)):
        # region Calculate and log time
        time_p = its_p * dt_n

        if itime_n == 1:
            if idbg_n == 1 or idbg_n == 0:
                print("========================================================\n")
                print("Working on timestep %g; time = %g s; process = %g\n" %
                      (its_p, time_p, rank_p))
                print("========================================================\n")
        # endregion

        # region Special treatment of first time step
        # initially increase height of topography only slowly
        topofact_p: float = min(1.0, float(time_p) / topotim_n)
        if its_p == 1:
            dtdx_p: float = dt_n / dx_n / 2.0
            dthetadt_p = None
            if imoist_n == 1 and idthdt_n == 1:
                # No latent heating for first time-step
                dthetadt_p = np.zeros((nx_p + 2 * nb_n, nz1_n))
            if idbg_n == 1:
                print("Using Euler forward step for 1. step ...\n")
        else:
            dtdx_p: float = dt_n / dx_n
        # endregion

        # region Time step for isentropic mass density
        snew_p = prog_isendens(sold_p, snow_p, unow_p,
                               dtdx_p, dthetadt=dthetadt_p, nx_p=nx_p)
        # endregion

        # region Time step for moisture scalars
        if imoist_n == 1:
            if idbg_n == 1:
                print("Add function call to prog_moisture")
            qvnew_p, qcnew_p, qrnew_p = prog_moisture(
                unow_p, qvold_p, qcold_p, qrold_p, qvnow_p, qcnow_p, qrnow_p, dtdx_p, dthetadt=dthetadt_p, nx=nx_p)

            if imicrophys_n == 2:
                ncnew_p, nrnew_p = prog_numdens(
                    unow_p, ncold_p, nrold_p, ncnow_p, nrnow_p, dtdx_p, dthetadt=dthetadt_p, nx=nx_p)
        # endregion

        # region Time step for momentum
        unew_p = prog_velocity(uold_p, unow_p, mtg_p,
                               dtdx_p, dthetadt_p, nx_p)
        # endregion

        # region Relaxation of prognostic fields
        if irelax_n == 1:
            if idbg_n == 1:
                print("Relaxing prognostic fields ...\n")
            snew = relax(snew, nx_p, nb_n, sbnd1_p, sbnd2_p)
            unew = relax(unew, nx_p + 1, nb_n, ubnd1_p, ubnd2_p)
            if imoist_n == 1:
                qvnew = relax(qvnew, nx_p, nb_n, qvbnd1_p, qvbnd2_p)
                qcnew = relax(qcnew, nx_p, nb_n, qcbnd1_p, qcbnd2_p)
                qrnew = relax(qrnew, nx_p, nb_n, qrbnd1_p, qrbnd2_p)

            # 2-moment scheme
            if imoist_n == 1 and imicrophys_n == 2:
                ncnew = relax(ncnew, nx_p, nb_n, ncbnd1_p, ncbnd2_p)
                nrnew = relax(nrnew, nx_p, nb_n, nrbnd1_p, nrbnd2_p)
        # endregion

        # region Exchange borders
        unew_p = exchange_borders_2d(unew_p, 100002)
        snew_p = exchange_borders_2d(snew_p, 100003)
        qvnew_p = exchange_borders_2d(qvnew_p, 100007)
        qcnew_p = exchange_borders_2d(qcnew_p, 100008)
        qrnew_p = exchange_borders_2d(qrnew_p, 100009)

        if imoist_n == 1 and imicrophys_n == 2:
            ncnew_p = exchange_borders_2d(ncnew_p, tag=100010)
            nrnew_p = exchange_borders_2d(nrnew_p, tag=100011)
        # endregion

        # region Diffusion and gravity wave absorber
        if imoist_n == 0:
            unew_p, snew_p = horizontal_diffusion(
                tau_p, unew_p, snew_p, nx=nx_p)
        else:
            if imicrophys_n == 2:
                unew_p, snew_p, qvnew_p, qcnew_p, qrnew_p, ncnew_p, nrnew_p = horizontal_diffusion(
                    tau_p,
                    unew_p,
                    snew_p,
                    qvnew=qvnew_p,
                    qcnew=qcnew_p,
                    qrnew=qrnew_p,
                    ncnew=ncnew_p,
                    nrnew=nrnew_p, 
                    nx=nx_p
                )
            else:
                unew_p, snew_p, qvnew_p, qcnew_p, qrnew_p = horizontal_diffusion(
                    tau_p, unew_p, snew_p, qvnew=qvnew_p, qcnew=qcnew_p, qrnew=qrnew_p, nx=nx_p
                )
        # endregion

        # region Diagnostic computation of pressure
        prs_p = diag_pressure(prs0_p, prs_p, snew_p)
        # endregion

        # region Calculate Exner function and Montgomery potential
        exn_p, mtg_p = diag_montgomery(prs_p, mtg_p, th0_p, topo_p, topofact_p)
        # endregion

        # region Calculation of geometric height (staggered)
        # needed for output and microphysics schemes
        zhtold_p[...] = zhtnow_p[...]
        zhtnow_p = diag_height(prs_p, exn_p, zhtnow_p,
                               th0_p, topo_p, topofact_p)
        # endregion

        if imoist_n == 1:
            # region Moisture: Clipping of negative values
            if idbg_n == 1:
                print("Implement moisture clipping")
            qvnew_p[qvnew_p < 0] = 0
            qcnew_p[qcnew_p < 0] = 0
            qrnew_p[qrnew_p < 0] = 0

            if imicrophys_n == 2:
                ncnew_p[ncnew_p < 0] = 0
                nrnew_p[nrnew_p < 0] = 0
            # endregion

        if imoist_n == 1 and imicrophys_n == 1:
            # region Kessler scheme ***
            if idbg_n == 1:
                print("Add function call to Kessler microphysics")
            lheat_p, qvnew_p, qcnew_p, qrnew_p, prec_p, tot_prec_p = kessler(
                snew_p, qvnew_p, qcnew_p, qrnew_p, prs_p, exn_p, zhtnow_p, th0_p, prec_p, tot_prec_p)
            # endregion
        elif imoist_n == 1 and imicrophys_n == 2:
            # region Two Moment Scheme
            if idbg_n == 1:
                print("Add function call to two moment microphysics")
            lheat_p, qvnew_p, qcnew_p, qrnew_p, tot_prec_p, prec_p, ncnew_p, nrnew_p = seifert(
                unew_p, th0_p, prs_p, snew_p, qvnew_p, qcnew_p, qrnew_p, exn_p, zhtold_p, zhtnow_p, tot_prec_p, prec_p, ncnew_p, nrnew_p, dthetadt_p)  # TODO
            # endregion

        if imoist_n == 1 and imicrophys_n > 0:
            if idthdt_n == 1:
                # Stagger lheat to model levels and compute tendency
                k = np.arange(1, nz_n)
                if imicrophys_n == 1:
                    dthetadt_p[:, k] = topofact_p * 0.5 * \
                        (lheat_p[:, k - 1] + lheat_p[:, k]) / dt_n
                else:
                    dthetadt_p[:, k] = topofact_p * 0.5 * \
                        (lheat_p[:, k - 1] + lheat_p[:, k]) / (2.0 * dt_n)

                # force dt_nhetadt_n to zeros at the bottom and at the top
                dthetadt_p[:, 0] = 0.0
                dthetadt_p[:, -1] = 0.0

                # periodic lateral boundary conditions
                # ----------------------------
                if irelax_n == 0:
                    dthetadt_p = periodic(dthetadt_p, nx_n, nb_n)
                else:
                    # Relax latent heat fields
                    # ----------------------------
                    dthetadt_p = relax(dthetadt_p, nx_n, nb_n,
                                       dthetadtbnd1_g, dthetadtbnd2_g)
            else:
                dthetadt_p = np.zeros((nx_n + 2 * nb_n, nz1_n))

        if idbg_n == 1:
            print("Preparing next time step ...\n")

        # region Exchange isentropic mass density and velocity
        if imicrophys_n == 2:
            ncold_p = ncnow_p
            ncnow_p = ncnew_p

            nrold_p = nrnow_p
            nrnow_p = nrnew_p

        sold_p = snow_p
        snow_p = snew_p

        uold_p = unow_p
        unow_p = unew_p

        if imoist_n == 1:
            qvold_p = qvnow_p
            qvnow_p = qvnew_p

            qcold_p = qcnow_p
            qcnow_p = qcnew_p

            qrold_p = qrnow_p
            qrnow_p = qrnew_p
            if idbg_n == 1:
                print("exchange moisture variables")

            if imicrophys_n == 2:
                if idbg_n == 1:
                    print("exchange number densitiy variables")
        # endregion

        # region Check maximum cfl criterion
        if iprtcfl_n == 1:
            u_max_p = np.amax(np.abs(unow_p))
            cfl_max_p = u_max_p * dtdx_p
            print("============================================================\n")
            print("CFL MAX: %g U MAX: %g m/s \n" % (cfl_max_p, u_max_p))
            if cfl_max_p > 1:
                print("!!! WARNING: CFL larger than 1 !!!\n")
            elif np.isnan(cfl_max_p):
                print("!!! MODEL ABORT: NaN values !!!\n")
            print("============================================================\n")
        # endregion

        # region Output every 'iout_n'-th time step
        # ---------------------------------
        if np.mod(its_p, iout_n) == 0:
            snow_g = gather_2d(snow_p, snow_g, nx_p, borderless_slice)
            zhtnow_g = gather_2d(zhtnow_p, zhtnow_g, nx_p, borderless_slice)

            # region The velocity is staggered, treat gathering separately
            if rank_p == 0:
                unow_buf = np.empty((rank_size * nx_p, nz_n))
            else:
                unow_buf = None

            comm.Gather(unow_p[nb_n + 1:-nb_n, :], unow_buf, root=0)

            if rank_p == 0:
                if irelax_n == 0:
                    unow_g[:nb_n, :] = unow_buf[-nb_n:, :]
                    unow_g[-nb_n:, :] = unow_buf[:nb_n, :]
                unow_g[2, :] = unow_p[2, :]
                unow_g[nb_n+1:-nb_n, :] = unow_buf[:, :]
            # endregion

            if imoist_n == 0:
                if rank_p == 0:
                    its_out_g, Z_g, U_g, S_g, T_g = makeoutput(
                        unow_g, snow_g, zhtnow_g, its_out_g, its_p, Z_g, U_g, S_g, T_g
                    )
            elif imoist_n == 1:
                qvnow_g = gather_2d(qvnow_p, qvnow_g, nx_p, borderless_slice)
                qvnow_g = gather_2d(qcnow_p, qvnow_g, nx_p, borderless_slice)
                qvnow_g = gather_2d(qrnow_p, qvnow_g, nx_p, borderless_slice)
                tot_prec_g = gather_1d(
                    tot_prec_p, tot_prec_g, nx_p, borderless_slice)
                prec_g = gather_1d(prec_p, prec_g, nx_p, borderless_slice)

                if imicrophys_n == 0 or imicrophys_n == 1:
                    if idthdt_n == 0:
                        if rank_p == 0:
                            its_out_g, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g = makeoutput(
                                unow_g,
                                snow_g,
                                zhtnow_g,
                                its_out_g,
                                its_p,
                                Z_g,
                                U_g,
                                S_g,
                                T_g,
                                qvnow=qvnow_g,
                                qcnow=qcnow_g,
                                qrnow=qrnow_g,
                                tot_prec=tot_prec_g,
                                prec=prec_g,
                                QV=QV_g,
                                QC=QC_g,
                                QR=QR_g,
                                TOT_PREC=TOT_PREC_g,
                                PREC=PREC_g,
                            )
                    elif idthdt_n == 1:
                        dthetadt_g = gather_1d(dthetadt_p, dthetadt_g,
                                               nx_p, borderless_slice)

                        if rank_p == 0:
                            its_out_g, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g, DTHETADT_g = makeoutput(
                                unow_g,
                                snow_g,
                                zhtnow_g,
                                its_out_g,
                                its_p,
                                Z_g,
                                U_g,
                                S_g,
                                T_g,
                                qvnow=qvnow_g,
                                qcnow=qcnow_g,
                                qrnow=qrnow_g,
                                tot_prec=tot_prec_g,
                                PREC=PREC_g,
                                prec=prec_g,
                                QV=QV_g,
                                QC=QC_g,
                                QR=QR_g,
                                TOT_PREC=TOT_PREC_g,
                                dthetadt=dthetadt_g,
                                DTHETADT=DTHETADT_g,
                            )
                if imicrophys_n == 2:
                    nrnow_g = gather_2d(
                        nrnow_p, nrnow_g, nx_p, borderless_slice)
                    ncnow_g = gather_2d(
                        ncnow_p, ncnow_g, nx_p, borderless_slice)

                    if idthdt_n == 0:
                        if rank_p == 0:
                            its_out_g, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g, NR_g, NC_g = makeoutput(
                                unow_g,
                                snow_g,
                                zhtnow_g,
                                its_out_g,
                                its_p,
                                Z_g,
                                U_g,
                                S_g,
                                T_g,
                                qvnow=qvnow_g,
                                qcnow=qcnow_g,
                                qrnow=qrnow_g,
                                tot_prec=tot_prec_g,
                                prec=prec_g,
                                nrnow=nrnow_g,
                                ncnow=ncnow_g,
                                QV=QV_g,
                                QC=QC_g,
                                QR=QR_g,
                                TOT_PREC=TOT_PREC_g,
                                PREC=PREC_g,
                                NR=NR_g,
                                NC=NC_g,
                            )
                    if idthdt_n == 1:
                        if rank_p == 0:
                            its_out_g, Z_g, U_g, S_g, T_g, QC_g, QV_g, QR_g, TOT_PREC_g, PREC_g, NR_g, NC_g, DTHETADT_g = makeoutput(
                                unow_g,
                                snow_g,
                                zhtnow_g,
                                its_out_g,
                                its_p,
                                Z_g,
                                U_g,
                                S_g,
                                T_g,
                                qvnow=qvnow_g,
                                qcnow=qcnow_g,
                                qrnow=qrnow_g,
                                tot_prec=tot_prec_g,
                                prec=prec_g,
                                nrnow=nrnow_g,
                                ncnow=ncnow_g,
                                QV=QV_g,
                                QC=QC_g,
                                QR=QR_g,
                                TOT_PREC=TOT_PREC_g,
                                PREC=PREC_g,
                                NR=NR_g,
                                NC=NC_g,
                                dthetadt=dthetadt_g,
                                DTHETADT=DTHETADT_g,
                            )
        # endregion

        if idbg_n == 1:
            print("\n\n")

    # -----------------------------------------------------------------------------
    # ########## END OF TIME LOOP ################################################
    if idbg_n > 0:
        print("\nEnd of time loop ...\n")

    tt = tm()
    print("Elapsed computation time on rank %g without writing: %g s\n" % (rank_p, tt - t0_p))

    # endregion

    # region Write output
    if rank_p == 0:
        print("Start wrtiting output.\n")
        if imoist_n == 0:
            write_output(nout, Z_g, U_g, S_g, T_g)
        elif imicrophys_n == 0 or imicrophys_n == 1:
            if idthdt_n == 1:
                write_output(
                    nout,
                    Z_g,
                    U_g,
                    S_g,
                    T_g,
                    QV=QV_g,
                    QC=QC_g,
                    QR=QR_g,
                    PREC=PREC_g,
                    TOT_PREC=TOT_PREC_g,
                    DTHETADT=DTHETADT_g,
                )
            else:
                write_output(
                    nout, Z_g, U_g, S_g, T_g, QV=QV_g, QC=QC_g, QR=QR_g, PREC=PREC_g, TOT_PREC=TOT_PREC_g
                )
        elif imicrophys_n == 2:
            if idthdt_n == 1:
                write_output(
                    nout,
                    Z_g,
                    U_g,
                    S_g,
                    T_g,
                    QV=QV_g,
                    QC=QC_g,
                    QR=QR_g,
                    PREC=PREC_g,
                    TOT_PREC=TOT_PREC_g,
                    NR=NR_g,
                    NC=NC_g,
                    DTHETADT=DTHETADT_g,
                )
            else:
                write_output(
                    nout,
                    Z_g,
                    U_g,
                    S_g,
                    T_g,
                    QV=QV_g,
                    QC=QC_g,
                    QR=QR_g,
                    PREC=PREC_g,
                    TOT_PREC=TOT_PREC_g,
                    NR=NR_g,
                    NC=NC_g,
                )
    # endregion

    # region Benchmarking
    t1 = tm()
    if itime_n == 1:
        print("Total elapsed computation time: %g s\n" % (t1 - t0_p))
    # endregion


if __name__ == '__main__':
    run_optimized()

# END OF SOLVER.PY
