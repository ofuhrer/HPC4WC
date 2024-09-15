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
* improvements by Mathias Hauser, Deniz Ural,                 *
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

import numpy as np  # Scientific computing with Python
from time import time as tm  # Benchmarking tools
import sys

# import model functions
from nmwc_model.makesetup import maketopo, makeprofile
from nmwc_model.boundary import periodic, relax
from nmwc_model.prognostics import (
    prog_isendens,
    prog_velocity,
    prog_moisture,
    prog_numdens,
)
from nmwc_model.diagnostics import diag_montgomery, diag_pressure, diag_height
from nmwc_model.diffusion import horizontal_diffusion
from nmwc_model.output import makeoutput, write_output
from nmwc_model.microphysics import kessler, seifert

# import global namelist variables
from nmwc_model.namelist import (
    imoist,
    imicrophys,
    irelax,
    idthdt,
    idbg,
    iprtcfl,
    nts,
    dt,
    iiniout,
    nout as _nout,
    iout,
    dx,
    nx,
    nx1,
    nb,
    nxb,
    nxb1,
    nz,
    nz1,
    nab,
    rdcp,
    g,
    diff,
    diffabs,
    topotim,
    itime,
)


def run_unoptimized():
    # Print the full precision
    # DL: REMOVE FOR STUDENT VERSION
    np.set_printoptions(threshold=sys.maxsize)

    # increase number of output steps by 1 for initial profile
    nout = _nout
    if iiniout == 1:
        nout += 1

    # Define physical fields
    # -------------------------

    # topography
    topo = np.zeros((nxb, 1))

    # height in z-coordinates
    zhtold = np.zeros((nxb, nz1))
    zhtnow = np.zeros_like(zhtold)
    Z = np.zeros((nout, nz1, nx))  # auxilary field for output

    # horizontal velocity
    uold = np.zeros((nxb1, nz))
    unow = np.zeros_like(uold)
    unew = np.zeros_like(uold)
    U = np.zeros((nout, nz, nx))  # auxilary field for output

    # isentropic density
    sold = np.zeros((nxb, nz))
    snow = np.zeros_like(sold)
    snew = np.zeros_like(sold)
    S = np.zeros((nout, nz, nx))  # auxilary field for output

    # Montgomery potential
    mtg = np.zeros((nxb, nz))
    mtgnew = np.zeros_like(mtg)

    # Exner function
    exn = np.zeros((nxb, nz1))

    # pressure
    prs = np.zeros((nxb, nz1))

    # output time vector
    T = np.arange(1, nout + 1)

    if imoist == 1:
        # precipitation
        prec = np.zeros(nxb)
        PREC = np.zeros((nout, nx))  #  auxiliary field for output

        # accumulated precipitation
        tot_prec = np.zeros(nxb)
        TOT_PREC = np.zeros((nout, nx))  #  auxiliary field for output

        # specific humidity
        qvold = np.zeros((nxb, nz))
        qvnow = np.zeros_like(qvold)
        qvnew = np.zeros_like(qvold)
        QV = np.zeros((nout, nz, nx))  # auxiliary field for output

        # specific cloud water content
        qcold = np.zeros((nxb, nz))
        qcnow = np.zeros_like(qcold)
        qcnew = np.zeros_like(qcold)
        QC = np.zeros((nout, nz, nx))  # auxiliary field for output

        # specific rain water content
        qrold = np.zeros((nxb, nz))
        qrnow = np.zeros_like(qrold)
        qrnew = np.zeros_like(qrold)
        QR = np.zeros((nout, nz, nx))  # auxiliary field for output

        if imicrophys == 2:
            # cloud droplet number density
            ncold = np.zeros((nxb, nz))
            ncnow = np.zeros_like(ncold)
            ncnew = np.zeros_like(ncold)
            NC = np.zeros((nout, nz, nx))  # auxiliary field for output

            # rain-droplet number density
            nrold = np.zeros((nxb, nz))
            nrnow = np.zeros_like(nrold)
            nrnew = np.zeros_like(nrold)
            NR = np.zeros((nout, nz, nx))  # auxiliary field for output

        if idthdt == 1:
            # latent heating
            dthetadt = np.zeros((nxb, nz1))
            DTHETADT = np.zeros((nout, nz, nx))  # auxiliary field for output

    # Define fields at lateral boundaries
    # 1 denotes the left boundary
    # 2 denotes the right boundary
    # ----------------------------------------------------------------------------
    # topography
    tbnd1 = 0.0
    tbnd2 = 0.0

    # isentropic density
    sbnd1 = np.zeros(nz)
    sbnd2 = np.zeros(nz)

    # horizontal velocity
    ubnd1 = np.zeros(nz)
    ubnd2 = np.zeros(nz)

    if imoist == 1:
        # specific humidity
        qvbnd1 = np.zeros(nz)
        qvbnd2 = np.zeros(nz)

        # specific cloud water content
        qcbnd1 = np.zeros(nz)
        qcbnd2 = np.zeros(nz)

        # specific rain water content
        qrbnd1 = np.zeros(nz)
        qrbnd2 = np.zeros(nz)

    if idthdt == 1:
        # latent heating
        dthetadtbnd1 = np.zeros(nz1)
        dthetadtbnd2 = np.zeros(nz1)

    if imicrophys == 2:
        # cloud droplet number density
        ncbnd1 = np.zeros(nz)
        ncbnd2 = np.zeros(nz)

        # rain droplet number density
        nrbnd1 = np.zeros(nz)
        nrbnd2 = np.zeros(nz)

    # Set initial conditions
    # -----------------------------------------------------------------------------
    if idbg == 1:
        print("Setting initial conditions ...\n")

    if imoist == 0:
        # Dry atmosphere
        th0, exn0, prs0, z0, mtg0, s0, u0, sold, snow, uold, unow, mtg, mtgnew = makeprofile(
            sold, uold, mtg, mtgnew
        )
    else:
        if imicrophys == 0 or imicrophys == 1:
            # moist atmosphere with kessler scheme
            [
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
            ] = makeprofile(
                sold,
                uold,
                qvold=qvold,
                qvnow=qvnow,
                qcold=qcold,
                qcnow=qcnow,
                qrold=qrold,
                qrnow=qrnow,
            )
        elif imicrophys == 2:
            # moist atmosphere with 2-moment scheme
            [
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
            ] = makeprofile(
                sold,
                uold,
                qvold=qvold,
                qvnow=qvnow,
                qcold=qcold,
                qcnow=qcnow,
                qrold=qrold,
                qrnow=qrnow,
                ncold=ncold,
                ncnow=ncnow,
                nrold=nrold,
                nrnow=nrnow,
            )

    # Save boundary values for the lateral boundary relaxation
    if irelax == 1:
        if idbg == 1:
            print("Saving initial lateral boundary values ...\n")

        sbnd1[:] = snow[0, :]
        sbnd2[:] = snow[-1, :]

        ubnd1[:] = unow[0, :]
        ubnd2[:] = unow[-1, :]

        if imoist == 1:
            qvbnd1[:] = qvnow[0, :]
            qvbnd2[:] = qvnow[-1, :]

            qcbnd1[:] = qcnow[0, :]
            qcbnd2[:] = qcnow[-1, :]

        if imicrophys != 0:
            qrbnd1[:] = qrnow[0, :]
            qrbnd2[:] = qrnow[-1, :]

        # 2-moment microphysics scheme
        if imicrophys == 2:
            k = np.arange(0, nz)
            ncbnd1[:] = ncnow[0, :]
            ncbnd2[:] = ncnow[-1, :]

            nrbnd1[:] = nrnow[0, :]
            nrbnd2[:] = nrnow[-1, :]

        if idthdt == 1:
            dthetadtbnd1[:] = dthetadt[0, :]
            dthetadtbnd2[:] = dthetadt[-1, :]

    # Make topography
    # ----------------
    topo = maketopo(topo, nxb)

    # switch between boundary relaxation / periodic boundary conditions
    # ------------------------------------------------------------------
    if irelax == 1:  # boundary relaxation
        if idbg == 1:
            print("Relax topography ...\n")

        # save lateral boundary values of topography
        tbnd1 = topo[0]
        tbnd2 = topo[-1]

        # relax topography
        topo = relax(topo, nx, nb, tbnd1, tbnd2)
    else:
        if idbg == 1:
            print("Periodic topography ...\n")

        # make topography periodic
        topo = periodic(topo, nx, nb)

    # calculate geometric height (staggered)
    zhtnow = diag_height(
        prs0[np.newaxis, :], exn0[np.newaxis, :], zhtnow, th0, topo, 0.0
    )

    # Height-dependent diffusion coefficient
    # --------------------------------------
    tau = diff * np.ones(nz)

    # *** Exercise 3.1 height-dependent diffusion coefficient ***
    # *** edit here ***

    k = np.arange(nz)
    tau = diff + (diffabs -diff)*np.sin(np.pi/2*(k-nz+nab-1)/nab)**2
    tau[0:nz-nab] = diff
    # *** Exercise 3.1 height-dependent diffusion coefficient ***

    # output initial fields
    its_out = -1  # output index
    if iiniout == 1 and imoist == 0:
        its_out, Z, U, S, T = makeoutput(unow, snow, zhtnow, its_out, 0, Z, U, S, T)
    elif iiniout == 1 and imoist == 1:
        if imicrophys == 0 or imicrophys == 1:
            if idthdt == 0:
                [its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC] = makeoutput(
                    unow,
                    snow,
                    zhtnow,
                    its_out,
                    0,
                    Z,
                    U,
                    S,
                    T,
                    qvnow=qvnow,
                    qcnow=qcnow,
                    qrnow=qrnow,
                    tot_prec=tot_prec,
                    prec=prec,
                    QV=QV,
                    QC=QC,
                    QR=QR,
                    TOT_PREC=TOT_PREC,
                    PREC=PREC,
                )
            elif idthdt == 1:
                [
                    its_out,
                    Z,
                    U,
                    S,
                    T,
                    QC,
                    QV,
                    QR,
                    TOT_PREC,
                    PREC,
                    DTHETADT,
                ] = makeoutput(
                    unow,
                    snow,
                    zhtnow,
                    its_out,
                    0,
                    Z,
                    U,
                    S,
                    T,
                    qvnow=qvnow,
                    qcnow=qcnow,
                    qrnow=qrnow,
                    tot_prec=tot_prec,
                    prec=prec,
                    QV=QV,
                    QC=QC,
                    QR=QR,
                    TOT_PREC=TOT_PREC,
                    PREC=PREC,
                    dthetadt=dthetadt,
                    DTHETADT=DTHETADT,
                )
        elif imicrophys == 2:
            if idthdt == 0:
                [its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC, NC, NR] = makeoutput(
                    unow,
                    snow,
                    zhtnow,
                    its_out,
                    0,
                    Z,
                    U,
                    S,
                    T,
                    qvnow=qvnow,
                    qcnow=qcnow,
                    qrnow=qrnow,
                    tot_prec=tot_prec,
                    prec=prec,
                    nrnow=nrnow,
                    ncnow=ncnow,
                    QV=QV,
                    QC=QC,
                    QR=QR,
                    TOT_PREC=TOT_PREC,
                    PREC=PREC,
                    NC=NC,
                    NR=NR,
                )
            elif idthdt == 1:
                [
                    its_out,
                    Z,
                    U,
                    S,
                    T,
                    QC,
                    QV,
                    QR,
                    TOT_PREC,
                    PREC,
                    NC,
                    NR,
                    DTHETADT,
                ] = makeoutput(
                    unow,
                    snow,
                    zhtnow,
                    its_out,
                    0,
                    Z,
                    U,
                    S,
                    T,
                    qvnow=qvnow,
                    qcnow=qcnow,
                    qrnow=qrnow,
                    tot_prec=tot_prec,
                    prec=prec,
                    nrnow=nrnow,
                    ncnow=ncnow,
                    QV=QV,
                    QC=QC,
                    QR=QR,
                    TOT_PREC=TOT_PREC,
                    PREC=PREC,
                    NC=NC,
                    NR=NR,
                    dthetadt=dthetadt,
                    DTHETADT=DTHETADT,
                )

    # ########## TIME LOOP #######################################################
    # ----------------------------------------------------------------------------
    # Loop over all time steps
    # ----------------------------------------------------------------------------
    if idbg == 1:
        print("Starting time loop ...\n")

    t0 = tm()
    for its in range(1, int(nts + 1)):
        # calculate time
        time = its * dt

        if itime == 1:
            if idbg == 1 or idbg == 0:
                print("========================================================\n")
                print("Working on timestep %g; time = %g s\n" % (its, time))
                print("========================================================\n")

        # initially increase height of topography only slowly
        topofact = min(1.0, float(time) / topotim)

        # Special treatment of first time step
        # -------------------------------------------------------------------------
        if its == 1:
            dtdx = dt / dx / 2.0
            dthetadt = None
            if imoist == 1 and idthdt == 1:
                # No latent heating for first time-step
                dthetadt = np.zeros((nxb, nz1))
            if idbg == 1:
                print("Using Euler forward step for 1. step ...\n")
        else:
            dtdx = dt / dx

        # *** Exercise 2.1 isentropic mass density ***
        # *** time step for isentropic mass density ***
        #

        # *** edit here ***

        snew = prog_isendens(sold, snow, unow, dtdx, dthetadt = dthetadt)
        #
        # *** Exercise 2.1 isentropic mass density ***

        # *** Exercise 4.1 / 5.1 moisture ***
        # *** time step for moisture scalars ***
        # *** edit here ***
        #

        if imoist == 1:
            if idbg == 1:
                print("Add function call to prog_moisture")
            qvnew,qcnew,qrnew = prog_moisture(unow, qvold, qcold, qrold, qvnow, qcnow, qrnow, dtdx, dthetadt=dthetadt)

            if imicrophys == 2:
                ncnew, nrnew = prog_numdens(unow, ncold, nrold, ncnow, nrnow, dtdx, dthetadt=dthetadt)
            
        #
        # *** Exercise 4.1 / 5.1 moisture scalars ***

        # *** Exercise 2.1 velocity ***
        # *** time step for momentum ***
        #

        # *** edit here ***
        unew = prog_velocity(uold, unow, mtg, dtdx, dthetadt = dthetadt)
        #
        # *** Exercise 2.1 velocity ***

        # exchange boundaries if periodic
        # -------------------------------------------------------------------------
        if irelax == 0:
            snew = periodic(snew, nx, nb)
            unew = periodic(unew, nx, nb)

            if imoist == 1:
                qvnew = periodic(qvnew, nx, nb)
                qcnew = periodic(qcnew, nx, nb)
                qrnew = periodic(qrnew, nx, nb)

            # 2-moment scheme
            if imoist == 1 and imicrophys == 2:
                ncnew = periodic(ncnew, nx, nb)
                nrnew = periodic(nrnew, nx, nb)

        # relaxation of prognostic fields
        # -------------------------------------------------------------------------
        if irelax == 1:
            if idbg == 1:
                print("Relaxing prognostic fields ...\n")
            snew = relax(snew, nx, nb, sbnd1, sbnd2)
            unew = relax(unew, nx1, nb, ubnd1, ubnd2)
            if imoist == 1:

                qvnew = relax(qvnew, nx, nb, qvbnd1, qvbnd2)
                qcnew = relax(qcnew, nx, nb, qcbnd1, qcbnd2)
                qrnew = relax(qrnew, nx, nb, qrbnd1, qrbnd2)

            # 2-moment scheme
            if imoist == 1 and imicrophys == 2:
                ncnew = relax(ncnew, nx, nb, ncbnd1, ncbnd2)
                nrnew = relax(nrnew, nx, nb, nrbnd1, nrbnd2)

        # Diffusion and gravity wave absorber
        # ------------------------------------

        if imoist == 0:
            [unew, snew] = horizontal_diffusion(tau, unew, snew)
        else:
            if imicrophys == 2:
                [unew, snew, qvnew, qcnew, qrnew, ncnew, nrnew] = horizontal_diffusion(
                    tau,
                    unew,
                    snew,
                    qvnew=qvnew,
                    qcnew=qcnew,
                    qrnew=qrnew,
                    ncnew=ncnew,
                    nrnew=nrnew,
                )
            else:
                [unew, snew, qvnew, qcnew, qrnew] = horizontal_diffusion(
                    tau, unew, snew, qvnew=qvnew, qcnew=qcnew, qrnew=qrnew
                )

        # *** Exercise 2.2 Diagnostic computation of pressure ***
        # *** Diagnostic computation of pressure ***
        #

        # *** edit here ***
        prs = diag_pressure(prs0, prs, snew)
        #
        # *** Exercise 2.2 Diagnostic computation of pressure ***

        # *** Exercise 2.2 Diagnostic computation of Montgomery ***
        # *** Calculate Exner function and Montgomery potential ***
        #

        # *** edit here ***
        exn, mtg = diag_montgomery(prs, mtg, th0, topo, topofact)
        #
        # *** Exercise 2.2 Diagnostic computation of Montgomery ***

        # Calculation of geometric height (staggered)
        # needed for output and microphysics schemes
        # ---------------------------------
        zhtold[...] = zhtnow[...]
        zhtnow = diag_height(prs, exn, zhtnow, th0, topo, topofact)

        if imoist == 1:
            # *** Exercise 4.1 Moisture ***
            # *** Clipping of negative values ***
            # *** edit here ***
            #

            if idbg == 1:
                print("Implement moisture clipping")
            qvnew[qvnew < 0] = 0
            qcnew[qcnew < 0] = 0
            qrnew[qrnew < 0] = 0

            if imicrophys == 2: 
                ncnew[ncnew < 0] = 0
                nrnew[nrnew < 0] = 0

            #
            # *** Exercise 4.1 Moisture ***

        if imoist == 1 and imicrophys == 1:
            # *** Exercise 4.2 Kessler ***
            # *** Kessler scheme ***
            # *** edit here ***
            #

            if idbg == 1:
                print("Add function call to Kessler microphysics")
            [lheat,qvnew,qcnew,qrnew,prec,prec_tot] = kessler(snew, qvnew, qcnew, qrnew, prs, exn, zhtnow, th0, prec, tot_prec)

            #
            # *** Exercise 4.2 Kessler ***
        elif imoist == 1 and imicrophys == 2:
            # *** Exercise 5.1 Two Moment Scheme ***
            # *** Two Moment Scheme ***
            # *** edit here ***
            #

            if idbg == 1:
                print("Add function call to two moment microphysics")
            [lheat,qvnew,qcnew,qrnew,tot_prec,prec,ncnew,nrnew] = seifert(unew,th0,prs,snew,qvnew,qcnew,qrnew,exn,zhtold,zhtnow,tot_prec,prec,ncnew,nrnew,dthetadt)
            #
            # *** Exercise 5.1 Two Moment Scheme ***

        if imoist == 1 and imicrophys > 0:
            if idthdt == 1:
                # Stagger lheat to model levels and compute tendency
                k = np.arange(1, nz)
                if imicrophys == 1:
                    dthetadt[:, k] = topofact * 0.5 * (lheat[:, k - 1] + lheat[:, k]) / dt
                else:
                    dthetadt[:, k] = topofact * 0.5 * (lheat[:, k - 1] + lheat[:, k]) / (2.0 * dt)

                # force dthetadt to zeros at the bottom and at the top
                dthetadt[:, 0] = 0.0
                dthetadt[:, -1] = 0.0

                # periodic lateral boundary conditions
                # ----------------------------
                if irelax == 0:
                    dthetadt = periodic(dthetadt, nx, nb)
                else:
                    # Relax latent heat fields
                    # ----------------------------
                    dthetadt = relax(dthetadt, nx, nb, dthetadtbnd1, dthetadtbnd2)
            else:
                dthetadt = np.zeros((nxb, nz1))

        if idbg == 1:
            print("Preparing next time step ...\n")

        # *** Exercise 2.1 / 4.1 / 5.1 ***
        # *** exchange isentropic mass density and velocity ***
        # *** (later also qv,qc,qr,nc,nr) ***
        # *** edit here ***
        if imicrophys == 2:
            ncold = ncnow
            ncnow = ncnew

            nrold = nrnow
            nrnow = nrnew

        sold = snow
        snow = snew

        uold = unow
        unow = unew

        if imoist == 1:
            qvold = qvnow
            qvnow = qvnew

            qcold = qcnow
            qcnow = qcnew

            qrold = qrnow
            qrnow = qrnew
            if idbg == 1:
                print("exchange moisture variables")

            if imicrophys == 2:
                if idbg == 1:
                    print("exchange number densitiy variables")

        #
        # *** Exercise 2.1 / 4.1 / 5.1 ***

        # check maximum cfl criterion
        # ---------------------------------
        if iprtcfl == 1:
            u_max = np.amax(np.abs(unow))
            cfl_max = u_max * dtdx
            print("============================================================\n")
            print("CFL MAX: %g U MAX: %g m/s \n" % (cfl_max, u_max))
            if cfl_max > 1:
                print("!!! WARNING: CFL larger than 1 !!!\n")
            elif np.isnan(cfl_max):
                print("!!! MODEL ABORT: NaN values !!!\n")
            print("============================================================\n")

        # output every 'iout'-th time step
        # ---------------------------------
        if np.mod(its, iout) == 0:
            if imoist == 0:
                its_out, Z, U, S, T = makeoutput(
                    unow, snow, zhtnow, its_out, its, Z, U, S, T
                )
            elif imoist == 1:
                if imicrophys == 0 or imicrophys == 1:
                    if idthdt == 0:
                        its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC = makeoutput(
                            unow,
                            snow,
                            zhtnow,
                            its_out,
                            its,
                            Z,
                            U,
                            S,
                            T,
                            qvnow=qvnow,
                            qcnow=qcnow,
                            qrnow=qrnow,
                            tot_prec=tot_prec,
                            prec=prec,
                            QV=QV,
                            QC=QC,
                            QR=QR,
                            TOT_PREC=TOT_PREC,
                            PREC=PREC,
                        )
                    elif idthdt == 1:
                        its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC, DTHETADT = makeoutput(
                            unow,
                            snow,
                            zhtnow,
                            its_out,
                            its,
                            Z,
                            U,
                            S,
                            T,
                            qvnow=qvnow,
                            qcnow=qcnow,
                            qrnow=qrnow,
                            tot_prec=tot_prec,
                            PREC=PREC,
                            prec=prec,
                            QV=QV,
                            QC=QC,
                            QR=QR,
                            TOT_PREC=TOT_PREC,
                            dthetadt=dthetadt,
                            DTHETADT=DTHETADT,
                        )
                if imicrophys == 2:
                    if idthdt == 0:
                        its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC, NR, NC = makeoutput(
                            unow,
                            snow,
                            zhtnow,
                            its_out,
                            its,
                            Z,
                            U,
                            S,
                            T,
                            qvnow=qvnow,
                            qcnow=qcnow,
                            qrnow=qrnow,
                            tot_prec=tot_prec,
                            prec=prec,
                            nrnow=nrnow,
                            ncnow=ncnow,
                            QV=QV,
                            QC=QC,
                            QR=QR,
                            TOT_PREC=TOT_PREC,
                            PREC=PREC,
                            NR=NR,
                            NC=NC,
                        )
                    if idthdt == 1:
                        its_out, Z, U, S, T, QC, QV, QR, TOT_PREC, PREC, NR, NC, DTHETADT = makeoutput(
                            unow,
                            snow,
                            zhtnow,
                            its_out,
                            its,
                            Z,
                            U,
                            S,
                            T,
                            qvnow=qvnow,
                            qcnow=qcnow,
                            qrnow=qrnow,
                            tot_prec=tot_prec,
                            prec=prec,
                            nrnow=nrnow,
                            ncnow=ncnow,
                            QV=QV,
                            QC=QC,
                            QR=QR,
                            TOT_PREC=TOT_PREC,
                            PREC=PREC,
                            NR=NR,
                            NC=NC,
                            dthetadt=dthetadt,
                            DTHETADT=DTHETADT,
                        )
        if idbg == 1:
            print("\n\n")

    # -----------------------------------------------------------------------------
    # ########## END OF TIME LOOP ################################################
    if idbg > 0:
        print("\nEnd of time loop ...\n")

    tt = tm()
    print("Elapsed computation time without writing: %g s\n" % (tt - t0))

    # Write output
    # ---------------------------------
    print("Start wrtiting output.\n")
    if imoist == 0:
        write_output(nout, Z, U, S, T)
    elif imicrophys == 0 or imicrophys == 1:
        if idthdt == 1:
            write_output(
                nout,
                Z,
                U,
                S,
                T,
                QV=QV,
                QC=QC,
                QR=QR,
                PREC=PREC,
                TOT_PREC=TOT_PREC,
                DTHETADT=DTHETADT,
            )
        else:
            write_output(
                nout, Z, U, S, T, QV=QV, QC=QC, QR=QR, PREC=PREC, TOT_PREC=TOT_PREC
            )
    elif imicrophys == 2:
        if idthdt == 1:
            write_output(
                nout,
                Z,
                U,
                S,
                T,
                QV=QV,
                QC=QC,
                QR=QR,
                PREC=PREC,
                TOT_PREC=TOT_PREC,
                NR=NR,
                NC=NC,
                DTHETADT=DTHETADT,
            )
        else:
            write_output(
                nout,
                Z,
                U,
                S,
                T,
                QV=QV,
                QC=QC,
                QR=QR,
                PREC=PREC,
                TOT_PREC=TOT_PREC,
                NR=NR,
                NC=NC,
            )
    t1 = tm()

    if itime == 1:
        print("Total elapsed computation time: %g s\n" % (t1 - t0))

if __name__ == "__main__":
    run_unoptimized()

# END OF SOLVER.PY
