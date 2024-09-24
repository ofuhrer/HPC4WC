import numpy as np

from nmwc_model.diagnostics import diag_density_and_temperature
from nmwc_model.meteo_utilities import eswat1
from nmwc_model.namelist import (
    nz,
    nxb,
    dt,
    cp,
    dth,
    vt_mult,
    autoconv_th,
    autoconv_mult,
    iern,
    r,
    r_v,
    sediment_on,
    dx,
    idthdt,
)


def get_kessler_terminal_velocity(qr, rho):
    """ Compute the rain sedimentation velocity as prescribed by the Kessler scheme.

    Parameters
    ----------
    qr : np.ndarray
        The mass fraction of precipitation water in [g g^-1].
    rho : np.ndarray
        The air density in [kg m^-3].

    Returns
    -------
    np.ndarray :
        The terminal velocity in [m s^-1].
    """
    vt = (
        36.34
        * (0.001 * rho * np.where(qr > 0.0, qr, 0.0)) ** 0.1346
        * np.sqrt(rho[:, 0:1] / rho)
    )
    return vt


def get_sedimentation_substeps(dts, zht, vt, crmax):
    """ Get the number of substeps required to obey the CFL condition associated
    with sedimentation.

    Parameters
    ----------
    dts : float
        The current timestep used to integrate the sedimentation in [s].
    zht : np.ndarray
        The vertically staggered geometric height in [m].
    vt : np.ndarray
        The sedimentation velocity in [m s^-1].
    crmax : float
        The maximum Courant number allowed.

    Returns
    -------
    int :
        The required number of substeps.
    """
    # compute courant number associated with sedimentation
    cr = dts * vt / (zht[:, 1:] - zht[:, :-1])

    # compute number of substeps required to retain stability
    nfall = np.maximum(1.0, np.round(0.5 + cr / crmax)).max()

    return nfall


def kessler(s, qv, qc, qr, prs, exn, zht, th0, prec, tot_prec):
    """ The Kessler microphysics scheme.

    The following tasks are carried out one after the other:

        - integrate the sedimentation flux;
        - compute and integrated the tendencies due to microphysics processes;
        - perform the saturation adjustment.

    Parameters
    ----------
    s : np.ndarray
        The isentropic density in [kg m^-2 K^-1].
    qv : np.ndarray
        The mass fraction of water vapor in [g g^-1].
        *Remark:* this array is modified in place.
    qc : np.ndarray
        The mass fraction of cloud liquid water in [g g^-1].
        *Remark:* this array is modified in place.
    qr : np.ndarray
        The mass fraction of precipitation water in [g g^-1].
        *Remark:* this array is modified in place.
    prs : np.ndarray
        The vertically staggered pressure field in [Pa].
    exn : np.ndarray
        The vertically staggered Exner function in [J kg^-1 K^-1].
    zht : np.ndarray
        The vertically staggered geometric height in [m].
    th0 : np.ndarray
        The potential temperature at the interface vertical levels in [K].
    prec : np.ndarray
        The precipitation rate in [mm hr^-1].
        *Remark:* this array is modified in place.
    tot_prec : np.ndarray
        The accumulated precipitation in [mm].
        *Remark:* this array is modified in place.

    Returns
    -------
    lheat : np.ndarray
        The latent heat absorbed by cloud/rain evaporation and released by vapor condensation.
    qv : np.ndarray
        The input array ``qv`` modified.
    qc : np.ndarray
        The input array ``qc`` modified.
    qr : np.ndarray
        The input array ``qr`` modified.
    prec : np.ndarray
        The input array ``prec`` modified.
    tot_prec : np.ndarray
        The input array ``tot_prec`` modified.
    """
    # define constants
    c1 = 0.001 * autoconv_mult
    c2 = autoconv_th  # originally 0.001
    c3 = 2.2
    c4 = 0.875
    xlv = 2.5e6
    max_cr_sedimentation = 0.75
    rhowater = 1000.0

    # diagnose air density and temperature
    rho, temp = diag_density_and_temperature(s, exn, zht, th0)

    # saturation mixing ratio of water vapor
    qvs = (r / r_v) * 100.0 * eswat1(temp) / (0.5 * (prs[:, :-1] + prs[:, 1:]))

    #
    # sedimentation
    #
    if sediment_on:
        # initialize short timestep and number of substeps
        dts = dt
        nt = 1

        while nt > 0:
            # get sedimentation velocity
            vt = vt_mult * get_kessler_terminal_velocity(qr, rho)

            # update number of substeps
            nfall = get_sedimentation_substeps(dts, zht, vt, max_cr_sedimentation)

            # adjust short timestep
            if nfall > nt:
                dts /= nfall
                nt *= nfall

            # sedimentation flux
            sflux = rho * vt * qr

            # step mass fraction of precipitation water
            k = np.arange(0, nz - 1)
            qr[:, k] += (
                dts
                * (sflux[:, k + 1] - sflux[:, k])
                / (rho[:, k] * 0.5 * (zht[:, k + 2] - zht[:, k]))
            )

            # update precipitation and total precipitation
            prec[...] = 3600 * 1000 * sflux[:, 0] / rhowater
            tot_prec += dts * prec / 3600

            # update the control variable
            nt -= 1

    #
    # microphysics
    #
    # autoconversion
    ar = c1 * np.where(qc > c2, qc - c2, 0.0)

    # accretion
    cr = c3 * qc * np.where(qr > 0.0, qr ** c4, 0.0)

    # rain evaporation
    if iern:
        er = np.where(
            qr > 0.0, 0.0484794 * (qvs - qv) * (rho * qr) ** (13.0 / 20.0), 0.0
        )
    else:
        er = 0.0

    # update the mass fraction of water species
    qv += dt * er
    qc -= dt * (ar + cr)
    qr += dt * (ar + cr - er)

    #
    # saturation adjustment
    #
    # extra amount of water
    sat = (qvs - qv) / (1.0 + qvs * (xlv ** 2) / (cp * r_v * (temp ** 2)))
    dq = np.where(sat <= qc, sat, qc)

    # perform the adjustment
    qv += dq
    qc -= dq

    # latent heat of vaporization and condensation
    lheat = - xlv / (0.5 * (exn[:, :-1] + exn[:, 1:])) * (dt * er + dq)

    return lheat, qv, qc, qr, prec, tot_prec


def seifert(
    u,
    t,
    pres,
    snew,
    qv,
    qc,
    qr,
    exn,
    zhtold,
    zhtnow,
    rainnc,
    rainncv,
    nc,
    nr,
    dthetadt=None,
):
    """
    ***********************************************
    Two-moment microphysical scheme (Seifert, 2001/2006)
    adapted from COSMO, Annette Miltenberger and Lukas Papritz (2012)
    ***********************************************
    """

    # define constants
    # svp1 = 0.6112
    svp2 = 17.67
    svp3 = 29.65
    svpt0 = 273.15
    ep2 = r / r_v
    xlv = 2.5e06

    # store specific humidity
    qv_ini = qv
    # qr_ini = qr
    # qc_ini = qc
    # nr_ini = nr
    # nc_ini = nc

    # constants
    # ----------

    rho0 = 1.225
    rho_w = 1000  # density of liquid water
    L_wd = 2.4 * 10 ** 6  # heat of vaporisation
    K_T = 2.500 * 10 ** (-2)  # heat conductivity
    c_r = 1.0 / 2

    # characteristics of cloud droplet distribution (cloud_nue1mue1)
    nu = 1  # parameters describing assumed distribution
    x_max = 2.6 * 10 ** (-10)  # maximal droplet mass
    x_min = 4.20 * 10 ** (-15)  # minimal droplet mass

    # characteristics of rain droplet distribution (rainULI)
    rain_x_min = 2.6 * 10 ** (-10)  # minimale Teilchenmasse
    rain_x_max = 3.0 * 10 ** (-6)  # maximale Teilchenmasse
    a_geo = 1.24 * 10 ** (-1)  # Koeff. Geometrie
    b_geo = 0.333333  # Koeff. Geometrie
    a_ven = 0.780000  # Koeff. Ventilation (PK)
    b_ven = 0.308000  # Koeff. Ventilation (PK)
    rain_nu = 0  # Breiteparameter der Verteilung

    # parameters for autoconversion
    k_c = 9.44 * 10 ** 9  # Long-Kernel
    k_1 = 600  # Parameter fuer Phi-Fkt. (autoconversion)
    k_2 = 0.68  # Parameter fuer Phi-Fkt. (autoconversion)
    k_au = (
        k_c / (20.0 * x_max) * (nu + 2) * (nu + 4) / (nu + 1) ** 2
    )  # autoconversion constant

    # parameters for accretion
    k_3 = 5 * 10 ** (-4)  # Parameter fuer Phi-Fkt. (accretion)
    k_r = 5.78  # Parameter Kernel (accretion)

    # parameter for rain selfcollection and break-up
    k_sc = k_c * (nu + 2) / (nu + 1)  # selfcollection constant
    k_rr = 4.33
    k_br = 1000
    D_br = 1.1 * 10 ** (-3)

    # parameters for rain evaporation and sedimentation
    rain_cmu0 = 6
    rain_cmu1 = 30
    rain_cmu2 = 10.0 ** 3
    rain_cmu3 = 1.1 * 10.0 ** (-3)
    rain_cmu4 = 1
    rain_cmu5 = 2
    N_sc = 0.710  # Schmidt-Zahl (PK)
    n_f = 0.333  # Exponent von N_sc im Vent-koeff.
    m_f = 0.5  # Exponent von N_re im Vent-koeff.
    nu_l = 1.460 * 10.0 ** (-5)  # Kinem. Visc. von Luft
    aa = 9.65
    bb = 10.3
    cc = 600
    alf = 9.65
    bet = 10.3
    gamma = 600

    # transpose input fields
    rainnc_tr = rainnc.T
    rainncv_tr = rainncv.T
    t_tr = t.T

    # reset rain rate to zero
    rainncv_tr = 0.0

    # compute density
    # ------------------------
    i = np.arange(0, nxb)
    k = np.arange(0, nz)
    ii, kk = np.ix_(i, k)
    pii = exn / cp
    rho = np.zeros((nxb, nz))
    rho[:, k] = snew[:, k] * dth / (zhtnow[:, k + 1] - zhtnow[:, k])
    t = 0.5 * (
        pii[ii, kk + 1] * np.tile(t_tr[k + 1], (nxb, 1))
        + pii[ii, kk] * np.tile(t_tr[k], (nxb, 1))
    )
    p = 0.5 * (
        pres[ii, kk] + pres[ii, kk + 1]
    )  # 1E05* (0.5*(pii[ii,kk+1]+pii[ii,kk])**(1004./287.))
    gam = 2.5e06 / (
        1004.0 * 0.5 * (pii[ii, kk] + pii[ii, kk + 1])
    )  # L / (cp_d * exn/cp)

    rrho_c = rho0 / rho
    rrho_04 = (rho0 / rho) ** 0.5

    f5 = svp2 * (svpt0 - svp3) * xlv / cp

    # vertical wind
    i = np.arange(1, (nxb - 1))
    k = np.arange(0, nz)
    ii, kk = np.ix_(i, k)
    dz_dx = np.zeros((nxb, nz))
    dz_dx[ii, kk] = (
        zhtnow[ii + 1, kk + 1]
        + zhtnow[ii + 1, kk]
        - zhtnow[ii - 1, kk + 1]
        - zhtnow[ii - 1, kk]
    ) / (4.0 * dx)
    w = np.zeros((nxb, nz))
    # w[ii,kk] = 0.5*(zhtnow[ii,kk]+zhtnow[ii,kk+1] - zhtold[ii,kk]-zhtold[ii,kk+1]) / dt + 0.5*(u[ii+1,kk)]+u[ii,kk])*dz_dx[ii,kk]
    w[ii, kk] = 0.5 * (u[ii + 1, kk] + u[ii, kk]) * dz_dx[ii, kk]
    if idthdt == 1:
        w[ii, kk] = (
            w[ii, kk]
            + 0.5
            * (dthetadt[ii, kk] + dthetadt[ii, kk + 1])
            * snew[ii, kk]
            / rho[ii, kk]
        )
    w[0, k] = 0.0
    w[nxb - 1, k] = 0.0

    # nucleation
    # -----------
    # HUCM continental case (Texas CCN)
    # N_ccn = 1260.*10**6
    # N_max = 3000.*10**6
    # N_min = 300.*10**6
    # S_max = 20
    # k_ccn = 0.308

    wcb_min = 0.1
    scb_min = 0.0

    T_3 = 273.2  # triple point water
    e_3 = 6.1078 * 100  # saturation vapor pressure at triple point
    A_w = 17.2693882  # constant f. saturation vapor pressure (water)
    B_w = 35.86  # constant f. saturation vapor pressure (water)
    e_ws_vec = lambda ta: e_3 * np.exp(A_w * (ta - T_3) / (ta - B_w))

    ssw = r_v * rho * qv * t / e_ws_vec(t) - 1.0

    qr = qr * rho
    qv = qv * rho
    qc = qc * rho
    nr = nr * rho
    nc = nc * rho

    w_cb = np.zeros((nxb, nz))
    for k in range(1, nz):
        ind = (w[:, k] > wcb_min) & (
            ssw[:, k] >= scb_min
        )  # & (ssw[:,k] > ssw[:,np.min(k-1,1)])
        if np.any(ind):
            w_cb[ind, k] = w[ind, k]

    # parameter for exponential decrease of N_ccn with height:
    z0_nccn = 4000.0  # up to this height (m) constant unchanged value:
    z1e_nccn = (
        2000.0
    )  # height interval at which N_ccn decreases by factor 1/e above z0_nccn:

    # characteristics of different kinds of prototype CN: intermediate case
    N_cn0 = 5000 * 10 ** 6
    etas = 0.8  # soluble fraction

    # Look-up tables (for r2 = 0.03 mum, lsigs = 0.4)
    wcb_ind = np.array([0, 0.5, 1.0, 2.5, 5.0])
    ncn_ind = (
        np.array([0, 50, 100, 200, 400, 800, 1600, 3200, 6400], dtype=np.float64)
        * 10 ** 6
    )  # fix for windows

    ltab_nuc = np.array(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 37.2, 67.1, 119.5, 206.7, 340.5, 549.4, 549.4, 549.4],
            [0.0, 39.0, 77.9, 141.2, 251.8, 436.7, 708.7, 1117.7, 1117.7],
            [0.0, 42.3, 84.7, 169.3, 310.3, 559.5, 981.7, 1611.6, 2455.6],
            [0.0, 44.0, 88.1, 176.2, 352.3, 647.8, 1173.0, 2049.7, 3315.6],
        ]
    )
    ltab_nuc = ltab_nuc * 10 ** 6

    # hard upper limit for number conc that eliminates also unrealistic high value
    # that would come from the dynamical core
    nc[w_cb > 0] = np.minimum(nc[w_cb > 0], N_cn0)

    # N_cn depends on height (to avoid strong in-cloud nucleation)
    zml_k = 0.5 * (zhtnow[:, 0:nz] + zhtnow[:, 1 : nz + 1])
    n_cn = N_cn0 * np.minimum(
        np.exp((z0_nccn - zml_k) / z1e_nccn), 1.0
    )  # exponential decrease with height
    n_cn = np.float64(n_cn)  # fix for windows

    nccn = np.zeros((nxb, nz))
    lp = np.array(np.where(w_cb > 0))
    for j1, j2 in lp.T:
        maty, = np.where(wcb_ind > w_cb[j1, j2])  # ugly workaround: When idthdt ==1
        matx, = np.where(ncn_ind > n_cn[j1, j2])

        if not np.any(matx):
            matx = 1
        if not np.any(maty):
            maty = 1

        locy = np.min(maty) - 1
        locx = np.min(matx) - 1
        if (locx < 8) and (locy < 4):
            indx0 = (n_cn[j1, j2] - ncn_ind[locx]) / (ncn_ind[locx + 1] - ncn_ind[locx])
            # print indx0.type
            indx1 = 1 - indx0
            indy0 = (w_cb[j1, j2] - wcb_ind[locy]) / (wcb_ind[locy + 1] - wcb_ind[locy])
            indy1 = 1 - indy0
            nccn[j1, j2] = (
                (indx0 * ltab_nuc[locy, locx] + indx1 * ltab_nuc[locy, locx + 1])
                * indy0
                + (
                    indx0 * ltab_nuc[locy + 1, locx]
                    + indx1 * ltab_nuc[locy + 1, locx + 1]
                )
                * indy1
            )
        elif (locx < 8) and (locy >= 4):
            indx0 = (n_cn[j1, j2] - ncn_ind[locx]) / (ncn_ind[locx + 1] - ncn_ind[locx])
            indx1 = 1 - indx0
            locy = np.min([locy, 4])
            nccn[j1, j2] = (
                indx0 * ltab_nuc[locy, locx] + indx1 * ltab_nuc[locy, locx + 1]
            )
        elif (locy < 4) and (locx >= 8):
            indy0 = (w_cb[j1, j2] - wcb_ind[locy]) / (wcb_ind[locy + 1] - wcb_ind[locy])
            indy1 = 1 - indy0
            locx = np.min([locx, 8])
            nccn[j1, j2] = (
                ltab_nuc[locy, locx] * indy0 + ltab_nuc[locy + 1, locy] * indy1
            )
        else:
            locy = 4
            locx = 8
            nccn[j1, j2] = ltab_nuc[locy, locx]

    # If n_cn is outside the range of the lookup table values, resulting
    # NCCN are clipped to the margin values. For the case of these margin values
    # being larger than n_cn (which happens sometimes, unfortunately), limit NCCN by n_cn:
    nccn = np.minimum(nccn, n_cn)

    nuc_n = etas * nccn - nc
    nuc_n = np.maximum(nuc_n, 0.0)
    nuc_q = np.minimum(nuc_n * x_min, qv)
    nuc_q[nuc_q < 0] = 0
    nuc_n = nuc_q / x_min

    nc = nc + nuc_n
    qc = qc + nuc_q
    qv = qv - nuc_q

    # nucn_max=np.maximum(nuc_n,nucn_max)

    # autoconversion, accretion, selfcollection, break-up
    # ----------------------------------------------------
    # autoconversion

    if np.any(qc > 0):
        # print('autoconversion')
        au = np.zeros((nxb, nz))
        sc = np.zeros((nxb, nz))

        ind = qc > 0
        x_c = np.minimum(np.maximum(qc[ind] / nc[ind], x_min), x_max)
        au[ind] = k_au * qc[ind] ** 2.0 * x_c ** 2.0 * dt * rrho_c[ind]

        if np.any(qc > 10 ** (-6)):
            ind1 = qc > 10 ** -6
            tau = np.minimum(
                np.maximum(1 - (qc[ind1] / (qc[ind1] + qr[ind1])), 10 ** (-25)), 0.9
            )
            phi = k_1 * tau ** k_2 * (1 - tau ** k_2) ** 3
            au[ind1] = au[ind1] * (1 + phi / (1 - tau) ** 2)

        au[ind] = np.maximum(np.minimum(qc[ind], au[ind]), 0)
        sc[ind] = (
            k_sc * qc[ind] ** 2.0 * dt * rrho_c[ind]
        )  # selfcollection cloud droplets

        nr_au = au[ind] / x_max
        nc_au = np.minimum(nc[ind], sc[ind])

        qc = qc - au
        qr = qr + au
        nr[ind] = nr[ind] + nr_au
        nc[ind] = nc[ind] - nc_au

    # accretion
    ac = np.zeros((nxb, nz))
    if np.any((qc > 0) & (qr > 0)):
        # print('accretion')
        ind = (qc > 0) & (qr > 0)
        tau = np.minimum(np.maximum(1 - qc[ind] / (qc[ind] + qr[ind]), 10 ** (-25)), 1)
        phi = (tau / (tau + k_3)) ** 4
        ac[ind] = k_r * qc[ind] * qr[ind] * phi * rrho_04[ind] * dt
        ac = np.minimum(qc, ac)

        x_c = np.minimum(np.maximum(qc / nc, x_min), x_max)
        nc_ac = np.minimum(nc, ac / x_c)

        qr = qr + ac
        qc = qc - ac
        nc = nc - nc_ac

    # self-collection rain / breakup
    if np.any(qr > 0):
        # print('selfcollection')
        ind = qr > 0
        x_r = np.minimum(np.maximum(qr[ind] / nr[ind], rain_x_min), rain_x_max)
        D_r = a_geo * x_r ** b_geo

        # selfcollection
        sc = k_rr * nr[ind] * qr[ind] * rrho_04[ind] * dt

        # breakup
        br = sc * 0
        if np.any(D_r > 0.3 * 10 ** (-3)):
            ind1 = D_r > 0.3 * 10 ** (-3)
            phi1 = k_br * (D_r[ind1] - D_br) + 1
            br[ind1] = phi1 * sc[ind1]

        nr_sc = np.minimum(nr[ind], sc - br)

        nr[ind] = nr[ind] - nr_sc

    nr[nr < 0] = 0.0
    nc[nc < 0] = 0.0
    qc[qc < 0] = 0.0
    qr[qr < 0] = 0.0

    qc[np.isnan(qc)] = 0.0
    qr[np.isnan(qr)] = 0.0
    nc[np.isnan(nc)] = 0.0
    nr[np.isnan(nr)] = 0.0

    if iern == 1:

        # evaporation of rain droplets
        # -----------------------------
        e_d = qv * r_v * t
        e_sw = e_ws_vec(t)
        s_sw = e_d / e_sw - 1

        if np.any((s_sw < 0) & (qr > 0) & (qc < 10 ** (-9))):
            # condition for the occurence of evaporation
            ind = (s_sw < 0) & (qr > 0) & (qc < 10 ** (-9))

            eva_q = np.zeros((nxb, nz))
            eva_n = np.zeros((nxb, nz))

            d_vtp = 8.7602 * 10 ** (-5) * t[ind] ** (1.81) / p[ind]
            g_d = (
                4.0
                * np.pi
                / (
                    L_wd ** 2.0 / (K_T * r_v * t[ind] ** 2)
                    + r_v * t[ind] / (d_vtp * e_sw[ind])
                )
            )

            x_r = qr[ind] / (nr[ind] + 10 ** (-20))
            x_r = np.minimum(np.maximum(x_r, rain_x_min), rain_x_max)

            D_m = a_geo * x_r ** b_geo

            mue = np.empty(x_r.shape)
            mue[D_m <= rain_cmu3] = (
                rain_cmu0
                * np.tanh(
                    (4.0 * rain_cmu2 * (D_m[D_m <= rain_cmu3] - rain_cmu3)) ** rain_cmu5
                )
                + rain_cmu4
            )
            mue[D_m > rain_cmu3] = (
                rain_cmu1
                * np.tanh((rain_cmu2 * (D_m[D_m > rain_cmu3] - rain_cmu3)) ** rain_cmu5)
                + rain_cmu4
            )

            mue = mue.T

            lam = (np.pi / 6.0 * rho_w * (mue + 3) * (mue + 2) * (mue + 1) / x_r) ** (
                1.0 / 3.0
            )

            gfak = 1.357940435 + mue * (
                0.3033273220
                + mue
                * (
                    -0.1299313363 * 10 ** (-1)
                    + mue
                    * (0.4002257774 * 10 ** (-3) - mue * 0.4856703981 * 10 ** (-5))
                )
            )

            f_q = a_ven + b_ven * N_sc ** n_f * (
                aa / nu_l * rrho_04[ind]
            ) ** m_f * gfak / np.sqrt(lam) * (
                1.0
                - 1.0 / 2.0 * (bb / aa) * (lam / (cc + lam)) ** (mue + 5.0 / 2.0)
                - 1.0
                / 8.0
                * (bb / aa) ** 2.0
                * (lam / (2.0 * cc + lam)) ** (mue + 5.0 / 2.0)
                - 1.0
                / 16.0
                * (bb / aa) ** 3.0
                * (lam / (3.0 * cc + lam)) ** (mue + 5.0 / 2.0)
                - 5.0
                / 127.0
                * (bb / aa) ** 4.0
                * (lam / (4.0 * cc + lam)) ** (mue + 5.0 / 2.0)
            )

            gamma_eva = np.empty(x_r.shape)
            gamma_eva[gfak > 0] = (
                gfak[gfak > 0] * (1.1 * 10 ** (-3) / D_m) * np.exp(-0.2 * mue)
            )
            gamma_eva[gfak <= 0] = 1
            gamma_eva = gamma_eva.T

            eva_q[ind] = -g_d * c_r * nr[ind] * (mue + 1) / lam * f_q * s_sw[ind] * dt
            eva_n[ind] = gamma_eva * eva_q[ind] / x_r

            eva_q = np.maximum(eva_q, 0)
            eva_n = np.maximum(eva_n, 0)
            eva_q = np.minimum(eva_q, qr)
            eva_n = np.minimum(eva_n, nr)

            qv = qv + eva_q
            qr = qr - eva_q
            nr = nr - eva_n

    # conversion of mixing ratios to mass densities
    # -------------------------------------------------------------------------
    qr = qr / rho
    qv = qv / rho
    qc = qc / rho
    nr = nr / rho
    nc = nc / rho

    # saturation adjustment
    # ----------------------
    es = eswat1(t) * 100
    qvs = ep2 * es / (p - es)

    # saturation adjustment: condensation/evaporation
    produc = (qv - qvs) / (1 + p / (p - es) * qvs * f5 / (t - svp3) ** 2)

    produc = np.maximum(produc, -qc)  # no evaporation if no cloud water
    produc[nc <= 0] = np.minimum(
        0, produc[nc <= 0]
    )  # no condensation if no cloud droplets
    produc = np.minimum(qv, produc)  # limit condensation to qv

    qc = qc + produc
    qc[nc <= 0] = 0.0
    nc[qc <= 0] = 0.0
    qv = qv - produc

    # Limit rain drop size
    nr = np.maximum(nr, qr / rain_x_max)
    nr = np.minimum(nr, qr / rain_x_min)
    # nc = np.maximum(nc,qc/x_max)
    nc = np.minimum(nc, 5000.0 * 10 ** 6)

    nr[nr < 0] = 0.0
    nc[nc < 0] = 0.0
    qc[qc < 0] = 0.0
    qr[qr < 0] = 0.0

    # sedimentation of rain droplets
    # ------------------------------
    dzmin = 10 ** 10
    # density correction for fall velocities
    rhocorr = (rho0 / rho) ** 0.5
    adz = 1 / (zhtnow[:, 1 : nz + 1] - zhtnow[:, 0:nz])  # reciprocal vertical grid
    dzmin = np.minimum(1.0 / adz, dzmin)

    qr = qr * rho
    nr = nr * rho

    dt_sedi = np.minimum(dt, 0.7 * dzmin / 20.0)
    nt_sedi = int(np.max((np.ceil(np.max(dt / dt_sedi)), 1)))
    dt_sedi = dt / nt_sedi

    for n in range(0, nt_sedi):
        v_n_rain = np.zeros((nxb, nz))
        v_q_rain = np.zeros((nxb, nz))
        q_flux = np.zeros((nxb, nz))
        n_flux = np.zeros((nxb, nz))

        k = np.arange(0, nz)
        if np.any(qr[:, k] > 10 ** (-20)):
            ind = qr[:, k] > 10 ** (-20)

            x_r = np.zeros((nxb, nz))
            x_r[ind] = qr[ind] / nr[ind]
            x_r[ind] = np.minimum(np.maximum(x_r[ind], rain_x_min), rain_x_max)
            D_m = (6.0 / (rho_w * np.pi) * x_r) ** (1.0 / 3.0)

            mue = np.zeros((nxb, nz))
            if np.any((qc >= 10 ** (-20)) & (qr > 10 ** (-20))):
                mue[(qc >= 10 ** (-20)) & (qr > 10 ** (-20))] = (
                    rain_nu + 1
                ) / b_geo - 1
            if np.any((D_m[ind] <= rain_cmu3) & (qc[ind] <= 10 ** (-20))):
                ind1 = (D_m <= rain_cmu3) & (qr > 10 ** (-20)) & (qc <= 10 ** (-20))
                mue[ind1] = (
                    rain_cmu0
                    * np.tanh((4.0 * rain_cmu2 * (D_m[ind1] - rain_cmu3)) ** 2)
                    + rain_cmu4
                )
            if np.any((D_m[ind] > rain_cmu3) & (qc[ind] <= 10 ** (-20))):
                ind2 = (D_m > rain_cmu3) & (qr > 10 ** (-20)) & (qc <= 10 ** (-20))
                mue[ind2] = (
                    rain_cmu1 * np.tanh((rain_cmu2 * (D_m[ind2] - rain_cmu3)) ** 2)
                    + rain_cmu4
                )

            D_r = (D_m ** 3.0 / ((mue + 3.0) * (mue + 2.0) * (mue + 1.0))) ** (
                1.0 / 3.0
            )

            v_n = alf - bet / (1.0 + gamma * D_r) ** (mue + 1.0)
            v_q = alf - bet / (1.0 + gamma * D_r) ** (mue + 4.0)
            v_n = v_n * rhocorr
            v_q = v_q * rhocorr
            v_n = np.maximum(v_n, 0.1)
            v_q = np.maximum(v_q, 0.1)
            v_n = np.minimum(v_n, 20)
            v_q = np.minimum(v_q, 20)
            v_n_rain = -v_q  # fall velocity
            v_q_rain = -v_q  # fall velocity

        # lower boundary condition for fall velocity
        v_n_rain[:, 0] = v_n_rain[:, 1]
        v_q_rain[:, 0] = v_q_rain[:, 1]

        for k in range(nz - 2, -1, -1):

            v_nv = 0.5 * (v_n_rain[:, k + 1] + v_n_rain[:, k])
            v_qv = 0.5 * (v_q_rain[:, k + 1] + v_q_rain[:, k])
            # assuming v_nv, v_qv always_negative
            c_nv = -v_nv * adz[:, k] * dt_sedi
            c_qv = -v_qv * adz[:, k] * dt_sedi

            kk = k
            s_nv = np.zeros((nxb))
            s_nv[c_nv <= 1] = v_nv[c_nv <= 1] * nr[c_nv <= 1, k]
            if np.any(c_nv > 1):
                cflag = np.zeros((nxb), dtype=bool)
                while np.any(c_nv > 1) and (kk < nz - 1):
                    ind = c_nv > 1
                    cflag[ind] = True
                    s_nv[ind] = s_nv[ind] + nr[ind, kk] / adz[ind, kk]
                    c_nv[ind] = (c_nv[ind] - 1) * adz[ind, kk + 1] / adz[ind, kk]
                    kk = kk + 1
                s_nv[cflag] = s_nv[cflag] + nr[cflag, kk] / adz[cflag, kk] * np.minimum(
                    c_nv[cflag], 1.0
                )
                s_nv[cflag] = -s_nv[cflag] / dt_sedi

            kk = k
            s_qv = np.zeros((nxb))
            s_qv[c_qv <= 1] = v_qv[c_qv <= 1] * qr[c_qv <= 1, k]
            if np.any(c_qv > 1):
                cflag = np.zeros((nxb), dtype=bool)
                while np.any(c_qv > 1) and (kk < nz - 1):
                    ind = c_qv > 1
                    cflag[ind] = True
                    s_qv[ind] = s_qv[ind] + qr[ind, kk] / adz[ind, kk]
                    c_qv[ind] = (c_qv[ind] - 1) * adz[ind, kk + 1] / adz[ind, kk]
                    kk = kk + 1
                s_qv[cflag] = s_qv[cflag] + qr[cflag, kk] / adz[cflag, kk] * np.minimum(
                    c_qv[cflag], 1.0
                )
                s_qv[cflag] = -s_qv[cflag] / dt_sedi

            # Flux-limiter to avoid negative values
            n_flux[:, k] = np.maximum(
                s_nv, n_flux[:, k + 1] - nr[:, k] / (adz[:, k] * dt_sedi)
            )
            q_flux[:, k] = np.maximum(
                s_qv, q_flux[:, k + 1] - qr[:, k] / (adz[:, k] * dt_sedi)
            )

        # uppper boundary condition
        n_flux[:, nz - 1] = 0.0
        q_flux[:, nz - 1] = 0.0

        k = np.arange(0, nz - 1)
        nr[:, k] = nr[:, k] + (n_flux[:, k] - n_flux[:, k + 1]) * adz[:, k] * dt_sedi
        qr[:, k] = qr[:, k] + (q_flux[:, k] - q_flux[:, k + 1]) * adz[:, k] * dt_sedi

        rainncv = -q_flux[:, 0].T * 3600.0 / rho_w * 1000.0  # mm/h
        rainnc = rainnc - q_flux[:, 0].T * dt_sedi / rho_w * 1000.0  # mm

    # Sedimentation rates seem to be too slow for the sedimentation velocities
    # of 9 m/s

    qr = qr / rho
    nr = nr / rho

    qv[qv < 0] = 0.0
    qc[qc < 0] = 0.0
    nr[nr < 0] = 0.0
    nc[nc < 0] = 0.0
    # nc[qc < 10**(-20)] = np.minimum(nc[qc < 10**(-20)],qc[qc < 10**(-20)]/x_min)
    # nr[qr < 10**(-20)] = np.minimum(nr[qr < 10**(-20)],qr[qr < 10**(-20)]/rain_x_min)

    # finally update all variables (except temperature)
    lheat = gam * (qv_ini - qv)

    # for debugging
    # print('qc : ', np.max(qc))
    # print('qr : ', np.max(qr))
    # print('nc : ', np.max(nc))
    # print('nr : ', np.max(nr))

    return lheat, qv, qc, qr, rainnc, rainncv, nc, nr
