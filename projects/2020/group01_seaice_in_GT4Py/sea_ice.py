#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# pylint: disable=W0511
# pylint: disable=C0326
# pylint: disable=C0103

import numpy as np

OUT_VARS = ["tskin", "tprcp", "fice", "gflux", "ep", "stc", "tice", \
    "snowmt", "evap", "snwdph", "chh", "weasd", "hflx", "qsurf", \
    "hice", "cmm"]

def run(in_dict):
    """run function"""

    # setup output
    out_dict = {}
    for key in OUT_VARS:
        out_dict[key] = in_dict[key].copy()

    sfc_sice(
        in_dict['im'], in_dict['km'], in_dict['ps'],
        in_dict['t1'], in_dict['q1'], in_dict['delt'],
        in_dict['sfcemis'], in_dict['dlwflx'], in_dict['sfcnsw'],
        in_dict['sfcdsw'], in_dict['srflag'], in_dict['cm'],
        in_dict['ch'], in_dict['prsl1'], in_dict['prslki'],
        in_dict['islimsk'], in_dict['wind'], in_dict['flag_iter'],
        in_dict['lprnt'], in_dict['ipr'], in_dict['cimin'],
        out_dict['hice'], out_dict['fice'], out_dict['tice'],
        out_dict['weasd'], out_dict['tskin'], out_dict['tprcp'],
        out_dict['stc'], out_dict['ep'], out_dict['snwdph'],
        out_dict['qsurf'], out_dict['cmm'], out_dict['chh'],
        out_dict['evap'], out_dict['hflx'], out_dict['gflux'],
        out_dict['snowmt'])

    return out_dict


def sfc_sice(im, km, ps, t1, q1, delt, sfcemis, dlwflx, sfcnsw, sfcdsw, srflag,
             cm, ch, prsl1, prslki, islimsk, wind, flag_iter, lprnt, ipr, cimin,
             hice, fice, tice, weasd, tskin, tprcp, stc, ep, snwdph, qsurf, cmm, chh, 
             evap, hflx, gflux, snowmt):
    """run function"""

    # constant definition
    # TODO - this should be moved into a shared physics constants / physics functions module
    cp     = 1.0046e+3
    hvap   = 2.5e+6
    sbc    = 5.6704e-8
    tgice  = 2.712e+2
    rv     = 4.615e+2
    rd     = 2.8705e+2
    eps    = rd / rv
    epsm1  = rd / rv - 1.
    rvrdm1 = rv / rd - 1.
    t0c    = 2.7315e+2

    # constant parameterts
    kmi    = 2
    cpinv  = 1. / cp
    hvapi  = 1. / hvap
    elocp  = hvap / cp
    himax  = 8.          # maximum ice thickness allowed
    himin  = 0.1         # minimum ice thickness required
    hsmax  = 2.          # maximum snow depth allowed
    timin  = 173.        # minimum temperature allowed for snow/ice
    albfw  = 0.06        # albedo for lead
    dsi    = 1. / 0.33

    # arrays
    mode = "nan"
    ffw    = init_array([im], mode)
    evapi  = init_array([im], mode)
    evapw  = init_array([im], mode)
    sneti  = init_array([im], mode)
    snetw  = init_array([im], mode)
    hfd    = init_array([im], mode)
    hfi    = init_array([im], mode)
    focn   = init_array([im], mode)
    snof   = init_array([im], mode)
    rch    = init_array([im], mode)
    rho    = init_array([im], mode)
    snowd  = init_array([im], mode)
    theta1 = init_array([im], mode)
    flag   = init_array([im], mode)
    q0     = init_array([im], mode)
    qs1    = init_array([im], mode)
    stsice = init_array([im, kmi], mode)

#  --- ...  set flag for sea-ice

    flag = (islimsk == 2) & flag_iter
    i = flag_iter & (islimsk < 2)
    hice[i] = 0.
    fice[i] = 0.

    i = flag & (srflag > 0.)
    ep[i] = ep[i] * (1. - srflag[i])
    weasd[i] = weasd[i] + 1.e3 * tprcp[i] * srflag[i]
    tprcp[i] = tprcp[i] * (1. - srflag[i])

#  --- ...  update sea ice temperature
    i = flag
    stsice[i, :] = stc[i, 0:kmi]

#  --- ...  initialize variables. all units are supposedly m.k.s. unless specified
#           psurf is in pascals, wind is wind speed, theta1 is adiabatic surface
#           temp from level 1, rho is density, qs1 is sat. hum. at level1 and qss
#           is sat. hum. at surface
#           convert slrad to the civilized unit from langley minute-1 k-4

#         dlwflx has been given a negative sign for downward longwave
#         sfcnsw is the net shortwave flux (direction: dn-up)

    q0[i]     = np.maximum(q1[i], 1.0e-8)
    theta1[i] = t1[i] * prslki[i]
    rho[i]    = prsl1[i] / (rd * t1[i] * (1. + rvrdm1 * q0[i]))
    qs1[i]    = fpvs(t1[i])
    qs1[i]    = np.maximum(eps * qs1[i] / (prsl1[i] + epsm1 * qs1[i]), 1.e-8)
    q0[i]     = np.minimum(qs1[i], q0[i])

    i = flag & (fice < cimin)
    if any(i):
        print("warning: ice fraction is low:", fice[i])
        fice[i] = cimin
        tice[i] = tgice
        tskin[i]= tgice
        print('fix ice fraction: reset it to:', fice[i])

    i = flag
    ffw[i]    = 1.0 - fice[i]

    qssi = fpvs(tice[i])
    qssi = eps * qssi / (ps[i] + epsm1 * qssi)
    qssw = fpvs(tgice)
    qssw = eps * qssw / (ps[i] + epsm1 * qssw)

#  --- ...  snow depth in water equivalent is converted from mm to m unit

    snowd[i] = weasd[i] * 0.001

#  --- ...  when snow depth is less than 1 mm, a patchy snow is assumed and
#           soil is allowed to interact with the atmosphere.
#           we should eventually move to a linear combination of soil and
#           snow under the condition of patchy snow.

#  --- ...  rcp = rho cp ch v

    cmm[i] = cm[i] * wind[i]
    chh[i] = rho[i] * ch[i] * wind[i]
    rch[i] = chh[i] * cp

#  --- ...  sensible and latent heat flux over open water & sea ice

    evapi[i] = elocp * rch[i] * (qssi - q0[i])
    evapw[i] = elocp * rch[i] * (qssw - q0[i])

    snetw[i] = sfcdsw[i] * (1. - albfw)
    snetw[i] = np.minimum(3. * sfcnsw[i] / (1. + 2. * ffw[i]), snetw[i])
    sneti[i] = (sfcnsw[i] - ffw[i] * snetw[i]) / fice[i]

    t12 = tice[i] * tice[i]
    t14 = t12 * t12

#  --- ...  hfi = net non-solar and upir heat flux @ ice surface

    hfi[i] = -dlwflx[i] + sfcemis[i] * sbc * t14 + evapi[i] + \
            rch[i] * (tice[i] - theta1[i])
    hfd[i] = 4. * sfcemis[i] * sbc * tice[i] * t12 + \
            (1. + elocp * eps * hvap * qs1[i] / (rd * t12)) * rch[i]

    t12 = tgice * tgice
    t14 = t12 * t12

#  --- ...  hfw = net heat flux @ water surface (within ice)

    focn[i] = 2.   # heat flux from ocean - should be from ocn model
    snof[i] = 0.    # snowfall rate - snow accumulates in gbphys

    hice[i] = np.maximum(np.minimum(hice[i], himax), himin)
    snowd[i] = np.minimum(snowd[i], hsmax)

    i = flag & (snowd > 2. * hice)
    if any(i):
        print('warning: too much snow :', snowd[i])
        snowd[i] = hice[i] + hice[i]
        print('fix: decrease snow depth to:', snowd[i])

    # run the 3-layer ice model
    ice3lay(im, kmi, fice, flag, hfi, hfd, sneti, focn, delt, lprnt, ipr,
            snowd, hice, stsice, tice, snof, snowmt, gflux)

    i = flag & (tice < timin)
    if np.any(i):
        print('warning: snow/ice temperature is too low:', tice[i], ' i=', i)
        tice[i] = timin
        print('fix snow/ice temperature: reset it to:', timin)

    i = flag & (stsice[:, 0] < timin)
    if any(i):
        print('warning: layer 1 ice temp is too low:', stsice[i, 0], ' i=', i)
        stsice[i, 0] = timin
        print('fix layer 1 ice temp: reset it to:', timin)

    i = flag & (stsice[:, 1] < timin)
    if any(i):
        print('warning: layer 2 ice temp is too low:', stsice[i, 1], 'i=', i)
        stsice[i, 1] = timin
        print('fix layer 2 ice temp: reset it to:', timin)

    i = flag
    tskin[i] = tice[i] * fice[i] + tgice * ffw[i]

    stc[i, 0:kmi] = np.minimum(stsice[i, 0:kmi], t0c)

#  --- ...  calculate sensible heat flux (& evap over sea ice)

    hflxi = rch[i] * (tice[i] - theta1[i])
    hflxw = rch[i] * (tgice - theta1[i])
    hflx[i] = fice[i] * hflxi + ffw[i] * hflxw
    evap[i] = fice[i] * evapi[i] + ffw[i] * evapw[i]

#  --- ...  the rest of the output

    qsurf[i] = q1[i] + evap[i] / (elocp * rch[i])

#  --- ...  convert snow depth back to mm of water equivalent

    weasd[i] = snowd[i] * 1000.
    snwdph[i] = weasd[i] * dsi             # snow depth in mm

    tem = 1. / rho[i]
    hflx[i] = hflx[i] * tem * cpinv
    evap[i] = evap[i] * tem * hvapi


def ice3lay(im,kmi,fice,flag,hfi,hfd, sneti, focn, delt, lprnt, ipr, \
            snowd, hice, stsice, tice, snof, snowmt, gflux):
    """function ice3lay"""

    # constant parameters
    ds   = 330.     # snow (ov sea ice) density (kg/m^3)
    dw   = 1000.    # fresh water density  (kg/m^3)
    dsdw = ds / dw
    dwds = dw / ds
    ks   = 0.31     # conductivity of snow   (w/mk)
    i0   = 0.3      # ice surface penetrating solar fraction
    ki   = 2.03     # conductivity of ice  (w/mk)
    di   = 917.     # density of ice   (kg/m^3)
    didw = di / dw
    dsdi = ds / di
    ci   = 2054.    # heat capacity of fresh ice (j/kg/k)
    li   = 3.34e5   # latent heat of fusion (j/kg-ice)
    si   = 1.       # salinity of sea ice
    mu   = 0.054    # relates freezing temp to salinity
    tfi  = -mu * si # sea ice freezing temp = -mu*salinity
    tfw  = -1.8     # tfw - seawater freezing temp (c)
    tfi0 = tfi - 0.0001
    dici = di * ci
    dili = di * li
    dsli = ds * li
    ki4  = ki * 4.

    # TODO: move variable definition to separate file 
    t0c    = 2.7315e+2
    
    # vecotorize constants
    mode = "nan"
    ip = init_array([im], mode)
    tsf = init_array([im], mode)
    ai = init_array([im], mode)
    k12 = init_array([im], mode)
    k32 = init_array([im], mode)
    a1 = init_array([im], mode)
    b1 = init_array([im], mode)
    c1 = init_array([im], mode)
    tmelt = init_array([im], mode)
    h1 = init_array([im], mode)
    h2 = init_array([im], mode)
    bmelt = init_array([im], mode)
    dh = init_array([im], mode)
    f1 = init_array([im], mode)
    hdi = init_array([im], mode)
    wrk = init_array([im], mode)
    wrk1 = init_array([im], mode)
    bi = init_array([im], mode)
    a10 = init_array([im], mode)
    b10 = init_array([im], mode)

    dt2  = 2. * delt
    dt4  = 4. * delt
    dt6  = 6. * delt
    dt2i = 1. / dt2

    i = flag
    snowd[i] = snowd[i]  * dwds
    hdi[i] = (dsdw * snowd[i] + didw * hice[i])

    i = flag & (hice < hdi)
    snowd[i] = snowd[i] + hice[i] - hdi[i]
    hice[i]  = hice[i] + (hdi[i] - hice[i]) * dsdi

    i = flag
    snof[i] = snof[i] * dwds
    tice[i] = tice[i] - t0c
    stsice[i, 0] = np.minimum(stsice[i, 0] - t0c, tfi0)     # degc
    stsice[i, 1] = np.minimum(stsice[i, 1] - t0c, tfi0)     # degc

    ip[i] = i0 * sneti[i] # ip +v here (in winton ip=-i0*sneti)

    i = flag & (snowd > 0.)
    tsf[i] = 0.
    ip[i]  = 0.

    i = flag & ~i
    tsf[i] = tfi
    ip[i] = i0 * sneti[i]  # ip +v here (in winton ip=-i0*sneti)

    i = flag
    tice[i] = np.minimum(tice[i], tsf[i])

    # compute ice temperature

    bi[i] = hfd[i]
    ai[i] = hfi[i] - sneti[i] + ip[i] - tice[i] * bi[i] # +v sol input here
    k12[i] = ki4 * ks / (ks * hice[i] + ki4 * snowd[i])
    k32[i] = (ki + ki) / hice[i]

    wrk[i] = 1. / (dt6 * k32[i] + dici * hice[i])
    a10[i] = dici * hice[i] * dt2i + \
        k32[i] * (dt4 * k32[i] + dici * hice[i]) * wrk[i]
    b10[i] = -di * hice[i] * (ci * stsice[i, 0] + li * tfi / \
            stsice[i, 0]) * dt2i - ip[i] - k32[i] * \
            (dt4 * k32[i] * tfw + dici * hice[i] * stsice[i,1]) * wrk[i]

    wrk1[i] = k12[i] / (k12[i] + bi[i])
    a1[i] = a10[i] + bi[i] * wrk1[i]
    b1[i] = b10[i] + ai[i] * wrk1[i]
    c1[i]   = dili * tfi * dt2i * hice[i]

    stsice[i, 0] = -(np.sqrt(b1[i] * b1[i] - 4. * a1[i] * c1[i]) + b1[i]) / \
        (a1[i] + a1[i])
    tice[i] = (k12[i] * stsice[i, 0] - ai[i]) / (k12[i] + bi[i])

    i = flag & (tice > tsf)
    a1[i] = a10[i] + k12[i]
    b1[i] = b10[i] - k12[i] * tsf[i]
    stsice[i, 0] = -(np.sqrt(b1[i] * b1[i] - 4. * a1[i] * c1[i]) + b1[i]) / \
        (a1[i] + a1[i])
    tice[i] = tsf[i]
    tmelt[i] = (k12[i] * (stsice[i, 0] - tsf[i]) - (ai[i] + bi[i] * tsf[i])) * delt

    i = flag & ~i
    tmelt[i] = 0.
    snowd[i] = snowd[i] + snof[i] * delt

    i = flag
    stsice[i, 1] = (dt2 * k32[i] * (stsice[i, 0] + tfw + tfw) + \
        dici * hice[i] * stsice[i, 1]) * wrk[i]
    bmelt[i] = (focn[i] + ki4 * (stsice[i, 1] - tfw) / hice[i]) * delt

#  --- ...  resize the ice ...

    h1[i] = 0.5 * hice[i]
    h2[i] = 0.5 * hice[i]

#  --- ...  top ...
    i = flag & (tmelt <= snowd * dsli)
    snowmt[i] = tmelt[i] / dsli
    snowd[i] = snowd[i] - snowmt[i]

    i = flag & ~i
    snowmt[i] = snowd[i]
    h1[i] = h1[i] - (tmelt[i] - snowd[i] * dsli) / \
            (di * (ci - li / stsice[i, 0]) * (tfi - stsice[i, 0]))
    snowd[i] = 0.

#  --- ...  and bottom

    i = flag & (bmelt < 0.)
    dh[i] = -bmelt[i] / (dili + dici * (tfi - tfw))
    stsice[i, 1] = (h2[i] * stsice[i, 1] + dh[i] * tfw) / (h2[i] + dh[i])
    h2[i] = h2[i] + dh[i]

    i = flag & ~i
    h2[i] = h2[i] - bmelt[i] / (dili + dici * (tfi - stsice[i, 1]))

#  --- ...  if ice remains, even up 2 layers, else, pass negative energy back in snow

    i = flag
    hice[i] = h1[i] + h2[i]

    # begin if_hice_block
    i = flag & (hice > 0.)
    # begin if_h1_block
    j = i & (h1 > 0.5*hice)
    f1[j] = 1. - 2*h2[j]/hice[j]
    stsice[j, 1] = f1[j] * (stsice[j, 0] + li*tfi/ \
            (ci*stsice[j,0])) + (1. - f1[j])*stsice[j,1]

    # begin if_stsice_block
    k = j & (stsice[:,1] > tfi)
    hice[k] = hice[k] - h2[k]* ci*(stsice[k, 1] - tfi)/(li*delt)
    stsice[k, 1] = tfi
    # end if_stsice_block

    # else if_h1_block
    j = flag & ~j
    f1[j] = 2*h1[j]/hice[j]
    stsice[j, 0] = f1[j]*(stsice[j,0] + li*tfi/ \
            (ci*stsice[j,0])) + (1. - f1[j])*stsice[j,1]
    stsice[j,0]= (stsice[j,0] - np.sqrt(stsice[j,0]\
            *stsice[j,0] - 4.0*tfi*li/ci)) * 0.5
    # end if_h1_block

    k12[i] = ki4*ks / (ks*hice[i] + ki4*snowd[i])
    gflux[i] = k12[i]*(stsice[i,0] - tice[i])
    
    # else if_hice_block
    i = flag & ~i
    snowd[i] = snowd[i] + (h1[i]*(ci*(stsice[i, 0] - tfi)\
            - li*(1. - tfi/stsice[i, 0])) + h2[i]*(ci*\
            (stsice[i, 1] - tfi) - li)) / li
    hice[i] = np.maximum(0., snowd[i]*dsdi)
    snowd[i] = 0.
    stsice[i, 0] = tfw
    stsice[i, 1] = tfw
    gflux[i] = 0.

    # end if_hice_block
    i = flag
    gflux[i] = fice[i] * gflux[i]
    snowmt[i] = snowmt[i] * dsdw
    snowd[i] = snowd[i] * dsdw
    tice[i] = tice[i]     + t0c
    stsice[i,0] = stsice[i,0] + t0c
    stsice[i,1] = stsice[i,1] + t0c


# TODO - this hsould be moved into a shared physics functions module
def fpvs(t):
    """Compute saturation vapor pressure
       t: Temperature [K]
    fpvs: Vapor pressure [Pa]
    """

    # constants
    # TODO - this should be moved into a shared physics constants module
    con_psat = 6.1078e+2
    con_ttp  = 2.7316e+2
    con_cvap = 1.8460e+3
    con_cliq = 4.1855e+3
    con_hvap = 2.5000e+6
    con_rv   = 4.6150e+2
    con_csol = 2.1060e+3
    con_hfus = 3.3358e+5

    tliq = con_ttp
    tice = con_ttp - 20.0
    dldtl = con_cvap - con_cliq
    heatl = con_hvap
    xponal = -dldtl / con_rv
    xponbl = -dldtl / con_rv + heatl / (con_rv * con_ttp)
    dldti = con_cvap - con_csol
    heati = con_hvap + con_hfus
    xponai = -dldti / con_rv
    xponbi = -dldti / con_rv + heati / (con_rv * con_ttp)

    convert_to_scalar = False
    if np.isscalar(t):
        t = np.array(t)
        convert_to_scalar = True

    fpvs = np.empty_like(t)
    tr = con_ttp / t

    ind1 = t >= tliq
    fpvs[ind1] = con_psat * (tr[ind1]**xponal) * np.exp(xponbl*(1. - tr[ind1]))

    ind2 = t < tice
    fpvs[ind2] = con_psat * (tr[ind2]**xponai) * np.exp(xponbi*(1. - tr[ind2]))

    ind3 = ~np.logical_or(ind1, ind2)
    w = (t[ind3] - tice) / (tliq - tice)
    pvl = con_psat * (tr[ind3]**xponal) * np.exp(xponbl*(1. - tr[ind3]))
    pvi = con_psat * (tr[ind3]**xponai) * np.exp(xponbi*(1. - tr[ind3]))
    fpvs[ind3] = w * pvl + (1. - w) * pvi

    if convert_to_scalar:
        fpvs = fpvs.item()

    return fpvs


def init_array(shape, mode):
    arr = np.empty(shape)
    if mode == "none":
        pass
    if mode == "zero":
        arr[:] = 0.
    elif mode == "nan":
        arr[:] = np.nan
    return arr
