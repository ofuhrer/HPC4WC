#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import gt4py as gt
from gt4py import gtscript

import timeit

# BACKEND = "gtcuda"
# BACKEND = "gtx86"
BACKEND = "numpy"

DT_F = gtscript.Field[np.float64]
DT_I = gtscript.Field[np.int32]

SCALAR_VARS = ["delt", "cimin"]

IN_VARS = ["ps", "t1", "q1", "sfcemis", "dlwflx", "sfcnsw", "sfcdsw", "srflag",
    "cm", "ch", "prsl1", "prslki", "islimsk", "wind", "flag_iter"]

OUT_VARS = ["tskin", "tprcp", "fice", "gflux", "ep", "stc0", "stc1", "tice", \
    "snowmt", "evap", "snwdph", "chh", "weasd", "hflx", "qsurf", \
    "hice", "cmm"]

# mathematical constants
EULER = 2.7182818284590452353602874713527

# TODO: All these constants should be moved into separate const file
# physical constants
HVAP = 2.5e6       # lat heat H2O cond (J/kg)
RV = 4.615e2       # gas constant H2O (J/kg/K)
PSAT = 6.1078e2    # pres at H2O 3pt (Pa)
TTP = 2.7316e2     # temp at H2O 3pt (K)
CVAP = 1.846e3     # spec heat H2O gas (J/kg/K)
CLIQ = 4.1855e3    # spec heat H2O liq (J/kg/K)
CSOL = 2.106e3     # spec heat H2O ice (J/kg/K)
HFUS = 3.3358e5    # lat heat H2O fusion (J/kg)

# constant parameters
DS  = 330.      # snow (ov sea ice) density (kg/m^3)
DW  = 1000.     # fresh water density  (kg/m^3)
KS  = 0.31      # conductivity of snow   (w/mk)
I0  = 0.3       # ice surface penetrating solar fraction
KI  = 2.03      # conductivity of ice  (w/mk)
DI  = 917.      # density of ice   (kg/m^3)
CI  = 2054.     # heat capacity of fresh ice (j/kg/k)
LI  = 3.34e5    # latent heat of fusion (j/kg-ice)
SI  = 1.        # salinity of sea ice
MU  = 0.054     # relates freezing temp to salinity
TFW = -1.8      # TFW - seawater freezing temp (c)
T0C = 2.7315e+2 # temp (K) of 0C

# constants definition
CP    = 1.0046e+3  # spec heat air at p (J/kg/K)
SBC   = 5.6704e-8  # stefan-boltzmann (W/m^2/K^4)
TGICE = 2.712e+2
RD    = 2.8705e+2  # gas constant air (J/kg/K)

# fvps constants
FPVS_XPONAL = -(CVAP - CLIQ) / RV
FPVS_XPONBL = -(CVAP - CLIQ) / RV + HVAP / (RV * TTP)
FPVS_XPONAI = -(CVAP - CSOL) / RV
FPVS_XPONBI = -(CVAP - CSOL) / RV + (HVAP + HFUS) / (RV * TTP)

# ice3lay constants
DSDW = DS / DW
DWDS = DW / DS
DIDW = DI / DW
DSDI = DS / DI
TFI  = -MU * SI # sea ice freezing temp = -MU*salinity
TFI0 = TFI - 0.0001
DICI = DI * CI
DILI = DI * LI
DSLI = DS * LI
KI4  = KI * 4.

# sfc_sice constants
EPS    = RD / RV
EPSM1  = RD / RV - 1.
RVRDM1 = RV / RD - 1.
ELOCP  = HVAP / CP
HIMAX  = 8.          # maximum ice thickness allowed
HIMIN  = 0.1         # minimum ice thickness required
HSMAX  = 2.          # maximum snow depth allowed
TIMIN  = 173.        # minimum temperature allowed for snow/ice
ALBFW  = 0.06        # albedo for lead
DSI    = 1. / 0.33

# other constants
INIT_VALUE = 0.  # TODO - this should be float("NaN") but not possible 07/20


def numpy_to_gt4py_storage(arr, backend="numpy"):
    """convert numpy storage to gt4py storage"""
    data = np.reshape(arr, (arr.shape[0], 1, 1))
    if data.dtype == "bool":
        data = data.astype(np.int32)
    return gt.storage.from_array(data, backend=backend, default_origin=(0, 0, 0))


def gt4py_to_numpy_storage(arr):
    """convert gt4py storage to numpy storage"""
    data = arr.view(np.ndarray)
    return np.reshape(data, (data.shape[0]))


def run(in_dict, backend=BACKEND):
    """Run function for GFS thermodynamics surface ice model 

    With this function, the GFS thermodynamics surface ice model can be run
    as a standalone parameterization.
    """

    # special handling of stc
    stc = in_dict.pop("stc")
    in_dict["stc0"] = stc[:, 0]
    in_dict["stc1"] = stc[:, 1]

    # setup storages
    scalar_dict = {k: in_dict[k] for k in SCALAR_VARS}
    out_dict = {k: numpy_to_gt4py_storage(in_dict[k].copy(), backend=backend) for k in OUT_VARS}
    in_dict = {k: numpy_to_gt4py_storage(in_dict[k], backend=backend) for k in IN_VARS}

    # compile stencil
    sfc_sice = gtscript.stencil(definition=sfc_sice_defs, backend=backend, externals={})

    # set timer
    tic = timeit.default_timer()

    # call sea-ice parametrization
    sfc_sice(**scalar_dict, **in_dict, **out_dict)

    # set timer
    toc = timeit.default_timer()

    # calculateelapsed time
    elapsed_time = toc - tic

    # convert back to numpy for validation
    out_dict = {k: gt4py_to_numpy_storage(out_dict[k]) for k in OUT_VARS}

    # special handling of stc
    stc[:, 0] = out_dict.pop("stc0")[:]
    stc[:, 1] = out_dict.pop("stc1")[:]
    out_dict["stc"] = stc

    return out_dict, elapsed_time


@gtscript.function
def exp_fn(a):
    """Compute exponential function"""
    return EULER ** a


@gtscript.function
def fpvs_fn(t):
    """Compute saturation vapor pressure
       t: Temperature
    fpvs: Vapor pressure [Pa]
    """

    tr = TTP / t

    # over liquid
    pvl = PSAT * (tr ** FPVS_XPONAL) * exp_fn(FPVS_XPONBL * (1. - tr))

    # over ice
    pvi = PSAT * (tr ** FPVS_XPONAI) * exp_fn(FPVS_XPONBI * (1. - tr))

    # determine regime weight
    w = (t - TTP + 20.) / 20.

    fpvs = INIT_VALUE
    if w >= 1.0:
        fpvs = pvl
    elif w < 0.0:
        fpvs = pvi
    else:
        fpvs = w * pvl + (1. - w) * pvi

    return fpvs


@gtscript.function
def ice3lay(fice, flag, hfi, hfd, sneti, focn, delt, snowd, hice,
            stc0, stc1, tice, snof, snowmt, gflux):
    """three-layer sea ice vertical thermodynamics
                                                                           *
    based on:  m. winton, "a reformulated three-layer sea ice model",      *
    journal of atmospheric and oceanic technology, 2000                    *
                                                                           *
                                                                           *
          -> +---------+ <- tice - diagnostic surface temperature ( <= 0c )*
         /   |         |                                                   *
     snowd   |  snow   | <- 0-heat capacity snow layer                     *
         \   |         |                                                   *
          => +---------+                                                   *
         /   |         |                                                   *
        /    |         | <- t1 - upper 1/2 ice temperature; this layer has *
       /     |         |         a variable (t/s dependent) heat capacity  *
     hice    |...ice...|                                                   *
       \     |         |                                                   *
        \    |         | <- t2 - lower 1/2 ice temp. (fixed heat capacity) *
         \   |         |                                                   *
          -> +---------+ <- base of ice fixed at seawater freezing temp.   *
                                                                           *
    =====================  definition of variables  =====================  *
                                                                           *
    inputs:                                                         size   *
       fice     - real, sea-ice concentration                         im   *
       flag     - logical, ice mask flag                              1    *
       hfi      - real, net non-solar and heat flux @ surface(w/m^2)  im   *
       hfd      - real, heat flux derivatice @ sfc (w/m^2/deg-c)      im   *
       sneti    - real, net solar incoming at top  (w/m^2)            im   *
       focn     - real, heat flux from ocean    (w/m^2)               im   *
       delt     - real, timestep                (sec)                 1    *
                                                                           *
    input/outputs:                                                         *
       snowd    - real, surface pressure                              im   *
       hice     - real, sea-ice thickness                             im   *
       stc0     - real, temp @ midpt of ice levels (deg c), 1st layer im   *     
       stc1     - real, temp @ midpt of ice levels (deg c), 2nd layer im   *     
       tice     - real, surface temperature     (deg c)               im   *
       snof     - real, snowfall rate           (m/sec)               im   *
                                                                           *
    outputs:                                                               *
       snowmt   - real, snow melt during delt   (m)                   im   *
       gflux    - real, conductive heat flux    (w/m^2)               im   *
                                                                           *
    locals:                                                                *
       hdi      - real, ice-water interface     (m)                        *
       hsni     - real, snow-ice                (m)                        *
                                                                           *
    ====================================================================== *
    """

    # TODO - only initialize the arrays which are defined in an if-statement
    hdi = INIT_VALUE
    ip  = INIT_VALUE
    tsf = INIT_VALUE
    ai  = INIT_VALUE
    bi  = INIT_VALUE
    k12 = INIT_VALUE
    k32 = INIT_VALUE
    wrk = INIT_VALUE
    wrk1 = INIT_VALUE
    a10 = INIT_VALUE
    b10 = INIT_VALUE
    a1  = INIT_VALUE
    b1  = INIT_VALUE
    c1  = INIT_VALUE
    tmelt = INIT_VALUE
    bmelt = INIT_VALUE
    h1 = INIT_VALUE
    h2 = INIT_VALUE
    dh = INIT_VALUE
    f1 = INIT_VALUE

    if flag:
        snowd = snowd  * DWDS
        hdi = (DSDW * snowd + DIDW * hice)

        if hice < hdi:
            snowd = snowd + hice - hdi
            hice  = hice + (hdi - hice) * DSDI

        snof = snof * DWDS
        tice = tice - T0C
        stc0 = stc0 - T0C if stc0 - T0C < TFI0 else TFI0
        stc1 = TFI0 if TFI0 < stc1 - T0C else stc1 - T0C     # degc

        ip = I0 * sneti # ip +v here (in winton ip=-I0*sneti)
        if snowd > 0.:
            tsf = 0.
            ip  = 0.
        else:
            tsf = TFI

        tice = tsf if tsf < tice else tice

        # compute ice temperature

        bi = hfd
        ai = hfi - sneti + ip - tice * bi # +v sol input here
        k12 = KI4 * KS / (KS * hice + KI4 * snowd)
        k32 = (KI + KI) / hice

        wrk = 1. / (6. * delt * k32 + DICI * hice)
        a10 = DICI * hice * (0.5 / delt) + \
            k32 * (4. * delt * k32 + DICI * hice) * wrk
        b10 = -DI * hice * (CI * stc0 + LI * TFI / \
                stc0) * (0.5 / delt) - ip - k32 * \
                (4. * delt * k32 * TFW + DICI * hice * stc1) * wrk

        wrk1 = k12 / (k12 + bi)
        a1 = a10 + bi * wrk1
        b1 = b10 + ai * wrk1
        c1   = DILI * TFI * (0.5 / delt) * hice

        stc0 = -((b1 * b1 - 4. * a1 * c1)**0.5 + b1) / (a1 + a1)
        tice = (k12 * stc0 - ai) / (k12 + bi)

        if tice > tsf:
            a1 = a10 + k12
            b1 = b10 - k12 * tsf
            stc0 = -((b1 * b1 - 4. * a1 * c1)**0.5 + b1) / (a1 + a1)
            tice = tsf
            tmelt = (k12 * (stc0 - tsf) - (ai + bi * tsf)) * delt
        else:
            tmelt = 0.
            snowd = snowd + snof * delt

        stc1 = (2. * delt * k32 * (stc0 + TFW + TFW) + \
            DICI * hice * stc1) * wrk
        bmelt = (focn + KI4 * (stc1 - TFW) / hice) * delt

    #  --- ...  resize the ice ...

        h1 = 0.5 * hice
        h2 = 0.5 * hice

    #  --- ...  top ...
        if tmelt <= snowd * DSLI:
            snowmt = tmelt / DSLI
            snowd = snowd - snowmt
        else:
            snowmt = snowd
            h1 = h1 - (tmelt - snowd * DSLI) / \
                    (DI * (CI - LI / stc0) * (TFI - stc0))
            snowd = 0.

    #  --- ...  and bottom

        if bmelt < 0.:
            dh = -bmelt / (DILI + DICI * (TFI - TFW))
            stc1 = (h2 * stc1 + dh * TFW) / (h2 + dh)
            h2 = h2 + dh
        else:
            h2 = h2 - bmelt / (DILI + DICI * (TFI - stc1))

    #  --- ...  if ice remains, even up 2 layers, else, pass negative energy back in snow

        hice = h1 + h2

        # begin if_hice_block
        if hice > 0.:
            if h1 > 0.5 * hice:
                f1 = 1. - 2. * h2 / hice
                stc1 = f1 * (stc0 + LI*TFI/ \
                        (CI*stc0)) + (1. - f1)*stc1

                if stc1 > TFI:
                    hice = hice - h2 * CI*(stc1 - TFI)/(LI*delt)
                    stc1 = TFI

            else:
                f1 = 2*h1/hice
                stc0 = f1*(stc0 + LI*TFI/ \
                        (CI*stc0)) + (1. - f1)*stc1
                stc0= (stc0 - (stc0 * stc0 - 4.0*TFI*LI/CI)**0.5) * 0.5

            k12 = KI4*KS / (KS*hice + KI4*snowd)
            gflux = k12*(stc0 - tice)

        else:
            snowd = snowd + (h1*(CI*(stc0 - TFI)\
                    - LI*(1. - TFI/stc0)) + h2*(CI*\
                    (stc1 - TFI) - LI)) / LI
            hice = snowd*DSDI if snowd*DSDI < 0. else 0.
            snowd = 0.
            stc0 = TFW
            stc1 = TFW
            gflux = 0.

        gflux = fice * gflux
        snowmt = snowmt * DSDW
        snowd = snowd * DSDW
        tice = tice + T0C
        stc0 = stc0 + T0C
        stc1 = stc1 + T0C

    return snowd, hice, stc0, stc1, tice, snof, snowmt, gflux


def sfc_sice_defs(
        ps: DT_F,
        t1: DT_F,
        q1: DT_F,
        sfcemis: DT_F,
        dlwflx: DT_F,
        sfcnsw: DT_F,
        sfcdsw: DT_F,
        srflag: DT_F,
        cm: DT_F,
        ch: DT_F,
        prsl1: DT_F,
        prslki: DT_F,
        islimsk: DT_I,
        wind: DT_F,
        flag_iter: DT_I,
        hice: DT_F,
        fice: DT_F,
        tice: DT_F,
        weasd: DT_F,
        tskin: DT_F,
        tprcp: DT_F,
        stc0: DT_F,
        stc1: DT_F,
        ep: DT_F,
        snwdph: DT_F,
        qsurf: DT_F,
        cmm: DT_F,
        chh: DT_F,
        evap: DT_F,
        hflx: DT_F,
        gflux: DT_F,
        snowmt: DT_F,
        *,
        delt: float,
        cimin: float
         ):
    """This file contains the GFS thermodynamics surface ice model.
   
    =====================================================================
    description:                                                        
   
    usage:                                                              
   
   
    program history log:                                                
           2005  --  xingren wu created  from original progtm and added 
                       two-layer ice model                              
           200x  -- sarah lu    added flag_iter                         
      oct  2006  -- h. wei      added cmm and chh to output             
           2007  -- x. wu modified for mom4 coupling (i.e. cpldice)     
                                      (not used anymore)                
           2007  -- s. moorthi micellaneous changes                     
      may  2009  -- y.-t. hou   modified to include surface emissivity  
                       effect on lw radiation. replaced the confusing   
                       slrad with sfc net sw sfcnsw (dn-up). reformatted
                       the code and add program documentation block.    
      sep  2009 -- s. moorthi removed rcl, changed pressure units and   
                       further optimized                                
      jan  2015 -- x. wu change "cimin = 0.15" for both                 
                       uncoupled and coupled case                       
      jul  2020 -- ETH-students port to gt4py
   
   
    ====================  definition of variables  ==================== 
   
    inputs:                                                       size  
       ps       - real, surface pressure                            im
       t1       - real, surface layer mean temperature ( k )        im
       q1       - real, surface layer mean specific humidity        im
       sfcemis  - real, sfc lw emissivity ( fraction )              im
       dlwflx   - real, total sky sfc downward lw flux ( w/m**2 )   im
       sfcnsw   - real, total sky sfc netsw flx into ground(w/m**2) im
       sfcdsw   - real, total sky sfc downward sw flux ( w/m**2 )   im
       srflag   - real, snow/rain fraction for precipitation        im
       cm       - real, surface exchange coeff for momentum (m/s)   im
       ch       - real, surface exchange coeff heat & moisture(m/s) im
       prsl1    - real, surface layer mean pressure                 im
       prslki   - real,                                             im
       islimsk  - integer, sea/land/ice mask (=0/1/2)               im
       wind     - real,                                             im
       flag_iter- logical,                                          im
       delt     - real, time interval (second)                      1
       cimin    - real, minimum ice fraction                        1

    input/outputs:
       hice     - real, sea-ice thickness                           im
       fice     - real, sea-ice concentration                       im
       tice     - real, sea-ice surface temperature                 im
       weasd    - real, water equivalent accumulated snow depth (mm)im
       tskin    - real, ground surface skin temperature ( k )       im
       tprcp    - real, total precipitation                         im
       stc0     - real, soil temp (k), 1. layer                     im
       stc1     - real, soil temp (k), 2nd layer                    im
       ep       - real, potential evaporation                       im

    outputs:
       snwdph   - real, water equivalent snow depth (mm)            im
       qsurf    - real, specific humidity at sfc                    im
       snowmt   - real, snow melt (m)                               im
       gflux    - real, soil heat flux (w/m**2)                     im
       cmm      - real, surface exchange coeff for momentum(m/s)    im
       chh      - real, surface exchange coeff heat&moisture (m/s)  im  
       evap     - real, evaperation from latent heat flux           im  
       hflx     - real, sensible heat flux                          im  
                                                                        
    =====================================================================
    """

    from __gtscript__ import PARALLEL, computation, interval

    with computation(PARALLEL), interval(...):

        # arrays
        # TODO - only initialize the arrays which are defined in an if-statement
        q0      = INIT_VALUE
        theta1  = INIT_VALUE
        rho     = INIT_VALUE
        ffw     = INIT_VALUE
        snowd   = INIT_VALUE
        rch     = INIT_VALUE
        evapi   = INIT_VALUE
        evapw   = INIT_VALUE
        sneti   = INIT_VALUE
        snetw   = INIT_VALUE
        t12     = INIT_VALUE
        t14     = INIT_VALUE
        hfi     = INIT_VALUE
        hfd     = INIT_VALUE
        focn    = INIT_VALUE
        snof    = INIT_VALUE
        hflxi   = INIT_VALUE
        hflxw   = INIT_VALUE

    #  --- ...  set flag for sea-ice

        flag = (islimsk == 2) and flag_iter

        if flag_iter and (islimsk < 2):
            hice = 0.
            fice = 0.

        qs1 = fpvs_fn(t1)
        if flag:
            if srflag > 0.:
                ep = ep * (1. - srflag)
                weasd = weasd + 1.e3 * tprcp * srflag
                tprcp = tprcp * (1. - srflag)

    #  --- ...  initialize variables. all units are supposedly m.k.s. unless specified
    #           psurf is in pascals, wind is wind speed, theta1 is adiabatic surface
    #           temp from level 1, rho is density, qs1 is sat. hum. at level1 and qss
    #           is sat. hum. at surface
    #           convert slrad to the civilized unit from langley minute-1 k-4

    #         dlwflx has been given a negative sign for downward longwave
    #         sfcnsw is the net shortwave flux (direction: dn-up)

            q0     = q1 if q1 > 1.0e-8 else 1.0e-8
            theta1 = t1 * prslki
            rho    = prsl1 / (RD * t1 * (1. + RVRDM1 * q0))

            qs1 = EPS * qs1 / (prsl1 + EPSM1 * qs1)
            qs1 = qs1 if qs1 > 1.e-8 else 1.e-8
            q0 = qs1 if qs1 < q0 else q0

            if fice < cimin:
                # print("warning: ice fraction is low:", fice)
                #fice = cimin
                tice = TGICE
                tskin= TGICE
                # print('fix ice fraction: reset it to:', fice)

            fice = fice if fice > cimin else cimin

        qssi = fpvs_fn(tice)
        qssw = fpvs_fn(TGICE)
        if flag:
            ffw  = 1. - fice
            qssi = EPS * qssi / (ps + EPSM1 * qssi)
            qssw = EPS * qssw / (ps + EPSM1 * qssw)

    #  --- ...  snow depth in water equivalent is converted from mm to m unit

            snowd = weasd * 0.001

    #  --- ...  when snow depth is less than 1 mm, a patchy snow is assumed and
    #           soil is allowed to interact with the atmosphere.
    #           we should eventually move to a linear combination of soil and
    #           snow under the condition of patchy snow.

    #  --- ...  rcp = rho CP ch v

            cmm = cm * wind
            chh = rho * ch * wind
            rch = chh * CP

    #  --- ...  sensible and latent heat flux over open water & sea ice

            evapi = ELOCP * rch * (qssi - q0)
            evapw = ELOCP * rch * (qssw - q0)

            snetw = sfcdsw * (1. - ALBFW)
            t12 = 3. * sfcnsw / (1. + 2. * ffw)
            snetw =  t12 if t12 < snetw else snetw
            sneti = (sfcnsw - ffw * snetw) / fice

            t12 = tice * tice
            t14 = t12 * t12

        #  --- ...  hfi = net non-solar and upir heat flux @ ice surface

            hfi = -dlwflx + sfcemis * SBC * t14 + evapi + \
                    rch * (tice - theta1)
            hfd = 4. * sfcemis * SBC * tice * t12 + \
                    (1. + ELOCP * EPS * HVAP * qs1 / (RD * t12)) * rch

            t12 = TGICE * TGICE
            t14 = t12 * t12

        #  --- ...  hfw = net heat flux @ water surface (within ice)

            focn = 2.   # heat flux from ocean - should be from ocn model
            snof = 0.   # snowfall rate - snow accumulates in gbphys

            # hice = np.maximum(np.minimum(hice, HIMAX), HIMIN)
            hice = HIMAX if HIMAX < hice else hice
            hice = HIMIN if HIMIN > hice else hice
            snowd = HSMAX if HSMAX < snowd else snowd

            if snowd > 2. * hice:
                # print('warning: too much snow :', snowd[i])
                snowd = hice + hice
                # print('fix: decrease snow depth to:', snowd[i])

        # run the 3-layer ice model
        snowd, hice, stc0, stc1, tice, snof, snowmt, gflux = ice3lay(
            fice, flag, hfi, hfd, sneti, focn, delt,
            snowd, hice, stc0, stc1, tice, snof, snowmt, gflux)

        if flag:
            if tice < TIMIN:
                # print('warning: snow/ice temperature is too low:', tice, ' i=', i)
                tice = TIMIN
                # print('fix snow/ice temperature: reset it to:', TIMIN)

            if stc0 < TIMIN:
                # print('warning: layer 1 ice temp is too low:', stsice[i, 0], ' i=', i)
                stc0 = TIMIN
                # print('fix layer 1 ice temp: reset it to:', TIMIN)

            if stc1 < TIMIN:
                # print('warning: layer 2 ice temp is too low:', stsice[i, 1], 'i=', i)
                stc1 = TIMIN
                # print('fix layer 2 ice temp: reset it to:', TIMIN)

            tskin = tice * fice + TGICE * ffw
            stc0 = T0C if T0C < stc0 else stc0
            stc1 = T0C if T0C < stc1 else stc1

        #  --- ...  calculate sensible heat flux (& evap over sea ice)

            hflxi = rch * (tice - theta1)
            hflxw = rch * (TGICE - theta1)
            hflx = fice * hflxi + ffw * hflxw
            evap = fice * evapi + ffw * evapw

        #  --- ...  the rest of the output

            qsurf = q1 + evap / (ELOCP * rch)

        #  --- ...  convert snow depth back to mm of water equivalent

            weasd = snowd * 1000.
            snwdph = weasd * DSI             # snow depth in mm

            hflx = hflx / rho * 1. / CP
            evap = evap / rho * 1. / HVAP
