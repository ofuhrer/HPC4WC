import gt4py as gt
from gt4py import gtscript
# import sys
# sys.path.append("..")
from shalconv.kernels.stencils_part1 import *
from shalconv.kernels.stencils_part2 import *
from shalconv.kernels.stencils_part34 import *
from shalconv.serialization import read_data, compare_data, OUT_VARS, numpy_dict_to_gt4py_dict
from shalconv import *
from shalconv.physcons import (
    con_g     as grav,
    con_cp    as cp,
    con_hvap  as hvap,
    con_rv    as rv,
    con_fvirt as fv,
    con_t0c   as t0c,
    con_rd    as rd,
    con_cvap  as cvap,
    con_cliq  as cliq,
    con_eps   as eps,
    con_epsm1 as epsm1,
    con_e     as e
)


def samfshalcnv_func(data_dict):
    """
    Scale-Aware Mass-Flux Shallow Convection

    :param data_dict: Dict of parameters required by the scheme
    :type data_dict: Dict of either scalar or gt4py storage
    """

    ############################ INITIALIZATION ############################

    ### Input variables and arrays ###
    im = data_dict["im"]
    ix = data_dict["ix"]
    km = data_dict["km"]
    itc = data_dict["itc"]
    ntc = data_dict["ntc"]
    ntk = data_dict["ntk"]
    ntr = data_dict["ntr"]
    ncloud = data_dict["ncloud"]
    clam = data_dict["clam"]
    c0s = data_dict["c0s"]
    c1 = data_dict["c1"]
    asolfac = data_dict["asolfac"]
    pgcon = data_dict["pgcon"]
    delt = data_dict["delt"]
    islimsk = data_dict["islimsk"]
    psp = data_dict["psp"]
    delp = data_dict["delp"]
    prslp = data_dict["prslp"]
    garea = data_dict["garea"]
    hpbl = data_dict["hpbl"]
    dot = data_dict["dot"]
    phil = data_dict["phil"]
    #fscav = data_dict["fscav"]

    ### Output buffers ###
    kcnv = data_dict["kcnv"]
    kbot = data_dict["kbot"]
    ktop = data_dict["ktop"]
    qtr = data_dict["qtr"]
    q1 = data_dict["q1"]
    t1 = data_dict["t1"]
    u1 = data_dict["u1"]
    v1 = data_dict["v1"]
    rn = data_dict["rn"]
    cnvw = data_dict["cnvw"]
    cnvc = data_dict["cnvc"]
    ud_mf = data_dict["ud_mf"]
    dt_mf = data_dict["dt_mf"]

    shape = (1, ix, km)

    ### Local storages for 1D arrays (integer) ###
    kpbl = gt.storage.ones(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kb = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kbcon = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kbcon1 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    ktcon = gt.storage.ones(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    ktcon1 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    ktconn = gt.storage.ones(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kbm = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kmax = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)

    ### Local storages for 1D arrays ("bool") ###
    cnvflg = gt.storage.ones(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    flg = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)

    ### Local storages for 1D arrays (float) ###
    aa1 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    cina = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    tkemean = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    clamt = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ps = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    del0 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    prsl = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    umean = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    tauadv = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    gdx = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delhbar = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delq = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delq2 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delqbar = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delqev = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    deltbar = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    deltv = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dtconv = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    edt = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pdot = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    po = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qcond = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qevap = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    hmax = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    rntot = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    vshear = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xlamud = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xmb = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xmbmax = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delubar = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delvbar = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    c0 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    wc = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    scaldfunc = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    sigmagfm = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qlko_ktcon = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    sumx = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    tx1 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)

    ### Local storages for 2D arrays (float) ###
    pfld = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    to = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qo = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    uo = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    vo = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qeso = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    wu2 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    buo = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    drag = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellal = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dbyo = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zo = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xlamue = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    heo = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    heso = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellah = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellaq = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellau = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellav = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    hcko = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ucko = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    vcko = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qcko = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qrcko = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    eta = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pwo = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    c0t = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    cnvwt = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)

    ### Local storages for 2D arrays (float, tracers), this will contain slices along n-axis ###
    shape_2d = (im, ntr)
    delebar = np.zeros(shape_2d, dtype=DTYPE_FLOAT)

    ### Local storages for 3D arrays (float, tracers), this will contain slices along n-axis ###
    shape_3d = (im, km, ntr)
    ctr = np.zeros(shape_3d, dtype=DTYPE_FLOAT)
    ctro = np.zeros(shape_3d, dtype=DTYPE_FLOAT)
    dellae = np.zeros(shape_3d, dtype=DTYPE_FLOAT)
    ecko = np.zeros(shape_3d, dtype=DTYPE_FLOAT)
    #qaero = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)

    ### K-indices field ###
    k_idx = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)

    ### State buffer for 1D-2D interactions
    state_buf1 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    state_buf2 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)

    ### PART2 Specific
    heo_kb = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dot_kbcon = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kbcon = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kb = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kbcon1 = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qtr_slice = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ctr_slice = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ctro_slice = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ecko_slice = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### PART3 Specific
    dellae_slice = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi_ktcon1 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi_kbcon1 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### PART4 Specific
    delebar_slice = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)

    ### Local Parameters ###
    g = grav
    elocp = hvap / cp
    el2orc = hvap * hvap / (rv * cp)
    d0 = 0.001
    cm = 1.0
    delta = fv
    fact1 = (cvap - cliq) / rv
    fact2 = hvap / rv - fact1 * t0c
    clamd = 0.1
    tkemx = 0.65
    tkemn = 0.05
    dtke = tkemx - tkemn
    dthk = 25.0
    cinpcrmx = 180.0
    cinpcrmn = 120.0
    cinacrmx = -120.0
    cinacrmn = -80.0
    crtlamd = 3.0e-4
    dtmax = 10800.0
    dtmin = 600.0
    bet1 = 1.875
    cd1 = 0.506
    f1 = 2.0
    gam1 = 0.5
    betaw = 0.03
    dxcrt = 15.0e3
    h1 = 0.33333333
    tf = 233.16
    tcr = 263.16
    tcrf = 1.0 / (tcr - tf)

    ### Determine whether to perform aerosol transport ###
    do_aerosols = (itc > 0) and (ntc > 0) and (ntr > 0)
    if (do_aerosols):
        do_aerosols = (ntr >= itc)

    ### Compute preliminary quantities needed for the static and feedback control portions of the algorithm ###

    # Convert input Pa terms to Cb terms
    pa_to_cb(psp, prslp, delp, ps, prsl, del0)

    km1 = km - 1

    ### Initialize storages (simple initializations already done above with gt4py functionalities) ###

    # Initialize column-integrated and other single-value-per-column
    # variable arrays
    init_col_arr(kcnv, cnvflg, kbot, ktop, kbcon, kb, rn, gdx, garea, km=km)

    # Return to the calling routine if deep convection is present or the
    # surface buoyancy flux is negative
    if exit_routine(cnvflg, im): return

    # Initialize further parameters and arrays
    init_par_and_arr( islimsk, c0, t1, c0t, cnvw, cnvc, ud_mf, dt_mf,
                      c0s=c0s, asolfac=asolfac, d0=d0)

    dt2 = delt

    # Model tunable parameters are all here
    aafac = 0.05
    evfact = 0.3
    evfactl = 0.3
    w1l = -8.0e-3
    w2l = -4.0e-2
    w3l = -5.0e-3
    w4l = -5.0e-4
    w1s = -2.0e-4
    w2s = -2.0e-3
    w3s = -1.0e-3
    w4s = -2.0e-5

    # Initialize the rest
    init_kbm_kmax(kbm, k_idx, kmax, state_buf1, state_buf2, tx1, ps, prsl, km=km)
    init_final( kbm, k_idx, kmax, flg, cnvflg, kpbl, tx1,
                ps, prsl, zo, phil, zi, pfld, eta, hcko, qcko,
                qrcko, ucko, vcko, dbyo, pwo, dellal, to, qo,
                uo, vo, wu2, buo, drag, cnvwt, qeso, heo, heso, hpbl,
                t1, q1, u1, v1, km=km)
               
    # Tracers Loop
    for n in range(ntr):
        
        kk = n+2
        
        qtr_slice[...] = qtr[np.newaxis, :, :, kk]
        
        # Initialize tracers. Keep in mind that, qtr slice is for the 
        # (n+2)-th tracer, while the other storages are slices 
        # representing the n-th tracer.
        init_tracers( cnvflg, k_idx, kmax, ctr_slice, ctro_slice, ecko_slice, qtr_slice )

        ctr_slice.synchronize()
        ctro_slice.synchronize()
        ecko_slice.synchronize()
        ctr[:, :, n] = ctr_slice[0, :, :].view(np.ndarray)
        ctro[:, :, n] = ctro_slice[0, :, :].view(np.ndarray)
        ecko[:, :, n] = ecko_slice[0, :, :].view(np.ndarray)
        

    #=======================================PART2=====================================
    #=================================================================================
    ### Search in the PBL for the level of maximum moist static energy to start the ascending parcel.
    stencil_static0(cnvflg, hmax, heo, kb, k_idx, kpbl, kmax, zo, to, qeso, qo, po, uo, vo, heso, pfld)
    
    for n in range(ntr):
        ctro_slice[...] = ctro[np.newaxis, :, :, n]
        stencil_ntrstatic0(cnvflg, k_idx, kmax, ctro_slice)
        ctro[:, :, n] = ctro_slice[0, :, :]

    ### Search below the index "kbm" for the level of free convection (LFC) where the condition.
    get_1D_from_index(heo, heo_kb, kb, k_idx)
    stencil_static1(cnvflg, flg, kbcon, kmax, k_idx, kbm, kb, heo_kb, heso)

    ### If no LFC, return to the calling routine without modifying state variables.
    if exit_routine(cnvflg, ix): return

    ### Determine the vertical pressure velocity at the LFC.
    get_1D_from_index(dot, dot_kbcon, kbcon, k_idx)
    get_1D_from_index(pfld, pfld_kbcon, kbcon, k_idx)
    get_1D_from_index(pfld, pfld_kb, kb, k_idx)
    stencil_static2(cnvflg, pdot, dot_kbcon, islimsk, k_idx, kbcon, kb, pfld_kb, pfld_kbcon)

    ### If no LFC, return to the calling routine without modifying state variables.
    if exit_routine(cnvflg, ix): return

    ### turbulent entrainment rate assumed to be proportional to subcloud mean TKE
    if (ntk > 0):
        qtr_ntk = gt.storage.from_array(qtr[np.newaxis, :, :, ntk - 1], BACKEND, default_origin)
        stencil_static3(sumx, tkemean, cnvflg, k_idx, kb, kbcon, zo, qtr_ntk,
                        clamt, clam=clam)
    #    qtr[:,:,ntr] = qtr_ntr[0,:,:]
    # else:
    # stencil_static4(cnvflg, clamt, clam=clam )

    ### assume updraft entrainment rate is an inverse function of height
    stencil_static5(cnvflg, xlamue, clamt, zi, xlamud, k_idx, kbcon, kb,
                    eta, ktconn, kmax, kbm, hcko, ucko, vcko, heo, uo, vo)

    for n in range(ntr):
        ctro_slice[...] = ctro[np.newaxis, :, :, n]
        ecko_slice[...] = ecko[np.newaxis, :, :, n]
        stencil_ntrstatic1(cnvflg, k_idx, kb, ecko_slice, ctro_slice)
        ecko[:, :, n] = ecko_slice[0, :, :]

    stencil_static7(cnvflg, k_idx, kb, kmax, zi, xlamue, xlamud, hcko, heo, dbyo,
                    heso, ucko, uo, vcko, vo, pgcon=pgcon)

    for n in range(ntr):
        ctro_slice[...] = ctro[np.newaxis, :, :, n]
        ecko_slice[...] = ecko[np.newaxis, :, :, n]
        stencil_ntrstatic2(cnvflg, k_idx, kb, kmax, zi, xlamue, ecko_slice, ctro_slice)
        ecko[:, :, n] = ecko_slice[0, :, :]

    stencil_update_kbcon1_cnvflg(dbyo, cnvflg, kmax, kbm, kbcon, kbcon1, flg, k_idx)
    get_1D_from_index(pfld, pfld_kbcon1, kbcon1, k_idx)
    stencil_static9(cnvflg, pfld_kbcon, pfld_kbcon1)

    if exit_routine(cnvflg, ix): return

    ### calculate convective inhibition
    stencil_static10(cina, cnvflg, k_idx, kb, kbcon1, zo, qeso, to,
                     dbyo, qo, pdot, islimsk)

    if exit_routine(cnvflg, ix): return

    dt2 = delt
    stencil_static11(flg, cnvflg, ktcon, kbm, kbcon1, dbyo, kbcon, del0, xmbmax,
                     aa1, kb, qcko, qo, qrcko, zi, qeso, to, xlamue,
                     xlamud, eta, c0t, dellal, buo, drag, zo, k_idx, pwo,
                     cnvwt, c1=c1, dt2=dt2, ncloud=ncloud)

    if exit_routine(cnvflg, ix): return

    stencil_static12(cnvflg, aa1, flg, ktcon1, kbm, k_idx, ktcon, zo, qeso,
                     to, dbyo, zi, xlamue, xlamud, qcko, qrcko, qo, eta, del0,
                     c0t, pwo, cnvwt, buo, wu2, wc, sumx, kbcon1, drag, dellal,
                     c1=c1, ncloud=ncloud)

    if(ncloud > 0):
        stencil_static13(cnvflg, k_idx, ktcon, qeso, to, dbyo, qcko, qlko_ktcon)
    
    stencil_static14(cnvflg, vshear, k_idx, kb, ktcon, uo, vo, zi, edt)

    #=======================================PART34=====================================
    #=================================================================================
    # Calculate the tendencies of the state variables (per unit cloud base
    # mass flux) and the cloud base mass flux
    comp_tendencies( cnvflg, k_idx, kmax, kb, ktcon, ktcon1, kbcon1, kbcon,
                     dellah, dellaq, dellau, dellav, del0, zi, zi_ktcon1,
                     zi_kbcon1, heo, qo, xlamue, xlamud, eta, hcko,
                     qrcko, uo, ucko, vo, vcko, qcko, dellal,
                     qlko_ktcon, wc, gdx, dtconv, u1, v1, po, to,
                     tauadv, xmb, sigmagfm, garea, scaldfunc, xmbmax,
                     sumx, umean,
                     g=g, betaw=betaw, dtmin=dtmin, dt2=dt2, dtmax=dtmax, dxcrt=dxcrt)
                     
    # Calculate the tendencies of the state variables (tracers part)             
    for n in range(ntr):
        
        ctro_slice[...] = ctro[np.newaxis, :, :, n]
        ecko_slice[...] = ecko[np.newaxis, :, :, n]
        comp_tendencies_tr( cnvflg, k_idx, kmax, kb, ktcon,
                            dellae_slice, del0, eta, ctro_slice, ecko_slice, g=g )
        dellae[:, :, n] = dellae_slice[0, :, :]

    # For the "feedback control", calculate updated values of the state
    # variables by multiplying the cloud base mass flux and the
    # tendencies calculated per unit cloud base mass flux from the
    # static control
    feedback_control_update( cnvflg, k_idx, kmax, kb, ktcon, flg,
                             islimsk, ktop, kbot, kbcon, kcnv, qeso,
                             pfld, delhbar, delqbar, deltbar, delubar,
                             delvbar, qcond, dellah, dellaq, t1, xmb,
                             q1, u1, dellau, v1, dellav, del0, rntot,
                             delqev, delq2, pwo, deltv, delq, qevap, rn,
                             edt, cnvw, cnvwt, cnvc, ud_mf, dt_mf, eta,
                             dt2=dt2, g=g, evfact=evfact, evfactl=evfactl,
                             el2orc=el2orc, elocp=elocp)
                             
    # Calculate updated values of the state variables (tracers part)                 
    for n in range(0, ntr):
        
        kk = n+2
        qtr_slice[...] = qtr[np.newaxis, :, :, kk]
        
        ctr_slice[...] = ctr[np.newaxis, :, :, n]
        dellae_slice[...] = dellae[np.newaxis, :, :, n]
        
        feedback_control_upd_trr( cnvflg, k_idx, kmax, ktcon,
                                  del0, delebar_slice, ctr_slice, 
                                  dellae_slice, xmb, qtr_slice,
                                  dt2=dt2, g=g)
        
        delebar[:, n] = delebar_slice[0, :, 0]
        qtr[:, :, kk] = qtr_slice[0, :, :]
        ctr[:, :, n] = ctr_slice[0, :, :]
    
    # Separate detrained cloud water into liquid and ice species as a 
    # function of temperature only
    if ncloud > 0:
        
        qtr_0 = gt.storage.from_array(qtr[np.newaxis, :, :, 0], BACKEND, default_origin)
        qtr_1 = gt.storage.from_array(qtr[np.newaxis, :, :, 1], BACKEND, default_origin)
        
        separate_detrained_cw( cnvflg, k_idx, kbcon,
                               ktcon, dellal, xmb, t1, qtr_1, qtr_0,
                               dt2=dt2, tcr=tcr, tcrf=tcrf)
                               
        qtr[:, :, 0] = qtr_0[0, :, :]
        qtr[:, :, 1] = qtr_1[0, :, :]
    
    # Include TKE contribution from shallow convection
    if ntk > 0:
        
        qtr_ntk = gt.storage.from_array(qtr[np.newaxis, :, :, ntk - 1], BACKEND, default_origin)
        
        tke_contribution( cnvflg, k_idx, kb, ktop,
                          eta, xmb, pfld, t1, sigmagfm, qtr_ntk,
                          betaw=betaw)
                          
        qtr[:, :, ntk - 1] = qtr_ntk[0, :, :]
        

    return kcnv, kbot, ktop, q1, t1, u1, v1, rn, cnvw, cnvc, ud_mf, dt_mf, qtr

