import pytest
import gt4py as gt
from gt4py import gtscript
import sys
sys.path.append("..")
from shalconv.kernels.stencils_part1 import *
from read_serialization import *
from shalconv.serialization import *
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


def samfshalcnv_part1(data_dict):
    """
    Scale-Aware Mass-Flux Shallow Convection
    
    :param data_dict: Dict of parameters required by the scheme
    :type data_dict: Dict of either scalar or gt4py storage
    """

############################ INITIALIZATION ############################

    ### Input variables and arrays ###
    im         = data_dict["im"]
    ix         = data_dict["ix"]
    km         = data_dict["km"]
    itc        = data_dict["itc"]
    ntc        = data_dict["ntc"]
    ntk        = data_dict["ntk"]
    ntr        = data_dict["ntr"]
    ncloud     = data_dict["ncloud"]
    clam       = data_dict["clam"]
    c0s        = data_dict["c0s"]
    c1         = data_dict["c1"]
    asolfac    = data_dict["asolfac"]
    pgcon      = data_dict["pgcon"]
    delt       = data_dict["delt"]
    islimsk    = data_dict["islimsk"]
    psp        = data_dict["psp"]
    delp       = data_dict["delp"]
    prslp      = data_dict["prslp"]
    garea      = data_dict["garea"]
    hpbl       = data_dict["hpbl"]
    dot        = data_dict["dot"]
    phil       = data_dict["phil"]
    #fscav      = data_dict["fscav"]
    
    ### Output buffers ###
    kcnv       = data_dict["kcnv"]
    kbot       = data_dict["kbot"]
    ktop       = data_dict["ktop"]
    qtr        = data_dict["qtr"]
    q1         = data_dict["q1"]
    t1         = data_dict["t1"]
    u1         = data_dict["u1"]
    v1         = data_dict["v1"]
    rn         = data_dict["rn"]
    cnvw       = data_dict["cnvw"]
    cnvc       = data_dict["cnvc"]
    ud_mf      = data_dict["ud_mf"]
    dt_mf      = data_dict["dt_mf"]

    shape = (1, ix, km)
    
    ### Local storages for 1D arrays (integer) ###
    kpbl       = gt.storage.ones (BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kb         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kbcon      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kbcon1     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    ktcon      = gt.storage.ones (BACKEND, default_origin, shape, dtype=DTYPE_INT)
    ktcon1     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    ktconn     = gt.storage.ones (BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kbm        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    kmax       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    
    ### Local storages for 1D arrays ("bool") ###
    cnvflg     = gt.storage.ones (BACKEND, default_origin, shape, dtype=DTYPE_INT)
    flg        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    
    ### Local storages for 1D arrays (float) ###
    aa1        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    cina       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    tkemean    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    clamt      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ps         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    del0       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    prsl       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    umean      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    tauadv     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    gdx        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delhbar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delq       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delq2      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delqbar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delqev     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    deltbar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    deltv      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dtconv     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    edt        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pdot       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    po         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qcond      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qevap      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    hmax       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    rntot      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    vshear     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xlamud     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xmb        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xmbmax     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delubar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    delvbar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    c0         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    wc         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    scaldfunc  = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    sigmagfm   = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qlko_ktcon = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    sumx       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    tx1        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi_ktcon1  = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi_kbcon1  = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### Local storages for 2D arrays (float) ###
    pfld       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    to         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qo         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    uo         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    vo         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qeso       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    wu2        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    buo        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    drag       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellal     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dbyo       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zo         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    xlamue     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    heo        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    heso       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellah     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellaq     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellau     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellav     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    hcko       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ucko       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    vcko       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qcko       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qrcko      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    eta        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi         = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pwo        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    c0t        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    cnvwt      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### Local storages for 2D arrays (float, tracers), this will contain slices along n-axis ###
    delebar    = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### Local storages for 3D arrays (float, tracers), this will contain slices along n-axis ###
    ctr        = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ctro       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dellae     = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ecko       = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qaero      = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    ### K-indices field ###
    k_idx      = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)

    ### State buffer for 1D-2D interactions
    state_buf1 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    state_buf2 = gt.storage.zeros(BACKEND, default_origin, shape, dtype=DTYPE_INT)
    
    ### Local Parameters ###
    g          = grav
    elocp      = hvap/cp
    el2orc     = hvap * hvap/(rv * cp)
    d0         = 0.001
    cm         = 1.0
    delta      = fv
    fact1      = (cvap - cliq)/rv
    fact2      = hvap/rv - fact1 * t0c
    clamd      = 0.1
    tkemx      = 0.65
    tkemn      = 0.05
    dtke       = tkemx - tkemn
    dthk       = 25.0
    cinpcrmx   = 180.0
    cinpcrmn   = 120.0
    cinacrmx   = -120.0
    cinacrmn   = -80.0
    crtlamd    = 3.0e-4
    dtmax      = 10800.0
    dtmin      = 600.0
    bet1       = 1.875
    cd1        = 0.506
    f1         = 2.0
    gam1       = 0.5
    betaw      = 0.03
    dxcrt      = 15.0e3
    h1         = 0.33333333
    tf         = 233.16
    tcr        = 263.16
    tcrf       = 1.0/(tcr - tf)
    
    
    ### Determine whether to perform aerosol transport ###
    do_aerosols = (itc > 0) and (ntc > 0) and (ntr > 0)
    if (do_aerosols):
        do_aerosols = (ntr >= itc)
    
    ### Compute preliminary quantities needed for the static and feedback control portions of the algorithm ###
    
    # Convert input Pa terms to Cb terms
    pa_to_cb( psp, prslp, delp, ps, prsl, del0)
    
    km1 = km - 1
    
    ### Initialize storages (simple initializations already done above with gt4py functionalities) ###
    
    # Initialize column-integrated and other single-value-per-column 
    # variable arrays
    init_col_arr( kcnv, cnvflg, kbot, ktop,
                  kbcon, kb, rn, gdx, garea, km=km)
    
    # Return to the calling routine if deep convection is present or the 
    # surface buoyancy flux is negative
    if exit_routine(cnvflg, im): return
    
    # Initialize further parameters and arrays
    init_par_and_arr( islimsk, c0, t1, c0t, cnvw, cnvc, ud_mf, dt_mf,
                      c0s=c0s, asolfac=asolfac, d0=d0)
                      
    dt2   = delt
    
    # Model tunable parameters are all here
    aafac   = 0.05
    evfact  = 0.3
    evfactl = 0.3
    w1l     = -8.0e-3 
    w2l     = -4.0e-2
    w3l     = -5.0e-3 
    w4l     = -5.0e-4
    w1s     = -2.0e-4
    w2s     = -2.0e-3
    w3s     = -1.0e-3
    w4s     = -2.0e-5
    
    # Initialize the rest
    init_kbm_kmax( kbm, k_idx, kmax, state_buf1, state_buf2, tx1, ps, prsl, km=km )
    init_final( kbm, k_idx, kmax, flg, cnvflg, kpbl, tx1,
                ps, prsl, zo, phil, zi, pfld, eta, hcko, qcko, 
                qrcko, ucko, vcko, dbyo, pwo, dellal, to, qo, 
                uo, vo, wu2, buo, drag, cnvwt, qeso, heo, heso, hpbl,
                t1, q1, u1, v1, km=km )
                
    
    # Tracers loop (THIS GOES AT THE END AND POSSIBLY MERGED WITH OTHER 
    # TRACER LOOPS!) --> better use version below
    #for n in range(2, ntr+2):
        
        #kk = n-2
        
        #qtr_shift = gt.storage.from_array(slice_to_3d(qtr[:, :, n]), BACKEND, default_origin)
        
        # Initialize tracers. Keep in mind that, qtr slice is for the 
        # n-th tracer, while the other storages are slices representing 
        # the (n-2)-th tracer.
        #init_tracers( cnvflg, k_idx, kmax, ctr, ctro, ecko, qtr_shift )
    return {"heo":heo, "heso":heso, "qo":qo, "qeso":qeso,
            "km":km, "kbm":kbm, "kmax":kmax, "kb":kb, "kpbl":kpbl,
            "kbcon": kbcon, "ktcon":ktcon, #"ktcon1":ktcon1, "kbcon1":kbcon1
            "cnvflg":cnvflg, "tkemean":tkemean, "islimsk":islimsk,
            "dot":dot, "aa1":aa1, "cina":cina, "clamt":clamt, "del":del0,
            "pdot":pdot, "po":po, "hmax":hmax, "xlamud":xlamud, "pfld":pfld,
            "to":to, "uo":uo, "vo":vo, "wu2":wu2, "buo":buo, "drag":drag,
            "wc":wc, "dbyo":dbyo, "zo":zo, "xlamue":xlamue, "hcko":hcko,
            "ucko":ucko, "vcko":vcko, "qcko":qcko, "eta":eta, "zi":zi,
            "c0t":c0t, "sumx":sumx, "cnvwt":cnvwt, "dellal":dellal,
            "ktconn":ktconn, "pwo":pwo, "qlko_ktcon":qlko_ktcon, "qrcko":qrcko,
            "xmbmax":xmbmax,
            "u1":u1, "v1":v1, "gdx":gdx, "garea":garea, "dtconv":dtconv, "delp":delp,
            "cnvc":cnvc, "cnvw":cnvw, "t1":t1}


def test_part1_ser():
    data_dict   = read_data(0, True, path = DATAPATH)
    gt4py_dict  = numpy_dict_to_gt4py_dict(data_dict)
    out_dict_p2 = read_serialization_part2()
    out_dict_p3 = read_serialization_part3()
    out_dict_p4 = read_serialization_part4()
    
    ret_dict = samfshalcnv_part1(gt4py_dict)
    exp_data = view_gt4pystorage(ret_dict)
    
    ref_data           = view_gt4pystorage(out_dict_p2)
    ref_data["u1"]     = out_dict_p3["u1"].view(np.ndarray)
    ref_data["v1"]     = out_dict_p3["v1"].view(np.ndarray)
    ref_data["gdx"]    = out_dict_p3["gdx"].view(np.ndarray)
    ref_data["garea"]  = out_dict_p3["garea"].view(np.ndarray)
    ref_data["dtconv"] = out_dict_p3["dtconv"].view(np.ndarray)
    ref_data["delp"]   = out_dict_p3["delp"].view(np.ndarray)
    ref_data["cnvc"]   = out_dict_p4["cnvc"].view(np.ndarray)
    ref_data["cnvw"]   = out_dict_p4["cnvw"].view(np.ndarray)
    ref_data["t1"]     = out_dict_p4["t1"].view(np.ndarray)
    #ref_data = view_gt4pystorage({"heo":out_dict_p2["heo"], "heso":out_dict_p2["heso"],
    #                              "qo":out_dict_p2["qo"], "qeso":out_dict_p2["qeso"],
    #                              "km":out_dict_p2["km"], "kbm":out_dict_p2["kbm"],
    #                              "kmax":out_dict_p2["kmax"], "kcnv":out_dict_p4["kcnv"],
    #                              "cnvflg":out_dict_p2["cnvflg"]})
    
    compare_data(exp_data, ref_data)


if __name__ == "__main__":
    test_part1_ser()
