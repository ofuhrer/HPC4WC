import gt4py as gt
import sys
sys.path.append("..")
from read_serialization import *
from shalconv.kernels.utils import get_1D_from_index, exit_routine
from shalconv.kernels.stencils_part2 import *
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


def samfshalcnv_part2( ix,km,clam,pgcon,delt,c1,ncloud,ntk,ntr,
                       kpbl,kb,kbcon,kbcon1,ktcon,ktcon1,kbm,kmax,
                       po,to,qo,uo,vo,qeso,dbyo,zo,
                       heo,heso,hcko,ucko,vcko,qcko,
                       ecko,ecko_slice,ctro,ctro_slice,
                       aa1,cina,clamt,del0,wu2,buo,drag,wc,
                       pdot,hmax,xlamue,xlamud,pfld,qtr,tkemean,
                       eta,zi,c0t,sumx,cnvflg,flg,islimsk,dot,
                       k_idx,heo_kb,dot_kbcon,pfld_kbcon,pfld_kb,pfld_kbcon1,
                       cnvwt,dellal,ktconn,pwo,qlko_ktcon,qrcko,xmbmax ):

    ### Search in the PBL for the level of maximum moist static energy to start the ascending parcel.
    stencil_static0( cnvflg, hmax, heo, kb, k_idx, kpbl, kmax, zo, to, qeso, qo, po, uo, vo, heso, pfld )

    for n in range(ntr):
        ctro_slice[...] = ctro[np.newaxis, :, :, n]
        stencil_ntrstatic0( cnvflg, k_idx, kmax, ctro_slice )
        ctro[:, :, n] = ctro_slice[0, :, :]

    ### Search below the index "kbm" for the level of free convection (LFC) where the condition.
    get_1D_from_index( heo, heo_kb, kb, k_idx )
    stencil_static1( cnvflg, flg, kbcon, kmax, k_idx, kbm, kb, heo_kb, heso )

    ### If no LFC, return to the calling routine without modifying state variables.
    if exit_routine(cnvflg, ix): return

    ### Determine the vertical pressure velocity at the LFC.
    get_1D_from_index( dot, dot_kbcon, kbcon, k_idx )
    get_1D_from_index( pfld, pfld_kbcon, kbcon, k_idx )
    get_1D_from_index( pfld, pfld_kb, kb, k_idx )
    stencil_static2( cnvflg, pdot, dot_kbcon, islimsk, k_idx, kbcon, kb, pfld_kb, pfld_kbcon )
    
    ### If no LFC, return to the calling routine without modifying state variables.
    if exit_routine(cnvflg, ix): return

    ### turbulent entrainment rate assumed to be proportional to subcloud mean TKE
    if(ntk > 0):
        qtr_ntk = gt.storage.from_array( qtr[np.newaxis, :, :, ntk-1], BACKEND, default_origin )
        stencil_static3( sumx, tkemean, cnvflg, k_idx, kb, kbcon, zo, qtr_ntk, clamt, clam=clam )
    #    qtr[:,:,ntr] = qtr_ntr[0,:,:]
    #else:
    #stencil_static4( cnvflg, clamt, clam=clam )
    
    ### assume updraft entrainment rate is an inverse function of height
    stencil_static5( cnvflg, xlamue, clamt, zi, xlamud, k_idx, kbcon, kb,
                     eta, ktconn, kmax, kbm, hcko, ucko, vcko, heo, uo, vo )

    for n in range(ntr):
        ctro_slice[...] = ctro[np.newaxis, :, :, n]
        ecko_slice[...] = ecko[np.newaxis, :, :, n]
        stencil_ntrstatic1( cnvflg, k_idx, kb, ecko_slice, ctro_slice )
        ecko[:, :, n] = ecko_slice[0, :, :]


    stencil_static7( cnvflg, k_idx, kb, kmax, zi, xlamue, xlamud, hcko, heo, dbyo,
                     heso, ucko, uo, vcko, vo, pgcon=pgcon )

    for n in range(ntr):
        ctro_slice[...] = ctro[np.newaxis, :, :, n]
        ecko_slice[...] = ecko[np.newaxis, :, :, n]
        stencil_ntrstatic2( cnvflg, k_idx, kb, kmax, zi, xlamue, ecko_slice, ctro_slice )
        ecko[:, :, n] = ecko_slice[0, :, :]

    stencil_update_kbcon1_cnvflg( dbyo, cnvflg, kmax, kbm, kbcon, kbcon1, flg, k_idx )
    get_1D_from_index( pfld, pfld_kbcon1, kbcon1, k_idx )
    stencil_static9( cnvflg, pfld_kbcon, pfld_kbcon1 )
    
    if exit_routine(cnvflg, ix): return

    ### calculate convective inhibition
    stencil_static10( cina, cnvflg, k_idx, kb, kbcon1, zo, qeso, to,
                      dbyo, qo, pdot, islimsk )
    
    if exit_routine(cnvflg, ix): return

    dt2 = delt
    stencil_static11( flg, cnvflg, ktcon, kbm, kbcon1, dbyo, kbcon, del0, xmbmax,
                      aa1, kb, qcko, qo, qrcko, zi, qeso, to, xlamue,
                      xlamud, eta, c0t, dellal, buo, drag, zo, k_idx, pwo,
                      cnvwt, c1=c1, dt2=dt2, ncloud=ncloud )
    
    if exit_routine(cnvflg, ix): return

    stencil_static12( cnvflg, aa1, flg, ktcon1, kbm, k_idx, ktcon, zo, qeso, 
                      to, dbyo, zi, xlamue, xlamud, qcko, qrcko, qo, eta, del0,
                      c0t, pwo, cnvwt, buo, wu2, wc, sumx, kbcon1, drag, dellal,
                      c1=c1, ncloud=ncloud )

    #if(ncloud > 0):
    stencil_static13( cnvflg, k_idx, ktcon, qeso, to, dbyo, qcko, qlko_ktcon )
    # else:
    #     stencil_static14( cnvflg, vshear, k_idx, kb, ktcon, uo, vo, zi, edt )


def apply_arguments_part2( input_dict, data_dict ):
    
    clam        = input_dict['clam']
    pgcon       = input_dict['pgcon']
    delt        = input_dict['delt']
    c1          = input_dict['c1']
    ncloud      = input_dict['ncloud']
    ntk         = input_dict['ntk']
    ntr         = input_dict['ntr']
    qtr         = data_dict['qtr']
    tkemean     = data_dict['tkemean']
    ix          = data_dict['ix']
    km          = data_dict['km']
    islimsk     = data_dict['islimsk']
    dot         = data_dict['dot']
    kpbl        = data_dict['kpbl']
    kb          = data_dict['kb']
    kbcon       = data_dict['kbcon']
    kbcon1      = data_dict['kbcon1']
    ktcon       = data_dict['ktcon']
    ktcon1      = data_dict['ktcon1']
    kbm         = data_dict['kbm']
    kmax        = data_dict['kmax']
    aa1         = data_dict['aa1']
    cina        = data_dict['cina']
    clamt       = data_dict['clamt']
    del0        = data_dict['del']
    pdot        = data_dict['pdot']
    po          = data_dict['po']
    hmax        = data_dict['hmax']
    xlamud      = data_dict['xlamud']
    pfld        = data_dict['pfld']
    to          = data_dict['to']
    qo          = data_dict['qo']
    uo          = data_dict['uo']
    vo          = data_dict['vo']
    qeso        = data_dict['qeso']
    wu2         = data_dict['wu2']
    buo         = data_dict['buo']
    drag        = data_dict['drag']
    wc          = data_dict['wc']
    dbyo        = data_dict['dbyo']
    zo          = data_dict['zo']
    xlamue      = data_dict['xlamue']
    heo         = data_dict['heo']
    heso        = data_dict['heso']
    hcko        = data_dict['hcko']
    ucko        = data_dict['ucko']
    vcko        = data_dict['vcko']
    qcko        = data_dict['qcko']
    ecko        = data_dict['ecko']
    ctro        = data_dict['ctro']
    eta         = data_dict['eta']
    zi          = data_dict['zi']
    c0t         = data_dict['c0t']
    sumx        = data_dict['sumx']
    cnvflg      = data_dict['cnvflg']
    flg         = data_dict['flg']
    cnvwt       = data_dict['cnvwt']
    dellal      = data_dict['dellal']
    ktconn      = data_dict['ktconn']
    pwo         = data_dict['pwo']
    qlko_ktcon  = data_dict['qlko_ktcon']
    qrcko       = data_dict['qrcko']
    xmbmax      = data_dict['xmbmax']
    shape       = (1, ix, km)
    k_idx       = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)
    heo_kb      = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dot_kbcon   = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kbcon  = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kb     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kbcon1 = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ctro_slice  = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    ecko_slice  = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    
    samfshalcnv_part2( ix, km, clam, pgcon, delt, c1, ncloud, ntk, ntr,
                       kpbl, kb, kbcon, kbcon1, ktcon, ktcon1, kbm, kmax,
                       po, to, qo, uo, vo, qeso, dbyo, zo,
                       heo, heso, hcko, ucko, vcko, qcko,
                       ecko, ecko_slice, ctro, ctro_slice,
                       aa1, cina, clamt, del0, wu2, buo, drag, wc,
                       pdot, hmax, xlamue, xlamud, pfld, qtr, tkemean,
                       eta, zi, c0t, sumx, cnvflg, flg, islimsk, dot,
                       k_idx, heo_kb, dot_kbcon, pfld_kbcon, pfld_kb, pfld_kbcon1,
                       cnvwt, dellal, ktconn, pwo, qlko_ktcon, qrcko, xmbmax )
                       
    return cnvflg, kmax, kbcon, kbcon1, cnvwt, dellal, ecko, ctro, pwo, qlko_ktcon, qrcko, xmbmax


def apply_arguments_stencil0(input_dict, data_dict):
    cnvflg = data_dict['cnvflg']
    hmax   = data_dict['hmax']
    heo    = data_dict['heo']
    kb     = data_dict['kb']
    kpbl   = data_dict['kpbl']
    kmax   = data_dict['kmax']
    zo     = data_dict['zo']
    to     = data_dict['to']
    qeso   = data_dict['qeso']
    qo     = data_dict['qo']
    po     = data_dict['po']
    uo     = data_dict['uo']
    vo     = data_dict['vo']
    heso   = data_dict['heso']
    pfld   = data_dict['pfld']
    ix     = input_dict['ix']
    km     = input_dict['km']
    shape  = (1, ix, km)
    k_idx  = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)
    
    stencil_static0( cnvflg, hmax, heo, kb, k_idx, kpbl, kmax, zo, to, qeso, qo, po, uo, vo, heso, pfld )
    
    return hmax, heo, heso, kb, to, qeso, qo, po, uo, vo


def apply_arguments_stencil012(input_dict, data_dict):
    cnvflg  = data_dict['cnvflg']
    hmax    = data_dict['hmax']
    heo     = data_dict['heo']
    kb      = data_dict['kb']
    kpbl    = data_dict['kpbl']
    kmax    = data_dict['kmax']
    zo      = data_dict['zo']
    to      = data_dict['to']
    qeso    = data_dict['qeso']
    qo      = data_dict['qo']
    po      = data_dict['po']
    uo      = data_dict['uo']
    vo      = data_dict['vo']
    heso    = data_dict['heso']
    pfld    = data_dict['pfld']
    flg     = data_dict['flg']
    kbcon   = data_dict['kbcon']
    kbm     = data_dict['kbm']
    pdot    = data_dict['pdot']
    dot     = data_dict['dot']
    islimsk = data_dict['islimsk']
    ix      = input_dict['ix']
    km      = input_dict['km']
    
    shape      = (1, ix, km)
    k_idx      = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)
    heo_kb     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dot_kbcon  = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kbcon = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kb    = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)

    stencil_static0( cnvflg, hmax, heo, kb, k_idx, kpbl, kmax, zo, to, qeso, qo, po, uo, vo, heso, pfld )
    
    get_1D_from_index( heo, heo_kb, kb, k_idx )
    
    stencil_static1( cnvflg, flg, kbcon, kmax, k_idx, kbm, kb, heo_kb, heso )
    
    if exit_routine(cnvflg, ix): return
    
    get_1D_from_index( dot, dot_kbcon, kbcon, k_idx )
    get_1D_from_index( pfld, pfld_kbcon, kbcon, k_idx )
    get_1D_from_index( pfld, pfld_kb, kb, k_idx )
    
    stencil_static2( cnvflg, pdot, dot_kbcon, islimsk, k_idx, kbcon, kb, pfld_kb, pfld_kbcon )
    
    return cnvflg, pdot


def apply_arguments_stencil012345(input_dict, data_dict):
    clam    = input_dict['clam']
    ntk     = input_dict['ntk']
    qtr     = data_dict['qtr']
    tkemean = data_dict['tkemean']
    sumx    = data_dict['sumx']
    ix      = data_dict['ix']
    km      = data_dict['km']
    islimsk = data_dict['islimsk']
    dot     = data_dict['dot']
    kpbl    = data_dict['kpbl']
    kb      = data_dict['kb']
    kbcon   = data_dict['kbcon']
    kbm     = data_dict['kbm']
    kmax    = data_dict['kmax']
    clamt   = data_dict['clamt']
    pdot    = data_dict['pdot']
    po      = data_dict['po']
    hmax    = data_dict['hmax']
    xlamud  = data_dict['xlamud']
    pfld    = data_dict['pfld']
    to      = data_dict['to']
    qo      = data_dict['qo']
    uo      = data_dict['uo']
    vo      = data_dict['vo']
    qeso    = data_dict['qeso']
    zo      = data_dict['zo']
    xlamue  = data_dict['xlamue']
    heo     = data_dict['heo']
    heso    = data_dict['heso']
    hcko    = data_dict['hcko']
    ucko    = data_dict['ucko']
    vcko    = data_dict['vcko']
    eta     = data_dict['eta']
    zi      = data_dict['zi']
    cnvflg  = data_dict['cnvflg']
    flg     = data_dict['flg']
    ktconn  = data_dict['ktconn']
    
    shape       = (1, ix, km)
    k_idx       = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)
    heo_kb      = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    dot_kbcon   = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kbcon  = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kb     = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kbcon1 = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    qtr_ntk     = gt.storage.from_array(qtr[np.newaxis, :, :, ntk-1], BACKEND, default_origin)
    
    stencil_static0( cnvflg, hmax, heo, kb, k_idx, kpbl, kmax, zo, to, qeso, qo, po, uo, vo, heso, pfld )
    
    get_1D_from_index( heo, heo_kb, kb, k_idx )
    
    stencil_static1( cnvflg, flg, kbcon, kmax, k_idx, kbm, kb, heo_kb, heso )
    
    get_1D_from_index( dot, dot_kbcon, kbcon, k_idx )
    get_1D_from_index( pfld, pfld_kbcon, kbcon, k_idx )
    get_1D_from_index( pfld, pfld_kb, kb, k_idx )
    
    stencil_static2( cnvflg, pdot, dot_kbcon, islimsk, k_idx, kbcon, kb, pfld_kb, pfld_kbcon )
    stencil_static3( sumx, tkemean, cnvflg, k_idx, kb, kbcon, zo, qtr_ntk, clamt, clam )
    stencil_static5( cnvflg, xlamue, clamt, zi, xlamud, k_idx, kbcon, kb,
                     eta, ktconn, kmax, kbm, hcko, ucko, vcko, heo, uo, vo )
                     
    return clamt, xlamud, xlamue, eta, kmax, kbm, hcko, ucko, vcko, tkemean, sumx


def test_part2_1():
    input_dict = read_data(0, True, path=DATAPATH)
    data_dict  = read_serialization_part2()
    out_dict   = read_serialization_part2_1()
    
    hmax, heo, heso, kb, to, qeso, qo, po, uo, vo = apply_arguments_stencil0( input_dict, data_dict )
    
    exp_data = view_gt4pystorage( {"hmax":hmax, "heo":heo, "heso": heso, "kb":kb, "to":to,
                                   "qeso":qeso, "qo":qo, "po":po, "uo":uo, "vo":vo} )
    ref_data = view_gt4pystorage( {"hmax":out_dict["hmax"], "heo":out_dict["heo"],
                                   "heso": out_dict["heso"], "kb":out_dict["kb"],
                                   "to":out_dict["to"], "qeso":out_dict["qeso"],
                                   "qo":out_dict["qo"], "po":out_dict["po"],
                                   "uo":out_dict["uo"], "vo":out_dict["vo"]} )
                                   
    compare_data(exp_data, ref_data)


def test_part2_2():
    input_dict = read_data(0, True, path=DATAPATH)
    data_dict  = read_serialization_part2()
    out_dict   = read_serialization_part2_2()
    
    cnvflg, pdot = apply_arguments_stencil012( input_dict, data_dict )
    
    exp_data = view_gt4pystorage( {"cnvflg":cnvflg, "pdot": pdot} )
    ref_data = view_gt4pystorage( {"cnvflg":out_dict["cnvflg"],"pdot": out_dict["pdot"]} )
    
    compare_data(exp_data, ref_data)


def test_part2_3():
    input_dict = read_data(0, True, path=DATAPATH)
    data_dict  = read_serialization_part2()
    out_dict   = read_serialization_part2_3()
    
    clamt, xlamud, xlamue, eta, kmax, kbm, hcko, ucko, vcko, tkemean, sumx = apply_arguments_stencil012345( input_dict, data_dict )
    
    exp_data = view_gt4pystorage( {"clamt":clamt, "xlamud":xlamud, "xlamue": xlamue, "eta":eta, "kmax":kmax,
                                   "kbm":kbm, "hcko":hcko, "ucko":ucko, "vcko":vcko, "tkemean":tkemean,
                                   "sumx":sumx} )
    ref_data = view_gt4pystorage( {"clamt":out_dict["clamt"], "xlamud":out_dict["xlamud"],
                                   "xlamue": out_dict["xlamue"], "eta":out_dict["eta"],
                                   "kmax":out_dict["kmax"], "kbm":out_dict["kbm"],
                                   "hcko":out_dict["hcko"], "ucko":out_dict["ucko"],
                                   "vcko":out_dict["vcko"], "tkemean": out_dict["tkemean"],
                                   "sumx":out_dict["sumx"]} )
                                   
    compare_data(exp_data, ref_data)
    
    
def test_part2_4():
    data_dict = read_serialization_part2_3()
    out_dict  = read_serialization_part2_4()
    
    cnvflg = data_dict["cnvflg"]
    kb     = data_dict["kb"]
    kmax   = data_dict["kmax"]
    zi     = data_dict["zi"]
    xlamue = data_dict["xlamue"]
    xlamud = data_dict["xlamud"]
    hcko   = data_dict["hcko"]
    heo    = data_dict["heo"]
    dbyo   = data_dict["dbyo"]
    heso   = data_dict["heso"]
    pgcon  = data_dict["pgcon"]
    ucko   = data_dict["ucko"]
    uo     = data_dict["uo"]
    vcko   = data_dict["vcko"]
    vo     = data_dict["vo"]
    ix     = data_dict['ix']
    km     = data_dict['km']
    
    shape = (1, ix, km)
    k_idx = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)
    
    stencil_static7( cnvflg, k_idx, kb, kmax, zi, xlamue, xlamud, hcko, heo, dbyo,
                     heso, pgcon, ucko, uo, vcko, vo )
                     
    exp_data = view_gt4pystorage( {"hcko":hcko, "dbyo":dbyo, "ucko":ucko, "vcko":vcko} )
    ref_data = view_gt4pystorage( {"hcko": out_dict["hcko"], "dbyo": out_dict["dbyo"],
                                   "ucko": out_dict["ucko"], "vcko": out_dict["vcko"]} )
                                   
    compare_data(exp_data, ref_data)


def test_part2_5():
    data_dict = read_serialization_part2_4()
    out_dict  = read_serialization_part2_5()
    
    dbyo   = data_dict["dbyo"]
    cnvflg = data_dict["cnvflg"]
    kmax   = data_dict["kmax"]
    kbm    = data_dict["kbm"]
    kbcon  = data_dict["kbcon"]
    kbcon1 = data_dict["kbcon1"]
    flg    = data_dict["flg"]
    pfld   = data_dict["pfld"]
    ix     = data_dict['ix']
    km     = data_dict['km']
    
    shape       = (1, ix, km)
    k_idx       = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)
    pfld_kbcon  = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    pfld_kbcon1 = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)

    get_1D_from_index( pfld, pfld_kbcon, kbcon, k_idx )
    
    stencil_update_kbcon1_cnvflg( dbyo, cnvflg, kmax, kbm, kbcon, kbcon1, flg, k_idx )
    
    get_1D_from_index( pfld, pfld_kbcon1, kbcon1, k_idx )
    
    stencil_static9( cnvflg, pfld_kbcon, pfld_kbcon1 )
    
    exp_data = view_gt4pystorage( {"cnvflg":cnvflg, "kbcon1":kbcon1, "flg":flg} )
    ref_data = view_gt4pystorage( {"cnvflg": out_dict["cnvflg"], "kbcon1": out_dict["kbcon1"],
                                   "flg": out_dict["flg"]} )
                                   
    compare_data(exp_data, ref_data)


def test_part2_6():
    data_dict = read_serialization_part2_5()
    out_dict  = read_serialization_part2_6()
    
    cina    = data_dict["cina"]
    cnvflg  = data_dict["cnvflg"]
    kb      = data_dict["kb"]
    kbcon1  = data_dict["kbcon1"]
    zo      = data_dict["zo"]
    qeso    = data_dict["qeso"]
    to      = data_dict["to"]
    dbyo    = data_dict["dbyo"]
    qo      = data_dict["qo"]
    pdot    = data_dict["pdot"]
    islimsk = data_dict["islimsk"]
    ix      = data_dict['ix']
    km      = data_dict['km']
    
    shape = (1, ix, km)
    k_idx = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)
    
    stencil_static10( cina, cnvflg, k_idx, kb, kbcon1, zo, qeso, to,
                      dbyo, qo, pdot, islimsk )
                      
    exp_data = view_gt4pystorage( {"cnvflg": cnvflg, "cina": cina} )
    ref_data = view_gt4pystorage( {"cnvflg": out_dict["cnvflg"], "cina": out_dict["cina"]} )
    
    compare_data(exp_data, ref_data)


def test_part2():
    input_dict  = read_data(0, True, path = DATAPATH)
    data_dict   = read_serialization_part2()
    out_dict_p3 = read_serialization_part3()
    out_dict_p4 = read_serialization_part4()
    
    cnvflg, kmax, kbcon, kbcon1, cnvwt, dellal, ecko, ctro, pwo, qlko_ktcon, qrcko, xmbmax = apply_arguments_part2( input_dict, data_dict )
    
    exp_data = view_gt4pystorage( {"cnvflg":cnvflg, "ecko":ecko, "ctro":ctro,
                "kmax":kmax, "kbcon": kbcon, "kbcon1":kbcon1,
                "cnvwt":cnvwt, "dellal":dellal, "pwo":pwo,
                "qlko_ktcon":qlko_ktcon, "qrcko":qrcko, "xmbmax":xmbmax} )
    ref_data = view_gt4pystorage( {"cnvflg":out_dict_p3["cnvflg"], "ctro":out_dict_p3["ctro"],
                "ecko":out_dict_p3["ecko"],"kmax":out_dict_p3["kmax"],
                "kbcon":out_dict_p3["kbcon"],"kbcon1":out_dict_p3["kbcon1"],
                "cnvwt":out_dict_p4["cnvwt"], "dellal":out_dict_p3["dellal"],
                "pwo": out_dict_p4["pwo"], "qlko_ktcon": out_dict_p3["qlko_ktcon"],
                "qrcko":out_dict_p3["qrcko"],"xmbmax":out_dict_p3["xmbmax"]} )
                
    compare_data(exp_data, ref_data)


if __name__ == "__main__":
    #test_part2_1()
    #test_part2_2()
    #test_part2_3()
    #test_part2_4()
    #test_part2_5()
    #test_part2_6()
    test_part2()
