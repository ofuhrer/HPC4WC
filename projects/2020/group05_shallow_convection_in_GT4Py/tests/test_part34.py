import pytest
import gt4py as gt
from gt4py import gtscript
import sys
sys.path.append("..")
from read_serialization import read_serialization_part3, read_serialization_part4
from shalconv.kernels.stencils_part34 import *
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


def samfshalcnv_part3(input_dict, data_dict):
    """
    Scale-Aware Mass-Flux Shallow Convection
    
    :param data_dict: Dict of parameters required by the scheme
    :type data_dict: Dict of either scalar or gt4py storage
    """
    ix    = input_dict["ix"]
    km    = input_dict["km"]
    shape = (1, ix, km)
    
    g     = grav
    betaw = 0.03
    dtmin = 600.0
    dtmax = 10800.0
    dxcrt = 15.0e3
    
    dt2        = input_dict["delt"]
    cnvflg     = data_dict["cnvflg"]
    k_idx      = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)
    kmax       = data_dict["kmax"]
    kb         = data_dict["kb"]
    ktcon      = data_dict["ktcon"]
    ktcon1     = data_dict["ktcon1"]
    kbcon1     = data_dict["kbcon1"]
    kbcon      = data_dict["kbcon"]
    dellah     = data_dict["dellah"]
    dellaq     = data_dict["dellaq"]
    dellau     = data_dict["dellau"]
    dellav     = data_dict["dellav"]
    del0       = data_dict["del"]
    zi         = data_dict["zi"]
    zi_ktcon1  = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    zi_kbcon1  = gt.storage.empty(BACKEND, default_origin, shape, dtype=DTYPE_FLOAT)
    heo        = data_dict["heo"]
    qo         = data_dict["qo"]
    xlamue     = data_dict["xlamue"]
    xlamud     = data_dict["xlamud"]
    eta        = data_dict["eta"]
    hcko       = data_dict["hcko"]
    qrcko      = data_dict["qrcko"]
    uo         = data_dict["uo"]
    ucko       = data_dict["ucko"]
    vo         = data_dict["vo"]
    vcko       = data_dict["vcko"]
    qcko       = data_dict["qcko"]
    dellal     = data_dict["dellal"]
    qlko_ktcon = data_dict["qlko_ktcon"]
    wc         = data_dict["wc"]
    gdx        = data_dict["gdx"]
    dtconv     = data_dict["dtconv"]
    u1         = data_dict["u1"]
    v1         = data_dict["v1"]
    po         = data_dict["po"]
    to         = data_dict["to"]
    tauadv     = data_dict["tauadv"]
    xmb        = data_dict["xmb"]
    sigmagfm   = data_dict["sigmagfm"]
    garea      = data_dict["garea"]
    scaldfunc  = data_dict["scaldfunc"]
    xmbmax     = data_dict["xmbmax"]
    sumx       = data_dict["sumx"]
    umean      = data_dict["umean"]
    
    #import pdb; pdb.set_trace() 
    
    # Calculate the tendencies of the state variables (per unit cloud base 
    # mass flux) and the cloud base mass flux
    comp_tendencies( cnvflg, k_idx, kmax, kb, ktcon, ktcon1, kbcon1, kbcon,
                     dellah, dellaq, dellau, dellav, del0, zi, zi_ktcon1,
                     zi_kbcon1, heo, qo, xlamue, xlamud, eta, hcko,
                     qrcko, uo, ucko, vo, vcko, qcko, dellal, 
                     qlko_ktcon, wc, gdx, dtconv, u1, v1, po, to, 
                     tauadv, xmb, sigmagfm, garea, scaldfunc, xmbmax,
                     sumx, umean,
                     g=g, betaw=betaw, dtmin=dtmin, dt2=dt2, dtmax=dtmax, dxcrt=dxcrt )
                     
    return dellah, dellaq, dellau, dellav, dellal, xmb, sigmagfm


def samfshalcnv_part4(input_dict, data_dict):
    """
    Scale-Aware Mass-Flux Shallow Convection
    
    :param data_dict: Dict of parameters required by the scheme
    :type data_dict: Dict of either scalar or gt4py storage
    """
    ix    = input_dict["ix"]
    km    = input_dict["km"]
    shape = (1, ix, km)
    
    g       = grav
    evfact  = 0.3
    evfactl = 0.3
    elocp   = hvap/cp
    el2orc  = hvap * hvap/(rv * cp)
    
    dt2     = input_dict["delt"]
    cnvflg  = data_dict["cnvflg"]
    k_idx   = gt.storage.from_array(np.indices(shape)[2] + 1, BACKEND, default_origin, dtype=DTYPE_INT)
    kmax    = data_dict["kmax"]
    kb      = data_dict["kb"]
    ktcon   = data_dict["ktcon"]
    flg     = data_dict["flg"]
    islimsk = data_dict["islimsk"]
    ktop    = data_dict["ktop"]
    kbot    = data_dict["kbot"]
    kcnv    = data_dict["kcnv"]
    kbcon   = data_dict["kbcon"]
    qeso    = data_dict["qeso"]
    pfld    = data_dict["pfld"]
    delhbar = data_dict["delhbar"]
    delqbar = data_dict["delqbar"]
    deltbar = data_dict["deltbar"]
    delubar = data_dict["delubar"]
    delvbar = data_dict["delvbar"]
    qcond   = data_dict["qcond"]
    dellah  = data_dict["dellah"]
    dellaq  = data_dict["dellaq"]
    dellau  = data_dict["dellau"]
    dellav  = data_dict["dellav"]
    t1      = data_dict["t1"]
    q1      = data_dict["q1"]
    del0    = data_dict["del"]
    rntot   = data_dict["rntot"]
    delqev  = data_dict["delqev"]
    delq2   = data_dict["delq2"]
    pwo     = data_dict["pwo"]
    deltv   = data_dict["deltv"]
    delq    = data_dict["delq"]
    qevap   = data_dict["qevap"]
    rn      = data_dict["rn"]
    edt     = data_dict["edt"]
    cnvw    = data_dict["cnvw"]
    cnvwt   = data_dict["cnvwt"]
    cnvc    = data_dict["cnvc"]
    ud_mf   = data_dict["ud_mf"]
    dt_mf   = data_dict["dt_mf"]
    u1      = data_dict["u1"]
    v1      = data_dict["v1"]
    xmb     = data_dict["xmb"]
    eta     = data_dict["eta"]
    
    #import pdb; pdb.set_trace() 

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
                             el2orc=el2orc, elocp=elocp )
                     
    return kcnv, kbot, ktop, q1, t1, u1, v1, rn, cnvw, cnvc, ud_mf, dt_mf


def test_part3():
    
    input_dict = read_data(0, True, path = DATAPATH)
    data_dict  = read_serialization_part3()
    out_dict   = read_serialization_part4()
    
    dellah, dellaq, dellau, dellav, dellal, xmb, sigmagfm = samfshalcnv_part3( input_dict, data_dict )
    exp_data = view_gt4pystorage({"dellah":dellah,"dellaq":dellaq,
                                  "dellau":dellau,"dellav":dellav,
                                  "dellal":dellal,"xmb":xmb,"sigmagfm":sigmagfm})
    compare_data(exp_data,
                 {"dellah":out_dict["dellah"].view(np.ndarray),"dellaq":out_dict["dellaq"].view(np.ndarray),
                  "dellau":out_dict["dellau"].view(np.ndarray),"dellav":out_dict["dellav"].view(np.ndarray),
                  "dellal":out_dict["dellal"].view(np.ndarray),"xmb":out_dict["xmb"].view(np.ndarray),"sigmagfm":out_dict["sigmagfm"].view(np.ndarray)})
    
    
def test_part4():
    
    input_dict = read_data(0, True, path = DATAPATH)
    data_dict  = read_serialization_part4()
    out_dict   = read_data(0, False, path = DATAPATH)
    
    kcnv, kbot, ktop, q1, t1, u1, v1, rn, cnvw, cnvc, ud_mf, dt_mf = samfshalcnv_part4( input_dict, data_dict )
    exp_data = view_gt4pystorage( {"kcnv":kcnv[0,:,0],"kbot":kbot[0,:,0],"ktop":ktop[0,:,0],
                                   "q1":q1[0,:,:],"t1":t1[0,:,:],
                                   "u1":u1[0,:,:],"v1":v1[0,:,:],"rn":rn[0,:,0],
                                   "cnvw":cnvw[0,:,:],"cnvc":cnvc[0,:,:],"ud_mf":ud_mf[0,:,:],
                                   "dt_mf":dt_mf[0,:,:]} )
    
    compare_data( exp_data,
                  {"kcnv":out_dict["kcnv"],"kbot":out_dict["kbot"],"ktop":out_dict["ktop"],
                   "q1":out_dict["q1"],"t1":out_dict["t1"],
                   "u1":out_dict["u1"],"v1":out_dict["v1"],"rn":out_dict["rn"],
                   "cnvw":out_dict["cnvw"],"cnvc":out_dict["cnvc"],"ud_mf":out_dict["ud_mf"],
                   "dt_mf":out_dict["dt_mf"]} )
    

if __name__ == "__main__":
    test_part3()
    test_part4()
