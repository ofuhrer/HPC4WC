import numpy as np

import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
from gt4py.gtscript import PARALLEL, computation, interval
import sys
sys.path.append("..")
from shalconv.funcphys import fpvsx, fpvsx_gt

backend="numpy" # "debug", "numpy", "gtx86", "gtcuda"
dtype = np.float64

con_ttp      = 2.7316e+2      # Temperature at H2O 3pt
con_psat     = 6.1078e+2      # Pressure at H2O 3pt
con_rv       = 4.6150e+2      # Gas constant H2O
con_cvap     = 1.8460e+3      # Specific heat of H2O gas
con_cliq     = 4.1855e+3      # Specific heat of H2O liquid
con_csol     = 2.1060e+3      # Specific heat of H2O ice
con_hvap     = 2.5000e+6      # Latent heat of H2O condensation
con_hfus     = 3.3358e+5      # Latent heat of H2O fusion
con_dldtl    = con_cvap - con_cliq
con_dldti    = con_cvap - con_csol
con_xponal   = -con_dldtl/con_rv
con_xponbl   = -con_dldtl/con_rv + con_hvap/(con_rv * con_ttp)
con_xponai   = -con_dldti/con_rv
con_xponbi   = -con_dldti/con_rv + (con_hvap + con_hfus)/(con_rv * con_ttp)

def fpvsx(t):
    tr = con_ttp / t
    tliq = con_ttp
    tice = con_ttp - 20.0

    w = (t - tice) / (tliq - tice)
    pvl = con_psat * (tr ** con_xponal) * np.exp(con_xponbl * (1.0 - tr))
    pvi = con_psat * (tr ** con_xponai) * np.exp(con_xponbi * (1.0 - tr))
    res = np.zeros_like(t)
    res = np.where(t >= tliq, pvl, res)
    res = np.where(t < tice, pvi, res)
    res = np.where((t<tliq) & (t>=tice), w * pvl + (1.0 - w) * pvi, res)
    return res

@gtscript.stencil(backend=backend) # this decorator triggers compilation of the stencil
def fpvsx_stencil(t: gtscript.Field[dtype],res: gtscript.Field[dtype]):
    with computation(PARALLEL), interval(...):
        tr = con_ttp / t
        tliq = con_ttp
        tice = con_ttp - 20.0

        tmp_l = np.e ** (con_xponbl * (1.0 - tr))
        tmp_i = np.e ** (con_xponbi * (1.0 - tr))
        res = 0.0
        w = 0.0
        pvl = 0.0
        pvi = 0.0
        if t >= tliq:
            res = con_psat * (tr ** con_xponal) * tmp_l
        elif t < tice:
            res = con_psat * (tr ** con_xponai) * tmp_i
        else:
            w = (t - tice) / (tliq - tice)
            pvl = con_psat * (tr ** con_xponal) * tmp_l
            pvi = con_psat * (tr ** con_xponai) * tmp_i

            res = w * pvl + (1.0 - w) * pvi

shape = (10,10,10)
x = np.random.rand(*shape) + 10.5
y = np.zeros(shape)
x_gt = gt4py.storage.from_array(x, backend, (0,0,0), dtype=dtype)
y_gt = gt4py.storage.from_array(y, backend, (0,0,0), dtype=dtype)

fpvsx_stencil(x_gt, y_gt, domain = shape)
y_gt.synchronize()
y[...] = fpvsx(x)
print(np.allclose(y,y_gt.view(np.ndarray)))
