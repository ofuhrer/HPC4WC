import gt4py as gt
from gt4py import gtscript
import numpy as np
from . import *
from shalconv.physcons import (
    con_ttp,
    con_psat,
    con_xponal,
    con_xponbl,
    con_xponai,
    con_xponbi
)
from shalconv.kernels.utils import exp


### Global variables used in fpvs ###
c1xpvs = None
c2xpvs = None

### Look-up table for saturation vapor pressure ###
nxpvs = 7501                  # Size of look-up table
tbpvs = np.empty(nxpvs)       # Look-up table stored as a 1D numpy array

tbpvs_gt = gt.storage.from_array(tbpvs, BACKEND, (0,), dtype=DTYPE_FLOAT)


# Computes saturation vapor pressure table as a function of temperature 
# for the table lookup function fpvs
def gpvs():
    global c1xpvs
    global c2xpvs
    
    xmin   = 180.0
    xmax   = 330.0
    xinc   = (xmax - xmin)/(nxpvs - 1)
    c2xpvs = 1.0/xinc
    c1xpvs = -xmin * c2xpvs
    
    for jx in range(0, nxpvs):
        x = xmin + jx * xinc
        tbpvs[jx] = fpvsx(x)


# Compute saturation vapor pressure from the temperature. A linear 
# interpolation is done between values in a lookup table computed in 
# gpvs.
def fpvs(t):
    xj = min(max(c1xpvs + c2xpvs * t, 0.0), nxpvs - 1)
    jx = int(min(xj, nxpvs - 2))
    
    return tbpvs[jx] + (xj - jx) * (tbpvs[jx+1] - tbpvs[jx])


# Compute exact saturation vapor pressure from temperature
def fpvsx(t):
    tr   = con_ttp/t
    tliq = con_ttp
    tice = con_ttp - 20.0
    
    if t >= tliq:
        return con_psat * (tr**con_xponal) * np.exp(con_xponbl * (1.0 - tr))
    elif t < tice:
        return con_psat * (tr**con_xponai) * np.exp(con_xponbi * (1.0 - tr))
    else:
        w   = (t - tice)/(tliq - tice)
        pvl = con_psat * (tr**con_xponal) * np.exp(con_xponbl * (1.0 - tr))
        pvi = con_psat * (tr**con_xponai) * np.exp(con_xponbi * (1.0 - tr))
        
        return w * pvl + (1.0 - w) * pvi


# Function fpvsx as gtscript.function, to be used in stencils
@gtscript.function
def fpvsx_gt(t):
    tr   = con_ttp/t
    tliq = con_ttp
    tice = con_ttp - 20.0
    
    tmp_l = np.e**(con_xponbl * (1.0 - tr))
    tmp_i = np.e**(con_xponbi * (1.0 - tr))
    ret   = 0.0
    w     = 0.0
    pvl   = 0.0
    pvi   = 0.0
    
    if t >= tliq:
        ret = con_psat * (tr**con_xponal) * tmp_l
    elif t < tice:
        ret = con_psat * (tr**con_xponai) * tmp_i
    else:
        w   = (t - tice)/(tliq - tice)
        pvl = con_psat * (tr**con_xponal) * tmp_l
        pvi = con_psat * (tr**con_xponai) * tmp_i
        
        ret = w * pvl + (1.0 - w) * pvi
        
    return ret
