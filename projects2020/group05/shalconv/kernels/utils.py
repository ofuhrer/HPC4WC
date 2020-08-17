import gt4py as gt
from gt4py import gtscript
from gt4py.gtscript import PARALLEL, FORWARD, BACKWARD, interval, computation
import numpy as np
from shalconv import BACKEND, REBUILD, FIELD_INT, FIELD_FLOAT, DTYPE_INT, DTYPE_FLOAT

########################### USEFUL FUNCTIONS ###########################
# These should be moved in a separate file to avoid cluttering and be 
# reused in other places!
########################################################################

@gtscript.function
def sqrt(x):
    return x**0.5


@gtscript.function
def exp(x):
    return np.e**x
        

@gtscript.function
def min(x, y):
    return x if x <= y else y 
        

@gtscript.function
def max(x, y):
    return x if x >= y else y 
        

@gtscript.function
def log(x, a):
    return a * (x**(1.0/a)) - a
        

def slice_to_3d(slice):
    return slice[np.newaxis, :, :]
    
    
def exit_routine(cnvflg, im):
    cnvflg.synchronize()
    cnvflg_np = cnvflg[0,:im,0].view(np.ndarray)
    return cnvflg_np.sum() == 0


@gtscript.stencil(backend=BACKEND, rebuild=REBUILD)
def get_1D_from_index(
        infield : FIELD_FLOAT, #2D array X
        outfield: FIELD_FLOAT, #1D array X[i,ind(i)]
        ind     : FIELD_INT, #1D array ind
        k_idx   : FIELD_INT #1D array k_idx
):
    with computation(FORWARD), interval(0, 1):
        outfield = infield

    with computation(FORWARD), interval(1, None):
        outfield = infield
        if (k_idx > ind):
            outfield = outfield[0, 0, -1]

    with computation(BACKWARD), interval(0, -1):
        outfield = outfield[0, 0, 1]
        
########################################################################
