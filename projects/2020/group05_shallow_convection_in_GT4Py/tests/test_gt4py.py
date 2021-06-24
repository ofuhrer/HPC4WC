import numpy as np

import gt4py
import gt4py.gtscript as gtscript
import gt4py.storage as gt_storage
from gt4py.gtscript import PARALLEL, computation, interval
import sys
sys.path.append("..")
from shalconv.kernels.utils import exp


backend="numpy" # "debug", "numpy", "gtx86", "gtcuda"
dtype = np.float64

@gtscript.function
def f(x):
    return x

@gtscript.stencil(backend=backend, externals={"f":f, "exp":exp}) # this decorator triggers compilation of the stencil
def stencil_test(data: gtscript.Field[dtype]):
    with computation(PARALLEL), interval(...):
        x = exp(0)
        y = 0.0
        if 1:
            y = f(x)

data = gt4py.storage.empty(backend, (0,0,0), (10,10,10), dtype=dtype)

stencil_test(data, domain = (10,10,10))
