"""
write a generic nanmean function to give to scipy.generic_filter for calculating the temporal and spatial interpolations that is speed up using numba.

Borrowed from https://ilovesymposia.com/2017/03/12/scipys-new-lowlevelcallable-is-a-game-changer/

    @author: Verena Bessenbacher
    @date: 01 04 2020 no april's fool!

"""

import numpy as np
from numba import cfunc, carray
from numba.types import intc, intp, float64, voidptr
from numba.types import CPointer

# mean of footprint as I need it
@cfunc(intc(CPointer(float64), intp,
            CPointer(float64), voidptr))
def nbnanmean(values_ptr, len_values, result, data):
    values = carray(values_ptr, (len_values,), dtype=float64)
    result[0] = np.nan
    tmp = 0
    i = 0
    for v in values:
        if ~np.isnan(v):
            tmp = tmp + v
            i = i + 1
    if i != 0:
        result[0] = tmp / max(i,1)
    return 1

if __name__ == '__main__':
    data = np.arange(25.).reshape(5,5)
    data[np.random.rand(*data.shape) < 0.1] = np.nan
    print(data)

    footprint = np.full((3,3), 1)
    footprint[1,1] = 0

    # original slow scipy version
    from scipy.ndimage.filters import generic_filter
    res = generic_filter(data, np.nanmean, footprint=footprint)
    print(res)

    # fast numba version
    from scipy import LowLevelCallable
    res = generic_filter(data, LowLevelCallable(nbnanmean.ctypes), footprint=footprint)
    print(res)
