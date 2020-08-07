import cupy as cp
import timeit

def get_time(sync=True):
    if sync:
        cp.cuda.Device().synchronize()
    return timeit.default_timer()