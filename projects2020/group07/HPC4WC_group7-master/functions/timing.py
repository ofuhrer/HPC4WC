import numpy as np
try: 
    import cupy as cp
except ImportError:
    cp=np
import time

def get_time():
    try: 
        cp.cuda.Device().synchronize()
    except AttributeError:
        pass
    return time.time()