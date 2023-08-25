import os
import numpy as np
import IPython
from datetime import datetime
import matplotlib.pyplot as plt


# Initialize NumPy PRNG with a seed so that Notebooks are reproducible
rng = np.random.default_rng(1337)


def initialize_field(NX, NY, NZ, dim_order="ZYX", mode="random", num_halo=0, array_order="C", dtype=np.float64):
    """
    This function initializes the 3D fields with some patterns to help validating
    the stencil update functions.
    """

    assert 0 <= num_halo < NX // 2 and 0 <= num_halo < NY // 2
    
    # We do all the initialization assuming the dim_order="ZYX"
    field = np.zeros([NZ, NY, NX], dtype=dtype)

    assert isinstance(mode, str)
    if mode == "random":
        tmp = rng.random(size=[NZ, NY - 2 * num_halo, NX - 2 * num_halo], dtype=dtype)
        # Uniformly distributed in [-1, 1)
        field[:, num_halo : NY - num_halo, num_halo : NX - num_halo] = 2 * tmp - 1  
    elif mode == "horizontal-bars":
        field[:, num_halo : NY//2 - num_halo : 2, num_halo : NX - num_halo] = 1
    elif mode == "vertical-bars":
        field[:, num_halo : NY - num_halo, num_halo : NX//2 - num_halo : 2] = 1
    elif mode == "square":
        # num_halo is ignored in this mode
        field[:, NY//2 - NY//4 : NY//2 + NY//4, NX//2 - NX//4 : NX//2 + NX//4] = 1
    else:
        raise ValueError("Wrong mode")
    
    # Rearrange dimensions to dim_order
    assert isinstance(dim_order, str)
    if dim_order == "ZYX":
        pass
    elif dim_order == "XZY":
        field = np.transpose(field, axes=[2, 0, 1])
    elif dim_order == "YXZ":
        field = np.transpose(field, axes=[1, 2, 0])
    elif dim_order == "XYZ":
        field = np.transpose(field, axes=[2, 1, 0])
    elif dim_order == "ZXY":
        field = np.transpose(field, axes=[0, 2, 1])
    elif dim_order == "YZX":
        field = np.transpose(field, axes=[1, 0, 2])
    else:
        raise ValueError("Wrong dim order")
            
    
    # Enforce returning arrays using the right C or Fortran order
    assert isinstance(array_order, str)
    if array_order == "C":
        return np.ascontiguousarray(field)
    elif array_order == "F":
        return np.asfortranarray(field)
    else:
        raise ValueError("Wrong array order")
    
    
def plot_field(in_field, dim_order="ZYX", k=0):
    field = np.array(in_field)
    
    plt.figure(figsize=(7, 5), dpi=100)
    
    # Rearrange dimensions to the desire
    assert isinstance(dim_order, str)
    if dim_order == "ZYX":
        pass
    elif dim_order == "XZY":
        field = np.transpose(field, axes=[1, 2, 0])
    elif dim_order == "YXZ":
        field = np.transpose(field, axes=[2, 0, 1])
    elif dim_order == "XYZ":
        field = np.transpose(field, axes=[2, 1, 0])
    elif dim_order == "ZXY":
        field = np.transpose(field, axes=[0, 2, 1])
    elif dim_order == "YZX":
        field = np.transpose(field, axes=[1, 0, 2])
    else:
        raise ValueError("Wrong dim order")
    
    plt.imshow(field[k, :, :], origin='lower', vmin=-1, vmax=1);    
    plt.colorbar();

def save_result(result, test_name=None, file="results.csv", overwrite=False, header=False):
    if overwrite:
        open(file, "w").close()
    
    with open(file, "a") as f:
        if header:
            print("timestamp,function,hardware,timeit_avg,timeit_std", file=f)
        
        if result is not None:
            assert isinstance(result, IPython.core.magics.execution.TimeitResult)
            assert isinstance(test_name, str)
            print(f"{datetime.utcnow()},{test_name},{os.uname()[1]},{result.average:.2e},{result.stdev:.2e}", file=f)


def compare_results(a, b, mode="faster"):
    assert isinstance(mode, str)
    assert mode in ["faster", "faster-%"]

    if mode == "faster":
        "A is x times as fast as B"
        if a == 0:
            return "∞"
        res = b / a
        if res > 10:
            # We don't care about decimals in this case
            return f"~{b / a:.0f}"
        else:
            return f"~{b / a:.1f}"
    elif mode == "faster-%":
        "A is x% faster than B"
        assert a <= b
        if a == 0:
            return "∞"
        return f"~{(b - a) / a * 100:.0f}%"
    else:
        raise ValueError("Wrong comparison mode")
