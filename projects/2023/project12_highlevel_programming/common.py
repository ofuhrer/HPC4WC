import numpy as np
import matplotlib.pyplot as plt


def initialize_fields(NX, NY, NZ, mode="random", order="C", dtype=np.float64):
    """
    This function initializes the 3D fields with some patterns to help validating
    the stencil update functions.
    """
    
    # Initialize 3D fields
    rng = np.random.default_rng()

    in_field = np.zeros([NZ, NY, NX], order=order, dtype=dtype)
    
    if mode == "random":
        rng.random(size=[NZ, NY, NX], out=in_field, dtype=dtype)
        # Uniformly distributed in [-1, 1)
        in_field = 2 * in_field - 1  
    elif mode == "horizontal-bars":
        in_field[:,: NY//2 : 2, :] = 1
    elif mode == "vertical-bars":
        in_field[:, :,: NX//2 : 2] = 1
    elif mode == "square":
        in_field[:, NY//2 - NY//4 : NY//2 + NY//4, NX//2 - NX//4 : NX//2 + NX//4] = 1
    else:
        raise ValueError("Wrong mode")
    
    out_field = np.copy(in_field)
    
    return in_field, out_field

def plot_field(field, k=0):
    field = np.array(field)
    plt.imshow(field[k, :, :], origin='lower', vmin=-1, vmax=1);
    plt.colorbar();