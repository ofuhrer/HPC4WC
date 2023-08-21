import numpy as np
import matplotlib.pyplot as plt


def initialize_fields(NX, NY, NZ, mode="random", num_halo=0, order="C", dtype=np.float64):
    """
    This function initializes the 3D fields with some patterns to help validating
    the stencil update functions.
    """

    assert num_halo < NX // 2 and num_halo < NY // 2
    
    # Initialize 3D fields
    rng = np.random.default_rng()

    in_field = np.zeros([NZ, NY, NX], order=order, dtype=dtype)
    
    if mode == "random":
        tmp = rng.random(size=[NZ, NY - 2 * num_halo, NX - 2 * num_halo], dtype=dtype)
        # Uniformly distributed in [-1, 1)
        in_field[:, num_halo : NY - num_halo, num_halo : NX - num_halo] = 2 * tmp - 1  
    elif mode == "horizontal-bars":
        in_field[:, num_halo : NY//2 - num_halo : 2, num_halo : NX - num_halo] = 1
    elif mode == "vertical-bars":
        in_field[:, num_halo : NY - num_halo, num_halo : NX//2 - num_halo : 2] = 1
    elif mode == "square":
        # num_halo is ignored in this mode
        in_field[:, NY//2 - NY//4 : NY//2 + NY//4, NX//2 - NX//4 : NX//2 + NX//4] = 1
    else:
        raise ValueError("Wrong mode")
    
    out_field = np.copy(in_field)
    
    return in_field, out_field

def plot_field(field, k=0):
    field = np.array(field)
    plt.figure(figsize=(7, 5), dpi=100)
    plt.imshow(field[k, :, :], origin='lower', vmin=-1, vmax=1);
    plt.colorbar();