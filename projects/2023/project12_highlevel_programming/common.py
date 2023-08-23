import numpy as np
import matplotlib.pyplot as plt
import gt4py as gt

def initialize_fields(NX, NY, NZ, dim_order="ZYX", mode="random", num_halo=0, array_order="C", dtype=np.float64):
    """
    This function initializes the 3D fields with some patterns to help validating
    the stencil update functions.
    """

    assert num_halo < NX // 2 and num_halo < NY // 2
    
    # Initialize 3D fields
    rng = np.random.default_rng()
    
    # We do all the initialization assuming the dim_order="ZYX"
    in_field = np.zeros([NZ, NY, NX], dtype=dtype)

    assert isinstance(mode, str)
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
    
    # Rearrange dimensions to dim_order
    assert isinstance(dim_order, str)
    if dim_order == "ZYX":
        pass
    elif dim_order == "XZY":
        in_field = np.transpose(in_field, axes=[2, 0, 1])
    elif dim_order == "YXZ":
        in_field = np.transpose(in_field, axes=[1, 2, 0])
    elif dim_order == "XYZ":
        in_field = np.transpose(in_field, axes=[2, 1, 0])
    elif dim_order == "ZXY":
        in_field = np.transpose(in_field, axes=[0, 2, 1])
    elif dim_order == "YZX":
        in_field = np.transpose(in_field, axes=[1, 0, 2])
    else:
        raise ValueError("Wrong dim order")
            
    out_field = np.copy(in_field)
    
    # Enforce returning arrays using the right C or Fortran order
    assert isinstance(array_order, str)
    if array_order == "C":
        return np.ascontiguousarray(in_field), np.ascontiguousarray(out_field)
    elif array_order == "F":
        return np.asfortranarray(in_field), np.asfortranarray(out_field)
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
             

def array_to_gt_storage(in_field, out_field, dtype=np.float64, backend="numpy", index=(0, 0, 0)):
    in_field_gt = gt.storage.from_array(
        in_field,
        dtype=dtype,
        backend=backend,
        aligned_index=index
    )
    
    out_field_gt = gt.storage.from_array(
        out_field,
        dtype=dtype,
        backend=backend,
        aligned_index=index
    )
    
    return in_field_gt, out_field_gt
