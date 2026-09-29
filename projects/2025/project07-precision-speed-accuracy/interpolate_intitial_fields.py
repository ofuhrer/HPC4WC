import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sp_fft
from scipy.interpolate import RegularGridInterpolator
import os


def read_field_from_file(filename, num_halo=None) -> np.array:
    """
    Reads the field from a binary file to a numpy array.
    """
    dtypes = {
        16: np.float16,
        32: np.float32,
        64: np.float64,
        128: np.longdouble
    }


    (rank, nbits, num_halo, nx, ny, nz) = np.fromfile(filename, dtype=np.int32, count=6)
    offset = (3 + rank) * 32 // nbits
    try: 
        dtype=dtypes[nbits]
    except:
        raise KeyError(f"Error: Number of bits per value in file ({nbits}) does not match any of {list(dtypes.keys())}.")
    data = np.fromfile(filename, dtype=dtype, count=nz * ny * nx + offset)
    if rank == 3:
        return np.reshape(data[offset:], (nz, ny, nx))
    else:
        return np.reshape(data[offset:], (ny, nx))
    
def save_field_to_file(filename: str, field: np.array) -> None:
    """
    Saves the field to a binary file.
    """
    dtype = field.dtype
    rank = 3
    nbits = np.dtype(dtype).itemsize * 8
    num_halo = 3
    nz, ny, nx = field.shape 

    header = np.array([rank, nbits, num_halo, nx, ny, nz], dtype=np.int32)
    field = field.flatten()
    
    with open(filename, 'wb') as f:
        header.tofile(f)
        field.tofile(f)
    return

def interpolate2D(field: np.array, shape=tuple) -> np.array:
    """
    Interpolate the horizontal field to the new shape.
    """

    nz_orig, ny_orig, nx_orig = field.shape
    nz, ny_new, nx_new = shape
    lx, ly = 1.0, 1.0
    assert nz_orig == nz, "The number of z levels must be the same for both fields."

    x_orig = np.linspace(0, lx, nx_orig)
    y_orig = np.linspace(0, ly, ny_orig)

    # New grid coordinates
    x = np.linspace(0, lx, nx_new)
    y = np.linspace(0, ly, ny_new)

    # Prepare meshgrid of new coordinates
    y_grid, x_grid = np.meshgrid(y, x, indexing="ij")  # shape (ny_new, nx_new)

    # Allocate output
    new_field = np.zeros(shape, dtype=field.dtype)

    for k in range(nz):
        interp_func = RegularGridInterpolator(
            (y_orig, x_orig),
            field[k, :, :],
            method="linear",
            bounds_error=False,
            fill_value=None,
        )

        # Flatten coordinate pairs
        points = np.stack([y_grid.ravel(), x_grid.ravel()], axis=-1)  # shape (nx_new * ny_new, 2)
        # Interpolate
        interpolated_values = interp_func(points)
        # Reshape to (ny_new, nx_new)
        new_field[k, :, :] = interpolated_values.reshape((ny_new, nx_new))

    return new_field

def plot_field(benchmark: np.array, out_field: np.array, figname: str) -> None:
    """
    Plots the benchmark and output fields at the middle z level.
    Args:
        benchmark: 3D numpy array of the benchmark field (shape: (nz, ny, nx))
        out_field: 3D numpy array of the output field (shape: (nz, ny, nx))
        figname: string, name of the file to save the plot
    """
    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    im1 = axs[0].imshow(
        benchmark[benchmark.shape[0] // 2, :, :], origin="lower", vmin=1.1*np.min(benchmark), vmax=1.1*np.max(benchmark), cmap='viridis'
    )
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title("Benchmark")

    im2 = axs[1].imshow(
        out_field[out_field.shape[0] // 2, :, :], origin="lower", vmin=1.1*np.min(out_field), vmax=1.1*np.max(out_field), cmap='viridis'
    )
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title("Output Field")

    plt.savefig(figname)
    plt.close(fig)
    return

def make_and_save_new_fields(ns: list, nz: int, f0: str, mode:str, dtype, f_dtype, num_halo=3) -> None:
    """
    Makes and saves new fields with horizontal resolution given in ns by interpolating the field saved in f0.
    """
    fs = [f"./input_data/{mode}_in_field_nx_{n}_ny_{n}_nz_{nz}_{f_dtype}.dat" for n in ns]
    field = read_field_from_file(f0)
    for f, n in zip(fs, ns):
        new_field = np.zeros((field.shape[0], n+2*num_halo, n+2*num_halo), dtype=dtype)
        new_field[:,num_halo:-num_halo, num_halo:-num_halo] = interpolate2D(np.astype(field[:,num_halo:-num_halo, num_halo:-num_halo], dtype), shape=(field.shape[0], n, n))
    save_field_to_file(f, new_field)
    return

def test():
    ns = [400, 700]
    fs = [f"./input_data/u_in_field_nx_{n}_ny_{n}_nz_64_double.dat" for n in ns]
    f0 = "./input_data/u_in_field_nx_512_ny_512_nz_64_double.dat"
    num_halo = 3
    field = read_field_from_file(f0)
    for f,n in zip(fs,ns):
        other_field = read_field_from_file(f)
        plot_field(field, other_field, f"./figures/test_{n}")

# TODO: Pick horizontal resolution and file to interpolate below
dtype = np.float64
f_dtype = "double"
ns = [512] # desired horizontal resolution
nz = 1 # vertical resolution
f0u = "./input_data/u_in_field_nx_1024_ny_1024_nz_1_longdouble.dat" # file with field that is to be interpolated
f0v = "./input_data/v_in_field_nx_1024_ny_1024_nz_1_longdouble.dat" # file with field that is to be interpolated
make_and_save_new_fields(ns, nz, f0u, 'u', dtype, f_dtype)
make_and_save_new_fields(ns, nz, f0v, 'v', dtype, f_dtype)