import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sp_fft
from scipy.interpolate import RegularGridInterpolator
import os

dtypes = {
        "half": np.float16,
        "single": np.float32,
        "double": np.float64,
        "longdouble": np.longdouble
    }

def read_field_from_file(filename, num_halo=None) -> np.array:
    """
    Function to read a binary file containing a 2D or 3D field with a header. Returns the field as a numpy array without halo regions.
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
        return np.reshape(data[offset:], (nz, ny, nx))[:, num_halo:-num_halo, num_halo:-num_halo]
    else:
        return np.reshape(data[offset:], (ny, nx))[num_halo:-num_halo, num_halo:-num_halo]

def read_snapshots(filename, nx, ny, nz, n_halo, num_snapshots, dtype=np.float64()):
    """
    Function to read snapshots of velocity fields from binary files.
    """
    snapshots = []
    nx_tot = nx + 2 * n_halo
    ny_tot = ny + 2 * n_halo
    header_size = 24 #6 * 4  # 6 int32 fields × 4 bytes each (to skip metadata)
    footer_size = 24 # trailing bytes
    field_size = nx_tot * ny_tot * nz * np.dtype(dtype).itemsize
    k = 0 # Vertical level to plot

    with open(filename, "rb") as f:
        for i in range(num_snapshots):

            header_bytes = f.read(header_size)

            if len(header_bytes) < header_size:
                break

            field_bytes = f.read(field_size)
            if len(field_bytes) < field_size:
                break

            data = np.frombuffer(field_bytes, dtype=dtype).reshape((nz, ny_tot, nx_tot))

            inner = data[0, n_halo:n_halo+ny, n_halo:n_halo+nx]

            snapshots.append(inner)
    return np.array(snapshots)

def interpolate2D(field: np.array, shape: tuple) -> np.array:
    """
    Interpolate the horizontal field to the new shape.
    """

    ny_orig, nx_orig = field.shape
    ny_new, nx_new = shape
    lx, ly = 1.0, 1.0

    x_orig = np.linspace(0, lx, nx_orig)
    y_orig = np.linspace(0, ly, ny_orig)

    # New grid coordinates
    x = np.linspace(0, lx, nx_new)
    y = np.linspace(0, ly, ny_new)

    # Prepare meshgrid of new coordinates
    y_grid, x_grid = np.meshgrid(y, x, indexing="ij")  # shape (ny_new, nx_new)

    # Allocate output
    new_field = np.zeros(shape, dtype=field.dtype)


    interp_func = RegularGridInterpolator(
        (y_orig, x_orig),
        field,
        method="linear",
        bounds_error=False,
        fill_value=None,
    )

    # Flatten coordinate pairs
    points = np.stack([y_grid.ravel(), x_grid.ravel()], axis=-1)  # shape (nx_new * ny_new, 2)
    # Interpolate
    interpolated_values = interp_func(points)
    # Reshape to (ny_new, nx_new)
    new_field[:, :] = interpolated_values.reshape((ny_new, nx_new))

    return new_field

def calc_err(field: np.array, benchmark: np.array) -> np.float64:
    """
    Calculates the mean error between the field and the benchmark.
    """
    err2 = np.linalg.norm(field - benchmark)
    mean_err2 = err2/(benchmark.shape[0] * benchmark.shape[1])
    return mean_err2

def calc_err_psd(field: np.array, benchmark: np.array) -> np.float64:
    """
    Calculates the mean psd error between the field and the benchmark.
    """
    fft_field = sp_fft.fftshift(sp_fft.fft2(field))
    fft_benchmark = sp_fft.fftshift(sp_fft.fft2(benchmark))
    field_psd = np.abs(fft_field)**2
    benchmark_psd = np.abs(fft_benchmark)**2
    err2 = np.linalg.norm(field_psd - benchmark_psd)
    mean_err2 = err2/(benchmark.shape[0] * benchmark.shape[1])
    return mean_err2

def results(figname:str, benchmark_fname: str, output_fnames: list, benchmark_resolution: tuple, benchmark_precision: str, output_resolutions: list, output_precisions: list, num_snapshots: int, Tend: float) -> None:
    """
    Calculates the l2 and psd errors between the output fields and the benchmark field, and plots them over time.
    """
    benchmark = read_snapshots(benchmark_fname, *benchmark_resolution, n_halo=3, num_snapshots=num_snapshots, dtype=dtypes[benchmark_precision])
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    colors = ['r', 'g', 'b']
    linestyles = ['-', '--', ':']
    for i, res in enumerate(output_resolutions):
        for j, prec in enumerate(output_precisions):
            fname = output_fnames[i][j]
            output_field = read_snapshots(fname, *res, n_halo=3, num_snapshots=num_snapshots, dtype=dtypes[prec])
            errors = []
            psd_errors = []
            
            for t in range(num_snapshots):
                benchmark_field = benchmark[t,:,:]
                field = interpolate2D(output_field[t, :, :], shape=benchmark_field.shape)
                
                err = calc_err(field, benchmark_field)
                errors.append(err)
                err_psd = calc_err_psd(field, benchmark_field)
                psd_errors.append(err_psd)
            
            t = np.linspace(0, Tend, num_snapshots)
            axs[0].plot(t, errors, label=f"{prec}, {res}", color=colors[i], linestyle=linestyles[j])
            axs[1].plot(t, psd_errors, label=f"{prec}, {res}", color=colors[i], linestyle=linestyles[j])

    axs[0].set_xlabel("Time")
    axs[0].set_ylabel("L2 Error")
    axs[0].set_title("L2 Error over Time")
    axs[0].grid()
    axs[1].set_xlabel("Time")
    axs[1].set_ylabel("PSD Error")
    axs[1].set_title("PSD Error over Time")
    axs[1].legend()
    axs[1].grid()
    plt.tight_layout()
    plt.savefig(figname)

if __name__ == "__main__":
    benchmark_fname = "output_data/u_out_field_nx_1024_ny_1024_nz_1_longdouble.dat"
    resolutions = [(128, 128, 1), (256, 256, 1), (512, 512, 1)]
    precisions = ["single", "double", "longdouble"]
    output_resolutions = []
    output_precisions = []

    fnames = [["" for _ in precisions] for __ in resolutions]
    for i, res in enumerate(resolutions):
        nx, ny, nz = res
        for j, prec in enumerate(precisions):
            fname = f"output_data/u_out_field_nx_{nx}_ny_{ny}_nz_{nz}_{prec}.dat"
            fnames[i][j] = fname
    
    figname = "figures/error_analysis.png"
    results(figname, benchmark_fname, fnames, (1024, 1024, 1), "longdouble", resolutions, precisions, num_snapshots=99, Tend=1.0)