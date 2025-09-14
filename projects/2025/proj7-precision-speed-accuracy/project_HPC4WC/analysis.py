import numpy as np
import matplotlib.pyplot as plt
import scipy.fft as sp_fft
from scipy.interpolate import RegularGridInterpolator
import os

def read_field_from_file(filename, num_halo=None) -> np.array:
    dtypes = {
        16: np.float16,
        32: np.float32,
        64: np.float64,
        128: np.longdouble # note: np.longdouble is only 8 bytes long on macos and does not work there (but on the HPC cluster it does!)
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
        benchmark[benchmark.shape[0] // 2, :, :], origin="lower", vmin=1.1*np.min(benchmark), vmax=1.1*np.max(benchmark), cmap='bwr'
    )
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title("Benchmark")

    im2 = axs[1].imshow(
        out_field[out_field.shape[0] // 2, :, :], origin="lower", vmin=1.1*np.min(out_field), vmax=1.1*np.max(out_field), cmap='bwr'
    )
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title("Output Field")

    plt.savefig(figname, format='svg', dpi=1000)
    plt.close(fig)
    return

def plot_psd(benchmark: np.array, out_field: np.array, figname: str) -> None: 
    """
    Plots the benchmark and output power spectral density (psd) fields at the middle z level.
    Args:
        benchmark: 2D numpy array of the benchmark psd field (shape: (ny, nx))
        out_field: 2D numpy array of the output psd field (shape: (ny, nx))
        figname: string, name of the file to save the plot
    """
    # spectral frequencies
    ny, nx = benchmark.shape
    fx = sp_fft.fftfreq(nx, d=1.0/(nx-1.0)) 
    fy = sp_fft.fftfreq(ny, d=1.0/(ny-1.0))
    # Shift zero frequency to center
    fx = sp_fft.fftshift(fx)
    fy = sp_fft.fftshift(fy)
    benchmark = sp_fft.fftshift(benchmark)
    benchmark = np.log(benchmark + 1e-10)  # Avoid log(0) issues
    out_field = sp_fft.fftshift(out_field)
    out_field = np.log(out_field + 1e-10)  # Avoid log(0) issues

    fig, axs = plt.subplots(1, 2, figsize=(12, 4))

    im1 = axs[0].imshow(
        benchmark, origin="lower", vmin=np.min(benchmark) * 1.1, vmax=np.max(benchmark) * 1.1, cmap='viridis'
    )
    axs[0].set_xlabel("Frequency X")
    axs[0].set_ylabel("Frequency Y")
    axs[0].set_xticks(np.linspace(0, nx-1, 5))
    axs[0].set_xticklabels([f"{fx[int(i)]:.2f}" for i in np.linspace(0, nx-1, 5)])
    axs[0].set_yticks(np.linspace(0, ny-1, 5))
    axs[0].set_yticklabels([f"{fy[int(i)]:.2f}" for i in np.linspace(0, ny-1, 5)])
    fig.colorbar(im1, ax=axs[0])
    axs[0].set_title("Log Benchmark")
    print(np.min(benchmark))
    im2 = axs[1].imshow(
        out_field, origin="lower", vmin=np.min(out_field) * 1.1, vmax=np.max(out_field) * 1.1, cmap='viridis'
    )
    axs[1].set_xlabel("Frequency X")
    axs[1].set_ylabel("Frequency Y")
    axs[1].set_xticks(np.linspace(0, nx-1, 5))
    axs[1].set_xticklabels([f"{fx[int(i)]:.2f}" for i in np.linspace(0, nx-1, 5)])
    axs[1].set_yticks(np.linspace(0, ny-1, 5))
    axs[1].set_yticklabels([f"{fy[int(i)]:.2f}" for i in np.linspace(0, ny-1, 5)])
    fig.colorbar(im2, ax=axs[1])
    axs[1].set_title("Log Output Field")

    plt.savefig(figname, format='svg', dpi=1000)
    plt.close(fig)
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

def calc_err(field: np.array, benchmark: np.array) -> np.float64: 
    """
    Calculates the mean error between the field and the benchmark.
    """
    err2 = np.linalg.norm(field - benchmark)
    mean_err2 = err2/(benchmark.shape[0] * benchmark.shape[1] * benchmark.shape[2])
    return mean_err2

def calc_err_psd(field: np.array, benchmark: np.array) -> np.float64: 
    """
    Calculates the mean error between the field and the benchmark.
    """
    err2 = np.linalg.norm(field - benchmark)
    mean_err2 = err2/(benchmark.shape[0] * benchmark.shape[1])
    return mean_err2

def make_plot_infos(resolutions: list[tuple], precisions: list[str]) -> dict: 
    """
    Create a dictionary with information about the resolutions and precisions for plotting.
    Arguments:
    resolutions: list of tuples, each tuple contains the resolution (nx, ny, nz)
    precisions: list of strings (e.g., "quarter", "float", "double", "long_double")
    """
    infos = {
        "resolution": [[None for _ in precisions] for __ in resolutions],
        "precision": [[None for _ in precisions] for __ in resolutions],
        "input_fname": [[None for _ in precisions] for __ in resolutions],
        "output_fname": [[None for _ in precisions] for __ in resolutions],
        "figname": [[None for _ in precisions] for __ in resolutions],
        "psdname": [[None for _ in precisions] for __ in resolutions],
        "shape": (len(resolutions), len(precisions))
    }
    for i, res in enumerate(resolutions):
        for j, prec in enumerate(precisions):
            infos["resolution"][i][j] = res
            infos["precision"][i][j] = prec
            infos["input_fname"][i][j] = f"input_data/u_in_field_nx_{res[0]}_ny_{res[1]}_nz_{res[2]}_{prec}.dat"
            infos["output_fname"][i][j] = f"output_data/u_out_field_nx_{res[0]}_ny_{res[1]}_nz_{res[2]}_{prec}.dat"
            infos["figname"][i][j] = f"figures/fig_nx_{res[0]}_ny_{res[1]}_nz_{res[2]}_{prec}.svg"
            infos["psdname"][i][j] = f"figures/psd_nx_{res[0]}_ny_{res[1]}_nz_{res[2]}_{prec}.svg"
    return infos

def results(infos: dict, benchmark_fname: str) -> None:
    """
    Calculates the errors between the output fields and the benchmark field, and plots the fields and their psd.
    """
    benchmark = read_field_from_file(benchmark_fname)
    shape = infos["shape"]
    errors = np.zeros(shape, dtype=benchmark.dtype)
    psd_errors = np.zeros(shape, dtype=benchmark.dtype)

    for i in range(shape[0]):
        for j in range(shape[1]):
            out_field = read_field_from_file(infos["output_fname"][i][j])
            out_field = interpolate2D(out_field, benchmark.shape)
            nz, ny, nx = benchmark.shape
            dx, dy = 1.0 / (nx - 1.0) , 1.0 / (ny - 1.0)

            # Normalized 2D psd of the middle z level of the output and benchmark fields
            psd_out = np.abs(sp_fft.fft2(out_field[out_field.shape[0]//2,:,:]))**2 / (nx*ny*dx*dy)
            psd_benchmark = np.abs(sp_fft.fft2(benchmark[benchmark.shape[0]//2,:,:]))**2 / (nx*ny*dx*dy)

            # calculate errors
            errors[i][j] = calc_err(out_field, benchmark)
            psd_errors[i][j] = calc_err_psd(psd_out, psd_benchmark)
            
            # plot the fields and psd
            plot_field(
                benchmark=benchmark,
                out_field=out_field,
                figname=infos['figname'][i][j]
            )
            plot_psd(
                benchmark=psd_benchmark,
                out_field=psd_out,
                figname=infos['psdname'][i][j]
            )

    # Print the errors in a table format
    print("Error Table:")
    header = ["Resolution \\ Precision"] + infos['precision'][0]
    print("{:<25}".format(header[0]), end="")
    for col in header[1:]:
        print("{:<15}".format(col), end="")
    print()
    for i, res in enumerate(infos['resolution']):
        print("{:<25}".format(str(res[0])), end="")
        for j in range(len(infos['precision'][0])):
            print("{:<15.6e}".format(errors[i][j]), end="")
        print()
    print()
    print('_' * 25)
    print("PSD Error Table:")
    print("{:<25}".format(header[0]), end="")
    for col in header[1:]:
        print("{:<15}".format(col), end="")
    print()
    for i, res in enumerate(infos['resolution']):
        print("{:<25}".format(str(res[0])), end="")
        for j in range(len(infos['precision'][0])):
            print("{:<15.6e}".format(psd_errors[i][j]), end="")
        print()
    print()
    return

def main(benchmark_fname, resolutions, precisions):
    infos = make_plot_infos(resolutions, precisions)
    results(infos, benchmark_fname)


if __name__ == "__main__":

    #TODO: choose a benchmark + resolution and precisions to test
    benchmark_fname = "output_data/u_out_field_nx_800_ny_800_nz_1_longdouble.dat"
    resolutions = [(75, 75, 1), (150, 150, 1), (300, 300, 1)]
    precisions = ["single", "double", "longdouble"]
    main(benchmark_fname=benchmark_fname, resolutions=resolutions, precisions=precisions)