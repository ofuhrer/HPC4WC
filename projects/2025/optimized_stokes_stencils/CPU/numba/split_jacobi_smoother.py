import sys
import numba as nb
import numpy as np
from time import time

nb.config.THREADING_LAYER = 'omp'

@nb.njit(cache=True, inline="always")
def apply_vx_BC(vx, BC):
    vx[0, :]  = -BC * vx[1, :]
    vx[-1, :] = -BC * vx[-2, :]
    vx[:, 0]  = 0.0
    vx[:, -2:] = 0.0
    return vx

@nb.njit(cache=True, inline="always")
def apply_vy_BC(vy, BC):
    vy[:, 0]   = -BC * vy[:, 1]
    vy[:, -1]  = -BC * vy[:, -2]
    vy[0, :]   = 0.0
    vy[-2:, :] = 0.0
    return vy

@nb.njit(cache=True, inline="always")
def _compute_x_momentum_coefficients(dx, dy, etaA, etaB, eta1, eta2):
    vx1_coeff = 2.0 * etaA / (dx * dx)
    vx2_coeff = eta1 / (dy * dy)
    vx3_coeff = -(eta1 + eta2) / (dy * dy) - 2.0 * (etaA + etaB) / (dx * dx)
    vx4_coeff = eta2 / (dy * dy)
    vx5_coeff = 2.0 * etaB / (dx * dx)
    vy1_coeff = eta1 / (dx * dy)
    vy2_coeff = -eta2 / (dx * dy)
    vy3_coeff = -eta1 / (dx * dy)
    vy4_coeff = eta2 / (dx * dy)
    return (vx1_coeff, vx2_coeff, vx3_coeff, vx4_coeff, vx5_coeff,
            vy1_coeff, vy2_coeff, vy3_coeff, vy4_coeff)

@nb.njit(cache=True, inline="always")
def _compute_y_momentum_coefficients(dx, dy, etaA, etaB, eta1, eta2):
    vy1_coeff = eta1 / (dx * dx)
    vy2_coeff = 2.0 * etaA / (dy * dy)
    vy3_coeff = -2.0 * (etaA + etaB) / (dy * dy) - (eta1 + eta2) / (dx * dx)
    vy4_coeff = 2.0 * etaB / (dy * dy)
    vy5_coeff = eta2 / (dx * dx)
    vx1_coeff = eta1 / (dx * dy)
    vx2_coeff = -eta1 / (dx * dy)
    vx3_coeff = -eta2 / (dx * dy)
    vx4_coeff = eta2 / (dx * dy)
    return (vy1_coeff, vy2_coeff, vy3_coeff, vy4_coeff, vy5_coeff,
            vx1_coeff, vx2_coeff, vx3_coeff, vx4_coeff)

@nb.njit(cache=True, parallel=True)
def naive_velocity_smoother(nx1, ny1, dx, dy,
                            etap, etab,
                            vx, vy,
                            relax_v, BC,
                            vx_rhs, vy_rhs,
                            vx_new, vy_new,
                            max_iter):

    for iter in range(max_iter):
        # vx pass
        for i in nb.prange(1, ny1 - 1):
            for j in range(1, nx1 - 2):
                etaA = etap[i, j]
                etaB = etap[i, j + 1]
                eta1 = etab[i - 1, j]
                eta2 = etab[i, j]
        
                vx1, vx2, vx3, vx4, vx5, vy1, vy2, vy3, vy4 = \
                    _compute_x_momentum_coefficients(dx, dy, etaA, etaB, eta1, eta2)

                sum_neighbors = (
                    vx1 * vx[i, j - 1] +
                    vx2 * vx[i - 1, j] +
                    vx4 * vx[i + 1, j] +
                    vx5 * vx[i, j + 1] +
                    vy1 * vy[i - 1, j] +
                    vy2 * vy[i, j] +
                    vy3 * vy[i - 1, j + 1] +
                    vy4 * vy[i, j + 1]
                )
                diag = vx3
                vx_new[i, j] = (1.0 - relax_v) * vx[i, j] + relax_v * (vx_rhs[i, j] - sum_neighbors) / diag
        
        # vy pass
        for i in nb.prange(1, ny1 - 2):
            for j in range(1, nx1 - 1):
                etaA = etap[i, j]
                etaB = etap[i, j + 1]
                eta1 = etab[i - 1, j]
                eta2 = etab[i, j]
                

                vy1, vy2, vy3, vy4, vy5, vx1, vx2, vx3, vx4 = \
                    _compute_y_momentum_coefficients(dx, dy, etaA, etaB, eta1, eta2)

                sum_neighbors = (
                    vy1 * vy[i, j - 1] +
                    vy2 * vy[i - 1, j] +
                    vy4 * vy[i + 1, j] +
                    vy5 * vy[i, j + 1] +
                    vx1 * vx[i, j - 1] +
                    vx2 * vx[i + 1, j - 1] +
                    vx3 * vx[i, j] +
                    vx4 * vx[i + 1, j]
                )
                diag = vy3
                vy_new[i, j] = (1.0 - relax_v) * vy[i, j] + relax_v * (vy_rhs[i, j] - sum_neighbors) / diag

        apply_vx_BC(vx_new, BC)
        apply_vy_BC(vy_new, BC)
        vx, vy = vx_new, vy_new

    return vx, vy, vx_new, vy_new

if __name__ == "__main__":
    max_iter = 0
    nx = 0
    ny = 0

     # Ensure the script receives the correct number of command-line arguments.
    if len(sys.argv) != 4:
        print("Usage: python script_name.py <nx> <ny> <max_iter>")
        sys.exit(1) # Exit with an error code

    # Parse the command-line arguments.
    try:
        nx = int(sys.argv[1])
        ny = int(sys.argv[2])
        max_iter = int(sys.argv[3])
    except ValueError:
        print("Error: nx, ny, and max_iter must be integers.")
        sys.exit(1) # Exit if conversion fails

    dx = 1.0/(nx-1)
    dy = 1.0/(ny-1)
    nx1 = nx + 1
    ny1 = ny + 1

    etab = np.random.rand(ny1, nx1) * 1e19 + 1e19
    etap = np.random.rand(ny1, nx1) * 1e19 + 1e19

    vx = np.zeros((ny1, nx1))
    vy = np.zeros((ny1, nx1))
    vx_new = np.zeros((ny1, nx1))
    vy_new = np.zeros((ny1, nx1))
    vx_rhs = np.zeros((ny1, nx1))
    vy_rhs = np.zeros((ny1, nx1))

    BC = -1.0
    relax_v = 0.7

    # Warm-up
    naive_velocity_smoother(nx1, ny1, dx, dy,
                            etap, etab,
                            vx, vy,
                            relax_v, BC,
                            vx_rhs, vy_rhs,
                            vx_new, vy_new,
                            max_iter=1)


    start = time()
    naive_velocity_smoother(nx1, ny1, dx, dy,
                            etap, etab,
                            vx, vy,
                            relax_v, BC,
                            vx_rhs, vy_rhs,
                            vx_new, vy_new,
                            max_iter=max_iter)
    end = time()
    # print(f"Naive Jacobi time: {end - start:.4f} seconds for {max_iter} iterations")
    elapsed_time = end - start

    print(f"Grid size: {nx}x{ny}")
    print(f"Max iterations: {max_iter}")
    print(f"Threads: {nb.config.NUMBA_NUM_THREADS}")
    print(f"Elapsed time: {elapsed_time:.6f} seconds")