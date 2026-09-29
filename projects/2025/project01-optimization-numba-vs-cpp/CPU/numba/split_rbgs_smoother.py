import sys
import numba as nb
import numpy as np
from time import time

# Set Numba threading layer to 'omp' for parallel execution
nb.config.THREADING_LAYER = 'omp'

@nb.njit(cache=False)
def apply_vx_BC(vx, BC):
    # Top and bottom
    vx[0, :]  = -BC * vx[1, :]
    vx[-1, :] = -BC * vx[-2, :]
    # Left wall
    vx[:, 0]  = 0.0
    # Right + ghost
    vx[:, -2:] = 0.0
    return vx

@nb.njit(cache=False)
def apply_vy_BC(vy, BC):
    # Left and right
    vy[:, 0]   = -BC * vy[:, 1]
    vy[:, -1]  = -BC * vy[:, -2]
    # Top wall
    vy[0, :]   = 0.0
    # Bottom + ghost
    vy[-2:, :] = 0.0
    return vy

@nb.njit(cache=False, parallel=True)
def _vx_rb_gs_sweep(nx1, ny1,
                    dx, dy,
                    etap, etab,
                    vx, vy,
                    relax_v, rhs, BC):
    """
    In-place Red-Black Gauss-Seidel update for vx.
    """

    #----------------------------
    #  Red pass: (i + j) % 2 == 0
    #----------------------------
    for i in nb.prange(1, ny1 - 1):
        j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
        for j in range(j_start, nx1 - 2, 2):
            # 1) Gather local viscosities
            etaA = etap[i,   j]
            etaB = etap[i,   j+1]
            eta1 = etab[i-1, j]
            eta2 = etab[i,   j]

            # 2) Construct coefficients for x-momentum
            vx1_coeff = 2.0 * etaA / (dx * dx)
            vx2_coeff = eta1     / (dy * dy)
            vx3_coeff = -(eta1 + eta2) / (dy * dy) \
                        - 2.0*(etaA + etaB)/(dx * dx)
            vx4_coeff = eta2 / (dy * dy)
            vx5_coeff = 2.0 * etaB / (dx * dx)

            # Cross terms with vy
            vy1_coeff =  eta1 / (dx * dy)
            vy2_coeff = -eta2 / (dx * dy)
            vy3_coeff = -eta1 / (dx * dy)
            vy4_coeff =  eta2 / (dx * dy)

            # 3) Sum neighbor contributions
            sum_neighbors = (
                vx1_coeff * vx[i,   j-1] +
                vx2_coeff * vx[i-1, j  ] +
                vx4_coeff * vx[i+1, j  ] +
                vx5_coeff * vx[i,   j+1] +
                vy1_coeff * vy[i-1, j  ] +
                vy2_coeff * vy[i,   j  ] +
                vy3_coeff * vy[i-1, j+1] +
                vy4_coeff * vy[i,   j+1]
            )

            diag = vx3_coeff

            # Gauss-Seidel in-place update
            vx[i, j] = (1.0 - relax_v)*vx[i, j] \
                        + relax_v*(rhs[i, j] - sum_neighbors)/diag

    # Apply vx boundary conditions
    apply_vx_BC(vx, BC)

    #----------------------------
    #  Black pass: (i + j) % 2 == 1
    #----------------------------
    for i in nb.prange(1, ny1 - 1):
        j_start = 2 if i % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 2, 2):
            # 1) Gather local viscosities
            etaA = etap[i,   j]
            etaB = etap[i,   j+1]
            eta1 = etab[i-1, j]
            eta2 = etab[i,   j]

            # 2) Construct coefficients for x-momentum
            vx1_coeff = 2.0 * etaA / (dx * dx)
            vx2_coeff = eta1     / (dy * dy)
            vx3_coeff = -(eta1 + eta2) / (dy * dy) \
                        - 2.0*(etaA + etaB)/(dx * dx)
            vx4_coeff = eta2 / (dy * dy)
            vx5_coeff = 2.0 * etaB / (dx * dx)

            # Cross terms with vy
            vy1_coeff =  eta1 / (dx * dy)
            vy2_coeff = -eta2 / (dx * dy)
            vy3_coeff = -eta1 / (dx * dy)
            vy4_coeff =  eta2 / (dx * dy)

            # 3) Sum neighbor contributions
            sum_neighbors = (
                vx1_coeff * vx[i,   j-1] +
                vx2_coeff * vx[i-1, j  ] +
                vx4_coeff * vx[i+1, j  ] +
                vx5_coeff * vx[i,   j+1] +
                vy1_coeff * vy[i-1, j  ] +
                vy2_coeff * vy[i,   j  ] +
                vy3_coeff * vy[i-1, j+1] +
                vy4_coeff * vy[i,   j+1]
            )

            diag = vx3_coeff

            # Gauss-Seidel in-place update
            vx[i, j] = (1.0 - relax_v)*vx[i, j] \
                       + relax_v*(rhs[i, j] - sum_neighbors)/diag

    # Apply vx boundary conditions
    apply_vx_BC(vx, BC)

    return vx

@nb.njit(cache=False, parallel=True)
def _vy_red_black_gs_sweep(nx1, ny1,
                           dx, dy,
                           etap, etab,
                           vx, vy,
                           relax_v, rhs, BC):
    """
    In-place Red-Black Gauss-Seidel update for vy.
    """

    #----------------------------
    #  Red pass
    #----------------------------
    for i in nb.prange(1, ny1 - 2):
        j_start = 1 if i % 2 == 0 else 2  # Red pass starts on even (i+j)
        for j in range(j_start, nx1 - 1, 2):
            if (i + j) % 2 == 0:
                # 1) Gather local viscosities
                etaA = etap[i,   j]
                etaB = etap[i+1, j]
                eta1 = etab[i,   j-1]
                eta2 = etab[i,   j]

                # 2) Construct coefficients for y-momentum
                vy1_coeff = eta1 / (dx * dx)
                vy2_coeff = 2.0 * etaA / (dy * dy)
                vy3_coeff = -2.0 * etaA/(dy*dy) \
                            -2.0 * etaB/(dy*dy) \
                            - eta1/(dx*dx) \
                            - eta2/(dx*dx)
                vy4_coeff = 2.0 * etaB / (dy * dy)
                vy5_coeff = eta2 / (dx * dx)

                # Cross terms with vx
                vx1_coeff =  eta1 / (dx * dy)
                vx2_coeff = -eta1 / (dx * dy)
                vx3_coeff = -eta2 / (dx * dy)
                vx4_coeff =  eta2 / (dx * dy)

                # 3) Sum neighbor contributions
                sum_neighbors = (
                    # vy neighbors
                    vy1_coeff * vy[i,   j-1] +
                    vy2_coeff * vy[i-1, j  ] +
                    vy4_coeff * vy[i+1, j  ] +
                    vy5_coeff * vy[i,   j+1] +
                    # cross terms with vx
                    vx1_coeff * vx[i,   j-1] +
                    vx2_coeff * vx[i+1, j-1] +
                    vx3_coeff * vx[i,   j  ] +
                    vx4_coeff * vx[i+1, j  ]
                )

                diag = vy3_coeff

                # 4) Gauss-Seidel in-place update
                vy[i, j] = (1.0 - relax_v)*vy[i, j] \
                           + relax_v*(rhs[i, j] - sum_neighbors)/diag

    # Apply vy boundary conditions
    apply_vy_BC(vy, BC)

    #----------------------------
    #  Black pass
    #----------------------------
    for i in nb.prange(1, ny1 - 2):
        j_start = 2 if i % 2 == 0 else 1  # Black pass starts on odd (i+j)
        for j in range(j_start, nx1 - 1, 2):
                # 1) Gather viscosities
                etaA = etap[i,   j]
                etaB = etap[i+1, j]
                eta1 = etab[i,   j-1]
                eta2 = etab[i,   j]

                # 2) Coefficients for y-momentum
                vy1_coeff = eta1 / (dx * dx)
                vy2_coeff = 2.0 * etaA / (dy * dy)
                vy3_coeff = -2.0 * etaA/(dy*dy) \
                            -2.0 * etaB/(dy*dy) \
                            - eta1/(dx*dx) \
                            - eta2/(dx*dx)
                vy4_coeff = 2.0 * etaB / (dy * dy)
                vy5_coeff = eta2 / (dx * dx)

                # Cross terms with vx
                vx1_coeff =  eta1 / (dx * dy)
                vx2_coeff = -eta1 / (dx * dy)
                vx3_coeff = -eta2 / (dx * dy)
                vx4_coeff =  eta2 / (dx * dy)

                # 3) Sum neighbors
                sum_neighbors = (
                    vy1_coeff * vy[i,   j-1] +
                    vy2_coeff * vy[i-1, j  ] +
                    vy4_coeff * vy[i+1, j  ] +
                    vy5_coeff * vy[i,   j+1] +
                    vx1_coeff * vx[i,   j-1] +
                    vx2_coeff * vx[i+1, j-1] +
                    vx3_coeff * vx[i,   j  ] +
                    vx4_coeff * vx[i+1, j  ]
                )

                diag = vy3_coeff

                # 4) In-place update
                vy[i, j] = (1.0 - relax_v)*vy[i, j] \
                           + relax_v*(rhs[i, j] - sum_neighbors)/diag

    # Apply vy boundary conditions
    apply_vy_BC(vy, BC)

    return vy

@nb.njit(cache=False)
def velocity_smoother(nx1, ny1,
                      dx, dy,
                      etap, etab,
                      vx, vy,
                      relax_v, BC,
                      vx_rhs, vy_rhs, max_iter):
    """
    Full Uzawa smoother for velocity.
    """
    for _ in range(max_iter):
        vx = _vx_rb_gs_sweep(nx1, ny1,
                             dx, dy,
                             etap, etab,
                             vx, vy,
                             relax_v, vx_rhs, BC)

        vy = _vy_red_black_gs_sweep(nx1, ny1,
                                    dx, dy,
                                    etap, etab,
                                    vx, vy,
                                    relax_v, vy_rhs, BC)

    return vx, vy

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

    dx = 1.0 / (nx - 1)
    dy = 1.0 / (ny - 1)
    nx1 = nx + 1
    ny1 = ny + 1

    # Random material properties
    etab = np.random.rand(ny1, nx1) * 1e19 + 1e19
    etap = np.random.rand(ny1, nx1) * 1e19 + 1e19

    # Allocate fields
    vx = np.zeros((ny1, nx1))
    vy = np.zeros((ny1, nx1))
    vx_rhs = np.zeros_like(vx)
    vy_rhs = np.zeros_like(vy)

    BC = -1.0
    relax_v = 0.7

    # Dry run to compile
    velocity_smoother(nx1, ny1,
                      dx, dy,
                      etap, etab,
                      vx, vy,
                      relax_v, BC,
                      vx_rhs, vy_rhs,
                      max_iter=1)

    # Benchmark
    start = time()
    velocity_smoother(nx1, ny1,
                      dx, dy,
                      etap, etab,
                      vx, vy,
                      relax_v, BC,
                      vx_rhs, vy_rhs,
                      max_iter=max_iter)
    end = time()
    # print(f"Red-Black GS time: {end - start:.4f} seconds for {max_iter} iterations")
    elapsed_time = end - start

    print(f"Grid size: {nx}x{ny}")
    print(f"Max iterations: {max_iter}")
    print(f"Threads: {nb.config.NUMBA_NUM_THREADS}")
    print(f"Elapsed time: {elapsed_time:.6f} seconds")