#include <iostream>
#include <chrono>
#include <random>
#include <omp.h>
#include <cstdlib>
#include <cstring>
#include "benchmark.hpp"
#include "macros.hpp"
#include "utils.hpp"

void velocity_smoother(int nx1, int ny1, double dx, double dy,
                       const double* etap, const double* etab,
                       double* vx, double* vy,
                       double relax_v, double BC,
                       const double* vx_rhs, const double* vy_rhs,
                       double* vx_new, double* vy_new,
                       int max_iter) {

    for (int iter = 0; iter < max_iter; ++iter) {
        // vx pass
        #pragma omp parallel for collapse(2)
        for (int ii = 1; ii < ny1 - 1; ii += TILE_I) {
            for (int jj = 1; jj < nx1 - 2; jj += TILE_J) {
                int i_max = std::min(ii + TILE_I, ny1 - 1);
                int j_max = std::min(jj + TILE_J, nx1 - 2);

                for (int i = ii; i < i_max; ++i) {
                    for (int j = jj; j < j_max; ++j) {
                        double etaA = AT(etap, i, j, nx1);
                        double etaB = AT(etap, i, j + 1, nx1);
                        double eta1 = AT(etab, i - 1, j, nx1);
                        double eta2 = AT(etab, i, j, nx1);

                        double vx1, vx2, vx3, vx4, vx5;
                        double vy1, vy2, vy3, vy4;

                        compute_x_coeff(dx, dy, etaA, etaB, eta1, eta2,
                                        vx1, vx2, vx3, vx4, vx5,
                                        vy1, vy2, vy3, vy4);

                        double sum_neighbors =
                            vx1 * AT(vx, i, j - 1, nx1) +
                            vx2 * AT(vx, i - 1, j, nx1) +
                            vx4 * AT(vx, i + 1, j, nx1) +
                            vx5 * AT(vx, i, j + 1, nx1) +
                            vy1 * AT(vy, i - 1, j, nx1) +
                            vy2 * AT(vy, i, j, nx1) +
                            vy3 * AT(vy, i - 1, j + 1, nx1) +
                            vy4 * AT(vy, i, j + 1, nx1);

                        double diag = vx3;

                        AT(vx_new, i, j, nx1) = (1.0 - relax_v) * AT(vx, i, j, nx1)
                                              + relax_v * (AT(vx_rhs, i, j, nx1) - sum_neighbors) / diag;
                    }
                }
            }
        }

        // vy pass
        #pragma omp parallel for collapse(2)
        for (int ii = 1; ii < ny1 - 2; ii += TILE_I) {
            for (int jj = 1; jj < nx1 - 1; jj += TILE_J) {
                int i_max = std::min(ii + TILE_I, ny1 - 2);
                int j_max = std::min(jj + TILE_J, nx1 - 1);

                for (int i = ii; i < i_max; ++i) {
                    for (int j = jj; j < j_max; ++j) {
                        double etaA = AT(etap, i, j, nx1);
                        double etaB = AT(etap, i, j + 1, nx1);
                        double eta1 = AT(etab, i - 1, j, nx1);
                        double eta2 = AT(etab, i, j, nx1);

                        double vy1, vy2, vy3, vy4, vy5;
                        double vx1, vx2, vx3, vx4;

                        compute_y_coeff(dx, dy, etaA, etaB, eta1, eta2,
                                        vy1, vy2, vy3, vy4, vy5,
                                        vx1, vx2, vx3, vx4);

                        double sum_neighbors =
                            vy1 * AT(vy, i, j - 1, nx1) +
                            vy2 * AT(vy, i - 1, j, nx1) +
                            vy4 * AT(vy, i + 1, j, nx1) +
                            vy5 * AT(vy, i, j + 1, nx1) +
                            vx1 * AT(vx, i, j - 1, nx1) +
                            vx2 * AT(vx, i + 1, j - 1, nx1) +
                            vx3 * AT(vx, i, j, nx1) +
                            vx4 * AT(vx, i + 1, j, nx1);

                        double diag = vy3;

                        AT(vy_new, i, j, nx1) = (1.0 - relax_v) * AT(vy, i, j, nx1)
                                              + relax_v * (AT(vy_rhs, i, j, nx1) - sum_neighbors) / diag;
                    }
                }
            }
        }

        apply_vx_BC(vx_new, nx1, ny1, BC);
        apply_vy_BC(vy_new, nx1, ny1, BC);

        std::swap(vx, vx_new);
        std::swap(vy, vy_new);
    }
}

int main(int argc, char* argv[]) {
    // Read nx
    if (argc < 2) {
        std::cerr << "Usage: " << argv[0] << " <nx> [ny = nx] [max_iter = 10]" << std::endl;
        return 1;
    }

    const int nx = std::atoi(argv[1]);
    const int ny = (argc > 2) ? std::atoi(argv[2]) : nx;
    const int max_iter = (argc > 3) ? std::atoi(argv[3]) : 10;

    if (nx < 101 || ny < 101) {
        std::cerr << "Error: nx and ny must be at least 101 x 101." << std::endl;
        return 1;
    }

    // Run the benchmark
    run_benchmark(nx, ny, max_iter);


    return 0;
}