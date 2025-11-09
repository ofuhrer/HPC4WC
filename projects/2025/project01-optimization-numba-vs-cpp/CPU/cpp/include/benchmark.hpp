#ifndef BENCHMARK_HPP
#define BENCHMARK_HPP
#include <omp.h>
#include <iostream>
#include <chrono>
#include <random>
#include <cstdlib>
#include <omp.h>

extern void velocity_smoother(int nx1, int ny1, double dx, double dy,
                              const double* etap, const double* etab,
                              double* vx, double* vy,
                              double relax_v, double BC,
                              const double* vx_rhs, const double* vy_rhs,
                              double* vx_new, double* vy_new,
                              int max_iter);

double run_benchmark(int nx, int ny, int max_iter) {
    int nx1 = nx;
    int ny1 = ny;
    double dx = 1.0 / (nx - 1);
    double dy = 1.0 / (ny - 1);
    double relax_v = 0.7;
    double BC = -1.0;

    // Check that number of iterations is even
    if (max_iter % 2 != 0) {
        std::cerr << "Error: max_iter must be even." << std::endl;
        return -1.0;
    }

    size_t size = static_cast<size_t>(nx1) * ny1;
    double *etap    = (double*) aligned_alloc(ALIGN_BYTES, size * sizeof(double));
    double *etab    = (double*) aligned_alloc(ALIGN_BYTES, size * sizeof(double));
    double *vx      = (double*) aligned_alloc(ALIGN_BYTES, size * sizeof(double));
    double *vy      = (double*) aligned_alloc(ALIGN_BYTES, size * sizeof(double));
    double *vx_new  = (double*) aligned_alloc(ALIGN_BYTES, size * sizeof(double));
    double *vy_new  = (double*) aligned_alloc(ALIGN_BYTES, size * sizeof(double));
    double *vx_rhs  = (double*) aligned_alloc(ALIGN_BYTES, size * sizeof(double));
    double *vy_rhs  = (double*) aligned_alloc(ALIGN_BYTES, size * sizeof(double));

    std::mt19937_64 rng(42);
    std::uniform_real_distribution<double> dist(1e19, 2e19);
    for (size_t i = 0; i < size; ++i) {
        etap[i] = dist(rng);
        etab[i] = dist(rng);
        vx[i] = vy[i] = vx_rhs[i] = vy_rhs[i] = 0.0;
    }

    // Optional warmup
    velocity_smoother(nx1, ny1, dx, dy,
                      etap, etab,
                      vx, vy,
                      relax_v, BC,
                      vx_rhs, vy_rhs,
                      vx_new, vy_new,
                      1);

    // Timed run
    int n_threads = omp_get_max_threads();
    auto start = std::chrono::high_resolution_clock::now();
    velocity_smoother(nx1, ny1, dx, dy,
                      etap, etab,
                      vx, vy,
                      relax_v, BC,
                      vx_rhs, vy_rhs,
                      vx_new, vy_new,
                      max_iter);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> elapsed = end - start;

    free(etap); free(etab);
    free(vx); free(vy);
    free(vx_new); free(vy_new);
    free(vx_rhs); free(vy_rhs);

    #ifdef PRINT_CSV
    std::cout << nx1 << "," << ny1 << "," << max_iter << "," << n_threads << "," << elapsed.count() << "\n";
    #else
    std::cout << "Grid size: " << nx1 << "x" << ny1 << std::endl
              << "Max iterations: " << max_iter << std::endl
              << "Threads: " << n_threads << std::endl
              << "Elapsed time: " << elapsed.count() << " seconds\n";
    #endif

    return elapsed.count();
}
#endif // BENCHMARK_HPP