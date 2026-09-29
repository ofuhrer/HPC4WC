#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "../cpp_utils.hpp"

using Clock = std::chrono::high_resolution_clock;

void make_initial(double *grid, int nx, int ny)
{
    const double pi = std::acos(-1.0);

    for (int i = 0; i < nx; ++i)
    {
        const double x = std::sin(2.0 * pi * (double)i / (double)(nx - 1));
        for (int j = 0; j < ny; ++j)
        {
            const double y = std::sin(2.0 * pi * (double)j / (double)(ny - 1));
            grid[i * ny + j] = x * y;
        }
    }
}

void laplacian(const double *u, double *out, int nx, int ny)
{
#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 1; i < nx - 1; ++i)
    {
        for (int j = 1; j < ny - 1; ++j)
        {
            const int center = i * ny + j;
            out[center] =
                u[(i - 1) * ny + j] + u[(i + 1) * ny + j] + u[i * ny + (j - 1)] + u[i * ny + (j + 1)] - 4.0 * u[center];
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cerr << "Usage: laplacian_openmp <nx> <ny> <warmups> <repeats> <output_path>\n";
        return 1;
    }

    const int nx = std::atoi(argv[1]);
    const int ny = std::atoi(argv[2]);
    const int warmups = std::atoi(argv[3]);
    const int repeats = std::atoi(argv[4]);

    if (nx < 2 || ny < 2 || warmups < 0 || repeats < 1)
    {
        std::cerr << "Expected nx >= 2, ny >= 2, warmups >= 0, repeats >= 1\n";
        return 1;
    }

    double *u = (double *)std::malloc((std::size_t)nx * ny * sizeof(double));
    double *out = (double *)std::malloc((std::size_t)nx * ny * sizeof(double));
    double *timings = (double *)std::malloc((std::size_t)repeats * sizeof(double));

    if (u == nullptr || out == nullptr || timings == nullptr)
    {
        std::cerr << "Failed to allocate buffers\n";
        return 1;
    }

    make_initial(u, nx, ny);
    for (int k = 0; k < nx * ny; ++k)
    {
        out[k] = 0.0;
    }

    for (int run = 0; run < warmups; ++run)
    {
        laplacian(u, out, nx, ny);
    }

    for (int run = 0; run < repeats; ++run)
    {
        const auto start = Clock::now();
        laplacian(u, out, nx, ny);
        const auto end = Clock::now();

        timings[run] = std::chrono::duration<double, std::milli>(end - start).count();
    }

    if (!benchmark::write_array(argv[5], out, (std::size_t)nx * ny))
    {
        std::cerr << "Failed to write output array\n";
        return 1;
    }

    std::cout << std::fixed << std::setprecision(12)
              << "NX=" << nx << "\n"
              << "NY=" << ny << "\n"
              << "RUNTIME_MS=" << benchmark::median(timings, repeats) << "\n";

    std::free(u);
    std::free(out);
    std::free(timings);

    return 0;
}
