#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>

#include "../cpp_utils.hpp"

using Clock = std::chrono::steady_clock;

void make_initial(double *initial, int nx, int ny)
{
    for (int i = 0; i < nx; ++i)
    {
        for (int j = 0; j < ny; ++j)
        {
            const double delta = 1.0 / (4.0 * (1.0 + std::exp(-4.0 * i / (nx - 1) + 4.0 * j / (nx - 1)) / 3.2));
            initial[i * ny + j] = 0.75 - delta;
            initial[nx * ny + i * ny + j] = 0.75 + delta;
        }
    }
}

void advection_diffusion(const double *initial, double *out, int nx, int ny)
{
    // Fixed positive input velocities imply abs(u) = u and abs(v) = v.
    const double dx = 1.0 / (nx - 1);
    const double dt = dx * dx;

#pragma omp parallel for collapse(2) schedule(static)
    for (int i = 3; i < nx - 3; ++i)
    {
        for (int j = 3; j < ny - 3; ++j)
        {
            const int k = i * ny + j;
            const double u = initial[k], v = initial[nx * ny + k];
            for (int component = 0; component < 2; ++component)
            {
                const double *f = initial + component * nx * ny;
                const double c = f[k];
                const double adv_x = u / (60.0 * dx) * (45.0 * (f[k + ny] - f[k - ny]) - 9.0 * (f[k + 2 * ny] - f[k - 2 * ny]) + (f[k + 3 * ny] - f[k - 3 * ny])) - u / (60.0 * dx) * (f[k + 3 * ny] + f[k - 3 * ny] - 6.0 * (f[k + 2 * ny] + f[k - 2 * ny]) + 15.0 * (f[k + ny] + f[k - ny]) - 20.0 * c);
                const double adv_y = v / (60.0 * dx) * (45.0 * (f[k + 1] - f[k - 1]) - 9.0 * (f[k + 2] - f[k - 2]) + (f[k + 3] - f[k - 3])) - v / (60.0 * dx) * (f[k + 3] + f[k - 3] - 6.0 * (f[k + 2] + f[k - 2]) + 15.0 * (f[k + 1] + f[k - 1]) - 20.0 * c);
                const double diff_x = (-f[k - 2 * ny] + 16.0 * f[k - ny] - 30.0 * c + 16.0 * f[k + ny] - f[k + 2 * ny]) / (12.0 * dx * dx);
                const double diff_y = (-f[k - 2] + 16.0 * f[k - 1] - 30.0 * c + 16.0 * f[k + 1] - f[k + 2]) / (12.0 * dx * dx);
                out[component * nx * ny + k] = c + dt * (-(adv_x + adv_y) + 0.1 * (diff_x + diff_y));
            }
        }
    }
}

int main(int argc, char *argv[])
{
    if (argc != 6)
    {
        std::cerr << "Usage: advection_diffusion <nx> <ny> <warmups> <repeats> <output_path>\n";
        return 1;
    }

    const int nx = std::atoi(argv[1]);
    const int ny = std::atoi(argv[2]);
    const int warmups = std::atoi(argv[3]);
    const int repeats = std::atoi(argv[4]);

    if (nx < 7 || ny < 7 || warmups < 0 || repeats < 1)
    {
        std::cerr << "Expected nx >= 7, ny >= 7, warmups >= 0, repeats >= 1\n";
        return 1;
    }

    double *initial = (double *)std::malloc(2 * (std::size_t)nx * ny * sizeof(double));
    double *out = (double *)std::malloc(2 * (std::size_t)nx * ny * sizeof(double));
    double *timings = (double *)std::malloc((std::size_t)repeats * sizeof(double));

    if (!initial || !out || !timings)
    {
        std::cerr << "Failed to allocate buffers\n";
        std::free(initial);
        std::free(out);
        std::free(timings);
        return 1;
    }

    make_initial(initial, nx, ny);
    std::copy(initial, initial + 2 * (std::size_t)nx * ny, out);

    for (int run = 0; run < warmups; ++run)
    {
        advection_diffusion(initial, out, nx, ny);
    }

    for (int run = 0; run < repeats; ++run)
    {
        const auto start = Clock::now();
        advection_diffusion(initial, out, nx, ny);
        timings[run] = std::chrono::duration<double, std::milli>(Clock::now() - start).count();
    }

    const bool written = benchmark::write_array(argv[5], out, 2 * (std::size_t)nx * ny);

    if (written)
    {
        std::cout << std::fixed << std::setprecision(12)
                  << "NX=" << nx << "\n"
                  << "NY=" << ny << "\n"
                  << "RUNTIME_MS=" << benchmark::median(timings, repeats) << "\n";
    }
    else
    {
        std::cerr << "Failed to write output array\n";
    }

    std::free(initial);
    std::free(out);
    std::free(timings);

    return written ? 0 : 1;
}
