#include "device.cuh"
#include "host.cuh"

#include <chrono>
#include <fstream>
#include <sstream>

#ifndef CUDART_VERSION
#error "CUDART_VERSION not defined!"
#endif


using time_point = std::chrono::time_point<std::chrono::steady_clock>;


// run_simulation<T>():
// Runs the 4th-order diffusion simulation on a CUDA-enabled GPU and writes its results to some output files.
//
// Input:   xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          itrs                :: Number of timestep iterations
//          bdry                :: Number of boundary points
//          mode                :: Computation mode (double-5/13-point stencil(s) with/without shared memory)
//          T                   :: Numeric real type
// Output:  return (...)        :: Measured time (memory transfer + device allocation + computation) in seconds
template<typename T>
double run_simulation(std::size_t xsize, std::size_t ysize, std::size_t zsize, std::size_t itrs, std::size_t bdry, Mode mode) {

    constexpr T alpha = static_cast<T>(1) / 32;
    const std::size_t xmin = bdry, xmax = xsize - bdry;
    const std::size_t ymin = bdry, ymax = ysize - bdry;

    cudaStream_t stream;
    T *u, *v, *u_host;
    std::ofstream os;

    check(cudaMallocHost(&u_host, xsize * ysize * zsize * sizeof(T)));
    host::initialise(u_host, xsize, ysize, zsize);

    os.open("in_field.csv");
    host::write_file(os, u_host, xsize, ysize, zsize);
    os.close();

    const time_point begin = std::chrono::steady_clock::now();
    check(cudaStreamCreate(&stream));

    #if CUDART_VERSION >= 11020
    // Requires at least CUDA v11.2 to be installed.
    check(cudaMallocAsync(&u, xsize * ysize * zsize * sizeof(T), stream));
    check(cudaMallocAsync(&v, xsize * ysize * zsize * sizeof(T), stream));
    #else
    check(cudaMalloc(&u, xsize * ysize * zsize * sizeof(T)));
    check(cudaMalloc(&v, xsize * ysize * zsize * sizeof(T)));
    #endif

    check(cudaMemcpyAsync(u, u_host, xsize * ysize * zsize * sizeof(T), cudaMemcpyHostToDevice, stream));

    switch(mode) {
        case Mode::laplap_global: {
            for(std::size_t i = 0; i < itrs; ++i) {
                device::update_boundaries(stream, u, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                device::update_interior_double_laplacian(stream, u, v, alpha, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
            }
            break;
        }
        case Mode::laplap_shared: {
            for(std::size_t i = 0; i < itrs; ++i) {
                device::update_boundaries(stream, u, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                device::update_interior_double_laplacian_shared(stream, u, v, alpha, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
            }
            break;
        }
        case Mode::biharm_global: {
            for(std::size_t i = 0; i < itrs; ++i) {
                device::update_boundaries(stream, u, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                device::update_interior_biharmonic(stream, u, v, alpha, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
            }
            break;
        }
        case Mode::biharm_shared: {
            for(std::size_t i = 0; i < itrs; ++i) {
                device::update_boundaries(stream, u, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
                device::update_interior_biharmonic_shared(stream, u, v, alpha, xmin, xmax, ymin, ymax, xsize, ysize, zsize);
            }
            break;
        }
        default: __builtin_unreachable();
    }
    device::update_boundaries(stream, u, xmin, xmax, ymin, ymax, xsize, ysize, zsize);

    check(cudaMemcpyAsync(u_host, u, xsize * ysize * zsize * sizeof(T), cudaMemcpyDeviceToHost, stream));

    #if CUDART_VERSION >= 11020
    // Requires at least CUDA v11.2 to be installed.
    check(cudaFreeAsync(u, stream));
    check(cudaFreeAsync(v, stream));
    #else
    check(cudaFree(u));
    check(cudaFree(v));
    #endif

    check(cudaStreamDestroy(stream));
    check(cudaDeviceSynchronize());
    const time_point end = std::chrono::steady_clock::now();

    os.open("out_field.csv");
    host::write_file(os, u_host, xsize, ysize, zsize);
    os.close();

    check(cudaFreeHost(u_host));

    return std::chrono::duration<double, std::milli>(end - begin).count() / 1000;
}


// templated_main<T>():
// Main function with flexible numeric real type.
//
// Input:   argv            :: Input arguments
//          argc            :: Number of input arguments
//          T               :: Numeric real type
// Output:  return (...)    :: Exit code (EXIT_SUCCESS or EXIT_FAILURE)
template<typename T>
int templated_main(int argc, char const **argv) {

    if(argc == 7) {
        std::size_t nx, ny, nz, bdry, itrs;
        Mode mode;

        {
            std::istringstream nx_ss(argv[1]), ny_ss(argv[2]), nz_ss(argv[3]), bdry_ss(argv[4]), itrs_ss(argv[5]);
            nx_ss >> nx; ny_ss >> ny; nz_ss >> nz; bdry_ss >> bdry; itrs_ss >> itrs;
            mode = utils::mode_from_string(argv[6]);

            if(nx_ss.fail() || ny_ss.fail() || nz_ss.fail() || itrs_ss.fail() ||
               nx == 0 || ny == 0 || nz == 0 || bdry < 2 || itrs == 0 || mode == Mode::invalid) {

                utils::print_args_errmsg();
                return EXIT_FAILURE;
            }
        }

        std::cout << "================================================================================\n";
        std::cout << "                             Welcome to stencil2d!\n";
        std::cout << "Version    :: C++ with CUDA v" << CUDART_VERSION / 1000 << '.' << CUDART_VERSION / 10 % 100 << '\n';
        std::cout << "Interior   :: (" << nx << ", " << ny << ", " << nz << ")\n";
        std::cout << "Boundaries :: (" << bdry << ", " << bdry << ", " << 0 << ")\n";
        std::cout << "Iterations :: " << itrs << '\n';
        std::cout << "Real size  :: " << sizeof(T) << '\n';
        std::cout << "Exec. mode :: " << utils::get_mode_desc(mode) << '\n';
        std::cout << "================================================================================\n";

        const double time = run_simulation<T>(nx + 2 * bdry, ny + 2 * bdry, nz, itrs, bdry, mode);

        std::cout << "Runtime    :: " << std::fixed << std::setprecision(10) << time << "s\n";
        std::cout << "================================================================================\n";
    }
    else {
        utils::print_args_errmsg();
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}


int main(int argc, char const **argv) {
    #if !defined(REALSIZE) || REALSIZE == 8
    return templated_main<double>(argc, argv);
    #elif REALSIZE == 4
    return templated_main<float>(argc, argv);
    #else
    #error "Selected REALSIZE not supported!"
    #endif
}
