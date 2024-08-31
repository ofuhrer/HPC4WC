#pragma once

#include "kernels.cuh"


namespace device {

// update_boundaries<T>():
// Enforces periodic boundary conditions in x and y.
//
// Input:   stream              :: CUDA stream used
//          u                   :: Input field (located on the device)
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax          :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u                   :: Output field (located on the device)
template<typename T>
void update_boundaries(cudaStream_t &stream, T *u,
                       std::size_t xmin, std::size_t xmax,
                       std::size_t ymin, std::size_t ymax,
                       std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    // Kernel pointers:
    const void *update_south_kernel = reinterpret_cast<void*>(kernels::update_south<T>);
    const void *update_north_kernel = reinterpret_cast<void*>(kernels::update_north<T>);
    const void *update_west_kernel = reinterpret_cast<void*>(kernels::update_west<T>);
    const void *update_east_kernel = reinterpret_cast<void*>(kernels::update_east<T>);

    // Block dimensions:
    const dim3 block_dim_south(std::min(xmax - xmin, static_cast<std::size_t>(256)), 1, 1);
    const dim3 block_dim_north(std::min(xmax - xmin, static_cast<std::size_t>(256)), 1, 1);
    const dim3 block_dim_west(xmin, std::max(256 / xmin, static_cast<std::size_t>(1)), 1);
    const dim3 block_dim_east(xmin, std::max(256 / xmin, static_cast<std::size_t>(1)), 1);

    // Grid dimensions:
    const dim3 grid_dim_south((xmax - xmin + (block_dim_south.x - 1)) / block_dim_south.x,
                              (ymin + (block_dim_south.y - 1)) / block_dim_south.y,
                              (zsize + (block_dim_south.z - 1)) / block_dim_south.z);
    const dim3 grid_dim_north((xmax - xmin + (block_dim_north.x - 1)) / block_dim_north.x,
                              (ysize - ymax + (block_dim_north.y - 1)) / block_dim_north.y,
                              (zsize + (block_dim_north.z - 1)) / block_dim_north.z);
    const dim3 grid_dim_west((xmin + (block_dim_west.x - 1)) / block_dim_west.x,
                             (ysize + (block_dim_west.y - 1)) / block_dim_west.y,
                             (zsize + (block_dim_west.z - 1)) / block_dim_west.z);
    const dim3 grid_dim_east((xsize - xmax + (block_dim_east.x - 1)) / block_dim_east.x,
                             (ysize + (block_dim_east.y - 1)) / block_dim_east.y,
                             (zsize + (block_dim_east.z - 1)) / block_dim_east.z);

    // Additional kernel arguments:
    std::size_t xint = xmax - xmin;
    std::size_t yint = ymax - ymin;

    // Kernel argument arrays:
    void *update_south_args[] = {&u, &xmin, &xmax, &ymin, &yint, &xsize, &ysize, &zsize};
    void *update_north_args[] = {&u, &xmin, &xmax, &ymax, &yint, &xsize, &ysize, &zsize};
    void *update_west_args[] = {&u, &xmin, &xint, &xsize, &ysize, &zsize};
    void *update_east_args[] = {&u, &xmax, &xint, &xsize, &ysize, &zsize};

    // Kernel launches:
    check(cudaLaunchKernel(update_south_kernel, grid_dim_south, block_dim_south, update_south_args, 0, stream));
    check(cudaLaunchKernel(update_north_kernel, grid_dim_north, block_dim_north, update_north_args, 0, stream));
    check(cudaLaunchKernel(update_west_kernel, grid_dim_west, block_dim_west, update_west_args, 0, stream));
    check(cudaLaunchKernel(update_east_kernel, grid_dim_east, block_dim_east, update_east_args, 0, stream));
}


// update_interior_double_laplacian<T>():
// Performs the fourth-order diffusion update in the interior of the domain using two consecutive 5-point Laplacian stencils and no shared memory.
//
// Input:   stream              :: CUDA stream used
//          u                   :: Input field (located on the device)
//          v                   :: Temporary field to store intermediate results in (located on the device)
//          alpha               :: Multiplier in the explicit Euler update
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax          :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u                   :: Output field (located on the device)
template<typename T>
void update_interior_double_laplacian(cudaStream_t &stream, T *u, T *v, T alpha, std::size_t xmin,
                                      std::size_t xmax, std::size_t ymin, std::size_t ymax,
                                      std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    // Kernel pointers:
    const void *laplacian_kernel = reinterpret_cast<void*>(kernels::laplacian<T>);
    const void *laplacian_update_kernel = reinterpret_cast<void*>(kernels::laplacian_update<T>);

    // Block dimensions:
    constexpr dim3 block_dim(16, 16, 1);

    // Grid dimensions:
    const dim3 grid_dim_lap((xmax - xmin + 2 + (block_dim.x - 1)) / block_dim.x,
                            (ymax - ymin + 2 + (block_dim.y - 1)) / block_dim.y,
                            (zsize + (block_dim.z - 1)) / block_dim.z);
    const dim3 grid_dim_int((xmax - xmin + (block_dim.x - 1)) / block_dim.x,
                            (ymax - ymin + (block_dim.y - 1)) / block_dim.y,
                            (zsize + (block_dim.z - 1)) / block_dim.z);

    // Additional kernel arguments:
    std::size_t xmin_lap = xmin - 1;
    std::size_t xmax_lap = xmax + 1;
    std::size_t ymin_lap = ymin - 1;
    std::size_t ymax_lap = ymax + 1;

    // Kernel argument arrays:
    void *laplacian_args[] = {&u, &v, &xmin_lap, &xmax_lap, &ymin_lap, &ymax_lap, &xsize, &ysize, &zsize};
    void *laplacian_update_args[] = {&u, &v, &alpha, &xmin, &xmax, &ymin, &ymax, &xsize, &ysize, &zsize};

    // Kernel launches:
    check(cudaLaunchKernel(laplacian_kernel, grid_dim_lap, block_dim, laplacian_args, 0, stream));
    check(cudaLaunchKernel(laplacian_update_kernel, grid_dim_int, block_dim, laplacian_update_args, 0, stream));
}


// update_interior_double_laplacian_shared<T>():
// Performs the fourth-order diffusion update in the interior of the domain using two consecutive 5-point Laplacian stencils and shared memory.
//
// Input:   stream              :: CUDA stream used
//          u                   :: Input field (located on the device)
//          v                   :: Temporary field to store intermediate results in (located on the device)
//          alpha               :: Multiplier in the explicit Euler update
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax          :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u                   :: Output field (located on the device)
template<typename T>
void update_interior_double_laplacian_shared(cudaStream_t &stream, T *u, T *v, T alpha, std::size_t xmin,
                                             std::size_t xmax, std::size_t ymin, std::size_t ymax,
                                             std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    // Kernel pointers:
    const void *laplacian_shared_kernel = reinterpret_cast<void*>(kernels::laplacian_shared<T>);
    const void *laplacian_shared_update_kernel = reinterpret_cast<void*>(kernels::laplacian_shared_update<T>);

    // Block dimensions:
    constexpr dim3 block_dim(16, 16, 1);

    // Grid dimensions:
    const dim3 grid_dim_lap((xmax - xmin + 2 + (block_dim.x - 1)) / block_dim.x,
                            (ymax - ymin + 2 + (block_dim.y - 1)) / block_dim.y,
                            (zsize + (block_dim.z - 1)) / block_dim.z);
    const dim3 grid_dim_int((xmax - xmin + (block_dim.x - 1)) / block_dim.x,
                            (ymax - ymin + (block_dim.y - 1)) / block_dim.y,
                            (zsize + (block_dim.z - 1)) / block_dim.z);

    // Additional kernel arguments:
    std::size_t xmin_lap = xmin - 1;
    std::size_t xmax_lap = xmax + 1;
    std::size_t ymin_lap = ymin - 1;
    std::size_t ymax_lap = ymax + 1;

    // Kernel argument arrays:
    void *laplacian_shared_args[] = {&u, &v, &xmin_lap, &xmax_lap, &ymin_lap, &ymax_lap, &xsize, &ysize, &zsize};
    void *laplacian_shared_update_args[] = {&u, &v, &alpha, &xmin, &xmax, &ymin, &ymax, &xsize, &ysize, &zsize};

    // Shared memory size:
    constexpr std::size_t shared_size = (block_dim.x + 2) * (block_dim.y + 2) * sizeof(T);

    // Kernel launches:
    check(cudaLaunchKernel(laplacian_shared_kernel, grid_dim_lap, block_dim, laplacian_shared_args, shared_size, stream));
    check(cudaLaunchKernel(laplacian_shared_update_kernel, grid_dim_int, block_dim, laplacian_shared_update_args, shared_size, stream));
}


// update_interior_biharmonic<T>():
// Performs the fourth-order diffusion update in the interior of the domain using a single 13-point biharmonic stencil and no shared memory.
//
// Input:   stream              :: CUDA stream used
//          u                   :: Input field (located on the device)
//          v                   :: Temporary field to store intermediate results in (located on the device)
//          alpha               :: Multiplier in the explicit Euler update
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax          :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u                   :: Output field (located on the device)
template<typename T>
void update_interior_biharmonic(cudaStream_t &stream, T *u, T *v, T alpha, std::size_t xmin,
                                std::size_t xmax, std::size_t ymin, std::size_t ymax,
                                std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    // Kernel pointers:
    const void *biharmonic_operator_kernel = reinterpret_cast<void*>(kernels::biharmonic_operator<T>);
    const void *update_interior_kernel = reinterpret_cast<void*>(kernels::update_interior<T>);

    // Block dimensions:
    constexpr dim3 block_dim(16, 16, 1);

    // Grid dimensions:
    const dim3 grid_dim((xmax - xmin + (block_dim.x - 1)) / block_dim.x,
                        (ymax - ymin + (block_dim.y - 1)) / block_dim.y,
                        (zsize + (block_dim.z - 1)) / block_dim.z);

    // Kernel argument arrays:
    void *biharmonic_operator_args[] = {&u, &v, &xmin, &xmax, &ymin, &ymax, &xsize, &ysize, &zsize};
    void *update_interior_args[] = {&u, &v, &alpha, &xmin, &xmax, &ymin, &ymax, &xsize, &ysize, &zsize};

    // Kernel launches:
    check(cudaLaunchKernel(biharmonic_operator_kernel, grid_dim, block_dim, biharmonic_operator_args, 0, stream));
    check(cudaLaunchKernel(update_interior_kernel, grid_dim, block_dim, update_interior_args, 0, stream));
}


// update_interior_biharmonic_shared<T>():
// Performs the fourth-order diffusion update in the interior of the domain using a single 13-point biharmonic stencil and shared memory.
//
// Input:   stream              :: CUDA stream used
//          u                   :: Input field (located on the device)
//          v                   :: Temporary field to store intermediate results in (located on the device)
//          alpha               :: Multiplier in the explicit Euler update
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax          :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u                   :: Output field (located on the device)
template<typename T>
void update_interior_biharmonic_shared(cudaStream_t &stream, T *u, T *v, T alpha, std::size_t xmin,
                                       std::size_t xmax, std::size_t ymin, std::size_t ymax,
                                       std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    // Kernel pointers:
    const void *biharmonic_operator_shared_kernel = reinterpret_cast<void*>(kernels::biharmonic_operator_shared<T>);
    const void *update_interior_kernel = reinterpret_cast<void*>(kernels::update_interior<T>);

    // Block dimensions:
    constexpr dim3 block_dim(16, 16, 1);

    // Grid dimensions:
    const dim3 grid_dim((xmax - xmin + (block_dim.x - 1)) / block_dim.x,
                        (ymax - ymin + (block_dim.y - 1)) / block_dim.y,
                        (zsize + (block_dim.z - 1)) / block_dim.z);

    // Kernel argument arrays:
    void *biharmonic_operator_args[] = {&u, &v, &xmin, &xmax, &ymin, &ymax, &xsize, &ysize, &zsize};
    void *update_interior_args[] = {&u, &v, &alpha, &xmin, &xmax, &ymin, &ymax, &xsize, &ysize, &zsize};

    // Shared memory size:
    constexpr std::size_t shared_size = (block_dim.x + 4) * (block_dim.y + 4) * sizeof(T);

    // Kernel launches:
    check(cudaLaunchKernel(biharmonic_operator_shared_kernel, grid_dim, block_dim, biharmonic_operator_args, shared_size, stream));
    check(cudaLaunchKernel(update_interior_kernel, grid_dim, block_dim, update_interior_args, 0, stream));
}

} // namespace device
