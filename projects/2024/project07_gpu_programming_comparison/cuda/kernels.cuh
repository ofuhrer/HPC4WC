#pragma once

#include "utils.cuh"


namespace kernels {

// CUDA kernel: update_south<T>():
// Enforces periodic boundary conditions on the south of the domain.
//
// Input:   u                   :: Input field (located on the device)
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin                :: j must be >= ymin to access an interior point (i, j, k)
//          yint                :: Width of the interior of the domain in y direction
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u                   :: Output field (located on the device)
template<typename T>
__global__ void update_south(T *u, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                             std::size_t yint, std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    // Ranges:
    // i in [xmin, xmax[
    // j in [0, ymin[
    // k in [0, zsize[

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if(i < xmax && j < ymin && k < zsize)
        u[index(i, j, k, xsize, ysize)] = u[index(i, j + yint, k, xsize, ysize)];
}


// CUDA kernel: update_north<T>():
// Enforces periodic boundary conditions on the north of the domain.
//
// Input:   u                   :: Input field (located on the device)
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymax                :: j must be < ymax to access an interior point (i, j, k)
//          yint                :: Width of the interior of the domain in y direction
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u                   :: Output field (located on the device)
template<typename T>
__global__ void update_north(T *u, std::size_t xmin, std::size_t xmax, std::size_t ymax,
                             std::size_t yint, std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    // Ranges:
    // i in [xmin, xmax[
    // j in [ymax, ysize[
    // k in [0, zsize[

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymax;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if(i < xmax && j < ysize && k < zsize)
        u[index(i, j, k, xsize, ysize)] = u[index(i, j - yint, k, xsize, ysize)];
}


// CUDA kernel: update_west<T>():
// Enforces periodic boundary conditions on the west of the domain.
//
// Input:   u                   :: Input field (located on the device)
//          xmin                :: i must be >= xmin to access an interior point (i, j, k)
//          xint                :: Width of the interior of the domain in x direction
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u                   :: Output field (located on the device)
template<typename T>
__global__ void update_west(T *u, std::size_t xmin, std::size_t xint,
                            std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    // Ranges:
    // i in [0, xmin[
    // j in [0, ysize[
    // k in [0, zsize[

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if(i < xmin && j < ysize && k < zsize)
        u[index(i, j, k, xsize, ysize)] = u[index(i + xint, j, k, xsize, ysize)];
}


// CUDA kernel: update_east<T>():
// Enforces periodic boundary conditions on the east of the domain.
//
// Input:   u                   :: Input field (located on the device)
//          xmax                :: i must be < xmax to access an interior point (i, j, k)
//          xint                :: Width of the interior of the domain in x direction
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u                   :: Output field (located on the device)
template<typename T>
__global__ void update_east(T *u, std::size_t xmax, std::size_t xint,
                            std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    // Ranges:
    // i in [xmax, xsize[
    // j in [0, ysize[
    // k in [0, zsize[

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmax;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if(i < xsize && j < ysize && k < zsize)
        u[index(i, j, k, xsize, ysize)] = u[index(i - xint, j, k, xsize, ysize)];
}


// CUDA kernel: laplacian<T>():
// Evaluates the 5-point Laplacian stencil without using shared memory.
//
// Input:   u                   :: Input field (located on the device)
//          v                   :: Pre-allocated memory (located on the device)
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax          :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  v                   :: Laplacian of the input field (located on the device)
template<typename T>
__global__ void laplacian(const T *u, T *v, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                          std::size_t ymax, std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymin;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    // Stencil:
    //     1
    // 1  -4   1
    //     1

    if(i < xmax && j < ymax && k < zsize)
        v[index(i, j, k, xsize, ysize)]
            = 1 * (u[index(i + 1, j, k, xsize, ysize)] + u[index(i, j + 1, k, xsize, ysize)]
                +  u[index(i - 1, j, k, xsize, ysize)] + u[index(i, j - 1, k, xsize, ysize)])
            - 4 *  u[index(i, j, k, xsize, ysize)];
}


// CUDA kernel: laplacian_update<T>():
// Performs an explicit Euler update using the 5-point Laplacian stencil without using shared memory.
//
// Input:   u                   :: Input field (located on the device)
//          v                   :: Laplacian of the input field (located on the device)
//          alpha               :: Multiplier in the explicit Euler update
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax          :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u                   :: Input field (updated using explicit Euler)
template<typename T>
__global__ void laplacian_update(T *u, const T *v, T alpha, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                                 std::size_t ymax, std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymin;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    // Stencil:
    //     1
    // 1  -4   1
    //     1

    if(i < xmax && j < ymax && k < zsize)
        u[index(i, j, k, xsize, ysize)] -= alpha * (
              1 * (v[index(i + 1, j, k, xsize, ysize)] + v[index(i, j + 1, k, xsize, ysize)]
                +  v[index(i - 1, j, k, xsize, ysize)] + v[index(i, j - 1, k, xsize, ysize)])
            - 4 *  v[index(i, j, k, xsize, ysize)]);
}


// CUDA kernel: laplacian_shared<T>():
// Evaluates the 5-point Laplacian stencil using shared memory.
//
// Input:   u                   :: Input field (located on the device)
//          v                   :: Pre-allocated memory (located on the device)
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax          :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  v                   :: Laplacian of the input field (located on the device)
template<typename T>
__global__ void laplacian_shared(const T *u, T *v, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                                 std::size_t ymax, std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    extern __shared__ T b[];

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymin;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    const std::size_t li = threadIdx.x + 1;
    const std::size_t lj = threadIdx.y + 1;
    const std::size_t lxsize = blockDim.x + 2;
    const std::size_t lysize = blockDim.y + 2;

    // Stencil:
    //     1
    // 1  -4   1
    //     1

    if(i < xmax && j < ymax && k < zsize) {

        if(li == 1) b[index(li - 1, lj, 0, lxsize, lysize)] = u[index(i - 1, j, k, xsize, ysize)];
        if(lj == 1) b[index(li, lj - 1, 0, lxsize, lysize)] = u[index(i, j - 1, k, xsize, ysize)];
        if(li == lxsize - 2 || i == xmax - 1) b[index(li + 1, lj, 0, lxsize, lysize)] = u[index(i + 1, j, k, xsize, ysize)];
        if(lj == lysize - 2 || j == ymax - 1) b[index(li, lj + 1, 0, lxsize, lysize)] = u[index(i, j + 1, k, xsize, ysize)];

        b[index(li, lj, 0, lxsize, lysize)] = u[index(i, j, k, xsize, ysize)];
    }

    __syncthreads();

    if(i < xmax && j < ymax && k < zsize)
        v[index(i, j, k, xsize, ysize)]
            = 1 * (b[index(li + 1, lj, 0, lxsize, lysize)] + b[index(li, lj + 1, 0, lxsize, lysize)]
                +  b[index(li - 1, lj, 0, lxsize, lysize)] + b[index(li, lj - 1, 0, lxsize, lysize)])
            - 4 *  b[index(li, lj, 0, lxsize, lysize)];
}


// CUDA kernel: laplacian_shared_update<T>():
// Performs an explicit Euler update using the 5-point Laplacian stencil and using shared memory.
//
// Input:   u                   :: Input field (located on the device)
//          v                   :: Laplacian of the input field (located on the device)
//          alpha               :: Multiplier in the explicit Euler update
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax          :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u                   :: Input field (updated using explicit Euler)
template<typename T>
__global__ void laplacian_shared_update(T *u, const T *v, T alpha, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                                        std::size_t ymax, std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    extern __shared__ T b[];

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymin;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    const std::size_t li = threadIdx.x + 1;
    const std::size_t lj = threadIdx.y + 1;
    const std::size_t lxsize = blockDim.x + 2;
    const std::size_t lysize = blockDim.y + 2;

    // Stencil:
    //     1
    // 1  -4   1
    //     1

    if(i < xmax && j < ymax && k < zsize) {

        if(li == 1)                           b[index(li - 1, lj, 0, lxsize, lysize)] = v[index(i - 1, j, k, xsize, ysize)];
        if(lj == 1)                           b[index(li, lj - 1, 0, lxsize, lysize)] = v[index(i, j - 1, k, xsize, ysize)];
        if(li == lxsize - 2 || i == xmax - 1) b[index(li + 1, lj, 0, lxsize, lysize)] = v[index(i + 1, j, k, xsize, ysize)];
        if(lj == lysize - 2 || j == ymax - 1) b[index(li, lj + 1, 0, lxsize, lysize)] = v[index(i, j + 1, k, xsize, ysize)];

        b[index(li, lj, 0, lxsize, lysize)] = v[index(i, j, k, xsize, ysize)];
    }

    __syncthreads();

    if(i < xmax && j < ymax && k < zsize)
        u[index(i, j, k, xsize, ysize)] -= alpha * (
              1 * (b[index(li + 1, lj, 0, lxsize, lysize)] + b[index(li, lj + 1, 0, lxsize, lysize)]
                +  b[index(li - 1, lj, 0, lxsize, lysize)] + b[index(li, lj - 1, 0, lxsize, lysize)])
            - 4 *  b[index(li, lj, 0, lxsize, lysize)]);
}


// CUDA kernel: biharmonic_operator<T>():
// Evaluates the 13-point biharmonic stencil without using shared memory.
//
// Input:   u                   :: Input field (located on the device)
//          v                   :: Pre-allocated memory (located on the device)
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax          :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  v                   :: Biharmonic operator of the input field (located on the device)
template<typename T>
__global__ void biharmonic_operator(const T *u, T *v, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                                    std::size_t ymax, std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymin;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    // Stencil:
    //         1
    //     2  -8   2
    // 1  -8  20  -8   1
    //     2  -8   2
    //         1

    if(i < xmax && j < ymax && k < zsize)
        v[index(i, j, k, xsize, ysize)]
            = 1 * (u[index(i + 2, j, k, xsize, ysize)] + u[index(i, j + 2, k, xsize, ysize)]
                +  u[index(i - 2, j, k, xsize, ysize)] + u[index(i, j - 2, k, xsize, ysize)])
            + 2 * (u[index(i + 1, j + 1, k, xsize, ysize)] + u[index(i + 1, j - 1, k, xsize, ysize)]
                +  u[index(i - 1, j + 1, k, xsize, ysize)] + u[index(i - 1, j - 1, k, xsize, ysize)])
            - 8 * (u[index(i + 1, j, k, xsize, ysize)] + u[index(i, j + 1, k, xsize, ysize)]
                +  u[index(i - 1, j, k, xsize, ysize)] + u[index(i, j - 1, k, xsize, ysize)])
           + 20 *  u[index(i, j, k, xsize, ysize)];
}


// CUDA kernel: biharmonic_operator_shared<T>():
// Evaluates the 13-point biharmonic stencil using shared memory.
//
// Input:   u                   :: Input field (located on the device)
//          v                   :: Pre-allocated memory (located on the device)
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax          :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  v                   :: Biharmonic operator of the input field (located on the device)
template<typename T>
__global__ void biharmonic_operator_shared(const T *u, T *v, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                                           std::size_t ymax, std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    extern __shared__ T b[];

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymin;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    const std::size_t li = threadIdx.x + 2;
    const std::size_t lj = threadIdx.y + 2;
    const std::size_t lxsize = blockDim.x + 4;
    const std::size_t lysize = blockDim.y + 4;

    // Stencil:
    //         1
    //     2  -8   2
    // 1  -8  20  -8   1
    //     2  -8   2
    //         1

    if(i < xmax && j < ymax && k < zsize) {

        if(li == 2 || li == 3)                   b[index(li - 2, lj, 0, lxsize, lysize)] = u[index(i - 2, j, k, xsize, ysize)];
        if(lj == 2 || lj == 3)                   b[index(li, lj - 2, 0, lxsize, lysize)] = u[index(i, j - 2, k, xsize, ysize)];
        if(li == lxsize - 3 || i == xmax - 1)    b[index(li + 1, lj, 0, lxsize, lysize)] = u[index(i + 1, j, k, xsize, ysize)];
        if(lj == lysize - 3 || j == ymax - 1)    b[index(li, lj + 1, 0, lxsize, lysize)] = u[index(i, j + 1, k, xsize, ysize)];
        if(li == lxsize - 4)                     b[index(li + 3, lj, 0, lxsize, lysize)] = u[index(i + 3, j, k, xsize, ysize)];
        if(lj == lysize - 4)                     b[index(li, lj + 3, 0, lxsize, lysize)] = u[index(i, j + 3, k, xsize, ysize)];
        if(li == 3 && lj == 3)                   b[index(li - 2, lj - 2, 0, lxsize, lysize)] = u[index(i - 2, j - 2, k, xsize, ysize)];
        if(li == 3 && lj == lysize - 4)          b[index(li - 2, lj + 2, 0, lxsize, lysize)] = u[index(i - 2, j + 2, k, xsize, ysize)];
        if(li == lxsize - 4 && lj == 3)          b[index(li + 2, lj - 2, 0, lxsize, lysize)] = u[index(i + 2, j - 2, k, xsize, ysize)];
        if(li == lxsize - 4 && lj == lysize - 4) b[index(li + 2, lj + 2, 0, lxsize, lysize)] = u[index(i + 2, j + 2, k, xsize, ysize)];

        b[index(li, lj, 0, lxsize, lysize)] = u[index(i, j, k, xsize, ysize)];
    }

    __syncthreads();

    if(i < xmax && j < ymax && k < zsize)
        v[index(i, j, k, xsize, ysize)] =
              1 * (b[index(li + 2, lj, 0, lxsize, lysize)] + b[index(li, lj + 2, 0, lxsize, lysize)]
                +  b[index(li - 2, lj, 0, lxsize, lysize)] + b[index(li, lj - 2, 0, lxsize, lysize)])
            + 2 * (b[index(li + 1, lj + 1, 0, lxsize, lysize)] + b[index(li + 1, lj - 1, 0, lxsize, lysize)]
                +  b[index(li - 1, lj + 1, 0, lxsize, lysize)] + b[index(li - 1, lj - 1, 0, lxsize, lysize)])
            - 8 * (b[index(li + 1, lj, 0, lxsize, lysize)] + b[index(li, lj + 1, 0, lxsize, lysize)]
                +  b[index(li - 1, lj, 0, lxsize, lysize)] + b[index(li, lj - 1, 0, lxsize, lysize)])
           + 20 *  b[index(li, lj, 0, lxsize, lysize)];
}


// CUDA kernel: update_interior<T>():
// Performs an explicit Euler update.
//
// Input:   u                   :: Input field (located on the device)
//          v                   :: Biharmonic operator of the input field (located on the device)
//          alpha               :: Multiplier in the explicit Euler update
//          xmin, xmax          :: i must be in [xmin, xmax[ to access an interior point (i, j, k)
//          ymin, ymax          :: j must be in [ymin, ymax[ to access an interior point (i, j, k)
//          xsize, ysize, zsize :: Dimensions of the domain (including boundary points)
//          T                   :: Numeric real type
// Output:  u                   :: Input field (updated using explicit Euler)
template<typename T>
__global__ void update_interior(T *u, const T *v, T alpha, std::size_t xmin, std::size_t xmax, std::size_t ymin,
                                std::size_t ymax, std::size_t xsize, std::size_t ysize, std::size_t zsize) {

    const std::size_t i = blockDim.x * blockIdx.x + threadIdx.x + xmin;
    const std::size_t j = blockDim.y * blockIdx.y + threadIdx.y + ymin;
    const std::size_t k = blockDim.z * blockIdx.z + threadIdx.z;

    if(i < xmax && j < ymax && k < zsize)
        u[index(i, j, k, xsize, ysize)] -= alpha * v[index(i, j, k, xsize, ysize)];
}

} // namespace kernels
