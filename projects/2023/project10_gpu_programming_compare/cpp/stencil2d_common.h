#pragma once

#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

// Numeric precision
typedef float FloatType;

// Constants
constexpr int const_h = 2;
constexpr FloatType const_alpha = 1.0 / 32.0;

// Job description
template <typename FloatType>
struct JobData {
    int x;
    int y;
    int z;
    int h;
    FloatType alpha;
    int iter;
    int b;
    bool noshared;
    FloatType* d_fld;
    FloatType* d_swp;
};

// CUDA checks
void check(cudaError_t error, const char* const file, int const line);
#define CheckErrors(val) check((val), __FILE__, __LINE__)

// Forward declarations
extern "C" void stencil3d_occupancy(dim3 b, dim3 g);
extern "C" void stencil3d_launch(cudaStream_t stream, JobData<FloatType> job, JobData<FloatType>* d_job, bool noshared, bool occ);
void stencil2d_host(int x, int y, int z, int iter, const char* filename);
float stencil2d_cuda(int x, int y, int z, int iter, int b, bool noshared, bool occ, int runs, const char* filename);
