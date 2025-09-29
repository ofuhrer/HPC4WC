#pragma once
#include <cuda_runtime.h>

// CUDA kernel that updates halo regions for periodic boundary conditions in a 3D field
// PRE: field is allocated on the device with size xsize * ysize * zsize
//      xsize, ysize, zsize are the dimensions of the field including halo regions
//      halo is the width of the halo region
// POST: Halo regions are updated with periodic boundary conditions:
//       - Top/bottom edges copy from bottom/top interior regions respectively
//       - Left/right edges copy from right/left interior regions respectively
//       - Corner regions are handled appropriately to maintain periodicity
__global__ void updateHaloKernel(double* field, int xsize, int ysize, int zsize, int halo);


// CUDA kernel that performs one diffusion step using a 13-point stencil on a single z-level
// This is the version of the diffusion step that combines the two laplace operations into one kernel
// to reduce memory bandwidth usage and improve performance.
// PRE: inField, outField, and tmp1Field are allocated on the device with size xsize * ysize * zsize
//      xsize, ysize, zsize are the dimensions including halo regions
//      k_level is the z-index of the level to process (0 <= k_level < zsize)
//      halo is the width of the halo region (must be >= 2 for the 13-point stencil)
//      alpha is the diffusion coefficient
// POST: outField[k_level] contains the result of one diffusion step: out = in - alpha * laplacian(in)
//       tmp1Field[k_level] contains the intermediate Laplacian result
//       The 13-point stencil includes nearest neighbors, diagonal neighbors, and next-nearest neighbors
__global__ void diffusionStepKernel(double* inField, double* outField, double* tmp1Field,
                                    int xsize, int ysize, int zsize, int k_level, int halo, double alpha);
