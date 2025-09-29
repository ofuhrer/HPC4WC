#include "stencil_kernels.cuh"

__global__ void updateHaloKernel(double* field, int xsize, int ysize, int zsize, int halo) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Calculate total number of halo points per z-level
    int xInterior = xsize - 2 * halo;
    int yInterior = ysize - 2 * halo;
    
    // Halo points per z-level:
    // - Top/bottom edges: 2 * xInterior * halo
    // - Left/right edges: 2 * ysize * halo (includes corners)
    int haloPointsPerZ = 2 * xInterior * halo + 2 * ysize * halo;
    int totalHaloPoints = haloPointsPerZ * zsize;
    
    if (idx >= totalHaloPoints) return;
    
    // Determine which z-level this thread handles
    int k = idx / haloPointsPerZ;
    int localIdx = idx % haloPointsPerZ;
    
    // Base offset for this z-level
    int zOffset = k * xsize * ysize;
    
    if (localIdx < 2 * xInterior * halo) {
        // Handle top/bottom edges
        int edgeIdx = localIdx;
        int isBottom = (edgeIdx < xInterior * halo) ? 1 : 0;
        int pos = edgeIdx % (xInterior * halo);
        int row = pos / xInterior;
        int col = pos % xInterior;
        
        int srcI = halo + col;
        int srcJ = isBottom ? (row + yInterior) : (halo + row);
        int dstJ = isBottom ? row : (row + halo + yInterior);
        
        int srcIdx = zOffset + srcI + srcJ * xsize;
        int dstIdx = zOffset + srcI + dstJ * xsize;
        
        field[dstIdx] = field[srcIdx];
    } 
    else {
        // Handle left/right edges (including corners)
        int edgeIdx = localIdx - 2 * xInterior * halo;
        int isLeft = (edgeIdx < ysize * halo) ? 1 : 0;
        int pos = edgeIdx % (ysize * halo);
        int col = pos / ysize;
        int row = pos % ysize;

        int isCorner = row < halo ? 1 : 0;
        isCorner = row >= ysize - halo ? -1 : isCorner;
        
        int srcJ = row + isCorner * (yInterior);
        int srcI = isLeft ? col + xInterior : (halo + col);
        int dstI = isLeft ? col : (col + halo + xInterior);
        
        int srcIdx = zOffset + srcI + srcJ * xsize;
        int dstIdx = zOffset + dstI + row * xsize;
        
        field[dstIdx] = field[srcIdx];
    }
}

__global__ void diffusionStepKernel(double* inField, double* outField, double* tmp1Field,
                                   int xsize, int ysize, int zsize, int k_level, int halo, double alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Add halo offset
    i += halo;
    j += halo;
    
    if (i >= xsize - halo || j >= ysize - halo) return;
    
    int idx = i + j * xsize + k_level * xsize * ysize;
    
    // First Laplacian: inField -> tmp1Field
    tmp1Field[idx] = 20. * inField[idx] - 
                     8. * inField[idx - 1] - 8. * inField[idx + 1] - 8. * 
                     inField[idx - xsize] - 8. * inField[idx + xsize] +
                     2. * inField[idx - xsize - 1] + 2. * inField[idx + xsize + 1] +
                     2. * inField[idx - xsize + 1] + 2. * inField[idx + xsize - 1] +
                     inField[idx - 2 * xsize] + inField[idx + 2 * xsize] + inField[idx - 2] + inField[idx + 2];
    
    // Apply diffusion step: out = in - alpha * laplap
    outField[idx] = inField[idx] - alpha * tmp1Field[idx];
}