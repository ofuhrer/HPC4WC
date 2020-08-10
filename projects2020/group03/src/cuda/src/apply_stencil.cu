#include <iostream>
#include <cuda.h>
#include "utils.h"
#include "apply_stencil.cuh"

__global__
void apply_stencil(realType const *infield,
                   realType *outfield,
                   int const xMin,
                   int const xMax,
                   int const xSize,
                   int const yMin,
                   int const yMax,
                   int const ySize,
                   int const zMax,
                   realType const alpha) {
  // shared memory buffers
  __shared__ realType buffer1[10][10][1];
  __shared__ realType buffer2[10][10][1];
  // local 3D indexes for buffer1/buffer2
  int const li = threadIdx.x + 1;
  int const lj = threadIdx.y + 1;
  int const lk = threadIdx.z;
  // global 3D indexes for infield/outfield
  int const i = blockDim.x * blockIdx.x + li;
  int const j = blockDim.y * blockIdx.y + lj;
  int const k = blockDim.z * blockIdx.z + lk;
  // global 1D index for infield/outfield
  int const index = i + j * xSize + k * xSize * ySize;
  // xInterior/yInterior
  int const xInterior = xMax - xMin;
  int const yInterior = yMax - yMin;

  // utils (Edges)
  bool const south = (j == yMin - 1 && i >= xMin && i < xMax  && k < zMax);
  bool const north = (j == yMax     && i >= xMin && i < xMax  && k < zMax);
  bool const west  = (i == xMin - 1 && j >= yMin && j < yMax  && k < zMax);
  bool const east  = (i == xMax     && j >= yMin && j < yMax  && k < zMax);
  bool const inner = (i >= xMin && i < xMax && j >= yMin && j < yMax && k < zMax);

  // initialize shared memory to zero
  buffer1[li][lj][lk] = 0.0;
  buffer2[li][lj][lk] = 0.0;
  // east and west boundary initialization
  if (threadIdx.x == 0) {
    buffer1[li-1][lj][lk] = 0.0;
    buffer2[li-1][lj][lk] = 0.0;
  } else if (threadIdx.x == blockDim.x - 1) {
    buffer1[li+1][lj][lk] = 0.0;
    buffer2[li+1][lj][lk] = 0.0;
  } else { /* pass */ }
  // south and north boundary initialization
  if (threadIdx.y == 0) {
    buffer1[li][lj-1][lk] = 0.0;
    buffer2[li][lj-1][lk] = 0.0;
  } else if (threadIdx.y == blockDim.y - 1) {
    buffer1[li][lj+1][lk] = 0.0;
    buffer2[li][lj+1][lk] = 0.0;
  } else { /* pass */}

  // fill buffer1 based on halo update
  if (south) {
    buffer1[li][lj][lk] = infield[index + yInterior * xSize];
    buffer1[li][lj-1][lk] = infield[index + (yInterior - 1) * xSize];
  } else if (north) {
    buffer1[li][lj][lk] = infield[index - yInterior * xSize];
    buffer1[li][lj+1][lk] = infield[index - (yInterior + 1) * xSize];
  } else if (west)  {
    buffer1[li][lj][lk] = infield[index + xInterior];
    buffer1[li-1][lj][lk] = infield[index + xInterior - 1];
  } else if (east)  {
    buffer1[li][lj][lk] = infield[index - xInterior];
    buffer1[li+1][lj][lk] = infield[index - xInterior + 1];
  } else if (inner) {
    buffer1[li][lj][lk] = infield[index];
    // if left-most or right-most load border values from neighbor domain
    if (threadIdx.x == 0) {
      buffer1[li-1][lj][lk] = infield[index - 1];
    } else if (threadIdx.x == blockDim.x - 1) {
      buffer1[li+1][lj][lk] = infield[index + 1];
    } else { /*pass*/ }
    // if upper-most or lower-most load border values from neighbor domain
    if (threadIdx.y == 0) {
      buffer1[li][lj-1][lk] = infield[index - xSize];
    } else if (threadIdx.y == blockDim.y - 1) {
      buffer1[li][lj+1][lk] = infield[index + xSize];
    } else { /*pass*/ }
  }
  __syncthreads();

  // apply the initial laplacian
  if (i >= xMin - 1 && i < xMax + 1 &&
      j >= yMin - 1 && j < yMax + 1 && k < zMax) {
    realType const value = -4.0 * buffer1[li][lj][lk]
                              + buffer1[li-1][lj][lk]
                              + buffer1[li+1][lj][lk]
                              + buffer1[li][lj-1][lk]
                              + buffer1[li][lj+1][lk];
    buffer2[li][lj][lk] = value;
  }
  __syncthreads();

  // apply the second laplacian
  if (i >= xMin && i < xMax &&
      j >= yMin && j < yMax && k < zMax) {
    realType const value = -4.0 * buffer2[li][lj][lk]
                              + buffer2[li-1][lj][lk]
                              + buffer2[li+1][lj][lk]
                              + buffer2[li][lj-1][lk]
                              + buffer2[li][lj+1][lk];
    outfield[index] = buffer1[li][lj][lk] - alpha * value;
  }
}
