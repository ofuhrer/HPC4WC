#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
//#include <omp.h>
#include <cuda.h>

#ifdef CRAYPAT
#include "pat_api.h"
#endif
#include "utils.h"

namespace {

void updateHalo(Storage3D<double>& inField) {
  const int xInterior = inField.xMax() - inField.xMin();
  const int yInterior = inField.yMax() - inField.yMin();

  // bottom edge (without corners)
  for(std::size_t k = 0; k < inField.zMax(); ++k) {
    for(std::size_t j = 0; j < inField.yMin(); ++j) {
      for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j + yInterior, k);
      }
    }
  }

  // top edge (without corners)
  for(std::size_t k = 0; k < inField.zMax(); ++k) {
    for(std::size_t j = inField.yMax(); j < inField.ySize(); ++j) {
      for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j - yInterior, k);
      }
    }
  }

  // left edge (including corners)
  for(std::size_t k = 0; k < inField.zMax(); ++k) {
    for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
      for(std::size_t i = 0; i < inField.xMin(); ++i) {
        inField(i, j, k) = inField(i + xInterior, j, k);
      }
    }
  }

  // right edge (including corners)
  for(std::size_t k = 0; k < inField.zMax(); ++k) {
    for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
      for(std::size_t i = inField.xMax(); i < inField.xSize(); ++i) {
        inField(i, j, k) = inField(i - xInterior, j, k);
      }
    }
  }
}

__global__
void apply_stencil(double *infield, double *outfield, int xMin, int xMax, int xSize, int yMin, int yMax, int ySize, int zMax, double alpha) {
  // shared memory buffers
  __shared__ double buffer1[10][10][4];
  __shared__ double buffer2[10][10][4];
  // global 3D indexes for infield/outfield
  int i = blockDim.x * blockIdx.x + threadIdx.x + 1;
  int j = blockDim.y * blockIdx.y + threadIdx.y + 1;
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  // local 3D indexes for buffer1/buffer2
  int li = threadIdx.x + 1;
  int lj = threadIdx.y + 1;
  int lk = threadIdx.z;
  // global 1D index for infield/outfield
  int index = i + j * xSize + k * xSize * ySize;
  // xInterior/yInterior
  int xInterior = xMax - xMin;
  int yInterior = yMax - yMin;

  // initialize shared memory
  // east and west boundary initialization
  if (threadIdx.x == 0) {
    buffer1[li-1][lj][lk] = 0.0;
    buffer2[li-1][lj][lk] = 0.0;
  } else if (threadIdx.x == blockDim.x - 1) {
    buffer1[li+1][lj][lk] = 0.0;
    buffer2[li+1][lj][lk] = 0.0;
  } else {
    // pass
  }
  // south and north boundary initialization
  if (threadIdx.y == 0) {
    buffer1[li][lj-1][lk] = 0.0;
    buffer2[li][lj-1][lk] = 0.0;
  } else if (threadIdx.y == blockDim.y - 1) {
    buffer1[li][lj+1][lk] = 0.0;
    buffer2[li][lj+1][lk] = 0.0;
  } else {
    // pass
  }
  __syncthreads();

  bool south = (j >= 0    && j < yMin  && i >= xMin && i < xMax  && k < zMax);
  bool north = (j >= yMax && j < ySize && i >= xMin && i < xMax  && k < zMax);
  bool west  = (i >= 0    && i < xMin  && j >= yMin && j < yMax  && k < zMax);
  bool east  = (i >= xMax && i < xSize && j >= yMin && j < yMax  && k < zMax);
  bool inner = (i < xSize && j < ySize && k < zMax);

  bool sw_corner = (i == xMin - 1 && j == yMin - 1 && k < zMax);
  bool se_corner = (i == xMax     && j == yMin - 1 && k < zMax);
  bool nw_corner = (i == xMin - 1 && j == yMax     && k < zMax);
  bool ne_corner = (i == xMax     && j == yMax     && k < zMax);

  if     (sw_corner) {
    // buffer1
    buffer1[li][lj][lk] = 0.0; // corner
    buffer1[li-1][lj][lk] = 0.0; // left
    buffer1[li][lj-1][lk] = 0.0; // bottom
    buffer1[li-1][lj-1][lk] = 0.0; // other corner
    // buffer2
    buffer2[li][lj][lk] = 0.0; // corner
  } else if(se_corner) {
    // buffer1
    buffer1[li][lj][lk] = 0.0; // corner
    buffer1[li+1][lj][lk] = 0.0; // right
    buffer1[li][lj-1][lk] = 0.0; // bottom
    buffer1[li+1][lj-1][lk] = 0.0; // other corner
    // buffer2
    buffer2[li][lj][lk] = 0.0; // corner
  } else if(nw_corner) {
    // buffer1
    buffer1[li][lj][lk] = 0.0; // corner
    buffer1[li-1][lj][lk] = 0.0; // left
    buffer1[li][lj+1][lk] = 0.0; // top
    buffer1[li-1][lj+1][lk] = 0.0; // other corner
    // buffer2
    buffer2[li][lj][lk] = 0.0; // corner
  } else if(ne_corner) {
    // buffer1
    buffer1[li][lj][lk] = 0.0; // corner
    buffer1[li+1][lj][lk] = 0.0; // right
    buffer1[li][lj+1][lk] = 0.0; // top
    buffer1[li+1][lj+1][lk] = 0.0; // other corner
    // buffer2
    buffer2[li][lj][lk] = 0.0; // corner
  }
  __syncthreads();

  // fill buffer1 considering halo update
  if      (south) {
    buffer1[li][lj][lk] = infield[index + yInterior * xSize];
    buffer1[li][lj-1][lk] = infield[index + (yInterior-1) * xSize];
  } else if (north) {
    buffer1[li][lj][lk] = infield[index - yInterior * xSize];
    buffer1[li][lj+1][lk] = infield[index - (yInterior+1) * xSize];
  } else if (west)  {
    buffer1[li][lj][lk] = infield[index + xInterior];
    buffer1[li-1][lj][lk] = infield[index + xInterior - 1];
  } else if (east)  {
    buffer1[li][lj][lk] = infield[index - xInterior];
    buffer1[li+1][lj][lk] = infield[index - xInterior + 1];
  } else if (inner) {
    buffer1[li][lj][lk] = infield[index];
    // Left-most or right-most
    if      (threadIdx.x == 0)              { buffer1[li-1][lj][lk] = infield[index - 1]; }
    else if (threadIdx.x == blockDim.x - 1) { buffer1[li+1][lj][lk] = infield[index + 1]; }
    else { /*pass*/ }
    // Upper-most or lower-most
    if      (threadIdx.y == 0)              { buffer1[li][lj-1][lk] = infield[index - xSize]; }
    else if (threadIdx.y == blockDim.y - 1) { buffer1[li][lj+1][lk] = infield[index + xSize]; }
    else { /*pass*/ }
  }
  __syncthreads();

  // apply the initial laplacian
  if (i >= xMin - 1 && i < xMax + 1 &&
      j >= yMin - 1 && j < yMax + 1 && k < zMax) {
    double value = -4.0 * buffer1[li][lj][lk]
                        + buffer1[li - 1][lj][lk]
                        + buffer1[li + 1][lj][lk]
                        + buffer1[li][lj - 1][lk]
                        + buffer1[li][lj + 1][lk];
    buffer2[li][lj][lk] = value;
  }
  __syncthreads();

  // apply the second laplacian
  if (i >= xMin && i < xMax &&
      j >= yMin && j < yMax && k < zMax) {
    double value = -4.0 * buffer2[li][lj][lk]
                        + buffer2[li - 1][lj][lk]
                        + buffer2[li + 1][lj][lk]
                        + buffer2[li][lj - 1][lk]
                        + buffer2[li][lj + 1][lk];
    outfield[index] = buffer1[li][lj][lk] - alpha * value;
  }
}

void apply_diffusion_gpu(Storage3D<double>& inField, Storage3D<double>& outField,
                         double alpha, unsigned numIter, int x, int y, int z, int halo) {
  // Utils
  std::size_t const xSize = inField.xSize();
  std::size_t const ySize = inField.ySize();
  std::size_t const xMin = inField.xMin();
  std::size_t const yMin = inField.yMin();
  std::size_t const xMax = inField.xMax();
  std::size_t const yMax = inField.yMax();
  std::size_t const zMin = inField.zMin();
  std::size_t const zMax = inField.zMax();
  std::size_t const size = sizeof(double) * xSize * ySize * zMax;

  // Allocate space on device memory and copy data from host
  double *infield, *outfield;
  //cuInit(0);
  cudaMalloc((void **)&infield, size);
  cudaMalloc((void **)&outfield, size);
  cudaMemcpy(infield, &inField(0, 0, 0), size, cudaMemcpyHostToDevice);

  dim3 blockDim(8, 8, 4);
  dim3 gridDim((xMax + blockDim.x - 1) / blockDim.x,
               (yMax + blockDim.y - 1) / blockDim.y,
               (zMax + blockDim.z - 1) / blockDim.z);

  cudaEvent_t tic, toc;
  cudaEventCreate(&tic);
  cudaEventCreate(&toc);
  cudaEventRecord(tic);

  for(std::size_t iter = 0; iter < numIter; ++iter) {
    apply_stencil<<<gridDim, blockDim>>>(infield, outfield, xMin, xMax, xSize, yMin, yMax, ySize, zMax, alpha);
    cudaDeviceSynchronize();
    if ( iter != numIter - 1 ) std::swap(infield, outfield);
  }

  cudaEventRecord(toc);
  cudaEventSynchronize(toc);
  float telapsed = -1;
  cudaEventElapsedTime(&telapsed, tic, toc);
  std::cout << "telapsed: " << telapsed << std::endl;
  cudaEventDestroy(tic);
  cudaEventDestroy(toc);

  // Copy result from device to host and free device memory
  cudaMemcpy(&outField(0, 0, 0), outfield, size, cudaMemcpyDeviceToHost);
  cudaFree(infield);
  cudaFree(outfield);
}


void apply_diffusion(Storage3D<double>& inField, Storage3D<double>& outField, double alpha,
                     unsigned numIter, int x, int y, int z, int halo) {

  Storage3D<double> tmp1Field(x, y, z, halo);

  for(std::size_t iter = 0; iter < numIter; ++iter) {

    updateHalo(inField);

    for(std::size_t k = 0; k < inField.zMax(); ++k) {

      // apply the initial laplacian
      for(std::size_t j = inField.yMin() - 1; j < inField.yMax() + 1; ++j) {
        for(std::size_t i = inField.xMin() - 1; i < inField.xMax() + 1; ++i) {
          tmp1Field(i, j, 0) = -4.0 * inField(i, j, k) + inField(i - 1, j, k) +
                               inField(i + 1, j, k) + inField(i, j - 1, k) + inField(i, j + 1, k);
        }
      }

      // apply the second laplacian
      for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
        for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
          double laplap = -4.0 * tmp1Field(i, j, 0) + tmp1Field(i - 1, j, 0) +
                          tmp1Field(i + 1, j, 0) + tmp1Field(i, j - 1, 0) + tmp1Field(i, j + 1, 0);

          // and update the field
          if(iter == numIter - 1) {
            outField(i, j, k) = inField(i, j, k) - alpha * laplap;
          } else {
            inField(i, j, k) = inField(i, j, k) - alpha * laplap;
          }
        }
      }
    }
  }
}


void reportTime(const Storage3D<double>& storage, int nIter, double diff) {
  std::cout << "# ranks nx ny ny nz num_iter time\ndata = np.array( [ \\\n";
  int size = 1;
//#pragma omp parallel
//  {
//#pragma omp master
//    { size = omp_get_num_threads(); }
//  }
  std::cout << "[ " << size << ", " << storage.xMax() - storage.xMin() << ", "
            << storage.yMax() - storage.yMin() << ", " << storage.zMax() << ", " << nIter << ", "
            << diff << "],\n";
  std::cout << "] )" << std::endl;
}
} // namespace

int main(int argc, char const* argv[]) {
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif
  int x = atoi(argv[2]);
  int y = atoi(argv[4]);
  int z = atoi(argv[6]);
  int iter = atoi(argv[8]);
  int nHalo = 2;
  assert(x > 0 && y > 0 && z > 0 && iter > 0);
  Storage3D<double> input(x, y, z, nHalo);
  input.initialize();
  Storage3D<double> output(x, y, z, nHalo);
  output.initialize();

  double alpha = 1. / 32.;

  std::ofstream fout;
  fout.open("in_field.dat", std::ios::binary | std::ofstream::trunc);
  input.writeFile(fout);
  fout.close();
#ifdef CRAYPAT
  PAT_record(PAT_STATE_ON);
#endif
  auto start = std::chrono::steady_clock::now();

  apply_diffusion_gpu(input, output, alpha, iter, x, y, z, nHalo);

  auto end = std::chrono::steady_clock::now();
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif
  updateHalo(output);
  fout.open("out_field.dat", std::ios::binary | std::ofstream::trunc);
  output.writeFile(fout);
  fout.close();

  auto diff = end - start;
  double timeDiff = std::chrono::duration<double, std::milli>(diff).count() / 1000.;
  reportTime(output, iter, timeDiff);

  return 0;
}
