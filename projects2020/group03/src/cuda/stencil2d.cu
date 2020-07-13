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
  __shared__ double buffer1[8][8][4];
  __shared__ double buffer2[8][8][4];
  int i = blockDim.x * blockIdx.x + threadIdx.x;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.z * blockIdx.z + threadIdx.z;
  int li = threadIdx.x;
  int lj = threadIdx.y;
  int lk = threadIdx.z;
  int index = i + j * xSize + k * xSize * ySize;
  int xInterior = xMax - xMin;
  int yInterior = yMax - yMin;

  // perform halo update (local)
  if (i >= xMin && i < xMax &&
      j >= 0    && j < yMin && k < zMax) {
    buffer1[li][lj][lk] = infield[index + yInterior * xSize];
  } else if (i >= xMin && i < xMax &&
             j >= yMax && j < ySize && k < zMax) {
    buffer1[li][lj][lk] = infield[index - yInterior * xSize];
  } else if (i >= 0    && i < xMin &&
             j >= yMin && j < yMax && k < zMax) {
    buffer1[li][lj][lk] = infield[index + xInterior];
  } else if (i >= xMax && i < xSize &&
             j >= yMin && j < yMax && k < zMax) {
    buffer1[li][lj][lk] = infield[index - xInterior];
  } else if (i < xSize && j < ySize && k < zMax) {
    buffer1[li][lj][lk] = infield[index];
  } else {
    // pass
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
    outfield[index] = infield[index] - alpha * value;
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
  cudaMemcpy(outfield, infield, size, cudaMemcpyDeviceToDevice);

  dim3 blockDim(8, 8, 4);
  dim3 gridDim((xSize + blockDim.x - 1) / blockDim.x,
               (ySize + blockDim.y - 1) / blockDim.y,
               (zMax  + blockDim.z - 1) / blockDim.z);

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
  int nHalo = 3;
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
