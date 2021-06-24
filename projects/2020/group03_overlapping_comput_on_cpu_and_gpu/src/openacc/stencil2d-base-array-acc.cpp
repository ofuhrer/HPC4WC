#include <cassert>
#include <chrono>
#include <cmath>
#include <fstream>
#include <iostream>
#include <omp.h>

#ifdef CRAYPAT
#include "pat_api.h"
#endif
#include "../utils.h"

namespace {

// base versions for verification //

void updateHalo(Storage3D<double> &inField) {
  const int xInterior = inField.xMax() - inField.xMin();
  const int yInterior = inField.yMax() - inField.yMin();

  // bottom edge (without corners)
  for (std::size_t k = 0; k < inField.zMax(); ++k) {
    for (std::size_t j = 0; j < inField.yMin(); ++j) {
      for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j + yInterior, k);
      }
    }
  }

  // top edge (without corners)
  for (std::size_t k = 0; k < inField.zMax(); ++k) {
    for (std::size_t j = inField.yMax(); j < inField.ySize(); ++j) {
      for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j - yInterior, k);
      }
    }
  }

  // left edge (including corners)
  for (std::size_t k = 0; k < inField.zMax(); ++k) {
    for (std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
      for (std::size_t i = 0; i < inField.xMin(); ++i) {
        inField(i, j, k) = inField(i + xInterior, j, k);
      }
    }
  }

  // right edge (including corners)
  for (std::size_t k = 0; k < inField.zMax(); ++k) {
    for (std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
      for (std::size_t i = inField.xMax(); i < inField.xSize(); ++i) {
        inField(i, j, k) = inField(i - xInterior, j, k);
      }
    }
  }
}
void apply_diffusion(Storage3D<double> &inField, Storage3D<double> &outField,
                     double alpha, unsigned numIter, int x, int y, int z,
                     int halo) {

  Storage3D<double> tmp1Field(x, y, z, halo);

  for (std::size_t iter = 0; iter < numIter; ++iter) {

    updateHalo(inField);

    for (std::size_t k = 0; k < inField.zMax(); ++k) {

      // apply the initial laplacian
      for (std::size_t j = inField.yMin() - 1; j < inField.yMax() + 1; ++j) {
        for (std::size_t i = inField.xMin() - 1; i < inField.xMax() + 1; ++i) {
          tmp1Field(i, j, 0) = -4.0 * inField(i, j, k) + inField(i - 1, j, k) +
                               inField(i + 1, j, k) + inField(i, j - 1, k) +
                               inField(i, j + 1, k);
        }
      }

      // apply the second laplacian
      for (std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
        for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
          double laplap = -4.0 * tmp1Field(i, j, 0) + tmp1Field(i - 1, j, 0) +
                          tmp1Field(i + 1, j, 0) + tmp1Field(i, j - 1, 0) +
                          tmp1Field(i, j + 1, 0);

          // and update the field
          if (iter == numIter - 1) {
            outField(i, j, k) = inField(i, j, k) - alpha * laplap;
          } else {
            inField(i, j, k) = inField(i, j, k) - alpha * laplap;
          }
        }
      }
    }
  }
}

// todo : add open acc calls here, straightforward
void inline updateHalo(double *input, int32_t xsize, int32_t ysize,
                       int32_t zsize, int32_t halosize) {

  std::size_t xMin = halosize;
  std::size_t xMax = xsize - halosize;
  std::size_t yMin = halosize;
  std::size_t yMax = ysize - halosize;
  std::size_t zMin = 0;
  std::size_t zMax = zsize;

  const int xInterior = xMax - xMin;
  const int yInterior = yMax - yMin;

#pragma acc data present(input)
  {
#pragma acc parallel
    {
#pragma acc loop independent gang worker vector collapse(3)
      for (std::size_t k = 0; k < zMax; ++k) {
        for (std::size_t j = 0; j < yMin; ++j) {
          for (std::size_t i = xMin; i < xMax; ++i) {
            std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
            std::size_t ijp1k =
                i + ((j + yInterior) * xsize) + (k * xsize * ysize);

            input[ijk] = input[ijp1k];
            // input[k][j][i] = input[k][j + yInterior][i];
          }
        }
      }

#pragma acc loop independent gang worker vector collapse(3)
      for (std::size_t k = 0; k < zMax; ++k) {
        for (std::size_t j = yMax; j < ysize; ++j) {
          for (std::size_t i = xMin; i < xMax; ++i) {
            std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
            std::size_t ijm1k =
                i + ((j - yInterior) * xsize) + (k * xsize * ysize);

            input[ijk] = input[ijm1k];
            // input[k][j][i] = input[k][j - yInterior][i];
          }
        }
      }

#pragma acc loop independent gang worker vector collapse(3)
      for (std::size_t k = 0; k < zMax; ++k) {
        for (std::size_t j = yMin; j < yMax; ++j) {
          for (std::size_t i = 0; i < xMin; ++i) {
            std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
            std::size_t ip1jk =
                i + xInterior + (j * xsize) + (k * xsize * ysize);

            input[ijk] = input[ip1jk];
            // input[k][j][i] = input(i + xInterior, j, k);
          }
        }
      }

#pragma acc loop independent gang worker vector collapse(3)
      for (std::size_t k = 0; k < zMax; ++k) {
        for (std::size_t j = yMin; j < yMax; ++j) {
          for (std::size_t i = xMax; i < xsize; ++i) {
            std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
            std::size_t im1jk =
                i - xInterior + (j * xsize) + (k * xsize * ysize);

            input[ijk] = input[im1jk];
            // input[k][j][i] = input[i - xInterior][j][k];
          }
        }
      }
    }
  }
}

void inline updateHalo_output(double *output, int32_t xsize, int32_t ysize,
                              int32_t zsize, int32_t halosize) {
  std::size_t xMin = halosize;
  std::size_t xMax = xsize - halosize;
  std::size_t yMin = halosize;
  std::size_t yMax = ysize - halosize;
  std::size_t zMin = 0;
  std::size_t zMax = zsize;

  const int xInterior = xMax - xMin;
  const int yInterior = yMax - yMin;

  const std::size_t sizeOf3DField = (xsize) * (ysize) * (zsize);

  // todo : do you need the copy in?
  // #pragma acc data copyin(output [0:sizeOf3DField])
  {
// #pragma acc enter data
// copyin(output [0:sizeOf3DField])
// bottom edge (without corners)
#pragma acc data present(output)
    {
#pragma acc parallel
      {
#pragma acc loop independent gang worker vector collapse(3)
        for (std::size_t k = 0; k < zMax; ++k) {
          for (std::size_t j = 0; j < yMin; ++j) {
            for (std::size_t i = xMin; i < xMax; ++i) {
              std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
              std::size_t ijp1k =
                  i + ((j + yInterior) * xsize) + (k * xsize * ysize);

              output[ijk] = output[ijp1k];
              // output[k][j][i] = output[k][j + yInterior][i];
            }
          }
        }

        // top edge (without corners)
// #pragma acc parallel present(output) async(2)
#pragma acc loop independent gang worker vector collapse(3)
        for (std::size_t k = 0; k < zMax; ++k) {
          for (std::size_t j = yMax; j < ysize; ++j) {
            for (std::size_t i = xMin; i < xMax; ++i) {
              std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
              std::size_t ijm1k =
                  i + ((j - yInterior) * xsize) + (k * xsize * ysize);

              output[ijk] = output[ijm1k];
              // output[k][j][i] = output[k][j - yInterior][i];
            }
          }
        }

        // left edge (including corners)
// #pragma acc parallel present(output) async(3)
#pragma acc loop independent gang worker vector collapse(3)
        for (std::size_t k = 0; k < zMax; ++k) {
          for (std::size_t j = yMin; j < yMax; ++j) {
            for (std::size_t i = 0; i < xMin; ++i) {
              std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
              std::size_t ip1jk =
                  i + xInterior + (j * xsize) + (k * xsize * ysize);

              output[ijk] = output[ip1jk];
              // output[k][j][i] = output(i + xInterior, j, k);
            }
          }
        }

        // right edge (including corners)
// #pragma acc parallel present(output) async(4)
#pragma acc loop independent gang worker vector collapse(3)
        for (std::size_t k = 0; k < zMax; ++k) {
          for (std::size_t j = yMin; j < yMax; ++j) {
            for (std::size_t i = xMax; i < xsize; ++i) {
              std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
              std::size_t im1jk =
                  i - xInterior + (j * xsize) + (k * xsize * ysize);

              output[ijk] = output[im1jk];
              // output[k][j][i] = output[i - xInterior][j][k];
            }
          }
        }
      }
    }
  }
  // #pragma acc exit data
  // copyout(output [0:sizeOf3DField])
}

void apply_diffusion(double *input, double *output, double alpha,
                     unsigned numIter, int x, int y, int z, int halo) {
  std::size_t xsize = x;
  std::size_t xMin = halo;
  std::size_t xMax = xsize - halo;

  std::size_t ysize = y;
  std::size_t yMin = halo;
  std::size_t yMax = ysize - halo;

  std::size_t zMin = 0;
  std::size_t zMax = z;

  // Storage3D<double> tmp1Field(x, y, z, halo);
  std::size_t sizeOf2DField = (xsize) * (ysize);
  std::size_t sizeOf3DField = (xsize) * (ysize)*z;

  double *tmp1Field = new double[sizeOf2DField];

#pragma acc enter data create(tmp1Field [0:sizeOf2DField])
  {
    for (std::size_t iter = 0; iter < numIter; ++iter) {

      // TODO : make this an acc routine
      // todo : turn this into a cuda kernel and call from acc???
      updateHalo(input, xsize, ysize, z, halo);
      // #pragma acc update device(input[:sizeOf3DField])
      // todo : if this is not on the gpu, the gpu copy needs to be updated?
      // # pragma acc update device(input[0:n])

      for (std::size_t k = 0; k < zMax; ++k) {
#pragma acc parallel present(input, tmp1Field)
#pragma acc loop gang vector collapse(2)
        for (std::size_t j = yMin - 1; j < yMax + 1; ++j) {
          for (std::size_t i = xMin - 1; i < xMax + 1; ++i) {

            std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
            std::size_t ip1jk = (i + 1) + (j * xsize) + (k * xsize * ysize);
            std::size_t im1jk = (i - 1) + (j * xsize) + (k * xsize * ysize);
            std::size_t ijp1k = i + ((j + 1) * xsize) + (k * xsize * ysize);
            std::size_t ijm1k = i + ((j - 1) * xsize) + (k * xsize * ysize);

            tmp1Field[j * xsize + i] = -4.0 * input[ijk] + input[im1jk] +
                                       input[ip1jk] + input[ijm1k] +
                                       input[ijp1k];
          }
        }
#pragma acc parallel present(input, tmp1Field)
#pragma acc loop gang vector collapse(2)
        for (std::size_t j = yMin; j < yMax; ++j) {
          for (std::size_t i = xMin; i < xMax; ++i) {

            std::size_t ij = i + (j * xsize);
            std::size_t ip1j = (i + 1) + (j * xsize);
            std::size_t im1j = (i - 1) + (j * xsize);
            std::size_t ijp1 = i + ((j + 1) * xsize);
            std::size_t ijm1 = i + ((j - 1) * xsize);

            double laplap = -4.0 * tmp1Field[ij] + tmp1Field[im1j] +
                            tmp1Field[ip1j] + tmp1Field[ijm1] + tmp1Field[ijp1];

            // and update the field
            std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
            input[ijk] = input[ijk] - alpha * laplap;
          }
        }
      }
      if (iter == numIter - 1) {
#pragma acc parallel present(input, output)
#pragma acc loop independent gang worker vector collapse(3)
        for (std::size_t k = 0; k < zMax; ++k) {
          for (std::size_t j = yMin; j < yMax; ++j) {
            for (std::size_t i = xMin; i < xMax; ++i) {
              std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
              output[ijk] = input[ijk];
            }
          }
        }
      }
    }
  }
#pragma acc exit data delete (tmp1Field [0:sizeOf2DField])

  delete[] tmp1Field;
}

void reportTime(const Storage3D<double> &storage, int nIter, double diff) {
  std::cout << "# ranks nx ny ny nz num_iter time\ndata = np.array( [ \\\n";
  int size;
#pragma omp parallel
  {
#pragma omp master
    { size = omp_get_num_threads(); }
  }
  std::cout << "[ " << size << ", " << storage.xMax() - storage.xMin() << ", "
            << storage.yMax() - storage.yMin() << ", " << storage.zMax() << ", "
            << nIter << ", " << diff << "],\n";
  std::cout << "] )" << std::endl;
}
} // namespace

int main(int argc, char const *argv[]) {
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif
  int x = atoi(argv[2]);
  int y = atoi(argv[4]);
  int z = atoi(argv[6]);
  unsigned int iter = atoi(argv[8]);
  int nHalo = 3;
  assert(x > 0 && y > 0 && z > 0 && iter > 0);

  std::size_t zsize, xsize, ysize;
  xsize = x + 2 * nHalo;
  ysize = y + 2 * nHalo;
  zsize = z;

  Storage3D<double> input_3D(x, y, z, nHalo);
  Storage3D<double> output_3D(x, y, z, nHalo);

  input_3D.initialize();
  output_3D.initialize();

  std::size_t sizeOf3DField = (xsize) * (ysize) * (zsize);
  double *input = new double[sizeOf3DField];
  double *output = new double[sizeOf3DField];

  // zero initialize the newly allocated array
  for (std::size_t k = 0; k < zsize; ++k) {
    for (std::size_t j = 0; j < ysize; ++j) {
      for (std::size_t i = 0; i < xsize; ++i) {
        std::size_t index1D = i + (j * xsize) + (k * xsize * ysize);
        input[index1D] = 0;
        output[index1D] = 0;
      }
    }
  }

  // initial condition
  for (std::size_t k = zsize / 4.0; k < 3 * zsize / 4.0; ++k) {
    for (std::size_t j = nHalo + xsize / 4.; j < nHalo + 3. / 4. * xsize; ++j) {
      for (std::size_t i = nHalo + xsize / 4.; i < nHalo + 3. / 4. * xsize;
           ++i) {
        input[k * (ysize * xsize) + j * xsize + i] = 1;
        output[k * (ysize * xsize) + j * xsize + i] = 1;
      }
    }
  }

  double alpha = 1. / 32.;

  // copy input to input_3D for writing to file
  std::ofstream fout;
  fout.open("in_field.dat", std::ios::binary | std::ofstream::trunc);
  input_3D.writeFile(fout);
  fout.close();

#pragma acc enter data copyin(input [0:sizeOf3DField])                         \
    copyin(output [0:sizeOf3DField])

#ifdef CRAYPAT
  PAT_record(PAT_STATE_ON);
#endif
  auto start = std::chrono::steady_clock::now();

  // apply_diffusion(input_3D, output_3D, alpha, iter, x, y, z, nHalo);
  apply_diffusion(input, output, alpha, iter, xsize, ysize, zsize, nHalo);

  auto end = std::chrono::steady_clock::now();
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif

  // updateHalo(output_3D);
  updateHalo_output(output, xsize, ysize, zsize, nHalo);
#pragma acc exit data copyout(output [0:sizeOf3DField]) delete (               \
    input [0:sizeOf3DField])

  // copy output array to output_3D for writing to file
  for (std::size_t k = 0; k < zsize; ++k) {
    for (std::size_t j = 0; j < ysize; ++j) {
      for (std::size_t i = 0; i < xsize; ++i) {
        std::size_t index1D = i + (j * xsize) + (k * xsize * ysize);
        output_3D(i, j, k) = output[index1D];
      }
    }
  }

#ifdef VALIDATE
  //--------------------------//
  // run base and verification //
  //--------------------------//
  Storage3D<double> input_V(x, y, z, nHalo);
  input_V.initialize();
  Storage3D<double> output_V(x, y, z, nHalo);
  output_V.initialize();
  apply_diffusion(input_V, output_V, alpha, iter, x, y, z, nHalo);
  updateHalo(output_V);
  double L2_error = 0;
  for (std::size_t k = 0; k < zsize; ++k) {
    for (std::size_t j = 0; j < ysize; ++j) {
      for (std::size_t i = 0; i < xsize; ++i) {
        L2_error += std::pow((output_3D(i, j, k) - output_V(i, j, k)), 2);
      }
    }
  }
  L2_error = sqrt(L2_error);
  std::cout << " L2 Error = " << L2_error << std::endl;
  //--------------------------//
  // run base and verification //
  //--------------------------//
#endif

  fout.open("out_field.dat", std::ios::binary | std::ofstream::trunc);
  output_3D.writeFile(fout);
  fout.close();

  auto diff = end - start;
  double timeDiff =
      std::chrono::duration<double, std::milli>(diff).count() / 1000.;
  reportTime(output_3D, iter, timeDiff);

  std::ofstream os;
  os.open("time_1.dat", std::ofstream::app);
  os << timeDiff << std::endl;
  os.close();
  delete[] input;
  delete[] output;

  return 0;
}
