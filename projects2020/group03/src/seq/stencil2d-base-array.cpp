#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>

// #include "pat_api.h"
#include "../utils.h"

namespace {

// deprecated older version of the function, here for ... documentation reasons?
void updateHalo(Storage3D<double> &inField) {
  const int xInterior = inField.xMax() - inField.xMin();
  const int yInterior = inField.yMax() - inField.yMin();

  // bottom edge (without corners)
  for (std::size_t k = 0; k < inField.zMin(); ++k) {
    for (std::size_t j = 0; j < inField.yMin(); ++j) {
      for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j + yInterior, k);
      }
    }
  }

  // top edge (without corners)
  for (std::size_t k = 0; k < inField.zMin(); ++k) {
    for (std::size_t j = inField.yMax(); j < inField.ySize(); ++j) {
      for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j - yInterior, k);
      }
    }
  }

  // left edge (including corners)
  for (std::size_t k = 0; k < inField.zMin(); ++k) {
    for (std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
      for (std::size_t i = 0; i < inField.xMin(); ++i) {
        inField(i, j, k) = inField(i + xInterior, j, k);
      }
    }
  }

  // right edge (including corners)
  for (std::size_t k = 0; k < inField.zMin(); ++k) {
    for (std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
      for (std::size_t i = inField.xMax(); i < inField.xSize(); ++i) {
        inField(i, j, k) = inField(i - xInterior, j, k);
      }
    }
  }
}

// overloaded function with cstd arrays
void updateHalo(double *inField, int32_t xsize, int32_t ysize, int32_t zsize,
                int32_t halosize) {

  std::size_t xMin = halosize;
  std::size_t xMax = xsize - halosize;
  std::size_t yMin = halosize;
  std::size_t yMax = ysize - halosize;
  std::size_t zMin = 0;
  std::size_t zMax = zsize;

  const int xInterior = xMax - xMin;
  const int yInterior = yMax - yMin;

  // bottom edge (without corners)
  for (std::size_t k = 0; k < zMin; ++k) {
    for (std::size_t j = 0; j < yMin; ++j) {
      for (std::size_t i = xMin; i < xMax; ++i) {
        std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
        std::size_t ijp1k = i + ((j + yInterior) * xsize) + (k * xsize * ysize);

        inField[ijk] = inField[ijp1k];
        // inField[k][j][i] = inField[k][j + yInterior][i];
      }
    }
  }

  // top edge (without corners)
  for (std::size_t k = 0; k < zMin; ++k) {
    for (std::size_t j = yMax; j < ysize; ++j) {
      for (std::size_t i = xMin; i < xMax; ++i) {
        std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
        std::size_t ijm1k = i + ((j - yInterior) * xsize) + (k * xsize * ysize);

        inField[ijk] = inField[ijm1k];
        // inField[k][j][i] = inField[k][j - yInterior][i];
      }
    }
  }

  // left edge (including corners)
  for (std::size_t k = 0; k < zMin; ++k) {
    for (std::size_t j = yMin; j < yMax; ++j) {
      for (std::size_t i = 0; i < xMin; ++i) {
        std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
        std::size_t ip1jk = i + xInterior + (j * xsize) + (k * xsize * ysize);

        inField[ijk] = inField[ip1jk];
        // inField[k][j][i] = inField(i + xInterior, j, k);
      }
    }
  }

  // right edge (including corners)
  for (std::size_t k = 0; k < zMin; ++k) {
    for (std::size_t j = yMin; j < yMax; ++j) {
      for (std::size_t i = xMax; i < xsize; ++i) {
        std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
        std::size_t im1jk = i - xInterior + (j * xsize) + (k * xsize * ysize);

        inField[ijk] = inField[im1jk];
        // inField[k][j][i] = inField[i - xInterior][j][k];
      }
    }
  }
}

// deprecated older version of the function, here for ... documentation reasons?
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

// overloaded function with cstd arrays
void apply_diffusion(double *inField, double *outField, double alpha,
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
  double *tmp1Field = new double[sizeOf2DField];

  for (std::size_t iter = 0; iter < numIter; ++iter) {
    updateHalo(inField, xsize, ysize, z, halo);

    for (std::size_t k = 0; k < zMax; ++k) {

      // apply the initial laplacian
      for (std::size_t j = yMin - 1; j < yMax + 1; ++j) {
        for (std::size_t i = xMin - 1; i < xMax + 1; ++i) {

          std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
          std::size_t ip1jk = (i + 1) + (j * xsize) + (k * xsize * ysize);
          std::size_t im1jk = (i - 1) + (j * xsize) + (k * xsize * ysize);
          std::size_t ijp1k = i + ((j + 1) * xsize) + (k * xsize * ysize);
          std::size_t ijm1k = i + ((j - 1) * xsize) + (k * xsize * ysize);

          tmp1Field[j * xsize + i] = -4.0 * inField[ijk] + inField[im1jk] +
                                     inField[ip1jk] + inField[ijm1k] +
                                     inField[ijp1k];
        }
      }

      // apply the second laplacian
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
          if (iter == numIter - 1) {
            std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
            outField[ijk] = inField[ijk] - alpha * laplap;
            // outField(i, j, k) = inField[ijk] - alpha * laplap;
          } else {
            std::size_t ijk = i + (j * xsize) + (k * xsize * ysize);
            inField[ijk] = inField[ijk] - alpha * laplap;
            // inField(i, j, k) = inField(i, j, k) - alpha * laplap;
          }
        }
      }
    }
  }
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
  int iter = atoi(argv[8]);
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
  updateHalo(output, xsize, ysize, zsize, nHalo);

  // copy output array to output_3D for writing to file
  for (std::size_t k = 0; k < zsize; ++k) {
    for (std::size_t j = 0; j < ysize; ++j) {
      for (std::size_t i = 0; i < xsize; ++i) {
        std::size_t index1D = i + (j * xsize) + (k * xsize * ysize);
        output_3D(i, j, k) = output[index1D];
      }
    }
  }

  fout.open("out_field.dat", std::ios::binary | std::ofstream::trunc);
  output_3D.writeFile(fout);
  fout.close();

  auto diff = end - start;
  double timeDiff =
      std::chrono::duration<double, std::milli>(diff).count() / 1000.;
  reportTime(output_3D, iter, timeDiff);

  delete[] input;
  delete[] output;

  return 0;
}
