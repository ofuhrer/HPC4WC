#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>

#ifdef CRAYPAT
#include "pat_api.h"
#endif
#include "utils.h"

void updateHalo(Storage3D<realType>& inField) {
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

void apply_diffusion(Storage3D<realType>& inField, Storage3D<realType>& outField, realType alpha,
                     unsigned numIter, int x, int y, int z, int halo) {

  Storage3D<realType> tmp1Field(x, y, z, halo);

  for(std::size_t iter = 0; iter < numIter; ++iter) {

    updateHalo(inField);
    realType a1 = -1. * alpha;
    realType a2 = -2. * alpha;
    realType a8 = 8. * alpha;
    realType a20 = 1. - 20. * alpha;

    for(std::size_t k = 0; k < inField.zMax(); ++k) {

      // apply the full diffusion
#pragma omp parallel for
      for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
        for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
          outField(i, j, k) = a1 * inField(i, j - 2, k) + a2 * inField(i - 1, j - 1, k) +
                              a8 * inField(i, j - 1, k) + a2 * inField(i + 1, j - 1, k) +
                              a1 * inField(i - 2, j, k) + a8 * inField(i - 1, j, k) +
                              a20 * inField(i, j, k) + a8 * inField(i + 1, j, k) +
                              a1 * inField(i + 2, j, k) + a2 * inField(i - 1, j + 1, k) +
                              a8 * inField(i, j + 1, k) + a2 * inField(i + 1, j + 1, k) +
                              a1 * inField(i, j + 2, k);
        }
      }

      // update the field
#pragma omp parallel for
      for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
        for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
          if(iter != numIter - 1) {
            inField(i, j, k) = outField(i, j, k);
          }
        }
      }
    }
  }
}

void reportTime(const Storage3D<realType>& storage, int nIter, double diff) {
  std::cout << "# ranks nx ny ny nz num_iter time\ndata = np.array( [ \\\n";
  int size;
#pragma omp parallel
  {
#pragma omp master
    { size = omp_get_num_threads(); }
  }
  std::cout << "[ " << size << ", " << storage.xMax() - storage.xMin() << ", "
            << storage.yMax() - storage.yMin() << ", " << storage.zMax() << ", " << nIter << ", "
            << diff << "],\n";
  std::cout << "] )" << std::endl;
}

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
  Storage3D<realType> input(x, y, z, nHalo);
  input.initialize();
  Storage3D<realType> output(x, y, z, nHalo);
  output.initialize();

  realType alpha = 1. / 32.;

  std::ofstream fout;
  fout.open("in_field.dat", std::ios::binary | std::ofstream::trunc);
  input.writeFile(fout);
  fout.close();
#ifdef CRAYPAT
  PAT_record(PAT_STATE_ON);
#endif
  auto start = std::chrono::steady_clock::now();

  apply_diffusion(input, output, alpha, iter, x, y, z, nHalo);

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
