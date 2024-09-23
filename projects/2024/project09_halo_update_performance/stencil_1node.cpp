// This is a copied version of the code from the exercises in HPC4WC/day2
// It is used as a reference to verify that the MPI parallelized diffusion code produces the same data
// Minor changes include:
//    setting k-loops in updateHalo() from 0 to zMax() instead of zMin() (this was probably set to zMin() for testing in the original code)
//    j-loops for the left and right halo_update where only running on the interior of the y-axis and did not include the corners, inspite of the comment telling they do. This was corrected, so now they do include the corners.
//    changing output file location and naming convention
//    adding nHalo parameter to main function

#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <sstream>

#ifdef CRAYPAT
#include "pat_api.h"
#endif
#include "utils.h"

namespace {

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
    for (std::size_t j = 0; j < inField.ySize(); ++j) {
      for (std::size_t i = 0; i < inField.xMin(); ++i) {
        inField(i, j, k) = inField(i + xInterior, j, k);
      }
    }
  }

  // right edge (including corners)
  for (std::size_t k = 0; k < inField.zMax(); ++k) {
    for (std::size_t j = 0; j < inField.ySize(); ++j) {
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

void reportTime(const Storage3D<double> &storage, int nIter, double diff) {
  std::cout << "# ranks nx ny nz num_iter time\ndata = np.array( [ \\\n";
  int size;

  std::cout << "[ " << storage.xMax() - storage.xMin() << ", "
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
  int nHalo = atoi(argv[8]);
  int iter = atoi(argv[10]);
  assert(x > 0 && y > 0 && z > 0 && nHalo > 0 && iter > 0);
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

  apply_diffusion(input, output, alpha, iter, x, y, z, nHalo);

  auto end = std::chrono::steady_clock::now();
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif
  updateHalo(output);
  std::ostringstream filename;
  filename << "output_folder/out_fields/out_field_reference_nHalo_" << nHalo << ".dat";
  fout.open(filename.str(), std::ios::binary | std::ofstream::trunc);
  output.writeFile(fout);
  fout.close();

  auto diff = end - start;
  double timeDiff =
      std::chrono::duration<double, std::milli>(diff).count() / 1000.;
  reportTime(output, iter, timeDiff);

  return 0;
}
