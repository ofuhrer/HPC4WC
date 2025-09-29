#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>

#ifdef CRAYPAT
#include "pat_api.h"
#endif
#include "utils.h"

namespace {

// Apply diffusion using a double Laplacian stencil on a 3D field
// PRE: inField and outField are initialized Storage3D objects with dimensions x*y*z and halo width 'halo'
//      alpha is the diffusion coefficient 
//      numIter is the number of diffusion iterations to perform
//      x, y, z are the interior dimensions (excluding halo)
//      halo is the width of the halo region (must be >= 2 for the stencil)
// POST: outField contains the result after numIter diffusion steps
//       inField is modified during computation (contains intermediate results)
//       Each iteration applies: out = in - alpha * laplacian(laplacian(in))
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

// Report timing results in a format suitable for analysis
// PRE: storage is a valid Storage3D object
//      nIter is the number of iterations performed
//      diff is the elapsed time in seconds
// POST: Outputs timing data to stdout in numpy array format
//       Format: [num_ranks, nx, ny, nz, num_iter, time]
void reportTime(const Storage3D<double> &storage, int nIter, double diff) {
  std::cout << "# ranks nx ny nz num_iter time\ndata = np.array( [ \\\n";
  int size;
  std::cout << "[ " << size << ", " << storage.xMax() - storage.xMin() << ", "
            << storage.yMax() - storage.yMin() << ", " << storage.zMax() << ", "
            << nIter << ", " << diff << "],\n";
  std::cout << "] )" << std::endl;
}
} // namespace

// Main function: Parse command line arguments and run diffusion simulation
// Expected arguments: program -nx <x> -ny <y> -nz <z> -iter <iterations>
// PRE: argc >= 9 and argv contains valid integer arguments for dimensions and iterations
// POST: Writes input field to "in_field.dat", runs diffusion simulation,
//       writes output field to "out_field.dat", and reports timing results
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
  fout.open("out_field.dat", std::ios::binary | std::ofstream::trunc);
  output.writeFile(fout);
  fout.close();

  auto diff = end - start;
  double timeDiff =
      std::chrono::duration<double, std::milli>(diff).count() / 1000.;
  reportTime(output, iter, timeDiff);

  return 0;
}
