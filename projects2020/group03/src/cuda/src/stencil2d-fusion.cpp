#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>

#ifdef CRAYPAT
#include "pat_api.h"
#endif
#include "utils.h"
#include "update_halo.h"
#include "apply_diffusion_fused.h"

void reportTime(const Storage3D<realType>& storage, int nIter, double diff) {
  std::cout << "# ranks nx ny ny nz num_iter time\ndata = np.array( [ \\\n";
  int size = 1;
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

  apply_diffusion_fused(input, output, alpha, iter, x, y, z, nHalo);

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
