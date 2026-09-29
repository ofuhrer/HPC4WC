#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const *argv[]) {

  int n_iter = atoi(argv[1]);
  omp_set_num_threads(atoi(argv[2]));
  double itime, ftime;
  itime = omp_get_wtime();
  double sum = 0;
#pragma omp parallel for
  for (int i = 0; i < n_iter; ++i) {
#pragma omp atomic
    sum += i;
  }

  ftime = omp_get_wtime();
  std::cout << sum << std::endl;
  std::cout << ftime - itime << std::endl;
  return 0;
}
