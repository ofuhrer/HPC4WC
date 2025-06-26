#include <cmath>
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const *argv[]) {

  int n_iter = atoi(argv[1]);
  omp_set_num_threads(atoi(argv[2]));
  double itime, ftime;
  itime = omp_get_wtime();
  double sum;
  std::vector<double> localvals(n_iter, 0);
#pragma omp parallel for schedule(static, 2)
  for (int i = 0; i < n_iter; ++i) {
    localvals[i] = acos(cos(asin(sin(std::fabs((double)i / n_iter))))) + localvals[i];
  }
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < n_iter; ++i) {
    sum += localvals[i];
  }

  ftime = omp_get_wtime();
  std::cout << sum << std::endl;
  std::cout << ftime - itime << std::endl;
  return 0;
}
