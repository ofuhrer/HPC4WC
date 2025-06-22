#include <cmath>
#include <iostream>
#include <omp.h>

int main(int argc, char const *argv[]) {

  int n_iter = atoi(argv[1]);
  omp_set_num_threads(atoi(argv[2]));

  double sum;
#pragma omp parallel for reduction(+ : sum)
  for (int i = 0; i < n_iter; ++i) {
    sum += acos(cos(asin(sin(std::fabs((double)i / n_iter)))));
  }

  std::cout << sum << std::endl;
  return 0;
}
