#include <iostream>
#include <omp.h>
#include <vector>
int main(int argc, char const* argv[]) {
  int steps = 10000000;

  double sum;

  omp_set_num_threads(atoi(argv[1]));
  double tick = omp_get_wtime();
#pragma omp parallel for reduction(+ : sum)
  for(std::size_t t = 0; t < steps; ++t) {
    sum += (1.0 - 2 * (t % 2)) / (2 * t + 1);
  }

  double tock = omp_get_wtime();
#pragma omp parallel
  {
    if(omp_get_thread_num() == 0)
      std::cout << omp_get_num_threads() << "\t" << tock - tick << std::endl;
  }
  return 0;
}