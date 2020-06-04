#include <iostream>
#include <omp.h>
#include <vector>
int main(int argc, char const* argv[]) {
  int steps = 10000000;

  double sum;
#pragma omp parallel for reduction(+ : sum)
  for(std::size_t t = 0; t < steps; ++t) {
    sum += (1.0 - 2 * (t % 2)) / (2 * t + 1);
  }

  std::cout << sum * 4.0;
}

