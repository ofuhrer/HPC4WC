#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const* argv[]) {
  double sum;
#pragma omp parallel for reduction(+ : sum)
  for(std::size_t t = 0; t < 10;++t) {
    sum += t;
  }
    std::cout << sum<<std::endl;
}
