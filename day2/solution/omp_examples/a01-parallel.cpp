#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const *argv[]) {
  // this is a sequential region
  srand(712);
  double a = rand() % 100;
  std::cout << std::to_string(a) + "\n\n";

  // this is a parallel region
#pragma omp parallel
  {
    double b = rand() % 100;
    std::cout << std::to_string(b) + "\n";
  }

  // this is a sequential region
  std::cout << "\nthis is a sequential region again\n";

  std::cout << std::flush;
}
