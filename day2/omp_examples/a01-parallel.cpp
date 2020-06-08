#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const* argv[]) {
  srand(712);
  double a = rand() % 100;
  std::cout << std::to_string(a) + "\n\n";

#pragma omp parallel
  {
    double b = rand() % 100;
    std::cout << std::to_string(b) + "\n";
  }
  std::cout << std::flush;

    
}

