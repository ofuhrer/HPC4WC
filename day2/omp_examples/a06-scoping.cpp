#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const* argv[]) {

  int myvar = -1;
#pragma omp parallel for num_threads(3)
  for(int i = 0; i < 10; ++i) {
    myvar = i;
#pragma omp critical(output)
    std::cout << "i is " << i << " and myvar is " << myvar << std::endl;
  }

  std::cout << "myvar: " << myvar << std::endl;

    
}

