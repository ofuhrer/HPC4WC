#include <cstdlib>
#include <iostream>
#include <omp.h>

int main(int argc, char const* argv[]) {

  int myvar = 42;
#pragma omp parallel for private(myvar)
  for(std::size_t i = 0; i < 10; ++i) {
#pragma omp critical(output)
    std::cout << "before writing:\n  i is " << i << " and myvar is " << myvar << std::endl;
    myvar = i;
#pragma omp critical(output)
    std::cout << "after writing:\n\ti is " << i << " and myvar is " << myvar << std::endl;
  }
  std::cout << "myvar: " << myvar << std::endl;
}
