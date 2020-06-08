#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const* argv[]) {

    std::cout << "schedule(static, 2)" << std::endl;
#pragma omp parallel for schedule(static, 2)
  for(int i = 0; i < 10; ++i) {
#pragma omp critical (output)
    std::cout << "This is iteration " << i << " executed from thread " << omp_get_thread_num()
              << std::endl;
  }
    std::cout << "schedule(static, 1)" << std::endl;
#pragma omp parallel for schedule(static, 1)
  for(int i = 0; i < 10; ++i) {
      #pragma omp critical (output)
    std::cout << "This is iteration " << i << " executed from thread " << omp_get_thread_num()
              << std::endl;
  }

    
}

