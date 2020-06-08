#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const* argv[]) {

// parallel loop
#pragma omp parallel for
  for(int i = 0; i < 10; ++i) {
#pragma omp critical(output)
    std::cout << "This is iteration " << i << " executed from thread " << omp_get_thread_num()
              << std::endl;
  }

    
}

