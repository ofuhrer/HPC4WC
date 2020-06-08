#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const* argv[]) {

      // enable different numbers of threads
#pragma omp parallel num_threads(2)
  {
    int size = omp_get_num_threads();
    int rank = omp_get_thread_num();
#pragma omp critical(output)
    std::cout << "I am rank " << rank << " of a total of " << size << " threads" << std::endl;
  }

    
}

