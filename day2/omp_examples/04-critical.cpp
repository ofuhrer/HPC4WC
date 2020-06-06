#include <iostream>
#include <omp.h>

int main() {

#pragma omp parallel num_threads(10)
  {
#pragma omp critical(output)
    std::cout << "I am processor " << omp_get_thread_num() << std::endl;
  }
    
  return 0;
    
}
