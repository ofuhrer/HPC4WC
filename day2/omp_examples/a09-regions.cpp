#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const* argv[]) {

#pragma omp parallel
    {
      int size = omp_get_num_threads();
      int rank = omp_get_thread_num();
        #pragma omp single
        {
        std::cout << "thread " << rank <<  " is present in single" << std::endl;
        std::cout << "and the size here is : " << omp_get_num_threads() << std::endl;
    }
        #pragma omp master
        {
        std::cout << "thread " << rank <<  " is present in master" << std::endl;
    }
        #pragma omp critical(somethingHard)
        {
           std::cout << "thread " << rank <<  " is present in critical" << std::endl; 
        }
    }

    
}

