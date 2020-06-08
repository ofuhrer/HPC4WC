#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const* argv[]) {

#pragma omp parallel num_threads(5)
    {
      int size = omp_get_num_threads();
      int rank = omp_get_thread_num();
        #pragma omp critical(somethingHard)
        {
           std::string s = "thread "+ std::to_string(rank)+" is present in critical1\n";
            std::cout << s; 
        }   
        #pragma omp critical(somethingEasy)
        {
            std::string s = "thread "+ std::to_string(rank)+" is present in critical2\n";
           std::cout << s; 
        }
    }
}

