#include <cstdlib>
#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const* argv[]) {

#pragma omp parallel num_threads(3)
    {
      int size = omp_get_num_threads();
      int rank = omp_get_thread_num();
        #pragma omp for nowait
        for(int i = 0; i < 6; ++i){
        std::string s = "loop 1, iteration " + std::to_string(i) + "\n";
            std::cout << s;
            
        }
        
        #pragma omp for nowait
        for(int i = 0; i < 6; ++i){
        std::string s = "loop 2, iteration " + std::to_string(i) + "\n";
            std::cout << s;
            
        }
    }
}
