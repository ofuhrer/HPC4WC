#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const* argv[]) {
    
    std::vector<double> input(10000000,1);
    std::vector<double> output(10000000,0);
    
    omp_set_num_threads(atoi(argv[1]));
    
    double tic = omp_get_wtime();
    
#pragma omp parallel for schedule(static, 1)
  for (int i = 0; i < input.size(); ++i){
    output[i] = 2*input[i]; 
    input[i] = 0;
  }
    
    double toc = omp_get_wtime();
    
#pragma omp parallel
  {
    if(omp_get_thread_num() == 0)
      std::cout << omp_get_num_threads() << "\t" << toc - tic << std::endl;
  }
    
  return 0;
    
}
