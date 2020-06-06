#include <iostream>
#include <omp.h>

int main() {
    
#pragma omp parallel for ordered 
  for (int i = 0; i < 10; ++i){
    int j = (100+i)*10 / 7.1;
#pragma omp ordered
    std::cout << "This is iteration " << i << std::endl;
  }
    
  return 0;
    
}
