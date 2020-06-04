#include <iostream>
#include <omp.h>

int main() {
  int data = 0.;
  int flag = 0;
#pragma omp parallel num_threads(10)
  {
    if(flag == 0) {
      data += omp_get_thread_num() + 100;
      flag =  omp_get_thread_num() + 100;;
#pragma omp flush(data, flag)
    }
  }
  std::cout << data << std::endl;
  std::cout << flag << std::endl;
  return 0;
}