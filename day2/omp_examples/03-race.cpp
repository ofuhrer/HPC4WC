#include <iostream>
#include <omp.h>

int main(int argc, char const* argv[]) {
  int solution = -1;
#pragma omp parallel
  { solution = omp_get_thread_num(); }
  std::cout << solution << std::endl;
  return 0;
}

