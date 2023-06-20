#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const *argv[]) {

  int N = atoi(argv[1]);
  std::vector<int> values(N, -1);

#pragma omp parallel num_threads(10)
  {
#pragma omp single
    {
      for (std::size_t i = 0; i < N; ++i) {
#pragma omp task firstprivate(i)
        {
          int rank, iteration;

          rank = omp_get_thread_num();
          iteration = i;
          values[iteration] = rank;
          std::string output = "Thread " + std::to_string(rank) +
                               " executed loop iteration " +
                               std::to_string(iteration) + "\n";
          std::cout << output;
        }
      }
    }
  }

  return 0;
}