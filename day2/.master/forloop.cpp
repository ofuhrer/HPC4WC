// hpc4wc:student-begin
// hpc4wc:student | #include <iostream>
// hpc4wc:student | #include <omp.h>
// hpc4wc:student | #include <vector>
// hpc4wc:student |
// hpc4wc:student | int main(int argc, char const* argv[]) {
// hpc4wc:student |
// hpc4wc:student |   int N = atoi(argv[1]);
// hpc4wc:student |   std::vector<int> values(N, -1);
// hpc4wc:student |
// hpc4wc:student |   //
// hpc4wc:student |   // Pragmas here?
// hpc4wc:student |   //
// hpc4wc:student |   for(std::size_t i = 0; i < N; ++i) {
// hpc4wc:student |     //
// hpc4wc:student |     // Pragmas here?
// hpc4wc:student |     //
// hpc4wc:student |     int rank, iteration;
// hpc4wc:student |
// hpc4wc:student |     rank = 1;      // rank = YOUR IMPLEMENTATION
// hpc4wc:student |     iteration = i; // iteration = YOUR IMPLEMENTATION
// hpc4wc:student |     values[iteration] = rank;
// hpc4wc:student |     std::string output = "Thread " + std::to_string(rank) + " executed loop iteration " +
// hpc4wc:student |                          std::to_string(iteration) + "\n";
// hpc4wc:student |     std::cout << output;
// hpc4wc:student |     //
// hpc4wc:student |     // Pragmas here?
// hpc4wc:student |     //
// hpc4wc:student |   }
// hpc4wc:student |
// hpc4wc:student |   return 0;
// hpc4wc:student | }
// hpc4wc:student-end
// hpc4wc:solution-begin
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
// hpc4wc:solution-end
