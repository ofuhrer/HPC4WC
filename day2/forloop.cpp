#include <iostream>
#include <omp.h>
#include <vector>

int main(int argc, char const* argv[]) {

  int N = atoi(argv[1]);
  std::vector<int> values(N, -1);

  //
  // Pragmas here?
  //
  for(std::size_t i = 0; i < N; ++i) {
    //
    // Pragmas here?
    //
    int rank, iteration;

    rank = 1;      // rank = YOUR IMPLEMENTATION
    iteration = i; // iteartion = YOUR IMPLEMENTATION
    values[iteration] = rank;
    std::string output = "Thread " + std::to_string(rank) + " executed loop iteration " +
                         std::to_string(iteration) + "\n";
    std::cout << output;
    //
    // Pragmas here?
    //
  }

  return 0;
}