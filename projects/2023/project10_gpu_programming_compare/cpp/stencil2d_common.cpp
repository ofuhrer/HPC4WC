
#include <cuda_runtime_api.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA Checks
void check(cudaError_t error, const char* const file, int const line)
{
    if (error) {
        printf("CUDA ERROR at %s:%d: code=%d (%s)\n", file, line,
            static_cast<unsigned int>(error), cudaGetErrorName(error));
        exit(EXIT_FAILURE);
    }
}
