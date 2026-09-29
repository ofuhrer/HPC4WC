#include <cuda_runtime.h>

void cuda_sync_() {
    cudaDeviceSynchronize();
}