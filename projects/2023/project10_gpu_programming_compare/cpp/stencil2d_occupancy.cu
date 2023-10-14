
#include "stencil2d_common.h"
#include <cuda_runtime_api.h>
#include <vector>

__global__ void someWork(int* d_int)
{
    extern __shared__ char shr[];
    int num_iter = 50000;
    volatile int t = 0;
#pragma unroll 1
    for (int n = 0; n < num_iter; ++n) {
        t += 1;
    }
    *d_int = t;
}

extern "C" void stencil3d_occupancy(dim3 b, dim3 g)
{
    int bx = b.x;
    int by = b.y;
    int bz = b.z;

    int gx = g.x;
    int gy = g.y;
    int gz = g.z;

    int shrBytes = 2 * bx * by * bz * sizeof(FloatType);

    int bs = bx * by * bz;
    int bt = bx * by * bz;
    int bw = bt / 32;

    int gs = gx * gy * gz;
    int gt = gs * bs;
    int gw = gt / 32;

    // Data for GTX 980
    // int gpu_sms = 16;
    // int gpu_warps = 128;
    // int gpu_shrm = 65536;

    // Data for Tesla P100
    int gpu_sms = 56;
    int gpu_warps = 64;
    int gpu_shrm = 65536;

    printf("\n");
    printf("Occupancy Analysis\n");
    printf("------------------\n");

    printf("\n");
    printf("Config of Blocks\n");
    printf("Threads  : %dx%dx%d Threads = %d Threads per Block [<=1024]\n", bx, by, bz, bs);
    printf("Memory   : %d Bytes per Block [<=49152]\n", shrBytes);

    printf("\n");
    printf("Config of Grid\n");
    printf("Blocks   : %dx%dx%d Blocks = %d Blocks per Grid\n", gx, gy, gz, gs);
    printf("Threads  : %dx%dx%d Threads = %d Threads per Grid\n", gx * bx, gy * by, gz * bz, gt);

    printf("\n");
    printf("GPU information\n");
    printf("SMs      : %d SMs on GPU\n", gpu_sms);
    printf("Warps    : Max %d Resident Warps per SM\n", gpu_warps);
    printf("Memory   : Max %d Bytes Shared Memory per SM\n", gpu_shrm);

    printf("\n");
    printf("Segmentation in Warps\n");
    printf("Block    : %d Threads per Block / 32 Threads per Warp = %d Warps per Block [<=32]\n", bt, bw);
    printf("Grid     : %d Threads per Grid / 32 Threads per Warp = %d Warps per Grid\n", gt, gw);

    printf("\n");
    printf("Limits for SMs\n");
    printf("Warp     : %d Warps per SM / %d Warps per Block = %d Blocks per SM\n", gpu_warps, bw, gpu_warps / bw);
    printf("Memory   : %d Bytes per SM / %d Bytes per Block = %d Blocks per SM\n", gpu_shrm, shrBytes, gpu_shrm / shrBytes);
    int blocksPerSM = std::min(gpu_warps / bw, gpu_shrm / shrBytes);
    printf("Combined : %d Blocks per SM\n", blocksPerSM);
    printf("\n");

    printf("Limits for GPU\n");
    printf("Warps    : %d SMs * %d Warps per SM = %d Warps on GPU\n", gpu_sms, gpu_warps, gpu_sms * gpu_warps);
    printf("Blocks   : %d SMs * %d Blocks per SM = %d Blocks on GPU\n", gpu_sms, gpu_warps / bw, gpu_sms * blocksPerSM);
    printf("Grid     : %d Blocks on GPU / %d Blocks per Grid = %.1f%% of Grid on GPU\n", gpu_sms * blocksPerSM, gs, 100.0 * gpu_sms * blocksPerSM / gs);
    printf("\n");

    // dim3 gridDim(gx, gy, gz);
    // dim3 blockDim(bx, by, bz);
    // int s = 8;
    // std::vector<cudaStream_t> streams(s);
    // for (int k = 0; k < s; ++k) {
    //     CheckErrors(cudaStreamCreate(&streams[k]));
    // }
    // int* d_int;
    // CheckErrors(cudaMalloc(&d_int, sizeof(int)));
    // for (int n = 0; n < s; ++n) {
    //     void* args[] = { &d_int };
    //     CheckErrors(cudaLaunchKernel((void*)someWork, gridDim, blockDim, args, shrBytes, streams[n]));
    // }
    // CheckErrors(cudaFree(d_int));
    // for (int k = 0; k < s; ++k) {
    //     CheckErrors(cudaStreamDestroy(streams[k]));
    // }
}
