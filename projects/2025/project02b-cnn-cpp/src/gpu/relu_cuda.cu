#include "relu_cuda.hpp"
#include <cuda_runtime.h>

// Element‑wise ReLU kernel
__global__ static void relu_kernel(float* data, int total) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < total) {
        float v = data[idx];
        data[idx] = v > 0.0f ? v : 0.0f;
    }
}

Tensor4D ReLUCUDA::forward(const Tensor4D& batch) const {
    int N = batch.size();
    if (N == 0) return {};

    int C = batch[0].size();
    int H = batch[0][0].size();
    int W = batch[0][0][0].size();
    int total = N * C * H * W;

    // Flatten
    std::vector<float> h_data(total);
    for (int n=0, k=0; n<N; ++n)
    for (int c=0; c<C; ++c)
    for (int i=0; i<H; ++i)
    for (int j=0; j<W; ++j)
        h_data[k++] = batch[n][c][i][j];

    // Device buffer
    float* d_data=nullptr;
    size_t bytes = total * sizeof(float);
    cudaMalloc(&d_data, bytes);
    cudaMemcpy(d_data, h_data.data(), bytes, cudaMemcpyHostToDevice);

    // Launch
    int blockSize = 256;
    int gridSize = (total + blockSize - 1)/blockSize;
    relu_kernel<<<gridSize, blockSize>>>(d_data, total);
    cudaDeviceSynchronize();

    // Copy back
    cudaMemcpy(h_data.data(), d_data, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_data);

    // Unflatten
    Tensor4D out(N, Tensor3D(C, std::vector<std::vector<float>>(H, std::vector<float>(W))));
    for (int n=0, k=0; n<N; ++n)
    for (int c=0; c<C; ++c)
    for (int i=0; i<H; ++i)
    for (int j=0; j<W; ++j)
        out[n][c][i][j] = h_data[k++];

    return out;
}
