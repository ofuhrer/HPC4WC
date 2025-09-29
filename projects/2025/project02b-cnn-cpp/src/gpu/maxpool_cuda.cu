#include "maxpool_cuda.hpp"
#include <cuda_runtime.h>
#include <cfloat>
#include <stdexcept>

// One thread per output pixel
__global__ static void maxpool_kernel(
    const float* __restrict__ input,
    float* output,
    int N, int C, int H, int W,
    int kernel, int stride,
    int H_out, int W_out
) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int cc    = blockIdx.z;
    int c     = cc % C;
    int n     = cc / C;
    if (w_out >= W_out || h_out >= H_out) return;

    int out_idx = ((n*C + c)*H_out + h_out)*W_out + w_out;
    float m = -FLT_MAX;

    for (int kh=0; kh<kernel; ++kh) {
        for (int kw=0; kw<kernel; ++kw) {
            int in_h = h_out*stride + kh;
            int in_w = w_out*stride + kw;
            int in_idx = ((n*C + c)*H + in_h)*W + in_w;
            m = fmaxf(m, input[in_idx]);
        }
    }
    output[out_idx] = m;
}

MaxPool2DCUDA::MaxPool2DCUDA(int kernel_size, int stride)
  : kernel_size_(kernel_size), stride_(stride)
{}

Tensor4D MaxPool2DCUDA::forward(const Tensor4D& batch) const {
    int N = batch.size();
    if (N == 0) return {};

    int C = batch[0].size();
    int H = batch[0][0].size();
    int W = batch[0][0][0].size();

    if ((H - kernel_size_) % stride_ != 0 ||
        (W - kernel_size_) % stride_ != 0)
        throw std::invalid_argument("MaxPool2DCUDA: size not divisible");

    int H_out = (H - kernel_size_) / stride_ + 1;
    int W_out = (W - kernel_size_) / stride_ + 1;

    // Flatten
    std::vector<float> h_input(N*C*H*W);
    for (int n=0, k=0; n<N; ++n)
    for (int c=0; c<C; ++c)
    for (int i=0; i<H; ++i)
    for (int j=0; j<W; ++j)
        h_input[k++] = batch[n][c][i][j];

    // Device buffers
    float *d_in=nullptr, *d_out=nullptr;
    size_t in_b = h_input.size()*sizeof(float);
    size_t out_b = size_t(N*C*H_out*W_out)*sizeof(float);
    cudaMalloc(&d_in,  in_b);
    cudaMalloc(&d_out, out_b);
    cudaMemcpy(d_in, h_input.data(), in_b, cudaMemcpyHostToDevice);

    // Launch
    dim3 block(16,16);
    dim3 grid(
      (W_out+block.x-1)/block.x,
      (H_out+block.y-1)/block.y,
      N * C
    );
    maxpool_kernel<<<grid, block>>>(
      d_in, d_out,
      N, C, H, W,
      kernel_size_, stride_,
      H_out, W_out
    );
    cudaDeviceSynchronize();

    // Copy back
    std::vector<float> h_output(N*C*H_out*W_out);
    cudaMemcpy(h_output.data(), d_out, out_b, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    // Unflatten
    Tensor4D out(N,
        Tensor3D(C,
          std::vector<std::vector<float>>(H_out, std::vector<float>(W_out))));
    for (int n=0, k=0; n<N; ++n)
    for (int c=0; c<C; ++c)
    for (int i=0; i<H_out; ++i)
    for (int j=0; j<W_out; ++j)
        out[n][c][i][j] = h_output[k++];

    return out;
}
