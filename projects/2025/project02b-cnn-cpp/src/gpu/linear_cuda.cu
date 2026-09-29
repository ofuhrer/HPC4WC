#include "linear_cuda.hpp"
#include <cuda_runtime.h>
#include <random>
#include <cmath>
#include <stdexcept>

__global__ static void linear_kernel(
    const float* __restrict__ input,
    float* output,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    int B, int inF, int outF
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= B * outF) return;
    int b = idx / outF;
    int o = idx % outF;

    float sum = bias[o];
    const float* wptr = weights + o * inF;
    const float* iptr = input   + b * inF;
    for (int i = 0; i < inF; ++i)
        sum += iptr[i] * wptr[i];
    output[idx] = sum;
}

LinearCUDA::LinearCUDA(int in_features, int out_features)
  : in_f_(in_features), out_f_(out_features),
    d_weights_(nullptr), d_bias_(nullptr)
{
    std::vector<float> h_w(out_f_ * in_f_);
    std::vector<float> h_b(out_f_, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_f_));

    for (auto &w : h_w) w = dist(gen);

    size_t w_bytes = h_w.size() * sizeof(float);
    size_t b_bytes = h_b.size() * sizeof(float);
    cudaMalloc(&d_weights_, w_bytes);
    cudaMalloc(&d_bias_,    b_bytes);
    cudaMemcpy(d_weights_, h_w.data(), w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_,    h_b.data(), b_bytes, cudaMemcpyHostToDevice);
}

LinearCUDA::~LinearCUDA() {
    cudaFree(d_weights_);
    cudaFree(d_bias_);
}

Tensor2D LinearCUDA::forward(const Tensor2D& input) const {
    int B = input.size();
    if (B == 0) return {};

    if (int(input[0].size()) != in_f_)
        throw std::invalid_argument("LinearCUDA: feature size mismatch");

    // Flatten
    std::vector<float> h_in(B * in_f_);
    for (int b = 0; b < B; ++b)
        for (int i = 0; i < in_f_; ++i)
            h_in[b * in_f_ + i] = input[b][i];

    float *d_in=nullptr, *d_out=nullptr;
    size_t in_b  = h_in.size() * sizeof(float);
    size_t out_b = size_t(B * out_f_) * sizeof(float);
    cudaMalloc(&d_in,  in_b);
    cudaMalloc(&d_out, out_b);
    cudaMemcpy(d_in, h_in.data(), in_b, cudaMemcpyHostToDevice);

    int blockSize = 256;
    int gridSize  = (B*out_f_ + blockSize -1)/blockSize;
    linear_kernel<<<gridSize, blockSize>>>(d_in, d_out, d_weights_, d_bias_, B, in_f_, out_f_);
    cudaDeviceSynchronize();

    // Copy back
    std::vector<float> h_out(B * out_f_);
    cudaMemcpy(h_out.data(), d_out, out_b, cudaMemcpyDeviceToHost);
    cudaFree(d_in);
    cudaFree(d_out);

    // Unflatten
    Tensor2D out(B, std::vector<float>(out_f_));
    for (int b = 0; b < B; ++b)
        for (int o = 0; o < out_f_; ++o)
            out[b][o] = h_out[b * out_f_ + o];

    return out;
}
