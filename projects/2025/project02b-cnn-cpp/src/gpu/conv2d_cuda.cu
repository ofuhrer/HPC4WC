#include "conv2d_cuda.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <random>

// Kernel: one thread per output pixel
__global__ static void conv2d_kernel(
    const float* __restrict__ input,
    float* output,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    int N, int C, int H, int W,
    int outC, int kernel, int stride, int pad,
    int H_out, int W_out
) {
    int w_out = blockIdx.x * blockDim.x + threadIdx.x;
    int h_out = blockIdx.y * blockDim.y + threadIdx.y;
    int oc_idx = blockIdx.z;
    int oc = oc_idx % outC;
    int n  = oc_idx / outC;
    if (w_out >= W_out || h_out >= H_out) return;

    int out_index = ((n * outC + oc) * H_out + h_out) * W_out + w_out;
    float sum = bias[oc];

    for (int ic = 0; ic < C; ++ic) {
        for (int kh = 0; kh < kernel; ++kh) {
            for (int kw = 0; kw < kernel; ++kw) {
                int in_h = h_out * stride + kh - pad;
                int in_w = w_out * stride + kw - pad;
                if (in_h >= 0 && in_h < H && in_w >= 0 && in_w < W) {
                    int in_index = ((n * C + ic) * H + in_h) * W + in_w;
                    int w_index  = ((oc * C + ic) * kernel + kh) * kernel + kw;
                    sum += weights[w_index] * input[in_index];
                }
            }
        }
    }
    output[out_index] = sum;
}

Conv2DCUDA::Conv2DCUDA(int in_channels,
                       int out_channels,
                       int kernel_size,
                       int stride,
                       int padding)
  : in_channels_(in_channels),
    out_channels_(out_channels),
    kernel_size_(kernel_size),
    stride_(stride),
    padding_(padding),
    d_weights_(nullptr),
    d_bias_(nullptr)
{
    if (kernel_size_ != 3 || stride_ != 1 || padding_ != 1)
        throw std::invalid_argument(
            "Conv2DCUDA currently supports only kernel_size=3, stride=1, padding=1");

    // Allocate and initialize host weights & biases
    std::vector<float> h_weights(out_channels_ * in_channels_ * kernel_size_ * kernel_size_);
    std::vector<float> h_bias(out_channels_, 0.0f);

    std::random_device rd;
    std::mt19937 gen(rd());
    float fan_in = float(in_channels_) * kernel_size_ * kernel_size_;
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));

    for (auto &w : h_weights) w = dist(gen);

    // Copy to device
    size_t w_bytes = h_weights.size() * sizeof(float);
    size_t b_bytes = h_bias.size()    * sizeof(float);
    cudaMalloc(&d_weights_, w_bytes);
    cudaMalloc(&d_bias_,    b_bytes);
    cudaMemcpy(d_weights_, h_weights.data(), w_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_bias_,    h_bias.data(),    b_bytes, cudaMemcpyHostToDevice);
}

Conv2DCUDA::~Conv2DCUDA() {
    cudaFree(d_weights_);
    cudaFree(d_bias_);
}

Tensor4D Conv2DCUDA::forward(const Tensor4D& batch) const {
    int N = batch.size();
    if (N == 0) return {};
    int C = batch[0].size();
    int H = batch[0][0].size();
    int W = batch[0][0][0].size();

    int H_out = (H + 2*padding_ - kernel_size_) / stride_ + 1;
    int W_out = (W + 2*padding_ - kernel_size_) / stride_ + 1;

    // Flatten input
    std::vector<float> h_input(N * C * H * W);
    for (int n = 0; n < N; ++n) {
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    int idx = ((n*C + c)*H + i)*W + j;
                    h_input[idx] = batch[n][c][i][j];
                }
            }
        }
    }

    // Allocate device buffers
    float *d_input=nullptr, *d_output=nullptr;
    size_t in_bytes  = h_input.size() * sizeof(float);
    size_t out_bytes = size_t(N)*out_channels_*H_out*W_out * sizeof(float);
    cudaMalloc(&d_input,  in_bytes);
    cudaMalloc(&d_output, out_bytes);
    cudaMemcpy(d_input, h_input.data(), in_bytes, cudaMemcpyHostToDevice);

    // Launch kernel
    dim3 block(16,16);
    dim3 grid(
        (W_out + block.x -1)/block.x,
        (H_out + block.y -1)/block.y,
        N * out_channels_
    );
    conv2d_kernel<<<grid, block>>>(
        d_input, d_output, d_weights_, d_bias_,
        N, C, H, W,
        out_channels_, kernel_size_, stride_, padding_,
        H_out, W_out
    );
    cudaDeviceSynchronize();

    // Copy back
    std::vector<float> h_output(N * out_channels_ * H_out * W_out);
    cudaMemcpy(h_output.data(), d_output, out_bytes, cudaMemcpyDeviceToHost);

    // Free device input/output
    cudaFree(d_input);
    cudaFree(d_output);

    // Unflatten to Tensor4D
    Tensor4D result(N,
        Tensor3D(out_channels_,
            std::vector<std::vector<float>>(H_out, std::vector<float>(W_out))));
    for (int n = 0; n < N; ++n) {
        for (int oc = 0; oc < out_channels_; ++oc) {
            for (int i = 0; i < H_out; ++i) {
                for (int j = 0; j < W_out; ++j) {
                    int idx = ((n*out_channels_ + oc)*H_out + i)*W_out + j;
                    result[n][oc][i][j] = h_output[idx];
                }
            }
        }
    }
    return result;
}
