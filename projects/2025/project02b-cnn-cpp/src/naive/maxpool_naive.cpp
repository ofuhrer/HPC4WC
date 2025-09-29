// maxpool_naive.cpp
/**
 * @file maxpool_naive.cpp
 * @brief Naive 2D max pooling layer without padding.
 *
 * Applies channel wise max pooling with a fixed window and stride. Validates
 * that input size is divisible by the stride, then computes outputs with
 * nested loops over channels and spatial positions, scanning each window to
 * take the maximum. Execution is single threaded and follows [N][C][H][W]
 * conventions at the caller. No algorithmic changes.
 */

#include <algorithm>   // std::max
#include <stdexcept>   // std::invalid_argument
#include "maxpool_naive.hpp"

MaxPool2DNaive::MaxPool2DNaive(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride) {}

Tensor4D MaxPool2DNaive::forward(const Tensor4D& input) const {
    const int N = static_cast<int>(input.size());
    if (N == 0) return {};

    Tensor4D output;
    output.reserve(N);

    // Batch wrapper: process each sample independently using the single-sample kernel.
    for (int n = 0; n < N; ++n) {
        output.push_back(forwardSingle(input[n]));
    }
    return output;
}

Tensor3D MaxPool2DNaive::forwardSingle(const Tensor3D& input) const {
    const int C = static_cast<int>(input.size());
    if (C == 0) return {};

    const int H = static_cast<int>(input[0].size());
    const int W = static_cast<int>(input[0][0].size());

    // Shape checks: no padding. Dimensions must align with kernel/stride.
    if ((H - kernel_size_) % stride_ != 0 || (W - kernel_size_) % stride_ != 0) {
        throw std::invalid_argument("MaxPool2DNaive: input size not divisible by stride");
    }

    const int H_out = (H - kernel_size_) / stride_ + 1;
    const int W_out = (W - kernel_size_) / stride_ + 1;

    Tensor3D output(
        C,
        std::vector<std::vector<float>>(H_out, std::vector<float>(W_out, 0.0f))
    );

    // For each channel, compute the maximum over each pooling window.
    // The innermost loop walks unit-stride over window elements to keep cache-friendly access.
    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < H_out; ++i) {
            for (int j = 0; j < W_out; ++j) {
                float max_val = input[c][i * stride_][j * stride_];
                for (int m = 0; m < kernel_size_; ++m) {
                    for (int n = 0; n < kernel_size_; ++n) {
                        const int row = i * stride_ + m;
                        const int col = j * stride_ + n;
                        max_val = std::max(max_val, input[c][row][col]);
                    }
                }
                output[c][i][j] = max_val;
            }
        }
    }

    return output;
}
