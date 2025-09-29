/**
 * @file maxpool_stencil.cpp
 * @brief Implementation of MaxPool2DStencil forward passes.
 *
 * Performs 2D max pooling on 3D and 4D tensors. Assumes contiguous
 * layout in row-major order. Behavior identical to the naive version.
 */

#include "maxpool_stencil.hpp"

#include <algorithm>  // std::max

MaxPool2DStencil::MaxPool2DStencil(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride) {}

Tensor4D MaxPool2DStencil::forward(const Tensor4D& input) const {
  int N = static_cast<int>(input.size());
  if (N == 0) return {};

  Tensor4D output;
  output.reserve(N);

  // Process each sample in the batch independently
  for (int n = 0; n < N; ++n) {
    output.push_back(forwardSingle(input[n]));
  }
  return output;
}

Tensor3D MaxPool2DStencil::forwardSingle(const Tensor3D& input) const {
  int C = static_cast<int>(input.size());
  if (C == 0) return {};

  int H = static_cast<int>(input[0].size());
  int W = static_cast<int>(input[0][0].size());

  // Check divisibility by stride (no padding assumed)
  if ((H - kernel_size_) % stride_ != 0 || (W - kernel_size_) % stride_ != 0) {
    throw std::invalid_argument("MaxPool2DStencil: input size not divisible by stride");
  }

  int H_out = (H - kernel_size_) / stride_ + 1;
  int W_out = (W - kernel_size_) / stride_ + 1;

  Tensor3D output(
      C, std::vector<std::vector<float>>(H_out, std::vector<float>(W_out, 0.0f)));

  // Iterate [C][H_out][W_out] and compute max within each pooling window
  for (int c = 0; c < C; ++c) {
    for (int i = 0; i < H_out; ++i) {
      for (int j = 0; j < W_out; ++j) {
        float max_val = input[c][i * stride_][j * stride_];
        for (int m = 0; m < kernel_size_; ++m) {
          for (int n = 0; n < kernel_size_; ++n) {
            int row = i * stride_ + m;
            int col = j * stride_ + n;
            max_val = std::max(max_val, input[c][row][col]);
          }
        }
        output[c][i][j] = max_val;
      }
    }
  }

  return output;
}
