/**
 * @file maxpool_stencil_alt.cpp
 * @brief OpenMP-optimized 2D max pooling (stencil-style).
 *
 * Parallelizes across the batch in the 4D entry point. The per-image kernel
 * remains serial for predictable cache behavior. Adds an `omp simd` hint within
 * the tiny pooling window. Numerical behavior is unchanged.
 */

#include "maxpool_stencil_alt.hpp"

#include <algorithm>  // std::max
#include <stdexcept>
#ifdef _OPENMP
#include <omp.h>
#endif

MaxPool2DStencil::MaxPool2DStencil(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride) {}

// Forward on 4D input: [N × C × H × W].
// Strategy: parallelize over N. Pre-size output to avoid push_back races.
Tensor4D MaxPool2DStencil::forward(const Tensor4D& input) const {
  const int N = static_cast<int>(input.size());
  if (N == 0) return {};

  Tensor4D output;
  output.resize(N);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(input, output, N) if (N > 1)
#endif
  for (int n = 0; n < N; ++n) {
    output[n] = forwardSingle(input[n]);
  }
  return output;
}

// Per-image pooling on 3D: [C × H × W].
// Serial by design; keeps writes localized and avoids nested parallel regions.
Tensor3D MaxPool2DStencil::forwardSingle(const Tensor3D& input) const {
  const int C = static_cast<int>(input.size());
  if (C == 0) return {};

  const int H = static_cast<int>(input[0].size());
  const int W = static_cast<int>(input[0][0].size());

  if ((H - kernel_size_) % stride_ != 0 || (W - kernel_size_) % stride_ != 0) {
    throw std::invalid_argument("MaxPool2DStencil: input size not divisible by stride");
  }

  const int H_out = (H - kernel_size_) / stride_ + 1;
  const int W_out = (W - kernel_size_) / stride_ + 1;

  Tensor3D output(C, std::vector<std::vector<float>>(H_out, std::vector<float>(W_out, 0.0f)));

  for (int c = 0; c < C; ++c) {
    for (int i = 0; i < H_out; ++i) {
      for (int j = 0; j < W_out; ++j) {
        float max_val = input[c][i * stride_][j * stride_];
        for (int m = 0; m < kernel_size_; ++m) {
#ifdef _OPENMP
#pragma omp simd reduction(max : max_val)
#endif
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
