// maxpool_opt_alt.cpp
/**
 * @file maxpool_opt_alt.cpp
 * @brief Optimized 2D max pooling (alternate) with batch parallelism.
 *
 * Parallelizes across the batch in forward(); the per-image kernel remains
 * serial to avoid nested parallel regions, with an omp simd max reduction
 * over the small K×K window. Validates stride divisibility and matches the
 * naive layer’s behavior with no algorithm changes.
 */

#include "maxpool_opt_alt.hpp"

#include <algorithm>   // std::max
#include <stdexcept>   // std::invalid_argument

#ifdef _OPENMP
#include <omp.h>
#endif

MaxPool2DOptAlt::MaxPool2DOptAlt(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride) {}

Tensor4D MaxPool2DOptAlt::forward(const Tensor4D& input) const {
  const std::ptrdiff_t N = static_cast<std::ptrdiff_t>(input.size());
  if (N == 0) return {};

  // Preallocate output for thread-safe indexed writes
  Tensor4D output(N);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N > 1) default(none) \
    shared(input, output, N)
#endif
  for (std::ptrdiff_t n = 0; n < N; ++n) {
    output[n] = forward_single(input[n]);
  }
  return output;
}

Tensor3D MaxPool2DOptAlt::forward_single(const Tensor3D& input) const {
  const std::ptrdiff_t C = static_cast<std::ptrdiff_t>(input.size());
  if (C == 0) return {};

  const std::ptrdiff_t H = static_cast<std::ptrdiff_t>(input[0].size());
  const std::ptrdiff_t W = static_cast<std::ptrdiff_t>(input[0][0].size());

  if ((H - kernel_size_) % stride_ != 0 || (W - kernel_size_) % stride_ != 0) {
    throw std::invalid_argument(
        "MaxPool2DOptAlt: input dimensions must be divisible by stride");
  }

  const std::ptrdiff_t H_out = (H - kernel_size_) / stride_ + 1;
  const std::ptrdiff_t W_out = (W - kernel_size_) / stride_ + 1;

  Tensor3D output(
      C, std::vector<std::vector<float>>(H_out, std::vector<float>(W_out, 0.0f)));

  // Serial per-image kernel; no nested parallelism.
  for (std::ptrdiff_t c = 0; c < C; ++c) {
    for (std::ptrdiff_t i = 0; i < H_out; ++i) {
      for (std::ptrdiff_t j = 0; j < W_out; ++j) {
        float max_val = input[c][i * stride_][j * stride_];

        // Small pooling window, encourage vectorization with simd.
#ifdef _OPENMP
#pragma omp simd reduction(max : max_val)
#endif
        for (std::ptrdiff_t m = 0; m < kernel_size_; ++m) {
          for (std::ptrdiff_t n = 0; n < kernel_size_; ++n) {
            const std::ptrdiff_t row = i * stride_ + m;
            const std::ptrdiff_t col = j * stride_ + n;
            max_val = std::max(max_val, input[c][row][col]);
          }
        }
        output[c][i][j] = max_val;
      }
    }
  }
  return output;
}
