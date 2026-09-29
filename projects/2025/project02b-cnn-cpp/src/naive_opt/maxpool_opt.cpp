// maxpool_opt.cpp
/**
 * @file maxpool_opt.cpp
 * @brief Optimized 2D max pooling with batch and channel parallelism.
 *
 * Validates size and stride compatibility, then applies channel wise K by K
 * pooling without padding. Parallelizes across the batch in forward, and across
 * channels and output rows in the single sample kernel using OpenMP collapse.
 * Uses an omp simd max reduction across the pooling window. Matches the naive
 * layer in behavior with no algorithm changes.
 */


#include <algorithm>   // std::max
#include <stdexcept>   // std::invalid_argument
#include <cstddef>     // std::size_t
#ifdef _OPENMP
#include <omp.h>
#endif

#include "maxpool_opt.hpp"

MaxPool2DOpt::MaxPool2DOpt(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride) {}

Tensor4D MaxPool2DOpt::forward(const Tensor4D& input) const {
  const int N = static_cast<int>(input.size());
  if (N == 0) return {};

  // Preallocate outer container for thread-safe writes.
  Tensor4D output(static_cast<std::size_t>(N));

  // Parallelize across batch dimension.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) \
    shared(input, output, N) if (N > 1)
#endif
  for (int n = 0; n < N; ++n) {
    output[static_cast<std::size_t>(n)] =
        forward_single(input[static_cast<std::size_t>(n)]);
  }
  return output;
}

Tensor3D MaxPool2DOpt::forward_single(const Tensor3D& input) const {
  const int C = static_cast<int>(input.size());
  if (C == 0) return {};

  const int H = static_cast<int>(input[0].size());
  const int W = static_cast<int>(input[0][0].size());

  const int K = kernel_size_;
  const int S = stride_;

  if ((H - K) % S != 0 || (W - K) % S != 0) {
    throw std::invalid_argument("MaxPool2DOpt: input size not divisible by stride");
  }

  const int H_out = (H - K) / S + 1;
  const int W_out = (W - K) / S + 1;

  Tensor3D output(
      static_cast<std::size_t>(C),
      std::vector<std::vector<float>>(
          static_cast<std::size_t>(H_out),
          std::vector<float>(static_cast<std::size_t>(W_out))));

  // Parallelize over channels and output rows (collapse the next two loops).
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) default(none) \
    shared(input, output, C, H_out, W_out, K, S, H, W)              \
    if (C * H_out > 1)
#endif
  for (int c = 0; c < C; ++c) {
    for (int i = 0; i < H_out; ++i) {
      const int base_row = i * S;
      const auto& chan = input[static_cast<std::size_t>(c)];
      auto& out_row = output[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)];
      for (int j = 0; j < W_out; ++j) {
        const int base_col = j * S;

        // Local max over KxK window.
        float max_val =
            chan[static_cast<std::size_t>(base_row)][static_cast<std::size_t>(base_col)];

        for (int m = 0; m < K; ++m) {
          const auto& in_row = chan[static_cast<std::size_t>(base_row + m)];
#ifdef _OPENMP
#pragma omp simd reduction(max : max_val)
#endif
          for (int k = 0; k < K; ++k) {
            const float v = in_row[static_cast<std::size_t>(base_col + k)];
            max_val = std::max(max_val, v);
          }
        }

        out_row[static_cast<std::size_t>(j)] = max_val;
      }
    }
  }

  return output;
}
