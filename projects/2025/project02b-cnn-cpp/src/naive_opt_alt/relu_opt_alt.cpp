// relu_opt_alt.cpp
/**
 * @file relu_opt_alt.cpp
 * @brief Optimized ReLU (alternate) with batch and collapsed-loop parallelism.
 *
 * Works in-place on a copied tensor, then applies ReLU element wise. Parallelizes
 * across (n, c, h) for 4D and across batch for 2D, with an omp simd hint on the
 * innermost unit-stride loop. Behavior matches the naive version with no changes.
 */

#include "relu_opt_alt.hpp"

#include <algorithm>  // std::max
#include <cstddef>    // std::ptrdiff_t

#ifdef _OPENMP
#include <omp.h>
#endif

Tensor4D ReLUOptAlt::forward(const Tensor4D& input) const {
  Tensor4D output = input;  // copy shape and values

  const std::ptrdiff_t N = static_cast<std::ptrdiff_t>(output.size());
  if (N == 0) return output;

  // Precompute shape once so collapsed loops don't reference outer indices.
  const std::ptrdiff_t C = static_cast<std::ptrdiff_t>(output[0].size());
  const std::ptrdiff_t H = static_cast<std::ptrdiff_t>(output[0][0].size());
  const std::ptrdiff_t W = static_cast<std::ptrdiff_t>(output[0][0][0].size());

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) if (N > 1) \
    default(none) shared(output, N, C, H, W)
#endif
  for (std::ptrdiff_t n = 0; n < N; ++n) {
    for (std::ptrdiff_t c = 0; c < C; ++c) {
      for (std::ptrdiff_t h = 0; h < H; ++h) {
        auto& row = output[n][c][h];

#ifdef _OPENMP
#pragma omp simd
#endif
        for (std::ptrdiff_t w = 0; w < W; ++w) {
          row[w] = std::max(0.0f, row[w]);
        }
      }
    }
  }
  return output;
}

Tensor2D ReLUOptAlt::forward(const Tensor2D& input) const {
  Tensor2D output = input;

  const std::ptrdiff_t N = static_cast<std::ptrdiff_t>(output.size());
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N > 1) \
    default(none) shared(output, N)
#endif
  for (std::ptrdiff_t n = 0; n < N; ++n) {
    auto& vec = output[n];
    const std::ptrdiff_t F = static_cast<std::ptrdiff_t>(vec.size());

#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::ptrdiff_t f = 0; f < F; ++f) {
      vec[f] = std::max(0.0f, vec[f]);
    }
  }
  return output;
}
