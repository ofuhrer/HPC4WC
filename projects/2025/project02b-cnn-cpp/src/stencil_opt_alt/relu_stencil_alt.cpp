/**
 * @file relu_stencil_alt.cpp
 * @brief OpenMP-optimized ReLUStencil forward passes (stencil-style).
 *
 * Parallelizes across rows of the tensor while keeping the innermost loop
 * contiguous for cache efficiency. Adds `omp simd` to hint vectorization.
 * Numerical behavior remains identical to the naive/stencil versions.
 */

#include "relu_stencil_alt.hpp"

#include <algorithm>  // std::max
#ifdef _OPENMP
#include <omp.h>
#endif

// ReLU on 4D tensors: [N × C × H × W]
// Parallelize over (n, c, h). Each iteration touches a distinct row: output[n][c][h][0..W-1].
// The innermost loop over w is unit stride and marked with `omp simd` to encourage vectorization.
Tensor4D ReLUStencil::forward(const Tensor4D& input) const {
  Tensor4D output = input;  // copy shape and values
  if (output.empty()) return output;

  const int N = static_cast<int>(output.size());
  const int C = static_cast<int>(output[0].size());
  const int H = static_cast<int>(output[0][0].size());
  const int W = static_cast<int>(output[0][0][0].size());
  const int row_count = N * C * H;  // number of independent rows

#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) default(none)                  \
    shared(output, N, C, H, W, row_count) if (row_count > 1)
#endif
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        auto& row = output[n][c][h];
        // Unit-stride inner loop; safe to vectorize.
#ifdef _OPENMP
#pragma omp simd
#endif
        for (int w = 0; w < W; ++w) {
          row[w] = std::max(0.0f, row[w]);
        }
      }
    }
  }
  return output;
}

// ReLU on 2D tensors: [N × F]
// Parallelize over rows. Each row is independent; inner loop is unit stride.
Tensor2D ReLUStencil::forward(const Tensor2D& input) const {
  Tensor2D output = input;
  if (output.empty()) return output;

  const int H = static_cast<int>(output.size());
  const int W = static_cast<int>(output[0].size());

#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(output, H, W) if (H > 1)
#endif
  for (int h = 0; h < H; ++h) {
    auto& row = output[h];
#ifdef _OPENMP
#pragma omp simd
#endif
    for (int w = 0; w < W; ++w) {
      row[w] = std::max(0.0f, row[w]);
    }
  }
  return output;
}
