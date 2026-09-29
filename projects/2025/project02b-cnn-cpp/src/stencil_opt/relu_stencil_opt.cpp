// relu_stencil_opt.cpp
/**
 * @file relu_stencil_opt.cpp
 * @brief Implementation of ReLUStencilOpt forward passes.
 *
 * Applies elementwise ReLU activation to 2D and 4D tensors. Parallelized with
 * OpenMP over batch/channels/rows, with SIMD hints on innermost loops. No new
 * allocations or I/O occur inside hot loops. Behavior is identical to the naive version.
 */

#include <algorithm>  // std::max
#include <cassert>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "relu_stencil_opt.hpp"

// 4D: [N × C × H × W]
Tensor4D ReLUStencilOpt::forward(const Tensor4D& input) const {
  const int N = static_cast<int>(input.size());
  if (N == 0) return {};

  const int C = static_cast<int>(input[0].size());
  const int H = static_cast<int>(input[0][0].size());
  const int W = static_cast<int>(input[0][0][0].size());

#ifndef NDEBUG
  assert(C > 0 && H > 0 && W > 0);
#endif

  // Allocate output with identical shape.
  Tensor4D output(
      static_cast<std::size_t>(N),
      Tensor3D(static_cast<std::size_t>(C),
               std::vector<std::vector<float>>(static_cast<std::size_t>(H),
                                               std::vector<float>(static_cast<std::size_t>(W)))));

  // Parallelize over (n, c, h). Innermost w loop is unit-stride for SIMD.
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) default(none)                           \
    shared(input, output, N, C, H, W) if (N * C * H > 1)
#endif
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int h = 0; h < H; ++h) {
        const auto& in_row =
            input[static_cast<std::size_t>(n)][static_cast<std::size_t>(c)][static_cast<std::size_t>(h)];
        auto& out_row =
            output[static_cast<std::size_t>(n)][static_cast<std::size_t>(c)][static_cast<std::size_t>(h)];

#ifdef _OPENMP
#pragma omp simd
#endif
        for (int w = 0; w < W; ++w) {
          const float x = in_row[static_cast<std::size_t>(w)];
          out_row[static_cast<std::size_t>(w)] = std::max(x, 0.0f);
        }
      }
    }
  }

  return output;
}

// 2D: [N × F]
Tensor2D ReLUStencilOpt::forward(const Tensor2D& input) const {
  const int N = static_cast<int>(input.size());
  if (N == 0) return {};

  const int F = static_cast<int>(input[0].size());

#ifndef NDEBUG
  assert(F > 0);
#endif

  Tensor2D output(static_cast<std::size_t>(N), std::vector<float>(static_cast<std::size_t>(F)));

  // Parallelize over rows; SIMD across features (unit-stride).
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(input, output, N, F) if (N > 1)
#endif
  for (int n = 0; n < N; ++n) {
    const auto& in_vec = input[static_cast<std::size_t>(n)];
    auto& out_vec = output[static_cast<std::size_t>(n)];

#ifdef _OPENMP
#pragma omp simd
#endif
    for (int f = 0; f < F; ++f) {
      const float x = in_vec[static_cast<std::size_t>(f)];
      out_vec[static_cast<std::size_t>(f)] = std::max(x, 0.0f);
    }
  }

  return output;
}
