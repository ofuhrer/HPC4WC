// relu_opt.cpp
/**
 * @file relu_opt.cpp
 * @brief Optimized ReLU with batch parallelism and inner-loop SIMD.
 *
 * Allocates only the output and writes results directly. Parallelizes across
 * (n, c, h) for 4D tensors and across batch for 2D tensors, with an omp simd
 * hint on the unit stride inner loop. Uses branchless clamp via std::max.
 * Matches the naive behavior with no algorithm changes.
 */

#include <cstddef>
#include <algorithm>  // for std::max
#ifdef _OPENMP
#include <omp.h>
#endif

#include "relu_opt.hpp"

Tensor4D ReLUOpt::forward(const Tensor4D& input) const {
  const std::ptrdiff_t N = static_cast<std::ptrdiff_t>(input.size());
  if (N == 0) return {};

  const std::ptrdiff_t C = static_cast<std::ptrdiff_t>(input[0].size());
  const std::ptrdiff_t H = static_cast<std::ptrdiff_t>(input[0][0].size());
  const std::ptrdiff_t W = static_cast<std::ptrdiff_t>(input[0][0][0].size());

  // Allocate output shape only
  Tensor4D output(
      static_cast<std::size_t>(N),
      Tensor3D(static_cast<std::size_t>(C),
               std::vector<std::vector<float>>(
                   static_cast<std::size_t>(H),
                   std::vector<float>(static_cast<std::size_t>(W)))));

  // Parallelize across (n, c, h). Keep w as unit stride inner loop and apply SIMD.
  const std::ptrdiff_t work_items = N * C * H;
#ifdef _OPENMP
#pragma omp parallel for collapse(3) schedule(static) \
    default(none) shared(input, output, N, C, H, W, work_items) \
    if (work_items > 1)
#endif
  for (std::ptrdiff_t n = 0; n < N; ++n) {
    for (std::ptrdiff_t c = 0; c < C; ++c) {
      for (std::ptrdiff_t h = 0; h < H; ++h) {
        const auto& in_row = input[static_cast<std::size_t>(n)]
                                    [static_cast<std::size_t>(c)]
                                    [static_cast<std::size_t>(h)];
        auto& out_row = output[static_cast<std::size_t>(n)]
                               [static_cast<std::size_t>(c)]
                               [static_cast<std::size_t>(h)];
#ifdef _OPENMP
#pragma omp simd
#endif
        for (std::ptrdiff_t w = 0; w < W; ++w) {
          const float x = in_row[static_cast<std::size_t>(w)];
          out_row[static_cast<std::size_t>(w)] = std::max(x, 0.0f);
        }
      }
    }
  }

  return output;
}

Tensor2D ReLUOpt::forward(const Tensor2D& input) const {
  const std::ptrdiff_t N = static_cast<std::ptrdiff_t>(input.size());
  if (N == 0) return {};

  const std::ptrdiff_t F = static_cast<std::ptrdiff_t>(input[0].size());

  // Allocate output shape only
  Tensor2D output(static_cast<std::size_t>(N),
                  std::vector<float>(static_cast<std::size_t>(F)));

  // Parallelize across batch n. SIMD across features f with unit stride.
  const std::ptrdiff_t work_items = N;
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) \
    shared(input, output, N, F, work_items) if (work_items > 1)
#endif
  for (std::ptrdiff_t n = 0; n < N; ++n) {
    const auto& in_vec = input[static_cast<std::size_t>(n)];
    auto& out_vec = output[static_cast<std::size_t>(n)];
#ifdef _OPENMP
#pragma omp simd
#endif
    for (std::ptrdiff_t f = 0; f < F; ++f) {
      const float x = in_vec[static_cast<std::size_t>(f)];
      out_vec[static_cast<std::size_t>(f)] = std::max(x, 0.0f);
    }
  }

  return output;
}
