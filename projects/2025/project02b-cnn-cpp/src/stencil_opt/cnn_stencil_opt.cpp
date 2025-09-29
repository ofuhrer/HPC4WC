// cnn_stencil_opt.cpp
/**
 * @file cnn_stencil_opt.cpp
 * @brief Top level CNN forward using stencil optimized sublayers.
 *
 * Sublayers handle their own internal parallelism. This file adds safe
 * batch parallelism for flatten and the final write back, and a SIMD hint
 * for the innermost width loop during flatten. No algorithmic changes.
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#include "cnn_stencil_opt.hpp"

CNNStencilOpt::CNNStencilOpt(int in_channels, int num_classes)
    : conv1_(in_channels, 8, 3, 1, 1),
      relu1_(),
      pool1_(2, 2),
      conv2_(8, 16, 3, 1, 1),
      relu2_(),
      pool2_(2, 2),
      fc_(16 * 7 * 7, num_classes) {}

// Forward pass through the full CNN using the stencil path.
Tensor4D CNNStencilOpt::forward(const Tensor4D& batch) const {
  // Layer stack
  Tensor4D x = conv1_.forward(batch);
  x = relu1_.forward(x);
  x = pool1_.forward(x);

  x = conv2_.forward(x);
  x = relu2_.forward(x);
  x = pool2_.forward(x);

  // Flatten [N][C][H][W] → [N][F]
  const int N = static_cast<int>(x.size());
  if (N == 0) return {};

  const int C = static_cast<int>(x[0].size());
  const int H = static_cast<int>(x[0][0].size());
  const int W = static_cast<int>(x[0][0][0].size());
  const int F = C * H * W;

  Tensor2D flattened(static_cast<std::size_t>(N),
                     std::vector<float>(static_cast<std::size_t>(F)));

  // Parallelize across samples, SIMD across width for unit stride stores.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) \
    shared(x, flattened, N, C, H, W, F) if (N > 1)
#endif
  for (int n = 0; n < N; ++n) {
    const auto& x_n = x[static_cast<std::size_t>(n)];
    for (int c = 0; c < C; ++c) {
      const auto& x_nc = x_n[static_cast<std::size_t>(c)];
      for (int i = 0; i < H; ++i) {
        const auto& x_nci = x_nc[static_cast<std::size_t>(i)];
        const int base = (c * H + i) * W;
#ifdef _OPENMP
#pragma omp simd
#endif
        for (int j = 0; j < W; ++j) {
          flattened[static_cast<std::size_t>(n)]
                   [static_cast<std::size_t>(base + j)] =
              x_nci[static_cast<std::size_t>(j)];
        }
      }
    }
  }

  // Fully connected and reshape to [N][1][1][num_classes]
  Tensor2D fc_out = fc_.forward(flattened);
  const int num_classes = static_cast<int>(fc_out[0].size());

  Tensor4D out(static_cast<std::size_t>(N),
               Tensor3D(1, std::vector<std::vector<float>>(
                               1, std::vector<float>(static_cast<std::size_t>(num_classes)))));

#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) \
    shared(out, fc_out, N, num_classes) if (N > 1)
#endif
  for (int n = 0; n < N; ++n) {
    out[static_cast<std::size_t>(n)][0][0] = fc_out[static_cast<std::size_t>(n)];
  }

  return out;
}
