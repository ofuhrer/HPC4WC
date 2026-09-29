/**
 * @file cnn_stencil_alt.cpp
 * @brief Implementation of CNNStencil with OpenMP-optimized flattening.
 */

#include "cnn_stencil_alt.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

CNNStencil::CNNStencil(int in_channels, int num_classes)
    : conv1_(in_channels, 8, 3, 1, 1),  // 1→8 channels, 3×3 conv
      relu1_(),
      pool1_(2, 2),                      // 2×2 pool, stride=2
      conv2_(8, 16, 3, 1, 1),            // 8→16 channels, 3×3 conv
      relu2_(),
      pool2_(2, 2),                      // 2×2 pool, stride=2
      fc_(16 * 7 * 7, num_classes)       // fully connected: 784 → num_classes
{}

// Forward pass through the full CNN.
Tensor4D CNNStencil::forward(const Tensor4D& batch) const {
  Tensor4D x = conv1_.forward(batch);
  x = relu1_.forward(x);
  x = pool1_.forward(x);

  x = conv2_.forward(x);
  x = relu2_.forward(x);
  x = pool2_.forward(x);

  // Flatten for the fully connected layer.
  const int N = static_cast<int>(x.size());
  if (N == 0) return {};

  const int C = static_cast<int>(x[0].size());
  const int H = static_cast<int>(x[0][0].size());
  const int W = static_cast<int>(x[0][0][0].size());

  Tensor2D flattened(N, std::vector<float>(C * H * W));

#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(x, flattened, N, C, H, W) \
    if (N > 1)
#endif
  for (int n = 0; n < N; ++n) {
    int idx = 0;
    for (int c = 0; c < C; ++c) {
      for (int i = 0; i < H; ++i) {
        const auto& row = x[n][c][i];  // contiguous row
#ifdef _OPENMP
#pragma omp simd
#endif
        for (int j = 0; j < W; ++j) {
          flattened[n][idx++] = row[j];
        }
      }
    }
  }

  Tensor2D fc_out = fc_.forward(flattened);

  // Convert back to 4D tensor: [N × 1 × 1 × num_classes].
  const int num_classes = static_cast<int>(fc_out[0].size());
  Tensor4D out(N, Tensor3D(1, std::vector<std::vector<float>>(1, std::vector<float>(num_classes))));

#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(out, fc_out, N, num_classes) \
    if (N > 1)
#endif
  for (int n = 0; n < N; ++n) {
    out[n][0][0] = fc_out[n];
  }

  return out;
}
