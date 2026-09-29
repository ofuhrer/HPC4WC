// cnn_opt_alt.cpp
/**
 * @file cnn_opt_alt.cpp
 * @brief Top level CNN forward using optimized sublayers (alternate path).
 *
 * Sublayers run their own parallelism. This file adds batch parallelism for
 * flatten and the final write back, plus a SIMD hint on the unit stride width
 * loop during flatten. Shapes remain [N][C][H][W]. No algorithmic changes.
 */

#include "cnn_opt_alt.hpp"

#ifdef _OPENMP
#include <omp.h>
#endif

// Constructor sets up all layers with hardcoded architecture for MNIST
CNNOptAlt::CNNOptAlt(int in_channels, int num_classes)
    : conv1_(in_channels, 8, 3, 1, 1),  // 1→8 channels, 3×3 kernel, stride=1, pad=1
      relu1_(),
      pool1_(2, 2),                      // 2×2 pooling, stride=2
      conv2_(8, 16, 3, 1, 1),            // 8→16 channels, 3×3 kernel, stride=1, pad=1
      relu2_(),
      pool2_(2, 2),                      // 2×2 pooling, stride=2
      fc_(16 * 7 * 7, num_classes)       // fully connected: 784 → num_classes
{}

// Forward pass through the full CNN
Tensor4D CNNOptAlt::forward(const Tensor4D& batch) const {
  // Pass through layers
  Tensor4D x = conv1_.forward(batch);
  x = relu1_.forward(x);
  x = pool1_.forward(x);

  x = conv2_.forward(x);
  x = relu2_.forward(x);
  x = pool2_.forward(x);

  // Flatten for the fully connected layer
  const int N = static_cast<int>(x.size());
  const int C = static_cast<int>(x[0].size());
  const int H = static_cast<int>(x[0][0].size());
  const int W = static_cast<int>(x[0][0][0].size());
  const int F = C * H * W;

  Tensor2D flattened(N, std::vector<float>(F));

  // Parallelize flatten over batch; innermost loop has unit stride on width.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N > 1) default(none) \
    shared(x, flattened, N, C, H, W, F)
#endif
  for (int n = 0; n < N; ++n) {
    for (int c = 0; c < C; ++c) {
      for (int i = 0; i < H; ++i) {
        const int base = (c * H + i) * W;  // start index in flattened[n] for this row
#ifdef _OPENMP
#pragma omp simd
#endif
        for (int j = 0; j < W; ++j) {
          flattened[n][base + j] = x[n][c][i][j];
        }
      }
    }
  }

  Tensor2D fc_out = fc_.forward(flattened);

  // Convert back to 4D tensor: [N][1][1][num_classes]
  const int O = static_cast<int>(fc_out[0].size());
  Tensor4D out(N, Tensor3D(1, std::vector<std::vector<float>>(1, std::vector<float>(O))));

  // Parallelize final reshape over batch
#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N > 1) default(none) \
    shared(fc_out, out, N)
#endif
  for (int n = 0; n < N; ++n) {
    out[n][0][0] = fc_out[n];
  }

  return out;
}
