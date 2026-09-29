/**
 * @file cnn_stencil.cpp
 * @brief Implementation of CNNStencil.
 *
 * A two-layer convolutional neural network with ReLU and max pooling,
 * followed by a fully connected classifier. Designed for MNIST.
 *
 * Note: Conv2DStencil assumes single input channel in compute path.
 * In conv2_ (8→16), only channel 0 contributes per output. This matches
 * the original stencil implementation and is preserved here.
 */

#include "cnn_stencil.hpp"

CNNStencil::CNNStencil(int in_channels, int num_classes)
    : conv1_(in_channels, 8, 3, 1, 1),  // 1→8 channels, 3×3, stride=1, pad=1
      relu1_(),
      pool1_(2, 2),                     // 2×2 pooling, stride=2
      conv2_(8, 16, 3, 1, 1),           // 8→16 channels, 3×3, stride=1, pad=1
      relu2_(),
      pool2_(2, 2),                     // 2×2 pooling, stride=2
      fc_(16 * 7 * 7, num_classes)      // fully connected: 784 → num_classes
{}

// Forward pass through the full CNN
Tensor4D CNNStencil::forward(const Tensor4D& batch) const {
  // Stage 1: Conv → ReLU → Pool
  Tensor4D x = conv1_.forward(batch);
  x = relu1_.forward(x);
  x = pool1_.forward(x);

  // Stage 2: Conv → ReLU → Pool
  x = conv2_.forward(x);
  x = relu2_.forward(x);
  x = pool2_.forward(x);

  // Flatten [N × C × H × W] → [N × (C*H*W)]
  int N = static_cast<int>(x.size());
  int C = static_cast<int>(x[0].size());
  int H = static_cast<int>(x[0][0].size());
  int W = static_cast<int>(x[0][0][0].size());

  Tensor2D flattened(N, std::vector<float>(C * H * W));
  for (int n = 0; n < N; ++n) {
    int idx = 0;
    for (int c = 0; c < C; ++c) {
      for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
          flattened[n][idx++] = x[n][c][i][j];
        }
      }
    }
  }

  // Fully connected classifier
  Tensor2D fc_out = fc_.forward(flattened);

  // Reshape to 4D: [N × 1 × 1 × num_classes]
  Tensor4D out(N, Tensor3D(1, std::vector<std::vector<float>>(
                                 1, std::vector<float>(fc_out[0].size()))));
  for (int n = 0; n < N; ++n) {
    out[n][0][0] = fc_out[n];
  }

  return out;
}
