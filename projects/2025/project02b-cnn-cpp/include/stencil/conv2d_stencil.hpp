#pragma once
/**
 * @file conv2d_stencil.hpp
 * @brief 2D convolution with stencil-style implementation (3×3, stride 1, padding 1).
 *
 * Fixed-parameter 2D convolution intended for stencil-oriented pipelines.
 * Behavior matches the naive version while grouping under the stencil variant.
 * Assumes single input channel in the inner compute (in_channels = 1).
 *
 * Usage:
 * @code
 * Conv2DStencil conv(1, out_channels, 3, 1, 1);
 * Tensor4D y = conv.forward(x);  // x: [N×1×H×W], y: [N×out_channels×H×W]
 * @endcode
 */

#include <vector>

/// 3D tensor: [channels][height][width]
using Tensor3D = std::vector<std::vector<std::vector<float>>>;

/// 4D tensor: [batch][channels][height][width]
using Tensor4D = std::vector<Tensor3D>;

/**
 * @class Conv2DStencil
 * @brief 2D convolution with fixed kernel_size=3, stride=1, padding=1 (single in-channel).
 */
class Conv2DStencil {
 public:
  /**
   * @brief Construct the convolution layer.
   * @param in_channels Number of input channels (assumed 1 in compute path).
   * @param out_channels Number of output channels.
   * @param kernel_size Kernel size (must be 3).
   * @param stride Stride (must be 1).
   * @param padding Padding (must be 1).
   */
  Conv2DStencil(int in_channels, int out_channels, int kernel_size, int stride = 1,
                int padding = 1);

  /**
   * @brief Forward pass on a batch of inputs [N × C × H × W].
   * @param batch Input tensor.
   * @return Output tensor [N × out_channels × H_out × W_out].
   * @throws std::invalid_argument on shape mismatch.
   */
  Tensor4D forward(const Tensor4D& batch) const;

 private:
  int in_channels_;
  int out_channels_;
  int kernel_size_;
  int stride_;
  int padding_;

  // Weights: [out_channels][in_channels][kernel_size][kernel_size]
  std::vector<Tensor3D> weights_;
  // Biases: [out_channels]
  std::vector<float> biases_;

  /// Forward pass on a single input [C × H × W].
  Tensor3D forwardSingle(const Tensor3D& input) const;
};
