#pragma once
/**
 * @file conv2d_stencil_alt.hpp
 * @brief 2D convolution layer with stencil-style implementation (3×3 kernel).
 *
 * Fixed parameters: kernel_size = 3, stride = 1, padding = 1.
 * Intended for stencil-oriented CNN pipelines. Numerical behavior matches the
 * naive version; this variant adds OpenMP parallel execution.
 *
 * Input:  [N × C × H × W]
 * Output: [N × out_channels × H_out × W_out], with
 *   H_out = (H + 2 * padding - kernel_size) / stride + 1
 *   W_out = (W + 2 * padding - kernel_size) / stride + 1
 *
 * Note: current implementation assumes a single input channel in the compute path.
 */

#include <vector>

/// 3D tensor: [channels][height][width]
using Tensor3D = std::vector<std::vector<std::vector<float>>>;

/// 4D tensor: [batch][channels][height][width]
using Tensor4D = std::vector<Tensor3D>;

/**
 * @class Conv2DStencil
 * @brief Explicit 3×3 convolution using a stencil-style kernel.
 */
class Conv2DStencil {
 public:
  /**
   * @brief Construct a Conv2DStencil layer.
   * @param in_channels  Number of input channels.
   * @param out_channels Number of output channels.
   * @param kernel_size  Must be 3.
   * @param stride       Must be 1.
   * @param padding      Must be 1.
   */
  Conv2DStencil(int in_channels, int out_channels, int kernel_size, int stride = 1,
                int padding = 1);

  /**
   * @brief Forward pass on a batch of inputs.
   * @param batch Input tensor [N × C × H × W].
   * @return Output tensor [N × out_channels × H_out × W_out].
   * @throws std::invalid_argument if channel count mismatches.
   */
  Tensor4D forward(const Tensor4D& batch) const;

 private:
  int in_channels_;   ///< Number of input channels.
  int out_channels_;  ///< Number of output channels.
  int kernel_size_;   ///< Fixed at 3.
  int stride_;        ///< Fixed at 1.
  int padding_;       ///< Fixed at 1.

  /// Layer weights: [out_channels][in_channels][kernel_size][kernel_size]
  std::vector<Tensor3D> weights_;
  /// Bias terms: [out_channels]
  std::vector<float> biases_;

  /**
   * @brief Forward pass on a single image.
   * @param input Tensor [C × H × W].
   * @return Output tensor [out_channels × H_out × W_out].
   */
  Tensor3D forwardSingle(const Tensor3D& input) const;
};
