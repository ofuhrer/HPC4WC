#pragma once
/**
 * @file maxpool_stencil_alt.hpp
 * @brief Max pooling layer with stencil-style fixed pattern (OpenMP-optimized).
 *
 * Typical use: kernel_size = 2, stride = 2, no padding.
 * Applies non-overlapping max-pooling windows.
 *
 * Input:  [C × H × W] or batch of [N × C × H × W]
 * Output: [C × H_out × W_out], where
 *   H_out = (H - kernel_size) / stride + 1
 *   W_out = (W - kernel_size) / stride + 1
 */

#include <stdexcept>
#include <vector>

/// 3D tensor: [channels][height][width]
using Tensor3D = std::vector<std::vector<std::vector<float>>>;

/// 4D tensor: [batch][channels][height][width]
using Tensor4D = std::vector<Tensor3D>;

class MaxPool2DStencil {
 public:
  /**
   * @brief Construct a max pooling layer.
   * @param kernel_size Pooling window size (e.g., 2).
   * @param stride Stride between windows (e.g., 2).
   */
  MaxPool2DStencil(int kernel_size = 2, int stride = 2);

  /**
   * @brief Forward pass on a batch of 4D input tensors [N × C × H × W].
   * @param input Input tensor.
   * @return Output tensor [N × C × H_out × W_out].
   * @throws std::invalid_argument if input dimensions are incompatible.
   */
  Tensor4D forward(const Tensor4D& input) const;

 private:
  int kernel_size_;
  int stride_;

  /**
   * @brief Forward pass on a single 3D tensor [C × H × W].
   * @param input Input tensor.
   * @return Output tensor [C × H_out × W_out].
   */
  Tensor3D forwardSingle(const Tensor3D& input) const;
};
