#pragma once
// maxpool_stencil_opt.hpp
/**
 * @file maxpool_stencil_opt.hpp
 * @brief 2D MaxPooling for the stencil-optimized CNN pipeline.
 *
 * Provides configurable kernel size and stride (defaults: 2×2, stride 2).
 * Operates on 4D input [N×C×H×W] with output [N×C×H_out×W_out].
 * Behavior and outputs are identical to the naive version.
 */


#include "tensors.hpp"
#include <stdexcept>

class MaxPool2DStencilOpt {
public:
  /**
   * @brief Construct a MaxPool2DStencilOpt layer.
   * @param kernel_size Side length of the pooling window (default 2).
   * @param stride Step size of the pooling window (default 2).
   */
  MaxPool2DStencilOpt(int kernel_size = 2, int stride = 2);

  /**
   * @brief Forward pass on a batch of images.
   * @param input Input tensor [N×C×H×W].
   * @return Output tensor [N×C×H_out×W_out].
   */
  Tensor4D forward(const Tensor4D& input) const;

private:
  int kernel_size_;
  int stride_;

  /**
   * @brief Forward pass on a single image.
   * @param input Input tensor [C×H×W].
   * @return Output tensor [C×H_out×W_out].
   */
  Tensor3D forwardSingle(const Tensor3D& input) const;
};
