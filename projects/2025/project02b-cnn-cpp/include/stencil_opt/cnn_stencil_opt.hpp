#pragma once
// cnn_stencil_opt.hpp
/**
 * @file cnn_stencil_opt.hpp
 * @brief CNN composed of stencil conv layers with minor safe optimizations.
 *
 * Architecture for 28x28 grayscale:
 *   conv1: 1→8, 3x3, s=1, p=1  → 8x28x28
 *   ReLU
 *   pool1: 2x2, s=2            → 8x14x14
 *   conv2: 8→16, 3x3, s=1, p=1 → 16x14x14
 *   ReLU
 *   pool2: 2x2, s=2            → 16x7x7
 *   flatten → 16*7*7=784
 *   fc: 784→num_classes
 *
 * Behavior and outputs are identical to the existing stencil path.
 */

#include "tensors.hpp"
#include "conv2d_stencil_opt.hpp"
#include "relu_stencil_opt.hpp"
#include "maxpool_stencil_opt.hpp"
#include "linear_stencil_opt.hpp"

class CNNStencilOpt {
public:
  /**
   * @brief Construct the CNN using stencil optimized layers.
   * @param in_channels Number of input channels, for MNIST use 1.
   * @param num_classes Number of output classes, for MNIST use 10.
   */
  CNNStencilOpt(int in_channels = 1, int num_classes = 10);

  /**
   * @brief Forward pass on a batch.
   * @param batch Input tensor [N × C × H × W].
   * @return Output tensor [N × 1 × 1 × num_classes].
   */
  Tensor4D forward(const Tensor4D& batch) const;

private:
  // Stencil path layers
  Conv2DStencilOpt    conv1_;    // 1→8, 3x3, s=1, p=1
  ReLUStencilOpt      relu1_;
  MaxPool2DStencilOpt pool1_;    // 2x2, s=2

  Conv2DStencilOpt    conv2_;    // 8→16, 3x3, s=1, p=1
  ReLUStencilOpt      relu2_;
  MaxPool2DStencilOpt pool2_;    // 2x2, s=2

  LinearStencilOpt    fc_;       // 16*7*7 → num_classes
};
