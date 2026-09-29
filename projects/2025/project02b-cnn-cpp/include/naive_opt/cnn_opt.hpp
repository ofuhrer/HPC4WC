#pragma once

/**
 * @file cnn_opt.hpp
 * @brief Optimized two layer CNN for MNIST using the optimized sublayers.
 *
 * Architecture for 28x28 grayscale:
 *   conv1: 1→8, 3x3, s=1, p=1  → 8x28x28
 *   ReLU
 *   pool1: 2x2, s=2            → 8x14x14
 *   conv2: 8→16, 3x3, s=1, p=1 → 16x14x14
 *   ReLU
 *   pool2: 2x2, s=2            → 16x7x7
 *   flatten → 16*7*7 = 784
 *   fc: 784 → num_classes
 */

#include "tensors.hpp"
#include "conv2d_opt.hpp"
#include "relu_opt.hpp"
#include "maxpool_opt.hpp"
#include "linear_opt.hpp"

class CNNOpt {
public:
  /// @param in_channels Number of input feature maps (MNIST: 1)
  /// @param num_classes Number of output classes (MNIST: 10)
  CNNOpt(int in_channels = 1, int num_classes = 10);

  /**
   * @brief Forward pass on a batch of inputs.
   * @param batch Input tensor [N][C][H][W].
   * @return Output tensor [N][1][1][num_classes].
   */
  Tensor4D forward(const Tensor4D& batch) const;

private:
  // Optimized layers
  Conv2DOpt    conv1_;    // 1→8, 3x3, stride=1, pad=1
  ReLUOpt      relu1_;
  MaxPool2DOpt pool1_;    // 2x2, stride=2

  Conv2DOpt    conv2_;    // 8→16, 3x3, stride=1, pad=1
  ReLUOpt      relu2_;
  MaxPool2DOpt pool2_;    // 2x2, stride=2

  LinearOpt    fc_;       // 16*7*7 → num_classes
};
