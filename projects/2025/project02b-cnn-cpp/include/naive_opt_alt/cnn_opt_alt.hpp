#pragma once

/**
 * @file cnn_opt_alt.hpp
 * @brief Alternate optimized CNN for MNIST using optimized sublayers.
 *
 * Architecture for 28×28 grayscale inputs:
 *   conv1: 1→8, 3×3, s=1, p=1   → 8×28×28
 *   relu1
 *   pool1: 2×2, s=2             → 8×14×14
 *   conv2: 8→16, 3×3, s=1, p=1  → 16×14×14
 *   relu2
 *   pool2: 2×2, s=2             → 16×7×7
 *   flatten → 16*7*7 = 784
 *   fc: 784 → num_classes
 *
 * Forward output:
 *   [N × 1 × 1 × num_classes], where N is the batch size.
 */

#include "conv2d_opt_alt.hpp"
#include "linear_opt_alt.hpp"
#include "maxpool_opt_alt.hpp"
#include "relu_opt_alt.hpp"

class CNNOptAlt {
public:
  /**
   * @brief Construct the CNN with MNIST-oriented architecture.
   * @param in_channels Number of input channels (default 1).
   * @param num_classes Number of output classes (default 10).
   */
  CNNOptAlt(int in_channels = 1, int num_classes = 10);

  /**
   * @brief Forward pass on a batch of images.
   * @param batch Input tensor [N × C × H × W].
   * @return Output tensor [N × 1 × 1 × num_classes].
   * @throws std::invalid_argument if internal layer shapes mismatch.
   */
  Tensor4D forward(const Tensor4D& batch) const;

private:
  // Layer sequence
  Conv2DOptAlt    conv1_;    ///< 1→8, 3×3, s=1, p=1
  ReLUOptAlt      relu1_;    ///< elementwise ReLU
  MaxPool2DOptAlt pool1_;    ///< 2×2, s=2

  Conv2DOptAlt    conv2_;    ///< 8→16, 3×3, s=1, p=1
  ReLUOptAlt      relu2_;
  MaxPool2DOptAlt pool2_;    ///< 2×2, s=2

  LinearOptAlt    fc_;       ///< 16*7*7 → num_classes
};
