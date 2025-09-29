#pragma once
/**
 * @file cnn_stencil.hpp
 * @brief Simple CNN with stencil-style convolution layers for MNIST.
 *
 * Architecture for 28×28 grayscale MNIST images:
 *   conv1: 1→8 channels, 3×3 kernel, stride=1, padding=1 → 8×28×28
 *   relu1
 *   pool1: 2×2, stride=2 → 8×14×14
 *   conv2: 8→16 channels, 3×3, stride=1, padding=1 → 16×14×14
 *   relu2
 *   pool2: 2×2, stride=2 → 16×7×7
 *   flatten → 16*7*7 = 784
 *   fc: 784 → num_classes
 */

#include "conv2d_stencil.hpp"
#include "linear_stencil.hpp"
#include "maxpool_stencil.hpp"
#include "relu_stencil.hpp"

/**
 * @class CNNStencil
 * @brief Compact CNN using stencil-style convolution.
 *
 * Input: [N × C × H × W]  
 * Output: [N × 1 × 1 × num_classes]
 */
class CNNStencil {
 public:
  /**
   * @brief Construct the CNN with fixed MNIST-style architecture.
   * @param in_channels Number of input channels (MNIST: 1).
   * @param num_classes Number of output classes (MNIST: 10).
   */
  CNNStencil(int in_channels = 1, int num_classes = 10);

  /**
   * @brief Forward pass through the CNN.
   * @param batch Input tensor [N × C × H × W].
   * @return Output tensor [N × 1 × 1 × num_classes].
   * @throws std::invalid_argument on shape mismatch.
   */
  Tensor4D forward(const Tensor4D& batch) const;

 private:
  // Layer definitions
  Conv2DStencil conv1_;     ///< 1→8 channels, 3×3, stride=1, pad=1
  ReLUStencil relu1_;
  MaxPool2DStencil pool1_;  ///< 2×2, stride=2

  Conv2DStencil conv2_;     ///< 8→16 channels, 3×3, stride=1, pad=1
  ReLUStencil relu2_;
  MaxPool2DStencil pool2_;  ///< 2×2, stride=2

  LinearStencil fc_;        ///< fully connected: 16*7*7 → num_classes
};
