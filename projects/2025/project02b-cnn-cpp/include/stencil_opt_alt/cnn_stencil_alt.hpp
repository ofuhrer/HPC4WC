#pragma once
/**
 * @file cnn_stencil_alt.hpp
 * @brief CNN for MNIST classification using stencil-style layers (OpenMP-optimized).
 *
 * Architecture (for 28×28 grayscale images):
 *   conv1: 1→8 channels, 3×3 kernel, stride=1, padding=1 → 8×28×28
 *   relu1
 *   pool1: 2×2, stride=2                               → 8×14×14
 *   conv2: 8→16 channels, 3×3, stride=1, padding=1     → 16×14×14
 *   relu2
 *   pool2: 2×2, stride=2                               → 16×7×7
 *   flatten                                            → 16*7*7 = 784
 *   fc: 784 → num_classes
 */

#include "conv2d_stencil_alt.hpp"
#include "relu_stencil_alt.hpp"
#include "maxpool_stencil_alt.hpp"
#include "linear_stencil_alt.hpp"

/**
 * @class CNNStencil
 * @brief Simple convolutional neural network built from stencil-style layers.
 */
class CNNStencil {
 public:
  /**
   * @brief Construct the CNN.
   * @param in_channels Number of input channels (e.g. MNIST: 1).
   * @param num_classes Number of output classes (e.g. MNIST: 10).
   */
  CNNStencil(int in_channels = 1, int num_classes = 10);

  /**
   * @brief Forward pass on a batch of images.
   * @param batch Input tensor [N × C × H × W].
   * @return Output tensor [N × 1 × 1 × num_classes].
   * @throws std::invalid_argument if input shapes are inconsistent.
   */
  Tensor4D forward(const Tensor4D& batch) const;

 private:
  // Layer definitions
  Conv2DStencil conv1_;     ///< 1→8 channels, 3×3 conv
  ReLUStencil relu1_;
  MaxPool2DStencil pool1_;  ///< 2×2 max pool, stride=2

  Conv2DStencil conv2_;     ///< 8→16 channels, 3×3 conv
  ReLUStencil relu2_;
  MaxPool2DStencil pool2_;  ///< 2×2 max pool, stride=2

  LinearStencil fc_;        ///< fully connected: 784 → num_classes
};
