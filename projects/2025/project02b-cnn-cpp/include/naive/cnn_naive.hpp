#pragma once

/**
 * @file cnn_naive.hpp
 * @brief Naive two-block CNN for MNIST with Conv→ReLU→Pool stacks and a final FC layer.
 *
 * Architecture for 28×28 grayscale inputs:
 *   conv1: 1→8, 3×3, s=1, p=1     →  8×28×28
 *   relu1
 *   pool1: 2×2, s=2               →  8×14×14
 *   conv2: 8→16, 3×3, s=1, p=1    → 16×14×14
 *   relu2
 *   pool2: 2×2, s=2               → 16×7×7
 *   flatten                       → 16*7*7 = 784
 *   fc: 784 → num_classes
 *
 * Forward output:
 *   [N × 1 × 1 × num_classes], where N is the batch size.
 */

#include "conv2d_naive.hpp"
#include "relu_naive.hpp"
#include "maxpool_naive.hpp"
#include "linear_naive.hpp"

/// Naive 2-layer convolutional network for MNIST.
class CNNNaive {
public:
    /**
     * @brief Construct the CNN with MNIST-oriented architecture.
     * @param in_channels Number of input channels (default: 1).
     * @param num_classes Number of output classes (default: 10).
     */
    CNNNaive(int in_channels = 1, int num_classes = 10);

    /**
     * @brief Forward pass through the network.
     * @param batch Input tensor [N × C × H × W].
     * @return Output tensor [N × 1 × 1 × num_classes].
     * @throws std::invalid_argument if shapes mismatch (propagated from sublayers).
     */
    Tensor4D forward(const Tensor4D& batch) const;

private:
    // Layer sequence
    Conv2DNaive    conv1_;    ///< 1→8, 3×3, s=1, p=1
    ReLUNaive      relu1_;    ///< elementwise ReLU
    MaxPool2DNaive pool1_;    ///< 2×2, s=2

    Conv2DNaive    conv2_;    ///< 8→16, 3×3, s=1, p=1
    ReLUNaive      relu2_;
    MaxPool2DNaive pool2_;    ///< 2×2, s=2

    LinearNaive    fc_;       ///< 16*7*7 → num_classes
};
