#pragma once

/**
 * @file maxpool_naive.hpp
 * @brief Naive 2D max pooling layer using std::vector tensors.
 *
 * Applies max pooling independently per channel, with a fixed kernel size and stride.
 * No padding is supported. Reduces spatial dimensions by taking the maximum value
 * within each pooling window.
 *
 * Shapes:
 *   Input (single): [C × H × W]
 *   Input (batch):  [N × C × H × W]
 *   Output (single): [C × H_out × W_out]
 *   Output (batch):  [N × C × H_out × W_out]
 *
 * Output size formulas:
 *   H_out = (H - kernel_size) / stride + 1
 *   W_out = (W - kernel_size) / stride + 1
 *
 * Requirements:
 *   - (H - kernel_size) and (W - kernel_size) must be divisible by stride.
 *   - Throws std::invalid_argument on invalid sizes.
 */

#include <vector>

// 3D tensor: [channels][height][width]
using Tensor3D = std::vector<std::vector<std::vector<float>>>;
// 4D tensor: [batch][channels][height][width]
using Tensor4D = std::vector<Tensor3D>;

/// Naive reference MaxPool 2D layer (no padding).
class MaxPool2DNaive {
public:
    /**
     * @brief Construct a max pooling layer.
     * @param kernel_size Window size (default 2).
     * @param stride      Step size (default 2).
     */
    MaxPool2DNaive(int kernel_size = 2, int stride = 2);

    /**
     * @brief Forward pass on a batch of inputs.
     * @param input Input tensor [N × C × H × W].
     * @return Output tensor [N × C × H_out × W_out].
     */
    Tensor4D forward(const Tensor4D& input) const;

private:
    int kernel_size_;
    int stride_;

    /**
     * @brief Forward pass on a single input tensor.
     * @param input [C × H × W].
     * @return Output [C × H_out × W_out].
     */
    Tensor3D forwardSingle(const Tensor3D& input) const;
};
