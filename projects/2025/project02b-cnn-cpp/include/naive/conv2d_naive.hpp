#pragma once

#include <vector>

/**
 * @file conv2d_naive.hpp
 * @brief Naive 2D convolution layer using nested std::vector tensors.
 *
 * Implements a direct cross-correlation (commonly called convolution in DL) with
 * configurable in/out channels, kernel size, stride, and padding. No parallelism
 * or cache optimizations are applied.
 *
 * Shapes:
 *   Input (single): [C_in × H × W]
 *   Input (batch):  [N × C_in × H × W]
 *   Weights:        [C_out × C_in × K × K]
 *   Bias:           [C_out]
 *   Output (single): [C_out × H_out × W_out]
 *   Output (batch):  [N × C_out × H_out × W_out]
 *
 * Output size formulas:
 *   H_out = (H + 2*padding - K) / stride + 1
 *   W_out = (W + 2*padding - K) / stride + 1
 *
 * Requirements:
 *   - Input channels must match constructor in_channels.
 *   - (H + 2*padding - K) and (W + 2*padding - K) must be divisible by stride.
 *   - Throws std::invalid_argument on mismatched shapes or invalid sizes.
 */

// 3D tensor: [channels][height][width]
using Tensor3D = std::vector<std::vector<std::vector<float>>>;
// 4D tensor: [batch][channels][height][width]
using Tensor4D = std::vector<Tensor3D>;

/// Naive 2D convolution supporting arbitrary batch sizes.
class Conv2DNaive {
public:
    /**
     * @brief Construct a naive Conv2D layer.
     * @param in_channels  Number of input channels.
     * @param out_channels Number of output channels.
     * @param kernel_size  Spatial size of the kernel (square).
     * @param stride       Stride between applications of the kernel.
     * @param padding      Number of zero padding pixels on each side.
     */
    Conv2DNaive(int in_channels,
                int out_channels,
                int kernel_size,
                int stride = 1,
                int padding = 0);

    /**
     * @brief Forward pass on a batch of inputs.
     * @param batch Input tensor [N × C × H × W].
     * @return Output tensor [N × C_out × H_out × W_out].
     * @throws std::invalid_argument if input channels mismatch or dimensions invalid.
     */
    Tensor4D forward(const Tensor4D& batch) const;

private:
    int in_channels_;
    int out_channels_;
    int kernel_size_;
    int stride_;
    int padding_;

    // Weights: [out_channels][in_channels][kernel_size][kernel_size]
    std::vector<Tensor3D> weights_;
    // Bias: [out_channels]
    std::vector<float> biases_;

    /**
     * @brief Forward pass on a single input tensor.
     * @param input [C × H × W].
     * @return Output [C_out × H_out × W_out].
     */
    Tensor3D forwardSingle(const Tensor3D& input) const;
};
