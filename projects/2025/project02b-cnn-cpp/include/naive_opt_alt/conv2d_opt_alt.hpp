#pragma once
/**
 * @file conv2d_opt_alt.hpp
 * @brief Optimized 2D convolution (alternate) with optional padding and stride.
 *
 * Computes a direct cross-correlation per output channel:
 *   out[oc, i, j] = bias[oc] + Σ_ic Σ_m Σ_n in[ic, i*S + m, j*S + n] * W[oc, ic, m, n]
 * with square kernels, no dilation, and no groups. The .cpp parallelizes across
 * the batch and adds an omp simd hint over kernel columns; numerics match naive.
 *
 * Shapes:
 *   Input:   [N × C_in × H × W]
 *   Weights: [C_out × C_in × K × K]
 *   Bias:    [C_out]
 *   Output:  [N × C_out × H_out × W_out]
 *
 * Output size:
 *   H_out = (H + 2*padding - K) / stride + 1
 *   W_out = (W + 2*padding - K) / stride + 1
 */

 #include <vector>

// 3D tensor: [channels][height][width]
using Tensor3D = std::vector<std::vector<std::vector<float>>>;
// 4D tensor: [batch][channels][height][width]
using Tensor4D = std::vector<Tensor3D>;

class Conv2DOptAlt {
public:
  /**
   * @brief Construct a 2D convolution layer.
   * @param in_channels  Number of input channels.
   * @param out_channels Number of output channels.
   * @param kernel_size  Square kernel size (K × K).
   * @param stride       Stride (default 1).
   * @param padding      Zero padding on all sides (default 0).
   */
  Conv2DOptAlt(int in_channels, int out_channels, int kernel_size, int stride = 1, int padding = 0);

  /**
   * @brief Forward pass on a batch of inputs.
   * @param batch 4D tensor [N × C_in × H × W].
   * @return 4D tensor [N × C_out × H_out × W_out].
   * @throws std::invalid_argument on shape mismatch or incompatible H/W with stride.
   */
  Tensor4D forward(const Tensor4D& batch) const;

private:
  int in_channels_;
  int out_channels_;
  int kernel_size_;
  int stride_;
  int padding_;

  // [out_channels][in_channels][kernel_size][kernel_size]
  std::vector<Tensor3D> weights_;
  // [out_channels]
  std::vector<float> biases_;

  /// @brief Forward pass on a single input [C_in × H × W].
  Tensor3D forward_single(const Tensor3D& input) const;
};
