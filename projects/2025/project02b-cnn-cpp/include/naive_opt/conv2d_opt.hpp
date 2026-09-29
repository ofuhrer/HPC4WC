#pragma once

/**
 * @file conv2d_opt.hpp
 * @brief 2D convolution (batch-parallel) with cache-friendly loop ordering.
 *
 * Computes a direct cross-correlation with optional zero padding and stride.
 * The implementation parallelizes across the batch in the .cpp and preserves
 * the naive layer’s numerics.
 *
 * Shapes:
 *   input   [N][C_in][H_in][W_in]
 *   weights [C_out][C_in][K][K]
 *   bias    [C_out]
 *   output  [N][C_out][H_out][W_out]
 *
 * Output size formulas:
 *   H_out = (H_in + 2*P - K) / S + 1
 *   W_out = (W_in + 2*P - K) / S + 1
 */

#include "tensors.hpp"

class Conv2DOpt {
public:
  /**
   * @brief Construct a 2D convolution layer.
   * @param in_channels  Number of input channels.
   * @param out_channels Number of output channels.
   * @param kernel_size  Spatial kernel size (square).
   * @param stride       Stride between applications (default 1).
   * @param padding      Zero padding on each side (default 0).
   */
  Conv2DOpt(int in_channels, int out_channels, int kernel_size,
            int stride = 1, int padding = 0);

  /**
   * @brief Forward pass over a batch of inputs.
   * @param batch Input tensor [N][C_in][H_in][W_in].
   * @return Output tensor [N][C_out][H_out][W_out].
   */
  Tensor4D forward(const Tensor4D& batch) const;

private:
  int in_channels_;
  int out_channels_;
  int kernel_size_;
  int stride_;
  int padding_;

  // weights_[oc][ic][k][k]
  std::vector<Tensor3D> weights_;
  // biases_[oc]
  std::vector<float> biases_;

  /**
   * @brief Forward pass for a single sample.
   * @param input Input tensor [C_in][H_in][W_in].
   * @return Output tensor [C_out][H_out][W_out].
   */
  Tensor3D forward_single(const Tensor3D& input) const;
};
