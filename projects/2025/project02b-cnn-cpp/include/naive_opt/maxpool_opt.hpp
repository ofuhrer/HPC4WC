#pragma once

/**
 * @file maxpool_opt.hpp
 * @brief Optimized 2D max pooling layer with OpenMP parallelism.
 *
 * Applies channel wise max pooling with a fixed kernel size and stride.
 * Parallelizes across batch and across (channel, row) in the .cpp file,
 * and uses omp simd for the innermost pooling window. Matches naive
 * semantics without padding.
 *
 * Shapes:
 *   Input:  [N][C][H][W]
 *   Output: [N][C][H_out][W_out]
 *
 * Output size formulas:
 *   H_out = (H - K) / S + 1
 *   W_out = (W - K) / S + 1
 */

#include "tensors.hpp"

class MaxPool2DOpt {
public:
  /**
   * @brief Construct a 2D max pooling layer.
   * @param kernel_size Size of the pooling window (K).
   * @param stride Stride length (S).
   */
  MaxPool2DOpt(int kernel_size, int stride);

  /**
   * @brief Forward pass on a batch of inputs.
   * @param input Input tensor [N][C][H][W].
   * @return Output tensor [N][C][H_out][W_out].
   */
  Tensor4D forward(const Tensor4D& input) const;

private:
  /**
   * @brief Forward pass on a single input tensor.
   * @param input [C][H][W].
   * @return Output [C][H_out][W_out].
   */
  Tensor3D forward_single(const Tensor3D& input) const;

  int kernel_size_;
  int stride_;
};
