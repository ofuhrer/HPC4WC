#pragma once

/**
 * @file maxpool_opt_alt.hpp
 * @brief Alternate optimized 2D max pooling layer with OpenMP.
 *
 * Applies max pooling independently per channel using a fixed kernel
 * size and stride. Parallelizes across the batch in the .cpp and adds
 * an omp simd hint for the innermost pooling loop. Behavior matches
 * the naive version, with no padding supported.
 *
 * Shapes:
 *   Input:  [N × C × H × W]
 *   Output: [N × C × H_out × W_out]
 *
 * Output size:
 *   H_out = (H - kernel_size) / stride + 1
 *   W_out = (W - kernel_size) / stride + 1
 */

#include <vector>

/// 3D tensor: [channels][height][width]
using Tensor3D = std::vector<std::vector<std::vector<float>>>;
/// 4D tensor: [batch][channels][height][width]
using Tensor4D = std::vector<Tensor3D>;

class MaxPool2DOptAlt {
public:
  /**
   * @brief Construct a max pooling layer.
   * @param kernel_size Window size (default 2).
   * @param stride      Step size (default 2).
   */
  MaxPool2DOptAlt(int kernel_size = 2, int stride = 2);

  /**
   * @brief Forward pass on a batch of inputs.
   * @param input Input tensor [N × C × H × W].
   * @return Output tensor [N × C × H_out × W_out].
   * @throws std::invalid_argument if H or W are not compatible with stride.
   */
  Tensor4D forward(const Tensor4D& input) const;

private:
  int kernel_size_;
  int stride_;

  /// @brief Forward pass on a single sample [C × H × W].
  Tensor3D forward_single(const Tensor3D& input) const;
};
