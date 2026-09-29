#pragma once

/**
 * @file relu_stencil_alt.hpp
 * @brief ReLU activation for the stencil path with OpenMP parallelism.
 *
 * Applies ReLU(x) = max(0, x) element wise on 4D image tensors and 2D feature
 * tensors. Behavior matches the naive and stencil versions; the .cpp only adds
 * parallel execution. Shapes are preserved.
 */

#include <vector>

/// 3D tensor: [channels][height][width]
using Tensor3D = std::vector<std::vector<std::vector<float>>>;

/// 4D tensor: [batch][channels][height][width]
using Tensor4D = std::vector<Tensor3D>;

/// 2D tensor: [batch][features] (used after flattening)
using Tensor2D = std::vector<std::vector<float>>;

/**
 * @class ReLUStencil
 * @brief Applies ReLU element-wise to tensors (OpenMP-parallel implementation).
 */
class ReLUStencil {
 public:
  /**
   * @brief Forward pass on a 4D input tensor [N × C × H × W].
   * @param input Input tensor.
   * @return Tensor of same shape with ReLU applied element-wise.
   */
  Tensor4D forward(const Tensor4D& input) const;

  /**
   * @brief Forward pass on a 2D input tensor [N × F].
   * @param input Input tensor.
   * @return Tensor of same shape with ReLU applied element-wise.
   */
  Tensor2D forward(const Tensor2D& input) const;
};
