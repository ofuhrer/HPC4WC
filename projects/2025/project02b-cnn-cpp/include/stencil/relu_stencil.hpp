#pragma once
/**
 * @file relu_stencil.hpp
 * @brief ReLU activation layer with stencil-style naming.
 *
 * Provides element-wise application of the ReLU non-linearity:
 *   ReLU(x) = max(0, x)
 * This version preserves the behavior of the naive implementation
 * but is grouped under the stencil namespace for clarity.
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
 * @brief Applies ReLU activation element-wise to tensors.
 *
 * Usage:
 * @code
 * ReLUStencil relu;
 * Tensor4D output = relu.forward(input4d);
 * Tensor2D flat_out = relu.forward(input2d);
 * @endcode
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
