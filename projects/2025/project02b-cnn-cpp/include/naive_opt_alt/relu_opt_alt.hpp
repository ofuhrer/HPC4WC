#pragma once

/**
 * @file relu_opt_alt.hpp
 * @brief Optimized naive ReLU layer with OpenMP parallelization.
 *
 * Applies the element wise nonlinearity:
 *   ReLU(x) = max(0, x)
 * Input and output tensors preserve the same shape.
 */

#include <vector>

/// 3D tensor: [channels][height][width]
using Tensor3D = std::vector<std::vector<std::vector<float>>>;
/// 4D tensor: [batch][channels][height][width]
using Tensor4D = std::vector<Tensor3D>;
/// 2D tensor: [batch][features] (used after flattening)
using Tensor2D = std::vector<std::vector<float>>;

class ReLUOptAlt {
public:
  /**
   * @brief Forward pass on a 4D input tensor [N × C × H × W].
   * @param input Input tensor.
   * @return Tensor of the same shape with ReLU applied element wise.
   */
  Tensor4D forward(const Tensor4D& input) const;

  /**
   * @brief Forward pass on a 2D input tensor [N × F].
   * @param input Input tensor.
   * @return Tensor of the same shape with ReLU applied element wise.
   */
  Tensor2D forward(const Tensor2D& input) const;
};
