#pragma once

/**
 * @file relu_opt.hpp
 * @brief Optimized ReLU activation layer with OpenMP parallelism.
 *
 * Applies ReLU element wise (ReLU(x) = max(0, x)) on 4D image tensors
 * and 2D feature tensors. Parallelizes across batch and outer loops
 * in the .cpp, with an omp simd hint for unit stride inner loops.
 * Matches naive behavior but avoids unnecessary deep copies.
 */

#include "tensors.hpp"

class ReLUOpt {
public:
  /**
   * @brief Forward pass on a 4D tensor.
   * @param input Input tensor [N × C × H × W].
   * @return Output tensor of same shape with ReLU applied.
   */
  Tensor4D forward(const Tensor4D& input) const;

  /**
   * @brief Forward pass on a 2D tensor.
   * @param input Input tensor [N × F].
   * @return Output tensor of same shape with ReLU applied.
   */
  Tensor2D forward(const Tensor2D& input) const;
};
