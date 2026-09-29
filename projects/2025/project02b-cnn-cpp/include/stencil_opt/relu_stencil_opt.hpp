#pragma once
// relu_stencil_opt.hpp
/**
 * @file relu_stencil_opt.hpp
 * @brief ReLU activation for the stencil-optimized CNN path.
 *
 * Provides elementwise ReLU on 2D and 4D tensors. Shapes are preserved.
 * Parallelized with OpenMP across batch/channels/rows and SIMD on innermost
 * dimensions. Behavior is identical to the naive version.
 */


#include "tensors.hpp"

class ReLUStencilOpt {
public:
  /**
   * @brief Apply ReLU elementwise on a 4D tensor.
   * @param input [N×C×H×W]
   * @return Tensor of same shape with ReLU applied.
   *
   * Preconditions (unchecked in Release): if N>0 then C,H,W>0 and rectangular shapes.
   */
  Tensor4D forward(const Tensor4D& input) const;

  /**
   * @brief Apply ReLU elementwise on a 2D tensor.
   * @param input [N×F]
   * @return Tensor of same shape with ReLU applied.
   *
   * Preconditions (unchecked in Release): if N>0 then F>0 and rectangular shapes.
   */
  Tensor2D forward(const Tensor2D& input) const;
};
