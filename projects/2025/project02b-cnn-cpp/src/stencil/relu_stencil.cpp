/**
 * @file relu_stencil.cpp
 * @brief Implementation of ReLUStencil forward passes.
 *
 * Applies ReLU element-wise to 2D and 4D tensors. Behavior and
 * performance are identical to the naive version.
 */

#include "relu_stencil.hpp"

#include <algorithm>  // for std::max

Tensor4D ReLUStencil::forward(const Tensor4D& input) const {
  Tensor4D output = input;  // copy shape and values

  // Iterate [N][C][H][W] and apply ReLU element-wise
  for (auto& image : output) {
    for (auto& channel : image) {
      for (auto& row : channel) {
        for (float& val : row) {
          val = std::max(0.0f, val);
        }
      }
    }
  }
  return output;
}

Tensor2D ReLUStencil::forward(const Tensor2D& input) const {
  Tensor2D output = input;  // copy shape and values

  // Iterate [N][F] and apply ReLU element-wise
  for (auto& vec : output) {
    for (float& val : vec) {
      val = std::max(0.0f, val);
    }
  }
  return output;
}
