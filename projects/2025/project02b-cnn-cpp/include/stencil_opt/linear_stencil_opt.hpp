#pragma once
// linear_stencil_opt.hpp
/**
 * @file linear_stencil_opt.hpp
 * @brief Fully connected (linear) layer for the stencil-optimized pipeline.
 *
 * Input:  [N × In]
 * Output: [N × Out]
 * Behavior matches the naive stencil version; only minor OpenMP/SIMD hygiene applied.
 */

#include "tensors.hpp"
#include <stdexcept>

class LinearStencilOpt {
public:
  /**
   * @brief Construct a linear layer with given input/output feature sizes.
   * @param in_features  Number of input features per sample.
   * @param out_features Number of output features per sample.
   * @throws std::invalid_argument if any size is non-positive.
   */
  LinearStencilOpt(int in_features, int out_features);

  /**
   * @brief Forward pass on a batch.
   * @param input [N × In]
   * @return Output tensor [N × Out]
   * @throws std::invalid_argument if input feature size mismatches `in_features_`.
   */
  Tensor2D forward(const Tensor2D& input) const;

private:
  int in_features_;
  int out_features_;
  /// Weight matrix stored as [Out][In]
  std::vector<std::vector<float>> weights_;
  /// Bias vector stored as [Out]
  std::vector<float> biases_;
};
