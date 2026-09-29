#pragma once
/**
 * @file linear_stencil_alt.hpp
 * @brief Fully connected (linear) layer with stencil-style naming (OpenMP-optimized).
 *
 * Maps a batch of flat inputs to outputs via y = W x + b.
 * Shapes: input [N × in_features], output [N × out_features].
 */

#include <stdexcept>
#include <vector>

/// 1D tensor: [features]
using Tensor1D = std::vector<float>;

/// 2D tensor: [batch][features]
using Tensor2D = std::vector<Tensor1D>;

/**
 * @class LinearStencil
 * @brief Dense layer mapping [N × in_features] → [N × out_features].
 */
class LinearStencil {
 public:
  /**
   * @brief Construct a linear layer.
   * @param in_features Number of input features.
   * @param out_features Number of output features.
   */
  LinearStencil(int in_features, int out_features);

  /**
   * @brief Forward pass on a batch.
   * @param input Tensor of shape [N × in_features].
   * @return Tensor of shape [N × out_features].
   * @throws std::invalid_argument if input feature size mismatches.
   */
  Tensor2D forward(const Tensor2D& input) const;

 private:
  int in_features_;
  int out_features_;

  // Weights: [out_features][in_features]
  std::vector<std::vector<float>> weights_;
  // Biases: [out_features]
  std::vector<float> biases_;
};
