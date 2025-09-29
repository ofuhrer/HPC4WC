#pragma once
/**
 * @file linear_stencil.hpp
 * @brief Fully connected (linear) layer with stencil-style implementation.
 *
 * Provides a dense matrix multiplication with bias:
 *   y = Wx + b
 * Weights are initialized with He initialization.
 *
 * Usage:
 * @code
 * LinearStencil layer(in_features, out_features);
 * Tensor2D out = layer.forward(batch_input);
 * @endcode
 */

#include <vector>

/// 1D tensor: [features]
using Tensor1D = std::vector<float>;

/// 2D tensor: [batch][features]
using Tensor2D = std::vector<Tensor1D>;

/**
 * @class LinearStencil
 * @brief Fully connected layer mapping [N × in_features] → [N × out_features].
 */
class LinearStencil {
 public:
  /**
   * @brief Construct a linear layer with given dimensions.
   * @param in_features Number of input features.
   * @param out_features Number of output features.
   */
  LinearStencil(int in_features, int out_features);

  /**
   * @brief Forward pass on a batch of inputs.
   * @param input Tensor of shape [N × in_features].
   * @return Tensor of shape [N × out_features].
   * @throws std::invalid_argument if input feature size mismatches.
   */
  Tensor2D forward(const Tensor2D& input) const;

 private:
  int in_features_;
  int out_features_;

  // Weight matrix: [out_features][in_features]
  std::vector<std::vector<float>> weights_;

  // Bias vector: [out_features]
  std::vector<float> biases_;
};
