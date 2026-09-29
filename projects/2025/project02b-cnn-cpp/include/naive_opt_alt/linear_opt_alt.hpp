#pragma once

/**
 * @file linear_opt_alt.hpp
 * @brief Alternate optimized fully connected layer with OpenMP parallelism.
 *
 * Applies an affine transform Wx + b per sample. Parallelizes across
 * (batch, output) pairs in the .cpp, with omp simd reduction across
 * input features. Numerics match the naive layer.
 *
 * Shapes:
 *   Input:   [N × in_features]
 *   Weights: [out_features × in_features]
 *   Bias:    [out_features]
 *   Output:  [N × out_features]
 *
 * Initialization:
 *   - Weights ~ N(0, sqrt(2 / in_features)) (He initialization).
 *   - Biases initialized to zero.
 */

#include <vector>

/// 2D tensor: [batch][features]
using Tensor2D = std::vector<std::vector<float>>;

class LinearOptAlt {
public:
  /**
   * @brief Construct the fully connected layer.
   * @param in_features  Number of input features.
   * @param out_features Number of output features.
   */
  LinearOptAlt(int in_features, int out_features);

  /**
   * @brief Forward pass on a batch of inputs.
   * @param input Input tensor [N × in_features].
   * @return Output tensor [N × out_features].
   * @throws std::invalid_argument if feature size mismatches.
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
