#pragma once

/**
 * @file linear_opt.hpp
 * @brief Optimized fully connected layer with OpenMP parallelism.
 *
 * Applies an affine transform Wx + b per sample. Parallelizes across batch
 * and outputs in the .cpp file, with an omp simd reduction across input
 * features for the accumulation. Matches the naive layer numerics.
 *
 * Shapes:
 *   input   [B][In]
 *   weights [Out][In]
 *   bias    [Out]
 *   output  [B][Out]
 */

#include "tensors.hpp"

class LinearOpt {
public:
  /**
   * @brief Construct a fully connected layer.
   * @param in_features  Number of input features per sample.
   * @param out_features Number of output features.
   */
  LinearOpt(int in_features, int out_features);

  /**
   * @brief Forward pass on a batch of inputs.
   * @param input Input tensor [B][In].
   * @return Output tensor [B][Out].
   */
  Tensor2D forward(const Tensor2D& input) const;

private:
  int in_features_;
  int out_features_;
  // weights_[o][i]
  std::vector<std::vector<float>> weights_;
  // biases_[o]
  std::vector<float> biases_;
};
