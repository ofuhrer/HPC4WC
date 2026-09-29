/**
 * @file linear_stencil.cpp
 * @brief Implementation of LinearStencil.
 *
 * Initializes weights with He initialization and performs
 * dense forward passes on batches of flat inputs.
 */

#include "linear_stencil.hpp"

#include <cmath>
#include <random>
#include <stdexcept>

LinearStencil::LinearStencil(int in_features, int out_features)
    : in_features_(in_features), out_features_(out_features) {
  // He initialization: N(0, sqrt(2 / in_features))
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_features_));

  weights_.resize(out_features_);
  for (int o = 0; o < out_features_; ++o) {
    weights_[o].resize(in_features_);
    for (int i = 0; i < in_features_; ++i) {
      weights_[o][i] = dist(gen);
    }
  }

  biases_.assign(out_features_, 0.0f);
}

Tensor2D LinearStencil::forward(const Tensor2D& input) const {
  int batch_size = static_cast<int>(input.size());
  if (batch_size == 0) return {};

  if (static_cast<int>(input[0].size()) != in_features_) {
    throw std::invalid_argument("LinearStencil: input feature size mismatch");
  }

  Tensor2D output(batch_size, std::vector<float>(out_features_, 0.0f));

  // For each sample in batch: output[b] = W * input[b] + b
  for (int b = 0; b < batch_size; ++b) {
    for (int o = 0; o < out_features_; ++o) {
      float sum = biases_[o];
      for (int i = 0; i < in_features_; ++i) {
        sum += input[b][i] * weights_[o][i];
      }
      output[b][o] = sum;
    }
  }

  return output;
}
