// linear_opt_alt.cpp
/**
 * @file linear_opt_alt.cpp
 * @brief Optimized fully connected (alternate) with batch×output parallelism.
 *
 * Uses He initialization and applies Wx + b per sample. Parallelizes across
 * (batch, output) with OpenMP collapse and adds an omp simd reduction over
 * input features for the inner accumulation. Matches the naive layer’s
 * behavior with no algorithmic changes.
 */

#include "linear_opt_alt.hpp"

#include <cmath>        // std::sqrt
#include <cstddef>      // std::ptrdiff_t
#include <random>       // RNG & normal_distribution
#include <stdexcept>    // std::invalid_argument

#ifdef _OPENMP
#include <omp.h>
#endif

LinearOptAlt::LinearOptAlt(int in_features, int out_features)
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

Tensor2D LinearOptAlt::forward(const Tensor2D& input) const {
  const std::ptrdiff_t batch_size = static_cast<std::ptrdiff_t>(input.size());
  if (batch_size == 0) return {};

  if (static_cast<int>(input[0].size()) != in_features_) {
    throw std::invalid_argument("LinearOptAlt: input feature size mismatch");
  }

  Tensor2D output(batch_size, std::vector<float>(out_features_, 0.0f));

  // Each (b, o) pair is independent; innermost loop over i is unit-stride on both inputs.
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) if (batch_size * out_features_ > 1) \
    default(none) shared(input, output, batch_size, in_features_, out_features_, weights_, biases_)
#endif
  for (std::ptrdiff_t b = 0; b < batch_size; ++b) {
    for (std::ptrdiff_t o = 0; o < static_cast<std::ptrdiff_t>(out_features_); ++o) {
      float sum = biases_[static_cast<int>(o)];

#ifdef _OPENMP
#pragma omp simd reduction(+ : sum)
#endif
      for (std::ptrdiff_t i = 0; i < static_cast<std::ptrdiff_t>(in_features_); ++i) {
        sum += input[b][static_cast<std::size_t>(i)] *
               weights_[static_cast<std::size_t>(o)][static_cast<std::size_t>(i)];
      }
      output[b][static_cast<std::size_t>(o)] = sum;
    }
  }

  return output;
}
