// linear_opt.cpp
/**
 * @file linear_opt.cpp
 * @brief Optimized fully connected layer with batch and output parallelism.
 *
 * Validates shapes, uses He initialization, and applies Wx + b per sample.
 * Parallelizes across batch and output units with OpenMP, with an omp simd
 * reduction over input features for the inner accumulation. Behavior matches
 * the naive version with no algorithm changes.
 */

#include <cmath>
#include <random>
#include <stdexcept>
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "linear_opt.hpp"

LinearOpt::LinearOpt(int in_features, int out_features)
    : in_features_(in_features), out_features_(out_features) {
  if (in_features_ <= 0 || out_features_ <= 0) {
    throw std::invalid_argument("LinearOpt: features must be positive");
  }

  // He initialization: N(0, sqrt(2 / in_features))
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(
      0.0f, std::sqrt(2.0f / static_cast<float>(in_features_)));

  weights_.assign(static_cast<std::size_t>(out_features_),
                  std::vector<float>(static_cast<std::size_t>(in_features_)));
  for (int o = 0; o < out_features_; ++o) {
    auto& wrow = weights_[static_cast<std::size_t>(o)];
    for (int i = 0; i < in_features_; ++i) {
      wrow[static_cast<std::size_t>(i)] = dist(gen);
    }
  }

  biases_.assign(static_cast<std::size_t>(out_features_), 0.0f);
}

Tensor2D LinearOpt::forward(const Tensor2D& input) const {
  const int B = static_cast<int>(input.size());
  if (B == 0) return {};

  if (static_cast<int>(input[0].size()) != in_features_) {
    throw std::invalid_argument("LinearOpt: input feature size mismatch");
  }

  Tensor2D output(static_cast<std::size_t>(B),
                  std::vector<float>(static_cast<std::size_t>(out_features_), 0.0f));

  // Parallelize across (b, o). SIMD on inner accumulation over i.
  const std::ptrdiff_t work_items =
      static_cast<std::ptrdiff_t>(B) * static_cast<std::ptrdiff_t>(out_features_);
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) default(none) \
    shared(input, output, weights_, biases_, B) if (work_items > 1)
#endif
  for (int b = 0; b < B; ++b) {
    for (int o = 0; o < out_features_; ++o) {
      const auto& x = input[static_cast<std::size_t>(b)];
      const auto& wrow = weights_[static_cast<std::size_t>(o)];
      float sum = biases_[static_cast<std::size_t>(o)];
#ifdef _OPENMP
#pragma omp simd reduction(+ : sum)
#endif
      for (int i = 0; i < in_features_; ++i) {
        sum += x[static_cast<std::size_t>(i)] * wrow[static_cast<std::size_t>(i)];
      }
      output[static_cast<std::size_t>(b)][static_cast<std::size_t>(o)] = sum;
    }
  }

  return output;
}
