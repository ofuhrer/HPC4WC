// linear_stencil_opt.cpp
/**
 * @file linear_stencil_opt.cpp
 * @brief Implementation of LinearStencilOpt (fully connected layer).
 *
 * Computes y = xW^T + b for each sample independently.
 * Parallelism: outer loops over (batch, out_feature) with collapse(2);
 * SIMD: inner accumulation over input features with reduction(+ : sum).
 * No I/O or allocations in hot inner loops; behavior identical to the naive version.
 */

#include <random>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "linear_stencil_opt.hpp"

LinearStencilOpt::LinearStencilOpt(int in_features, int out_features)
    : in_features_(in_features), out_features_(out_features) {
  if (in_features_ <= 0 || out_features_ <= 0) {
    throw std::invalid_argument("LinearStencilOpt: features must be positive");
  }

  // He initialization: N(0, sqrt(2 / in_features))
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<float> dist(0.0f,
      std::sqrt(2.0f / static_cast<float>(in_features_)));

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

Tensor2D LinearStencilOpt::forward(const Tensor2D& input) const {
  const int B = static_cast<int>(input.size());
  if (B == 0) return {};

  if (static_cast<int>(input[0].size()) != in_features_) {
    throw std::invalid_argument("LinearStencilOpt: input feature size mismatch");
  }

  Tensor2D output(static_cast<std::size_t>(B),
                  std::vector<float>(static_cast<std::size_t>(out_features_), 0.0f));

  // Parallelize across (batch, out_features). Innermost accumulation is unit-stride over In.
#ifdef _OPENMP
#pragma omp parallel for collapse(2) schedule(static) default(none)                  \
    shared(input, output, B, in_features_, out_features_, weights_, biases_)        \
    if (B * out_features_ > 1)
#endif
  for (int b = 0; b < B; ++b) {
    for (int o = 0; o < out_features_; ++o) {
      const auto& x_row = input[static_cast<std::size_t>(b)];
      const auto& w_row = weights_[static_cast<std::size_t>(o)];

      float sum = biases_[static_cast<std::size_t>(o)];

      // Inner product over input features (unit-stride). SIMD-friendly reduction.
#ifdef _OPENMP
#pragma omp simd reduction(+ : sum)
#endif
      for (int i = 0; i < in_features_; ++i) {
        sum += x_row[static_cast<std::size_t>(i)] * w_row[static_cast<std::size_t>(i)];
      }

      output[static_cast<std::size_t>(b)][static_cast<std::size_t>(o)] = sum;
    }
  }

  return output;
}
