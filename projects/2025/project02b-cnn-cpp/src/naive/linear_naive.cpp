// linear_naive.cpp
/**
 * @file linear_naive.cpp
 * @brief Naive fully connected (linear) layer with He initialization.
 *
 * Applies an affine transform Wx + b to each sample in the batch using
 * nested loops over input features and output units. Weights are drawn
 * from a normal distribution with variance scaling. Execution is single
 * threaded and validates input feature size. No algorithmic changes.
 */

#include <cmath>       // std::sqrt
#include <random>      // std::random_device, std::mt19937, std::normal_distribution
#include <stdexcept>   // std::invalid_argument
#include "linear_naive.hpp"

LinearNaive::LinearNaive(int in_features, int out_features)
    : in_features_(in_features),
      out_features_(out_features) {
    // He initialization: weights ~ N(0, sqrt(2 / in_features)).
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / in_features_));

    weights_.resize(out_features_);
    for (int i = 0; i < out_features_; ++i) {
        weights_[i].resize(in_features_);
        for (int j = 0; j < in_features_; ++j) {
            weights_[i][j] = dist(gen);
        }
    }

    biases_.assign(out_features_, 0.0f);
}

Tensor2D LinearNaive::forward(const Tensor2D& input) const {
    const int batch_size = static_cast<int>(input.size());
    if (batch_size == 0) {
        return {};
    }

    if (static_cast<int>(input[0].size()) != in_features_) {
        throw std::invalid_argument("LinearNaive: input feature size mismatch");
    }

    Tensor2D output(batch_size, std::vector<float>(out_features_, 0.0f));

    // For each sample in the batch, apply affine transform: Wx + b.
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
