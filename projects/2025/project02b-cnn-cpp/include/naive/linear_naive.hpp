#pragma once

/**
 * @file linear_naive.hpp
 * @brief Naive fully connected (dense) layer using std::vector tensors.
 *
 * Applies an affine transform Wx + b to each sample. Supports arbitrary batch size.
 *
 * Shapes:
 *   Input:   [N × in_features]
 *   Weights: [out_features × in_features]
 *   Bias:    [out_features]
 *   Output:  [N × out_features]
 *
 * Initialization:
 *   Weights ~ N(0, sqrt(2 / in_features)) (He initialization).
 *   Biases initialized to zero.
 *
 * Requirements:
 *   - Input feature dimension must equal in_features.
 *   - Throws std::invalid_argument if shape mismatches.
 */

#include <vector>

// 2D tensor: [batch][features]
using Tensor2D = std::vector<std::vector<float>>;

/// Naive fully connected (dense) layer with He initialization.
class LinearNaive {
public:
    /**
     * @brief Construct the fully connected layer.
     * @param in_features   Number of input features.
     * @param out_features  Number of output features.
     */
    LinearNaive(int in_features, int out_features);

    /**
     * @brief Forward pass on a batch of inputs.
     * @param input Input tensor [N × in_features].
     * @return Output tensor [N × out_features].
     * @throws std::invalid_argument if input feature dimension mismatches.
     */
    Tensor2D forward(const Tensor2D& input) const;

private:
    int in_features_;
    int out_features_;

    // Weights: [out_features][in_features]
    std::vector<std::vector<float>> weights_;
    // Bias: [out_features]
    std::vector<float> biases_;
};
