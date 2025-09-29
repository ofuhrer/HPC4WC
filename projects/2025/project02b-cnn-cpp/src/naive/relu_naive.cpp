// relu_naive.cpp
/**
 * @file relu_naive.cpp
 * @brief Naive ReLU activation layer.
 *
 * Applies the rectified linear unit element wise, replacing negative values
 * with zero. Provides overloads for 4D tensors [N][C][H][W] and 2D tensors
 * [N][features]. Implementation copies the input, then loops in a cache
 * friendly unit stride order. Execution is single threaded. No changes in
 * algorithm or semantics.
 */

#include <algorithm>  // std::max
#include "relu_naive.hpp"

Tensor4D ReLUNaive::forward(const Tensor4D& input) const {
    // Copy input to preserve value semantics: output shape equals input shape.
    Tensor4D output = input;

    // Apply ReLU element-wise. The innermost loop iterates over contiguous
    // float values to preserve unit-stride access.
    for (auto& image : output) {
        for (auto& channel : image) {
            for (auto& row : channel) {
                for (float& val : row) {
                    val = std::max(0.0f, val);
                }
            }
        }
    }
    return output;
}

Tensor2D ReLUNaive::forward(const Tensor2D& input) const {
    Tensor2D output = input;

    // Apply ReLU element-wise over [batch][features].
    for (auto& vec : output) {
        for (float& val : vec) {
            val = std::max(0.0f, val);
        }
    }
    return output;
}
