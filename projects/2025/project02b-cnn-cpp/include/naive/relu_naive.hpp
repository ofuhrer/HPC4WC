#pragma once

/**
 * @file relu_naive.hpp
 * @brief Naive ReLU activation layer using std::vector tensors.
 *
 * Applies the rectified linear unit element-wise: ReLU(x) = max(0, x).
 * Provides overloads for both 4D image tensors and 2D feature tensors.
 * Stateless and intended as a simple baseline for correctness checks.
 *
 * Shapes:
 *   Input 4D: [N × C × H × W] → same shape
 *   Input 2D: [N × F]         → same shape
 */

#include <vector>

// 3D tensor: [channels][height][width]
using Tensor3D = std::vector<std::vector<std::vector<float>>>;
// 4D tensor: [batch][channels][height][width]
using Tensor4D = std::vector<Tensor3D>;
// 2D tensor: [batch][features]
using Tensor2D = std::vector<std::vector<float>>;

/// Naive reference ReLU layer.
class ReLUNaive {
public:
    /**
     * @brief Forward pass on a 4D tensor.
     * @param input [N × C × H × W].
     * @return Tensor of the same shape with ReLU applied element-wise.
     */
    Tensor4D forward(const Tensor4D& input) const;

    /**
     * @brief Forward pass on a 2D tensor.
     * @param input [N × F].
     * @return Tensor of the same shape with ReLU applied element-wise.
     */
    Tensor2D forward(const Tensor2D& input) const;
};
