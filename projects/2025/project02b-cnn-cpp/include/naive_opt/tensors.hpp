#pragma once

/**
 * @file tensors.hpp
 * @brief Canonical tensor type aliases for the CNN project.
 *
 * Provides nested std::vector definitions for 2D, 3D, and 4D tensors
 * consistently used across all layers. Shapes follow the DL convention:
 *
 *   Tensor2D: [N][F]        batch of feature vectors
 *   Tensor3D: [C][H][W]     single image or feature map
 *   Tensor4D: [N][C][H][W]  batch of images or feature maps
 */

#include <vector>

// 2D tensor: [N][F]
using Tensor2D = std::vector<std::vector<float>>;
// 3D tensor: [C][H][W]
using Tensor3D = std::vector<std::vector<std::vector<float>>>;
// 4D tensor: [N][C][H][W]
using Tensor4D = std::vector<Tensor3D>;
