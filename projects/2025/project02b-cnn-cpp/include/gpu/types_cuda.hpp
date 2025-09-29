#pragma once
#include <vector>

// Tensor aliases as in CPU ver.
using Tensor2D = std::vector<std::vector<float>>;
using Tensor3D = std::vector<std::vector<std::vector<float>>>;
using Tensor4D = std::vector<Tensor3D>;
