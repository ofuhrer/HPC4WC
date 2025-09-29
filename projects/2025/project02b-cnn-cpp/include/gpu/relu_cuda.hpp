#pragma once
#include "types_cuda.hpp"

class ReLUCUDA {
public:
    ReLUCUDA() = default;
    Tensor4D forward(const Tensor4D& batch) const;
};
