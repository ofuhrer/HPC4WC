#pragma once
#include "types_cuda.hpp"

class MaxPool2DCUDA {
public:
    MaxPool2DCUDA(int kernel_size, int stride);
    Tensor4D forward(const Tensor4D& batch) const;

private:
    int kernel_size_, stride_;
};
