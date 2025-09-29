#pragma once
#include "types_cuda.hpp"

class LinearCUDA {
public:
    LinearCUDA(int in_features, int out_features);
    ~LinearCUDA();
    Tensor2D forward(const Tensor2D& input) const;

private:
    int in_f_, out_f_;
    float *d_weights_, *d_bias_;
};
