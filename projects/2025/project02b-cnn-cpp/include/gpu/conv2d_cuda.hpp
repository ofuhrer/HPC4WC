#pragma once
#include "types_cuda.hpp"

class Conv2DCUDA {
public:
    Conv2DCUDA(int in_channels,
               int out_channels,
               int kernel_size,
               int stride,
               int padding);
    ~Conv2DCUDA();

    // Forward on a batch of shape [N][C][H][W]
    Tensor4D forward(const Tensor4D& batch) const;

private:
    int in_channels_, out_channels_, kernel_size_, stride_, padding_;
    // Device pointers for weights and biases
    float *d_weights_, *d_bias_;
};
