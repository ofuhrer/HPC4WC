#pragma once
#include "types_cuda.hpp"
#include "conv2d_cuda.hpp"
#include "relu_cuda.hpp"
#include "maxpool_cuda.hpp"
#include "linear_cuda.hpp"

class CNNCUDA {
public:
    CNNCUDA(int in_channels = 1, int num_classes = 10);
    Tensor4D forward(const Tensor4D& batch) const;

private:
    Conv2DCUDA  conv1_;
    ReLUCUDA    relu1_;
    MaxPool2DCUDA pool1_;
    Conv2DCUDA  conv2_;
    ReLUCUDA    relu2_;
    MaxPool2DCUDA pool2_;
    LinearCUDA  fc_;
};
