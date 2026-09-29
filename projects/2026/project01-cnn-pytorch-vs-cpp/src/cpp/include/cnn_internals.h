#ifndef CNN_INTERNALS_H
#define CNN_INTERNALS_H

#include <vector>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <arm_neon.h>

// ---------------------------------------------------------
// DATA STRUCTURES
// ---------------------------------------------------------

// A basic structure is defined to hold a multi-dimensional tensor in a flat 1D array.
struct Tensor {
    std::vector<float> data;
    int batches;
    int channels;
    int height;
    int width;
};

// ---------------------------------------------------------
// LAYER FORWARD PASSES
// ---------------------------------------------------------

// A 2D convolution forward pass is declared.
void conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                    int stride, int padding);

// A 2D convolution forward pass with blocking is declared.
void conv2d_forward_blocked(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                    int stride, int padding);

// A restructured 2D convolution forward pass is declared.
void conv2d_forward_restructured(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                    int stride, int padding);

// A hoisted 2D convolution forward pass is declared with restrict pointers for optimization.
void conv2d_forward_hoist_restrict(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                    int stride, int padding);

// Loops are reordered for better cache performance in this 2D convolution forward pass.
void conv2d_forward_reorder(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                    int stride, int padding);

// A specialized 2D convolution forward pass is declared, potentially optimized for specific input sizes or hardware.                    
void conv2d_forward_specialized(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output);                    

// A specialized 2D convolution forward pass with blocking is declared for further optimization.
void conv2d_forward_specialized_blocked(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output);

// A ReLU activation forward pass is declared.
void relu_forward(Tensor& tensor);

// A fully connected (dense) layer forward pass is declared.
void linear_forward(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output);

// A 2D Max Pooling forward pass is declared.
void maxpool2d_forward(const Tensor& input, Tensor& output, int pool_size, int stride);

// A specialized 2D Max Pooling forward pass is declared, potentially optimized for specific input sizes or hardware.
void maxpool2d_forward_specialized(const Tensor& input, Tensor& output);

// Adaptive Average Pooling (to 1x1) is declared.
void adaptive_avgpool2d_forward(const Tensor& input, Tensor& output);

// A 2D convolution forward pass using im2col / im2col matrix multiplication.
void conv2d_forward_im2col(const Tensor& input, const Tensor& weight, const Tensor& bias, Tensor& output,
                             int stride, int padding);



#endif // CNN_INTERNALS_H
