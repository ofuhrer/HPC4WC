// conv2d_naive.cpp
/**
 * @file conv2d_naive.cpp
 * @brief Naive 2D convolution layer with He initialization.
 *
 * Implements a basic Conv2D forward pass: input is zero-padded if requested,
 * then convolved with learned weights and biases. Output is built with nested
 * loops over output channels and spatial positions, accumulating over input
 * channels and kernel elements. Single threaded with no parallel hints.
 * Validates channel counts and stride divisibility. No algorithmic changes.
 */

#include <cmath>        // std::sqrt
#include <random>       // std::random_device, std::mt19937, std::normal_distribution
#include <stdexcept>    // std::invalid_argument
#include "conv2d_naive.hpp"

// File-local helper: build zero-padded view of the input without exposing it in the header.
static Tensor3D compute_padded_input(const Tensor3D& input, int padding) {
    const int C = static_cast<int>(input.size());
    const int H = static_cast<int>(input[0].size());
    const int W = static_cast<int>(input[0][0].size());

    Tensor3D padded(
        C,
        std::vector<std::vector<float>>(H + 2 * padding,
                                        std::vector<float>(W + 2 * padding, 0.0f))
    );

    for (int c = 0; c < C; ++c) {
        for (int i = 0; i < H; ++i) {
            for (int j = 0; j < W; ++j) {
                padded[c][i + padding][j + padding] = input[c][i][j];
            }
        }
    }
    return padded;
}

Conv2DNaive::Conv2DNaive(int in_channels,
                         int out_channels,
                         int kernel_size,
                         int stride,
                         int padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {
    // He initialization: weights ~ N(0, sqrt(2 / fan_in)), fan_in = C_in * K * K.
    std::random_device rd;
    std::mt19937 gen(rd());
    const float fan_in = static_cast<float>(in_channels_) * kernel_size_ * kernel_size_;
    std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));

    weights_.resize(out_channels_);
    for (int oc = 0; oc < out_channels_; ++oc) {
        weights_[oc].resize(in_channels_);
        for (int ic = 0; ic < in_channels_; ++ic) {
            weights_[oc][ic].resize(kernel_size_, std::vector<float>(kernel_size_));
            for (int m = 0; m < kernel_size_; ++m) {
                for (int n = 0; n < kernel_size_; ++n) {
                    weights_[oc][ic][m][n] = dist(gen);
                }
            }
        }
    }

    biases_.assign(out_channels_, 0.0f);
}

Tensor4D Conv2DNaive::forward(const Tensor4D& batch) const {
    const int N = static_cast<int>(batch.size());
    if (N == 0) return {};

    // Validate channel count on first sample.
    if (static_cast<int>(batch[0].size()) != in_channels_) {
        throw std::invalid_argument("Conv2DNaive: input batch channel mismatch");
    }

    Tensor4D output;
    output.reserve(N);

    // Batch wrapper: process each sample independently using the single-sample kernel.
    for (int n = 0; n < N; ++n) {
        output.push_back(forwardSingle(batch[n]));
    }
    return output;
}

Tensor3D Conv2DNaive::forwardSingle(const Tensor3D& input) const {
    // Validate channels.
    if (static_cast<int>(input.size()) != in_channels_) {
        throw std::invalid_argument("Conv2DNaive: channel count mismatch");
    }

    // Apply padding when requested.
    const Tensor3D* in_ptr = &input;
    Tensor3D padded;
    if (padding_ > 0) {
        padded = compute_padded_input(input, padding_);
        in_ptr = &padded;
    }
    const Tensor3D& in = *in_ptr;

    // Compute output dimensions and check divisibility: (H + 2P - K) and (W + 2P - K) must be multiples of stride.
    const int H_in = static_cast<int>(in[0].size());
    const int W_in = static_cast<int>(in[0][0].size());
    if ((H_in - kernel_size_) % stride_ != 0 || (W_in - kernel_size_) % stride_ != 0) {
        throw std::invalid_argument("Conv2DNaive: size not divisible by stride");
    }
    const int H_out = (H_in - kernel_size_) / stride_ + 1;
    const int W_out = (W_in - kernel_size_) / stride_ + 1;

    // Allocate output [C_out][H_out][W_out].
    Tensor3D out(
        out_channels_,
        std::vector<std::vector<float>>(H_out, std::vector<float>(W_out, 0.0f))
    );

    // Naive convolution loops: for each output channel and spatial position, accumulate over input channels and kernel.
    for (int oc = 0; oc < out_channels_; ++oc) {
        for (int i = 0; i < H_out; ++i) {
            for (int j = 0; j < W_out; ++j) {
                float sum = biases_[oc];
                for (int ic = 0; ic < in_channels_; ++ic) {
                    for (int m = 0; m < kernel_size_; ++m) {
                        for (int n = 0; n < kernel_size_; ++n) {
                            const int row = i * stride_ + m;
                            const int col = j * stride_ + n;
                            sum += in[ic][row][col] * weights_[oc][ic][m][n];
                        }
                    }
                }
                out[oc][i][j] = sum;
            }
        }
    }
    return out;
}
