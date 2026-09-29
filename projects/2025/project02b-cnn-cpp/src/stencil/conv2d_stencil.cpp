/**
 * @file conv2d_stencil.cpp
 * @brief Implementation of Conv2DStencil (3×3, stride 1, padding 1).
 *
 * Initializes weights with He initialization and applies a fixed 3×3 stencil.
 * The compute path assumes one input channel (uses weights_[oc][0] and in[0]).
 * Behavior and performance are identical to the original version.
 */

#include "conv2d_stencil.hpp"

#include <cassert>
#include <cmath>
#include <random>
#include <stdexcept>

// File-local helper: zero-pad an input tensor without polluting the header.
static Tensor3D compute_padded_input(const Tensor3D& input, int padding) {
  int C = static_cast<int>(input.size());
  int H = static_cast<int>(input[0].size());
  int W = static_cast<int>(input[0][0].size());

  Tensor3D padded(
      C, std::vector<std::vector<float>>(H + 2 * padding,
                                         std::vector<float>(W + 2 * padding, 0.0f)));

  // Copy into centered region.
  for (int c = 0; c < C; ++c) {
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
        padded[c][i + padding][j + padding] = input[c][i][j];
      }
    }
  }
  return padded;
}

Conv2DStencil::Conv2DStencil(int in_channels, int out_channels, int kernel_size, int stride,
                             int padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {
  // Enforce the fixed stencil parameters at construction time.
  assert(kernel_size == 3 && stride == 1 && padding == 1 &&
         "Conv2DStencil only supports kernel_size=3, stride=1, padding=1");

  // He initialization: N(0, sqrt(2 / fan_in))
  std::random_device rd;
  std::mt19937 gen(rd());
  float fan_in = static_cast<float>(in_channels_) * kernel_size_ * kernel_size_;
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

Tensor4D Conv2DStencil::forward(const Tensor4D& batch) const {
  int N = static_cast<int>(batch.size());
  if (N == 0) return {};
  if (static_cast<int>(batch[0].size()) != in_channels_) {
    throw std::invalid_argument("Conv2DStencil: input batch channel mismatch");
  }

  Tensor4D output;
  output.reserve(N);

  // Process each sample in the batch independently.
  for (int n = 0; n < N; ++n) {
    output.push_back(forwardSingle(batch[n]));
  }
  return output;
}

Tensor3D Conv2DStencil::forwardSingle(const Tensor3D& input) const {
  if (static_cast<int>(input.size()) != in_channels_) {
    throw std::invalid_argument("Conv2DStencil: channel count mismatch");
  }

  // Apply zero padding if requested; alias otherwise to avoid copies.
  const Tensor3D* in_ptr = &input;
  Tensor3D padded;
  if (padding_ > 0) {
    padded = compute_padded_input(input, padding_);
    in_ptr = &padded;
  }
  const Tensor3D& in = *in_ptr;

  int H_in = static_cast<int>(in[0].size());
  int W_in = static_cast<int>(in[0][0].size());
  if ((H_in - kernel_size_) % stride_ != 0 || (W_in - kernel_size_) % stride_ != 0) {
    throw std::invalid_argument("Conv2DStencil: size not divisible by stride");
  }

  int H_out = (H_in - kernel_size_) / stride_ + 1;
  int W_out = (W_in - kernel_size_) / stride_ + 1;

  Tensor3D out(out_channels_,
               std::vector<std::vector<float>>(H_out, std::vector<float>(W_out, 0.0f)));

  // Compute [out_channels][H_out][W_out] with unrolled 3×3 stencil.
  // Note: i,j iterate over output indices; inner access uses [i-1][j-1] due to 1-pixel padding.
  for (int oc = 0; oc < out_channels_; ++oc) {
    for (int i = 1; i < H_out + 1; ++i) {
      for (int j = 1; j < W_out + 1; ++j) {
        float sum = weights_[oc][0][0][0] * in[0][i - 1][j - 1] +
                    weights_[oc][0][0][1] * in[0][i - 1][j] +
                    weights_[oc][0][0][2] * in[0][i - 1][j + 1] +
                    weights_[oc][0][1][0] * in[0][i][j - 1] +
                    weights_[oc][0][1][1] * in[0][i][j] +
                    weights_[oc][0][1][2] * in[0][i][j + 1] +
                    weights_[oc][0][2][0] * in[0][i + 1][j - 1] +
                    weights_[oc][0][2][1] * in[0][i + 1][j] +
                    weights_[oc][0][2][2] * in[0][i + 1][j + 1] + biases_[oc];
        out[oc][i - 1][j - 1] = sum;
      }
    }
  }

  return out;
}
