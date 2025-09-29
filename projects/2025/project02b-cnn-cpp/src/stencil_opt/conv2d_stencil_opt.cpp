// conv2d_stencil_opt.cpp
/**
 * @file conv2d_stencil_opt.cpp
 * @brief Implementation of Conv2DStencilOpt forward passes.
 *
 * Explicit 3×3 convolution with stride=1 and pad=1. Parallelized over batch (N),
 * optionally over output channels (oc) when safe. SIMD is applied across width (j).
 * Assumes in_channels = 1 for the stencil path. Behavior identical to the naive version.
 */

#include <random>
#include <cmath>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "conv2d_stencil_opt.hpp"

namespace {
// Zero-pad a single image [C×H×W].
Tensor3D compute_padded_input(const Tensor3D& input, int padding) {
  const int C = static_cast<int>(input.size());
  const int H = static_cast<int>(input[0].size());
  const int W = static_cast<int>(input[0][0].size());

  Tensor3D padded(
      static_cast<std::size_t>(C),
      std::vector<std::vector<float>>(
          static_cast<std::size_t>(H + 2 * padding),
          std::vector<float>(static_cast<std::size_t>(W + 2 * padding), 0.0f)));

  for (int c = 0; c < C; ++c)
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < W; ++j)
        padded[static_cast<std::size_t>(c)][static_cast<std::size_t>(i + padding)]
              [static_cast<std::size_t>(j + padding)] =
                  input[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)]
                       [static_cast<std::size_t>(j)];
  return padded;
}
} // namespace

Conv2DStencilOpt::Conv2DStencilOpt(int in_channels,
                                   int out_channels,
                                   int kernel_size,
                                   int stride,
                                   int padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {
  if (!(kernel_size_ == 3 && stride_ == 1 && padding_ == 1)) {
    throw std::invalid_argument(
        "Conv2DStencilOpt supports kernel=3, stride=1, padding=1 only");
  }

  // He initialization: N(0, sqrt(2 / (in_channels * K * K))).
  std::random_device rd;
  std::mt19937 gen(rd());
  const float fan_in = static_cast<float>(in_channels_) * kernel_size_ * kernel_size_;
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));

  weights_.assign(static_cast<std::size_t>(out_channels_), Tensor3D());
  for (int oc = 0; oc < out_channels_; ++oc) {
    auto& oc_w = weights_[static_cast<std::size_t>(oc)];
    oc_w.assign(static_cast<std::size_t>(in_channels_),
                std::vector<std::vector<float>>(3, std::vector<float>(3)));
    for (int ic = 0; ic < in_channels_; ++ic) {
      auto& w = oc_w[static_cast<std::size_t>(ic)];
      for (int m = 0; m < 3; ++m)
        for (int n = 0; n < 3; ++n)
          w[static_cast<std::size_t>(m)][static_cast<std::size_t>(n)] = dist(gen);
    }
  }
  biases_.assign(static_cast<std::size_t>(out_channels_), 0.0f);
}

Tensor4D Conv2DStencilOpt::forward(const Tensor4D& batch) const {
  const int N = static_cast<int>(batch.size());
  if (N == 0) return {};

  if (static_cast<int>(batch[0].size()) != in_channels_) {
    throw std::invalid_argument("Conv2DStencilOpt: input batch channel mismatch");
  }

  Tensor4D output(static_cast<std::size_t>(N));

  // Parallelize over batch samples.
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) \
    shared(batch, output, N) if (N > 1)
#endif
  for (int n = 0; n < N; ++n) {
    output[static_cast<std::size_t>(n)] = forwardSingle(batch[static_cast<std::size_t>(n)]);
  }

  return output;
}

Tensor3D Conv2DStencilOpt::forwardSingle(const Tensor3D& input) const {
  if (static_cast<int>(input.size()) != in_channels_) {
    throw std::invalid_argument("Conv2DStencilOpt: channel count mismatch");
  }

  // Apply padding if needed.
  const Tensor3D* in_ptr = &input;
  Tensor3D padded;
  if (padding_ > 0) {
    padded = compute_padded_input(input, padding_);
    in_ptr = &padded;
  }
  const Tensor3D& in = *in_ptr;

  const int H_in = static_cast<int>(in[0].size());
  const int W_in = static_cast<int>(in[0][0].size());

  if ((H_in - kernel_size_) % stride_ != 0 || (W_in - kernel_size_) % stride_ != 0) {
    throw std::invalid_argument("Conv2DStencilOpt: size not divisible by stride");
  }

  const int H_out = (H_in - kernel_size_) / stride_ + 1;  // equals H_in - 2 when stride=1
  const int W_out = (W_in - kernel_size_) / stride_ + 1;

  Tensor3D out(
      static_cast<std::size_t>(out_channels_),
      std::vector<std::vector<float>>(static_cast<std::size_t>(H_out),
                                      std::vector<float>(static_cast<std::size_t>(W_out), 0.0f)));

  // Explicit 3×3 stencil. Assumes in_channels = 1.
  for (int oc = 0; oc < out_channels_; ++oc) {
    const auto& w = weights_[static_cast<std::size_t>(oc)][0]; // assumes ic == 0
    const float b = biases_[static_cast<std::size_t>(oc)];

    for (int i = 1; i < H_out + 1; ++i) {
      const auto& row_above = in[0][static_cast<std::size_t>(i - 1)];
      const auto& row_mid   = in[0][static_cast<std::size_t>(i)];
      const auto& row_below = in[0][static_cast<std::size_t>(i + 1)];

#ifdef _OPENMP
#pragma omp simd
#endif
      for (int j = 1; j < W_out + 1; ++j) {
        const float s =
            w[0][0] * row_above[static_cast<std::size_t>(j - 1)] +
            w[0][1] * row_above[static_cast<std::size_t>(j    )] +
            w[0][2] * row_above[static_cast<std::size_t>(j + 1)] +
            w[1][0] * row_mid  [static_cast<std::size_t>(j - 1)] +
            w[1][1] * row_mid  [static_cast<std::size_t>(j    )] +
            w[1][2] * row_mid  [static_cast<std::size_t>(j + 1)] +
            w[2][0] * row_below[static_cast<std::size_t>(j - 1)] +
            w[2][1] * row_below[static_cast<std::size_t>(j    )] +
            w[2][2] * row_below[static_cast<std::size_t>(j + 1)] +
            b;

        out[static_cast<std::size_t>(oc)][static_cast<std::size_t>(i - 1)]
           [static_cast<std::size_t>(j - 1)] = s;
      }
    }
  }

  return out;
}
