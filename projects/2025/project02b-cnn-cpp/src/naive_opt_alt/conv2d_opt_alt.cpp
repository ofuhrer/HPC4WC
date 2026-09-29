// conv2d_opt_alt.cpp
/**
 * @file conv2d_opt_alt.cpp
 * @brief Optimized 2D convolution (alternate) with batch parallelism and per-image tiling.
 *
 * Validates shapes, supports zero padding, and uses He initialization. Parallelizes
 * across the batch in forward(); inside the single-sample kernel, collapses over
 * (out_channel, out_row) when not already in a parallel region and applies an
 * omp simd reduction across kernel columns. Falls back to a serial per-image path
 * if omp_in_parallel() is true. Behavior matches the naive convolution.
 */

#include "conv2d_opt_alt.hpp"

#include <cmath>       // std::sqrt
#include <random>      // RNG, normal_distribution
#include <stdexcept>   // std::invalid_argument

#ifdef _OPENMP
#include <omp.h>
#endif

// File-local helper: zero-padding without exposing in the header.
// Writes input into a larger buffer with unit-stride inner dimension.
static Tensor3D compute_padded_input(const Tensor3D& input, int padding) {
  const int C = static_cast<int>(input.size());
  const int H = static_cast<int>(input[0].size());
  const int W = static_cast<int>(input[0][0].size());
  Tensor3D padded(C,
                  std::vector<std::vector<float>>(H + 2 * padding,
                                                  std::vector<float>(W + 2 * padding, 0.0f)));
  for (int c = 0; c < C; ++c)
    for (int i = 0; i < H; ++i)
      for (int j = 0; j < W; ++j)
        padded[c][i + padding][j + padding] = input[c][i][j];
  return padded;
}

Conv2DOptAlt::Conv2DOptAlt(int in_channels, int out_channels, int kernel_size, int stride, int padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {
  // He initialization: N(0, sqrt(2 / fan_in)), fan_in = in_channels * k * k
  std::random_device rd;
  std::mt19937 gen(rd());
  const float fan_in = static_cast<float>(in_channels_) * kernel_size_ * kernel_size_;
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));

  weights_.resize(out_channels_);
  for (int oc = 0; oc < out_channels_; ++oc) {
    weights_[oc].resize(in_channels_);
    for (int ic = 0; ic < in_channels_; ++ic) {
      weights_[oc][ic].resize(kernel_size_, std::vector<float>(kernel_size_));
      for (int m = 0; m < kernel_size_; ++m)
        for (int n = 0; n < kernel_size_; ++n)
          weights_[oc][ic][m][n] = dist(gen);
    }
  }
  biases_.assign(out_channels_, 0.0f);
}

Tensor4D Conv2DOptAlt::forward(const Tensor4D& batch) const {
  const int N = static_cast<int>(batch.size());
  if (N == 0) return {};

  // Validate channel count on first sample
  if (static_cast<int>(batch[0].size()) != in_channels_) {
    throw std::invalid_argument("Conv2DOptAlt: input batch channel mismatch");
  }

  // Preallocate for thread-safe indexed writes
  Tensor4D output(N);

#ifdef _OPENMP
#pragma omp parallel for schedule(static) if (N > 1) default(none) shared(batch, output, N)
#endif
  for (int n = 0; n < N; ++n) {
    output[n] = forward_single(batch[n]);
  }
  return output;
}

Tensor3D Conv2DOptAlt::forward_single(const Tensor3D& input) const {
  // Validate channels
  if (static_cast<int>(input.size()) != in_channels_) {
    throw std::invalid_argument("Conv2DOptAlt: channel count mismatch");
  }

  // Apply padding if needed
  const Tensor3D* in_ptr = &input;
  Tensor3D padded;
  if (padding_ > 0) {
    padded = compute_padded_input(input, padding_);
    in_ptr = &padded;
  }
  const Tensor3D& in = *in_ptr;

  // Compute dimensions and check divisibility
  const int H_in = static_cast<int>(in[0].size());
  const int W_in = static_cast<int>(in[0][0].size());
  if ((H_in - kernel_size_) % stride_ != 0 || (W_in - kernel_size_) % stride_ != 0) {
    throw std::invalid_argument("Conv2DOptAlt: size not divisible by stride");
  }

  const int H_out = (H_in - kernel_size_) / stride_ + 1;
  const int W_out = (W_in - kernel_size_) / stride_ + 1;

  // Allocate output [out_channels][H_out][W_out]
  Tensor3D out(out_channels_,
               std::vector<std::vector<float>>(H_out, std::vector<float>(W_out, 0.0f)));

  // Per-image compute. Prefer outer parallelism on (oc, i) when not already in a parallel region.
  // The innermost kernel width loop uses unit stride; small SIMD hint is added.
#ifdef _OPENMP
  if (!omp_in_parallel()) {
#pragma omp parallel for collapse(2) schedule(static) default(none) \
    shared(in, out, H_out, W_out) \
    shared(in_channels_, out_channels_, kernel_size_, stride_, biases_, weights_)
    for (int oc = 0; oc < out_channels_; ++oc) {
      for (int i = 0; i < H_out; ++i) {
        for (int j = 0; j < W_out; ++j) {
          float sum = biases_[oc];
          for (int ic = 0; ic < in_channels_; ++ic) {
            for (int m = 0; m < kernel_size_; ++m) {
              const int row = i * stride_ + m;
#pragma omp simd reduction(+ : sum)
              for (int n = 0; n < kernel_size_; ++n) {
                const int col = j * stride_ + n;
                sum += in[ic][row][col] * weights_[oc][ic][m][n];
              }
            }
          }
          out[oc][i][j] = sum;
        }
      }
    }
  } else
#endif
  {
    for (int oc = 0; oc < out_channels_; ++oc) {
      for (int i = 0; i < H_out; ++i) {
        for (int j = 0; j < W_out; ++j) {
          float sum = biases_[oc];
          for (int ic = 0; ic < in_channels_; ++ic) {
            for (int m = 0; m < kernel_size_; ++m) {
              const int row = i * stride_ + m;
#ifdef _OPENMP
#pragma omp simd reduction(+ : sum)
#endif
              for (int n = 0; n < kernel_size_; ++n) {
                const int col = j * stride_ + n;
                sum += in[ic][row][col] * weights_[oc][ic][m][n];
              }
            }
          }
          out[oc][i][j] = sum;
        }
      }
    }
  }

  return out;
}
