// conv2d_opt.cpp
/**
 * @file conv2d_opt.cpp
 * @brief Optimized 2D convolution with batch parallelism and inner-loop SIMD.
 *
 * Validates shapes, applies optional zero padding, then computes outputs with a
 * cache-friendly loop order over spatial positions. Parallelizes across the batch
 * with OpenMP and uses an omp simd reduction over kernel columns. Weights use He
 * initialization. Behavior matches the naive version with no algorithmic changes.
 */


#include <cmath>
#include <random>
#include <stdexcept>
#include <cstddef>
#ifdef _OPENMP
#include <omp.h>
#endif

#include "conv2d_opt.hpp"

// File-local helper: apply zero-padding to input tensor.
static Tensor3D compute_padded_input(const Tensor3D& input, int padding) {
  const int C = static_cast<int>(input.size());
  const int H = static_cast<int>(input[0].size());
  const int W = static_cast<int>(input[0][0].size());

  Tensor3D padded(
      static_cast<std::size_t>(C),
      std::vector<std::vector<float>>(
          static_cast<std::size_t>(H + 2 * padding),
          std::vector<float>(static_cast<std::size_t>(W + 2 * padding), 0.0f)));

  for (int c = 0; c < C; ++c) {
    for (int i = 0; i < H; ++i) {
      for (int j = 0; j < W; ++j) {
        padded[static_cast<std::size_t>(c)][static_cast<std::size_t>(i + padding)]
              [static_cast<std::size_t>(j + padding)] =
            input[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)]
                 [static_cast<std::size_t>(j)];
      }
    }
  }

  return padded;
}

Conv2DOpt::Conv2DOpt(int in_channels, int out_channels, int kernel_size,
                     int stride, int padding)
    : in_channels_(in_channels),
      out_channels_(out_channels),
      kernel_size_(kernel_size),
      stride_(stride),
      padding_(padding) {
  if (in_channels_ <= 0 || out_channels_ <= 0 || kernel_size_ <= 0 ||
      stride_ <= 0 || padding_ < 0) {
    throw std::invalid_argument("Conv2DOpt: invalid constructor arguments");
  }

  // He initialization: N(0, sqrt(2 / fan_in))
  std::random_device rd;
  std::mt19937 gen(rd());
  const float fan_in = static_cast<float>(in_channels_) *
                       static_cast<float>(kernel_size_) *
                       static_cast<float>(kernel_size_);
  std::normal_distribution<float> dist(0.0f, std::sqrt(2.0f / fan_in));

  weights_.assign(static_cast<std::size_t>(out_channels_), Tensor3D());
  for (int oc = 0; oc < out_channels_; ++oc) {
    auto& oc_w = weights_[static_cast<std::size_t>(oc)];
    oc_w.assign(static_cast<std::size_t>(in_channels_),
                std::vector<std::vector<float>>(
                    static_cast<std::size_t>(kernel_size_),
                    std::vector<float>(static_cast<std::size_t>(kernel_size_))));
    for (int ic = 0; ic < in_channels_; ++ic) {
      auto& w_ic = oc_w[static_cast<std::size_t>(ic)];
      for (int m = 0; m < kernel_size_; ++m) {
        for (int n = 0; n < kernel_size_; ++n) {
          w_ic[static_cast<std::size_t>(m)][static_cast<std::size_t>(n)] = dist(gen);
        }
      }
    }
  }
  biases_.assign(static_cast<std::size_t>(out_channels_), 0.0f);
}

Tensor4D Conv2DOpt::forward(const Tensor4D& batch) const {
  const int N = static_cast<int>(batch.size());
  if (N == 0) return {};

  // Sanity check: channel count of first sample
  if (static_cast<int>(batch[0].size()) != in_channels_) {
    throw std::invalid_argument("Conv2DOpt: input batch channel mismatch");
  }

  Tensor4D output(static_cast<std::size_t>(N));

  // Parallelize across the batch
#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) \
    shared(batch, output, N) if (N > 1)
#endif
  for (int n = 0; n < N; ++n) {
    output[static_cast<std::size_t>(n)] =
        forward_single(batch[static_cast<std::size_t>(n)]);
  }

  return output;
}

Tensor3D Conv2DOpt::forward_single(const Tensor3D& input) const {
  if (static_cast<int>(input.size()) != in_channels_) {
    throw std::invalid_argument("Conv2DOpt: channel count mismatch");
  }

  // Apply padding if requested
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
    throw std::invalid_argument("Conv2DOpt: size not divisible by stride");
  }

  const int H_out = (H_in - kernel_size_) / stride_ + 1;
  const int W_out = (W_in - kernel_size_) / stride_ + 1;

  Tensor3D out(
      static_cast<std::size_t>(out_channels_),
      std::vector<std::vector<float>>(
          static_cast<std::size_t>(H_out),
          std::vector<float>(static_cast<std::size_t>(W_out), 0.0f)));

  // Loop order: (i, j, oc, ic, m, n) to maximize cache reuse of input patches.
  for (int i = 0; i < H_out; ++i) {
    const int base_row = i * stride_;
    for (int j = 0; j < W_out; ++j) {
      const int base_col = j * stride_;

      for (int oc = 0; oc < out_channels_; ++oc) {
        float sum = biases_[static_cast<std::size_t>(oc)];

        for (int ic = 0; ic < in_channels_; ++ic) {
          const auto& in_ic   = in[static_cast<std::size_t>(ic)];
          const auto& w_oc_ic = weights_[static_cast<std::size_t>(oc)][static_cast<std::size_t>(ic)];

          for (int m = 0; m < kernel_size_; ++m) {
            const int row = base_row + m;
            const auto& in_row = in_ic[static_cast<std::size_t>(row)];
            const auto& w_row  = w_oc_ic[static_cast<std::size_t>(m)];

            // SIMD across kernel columns with accumulation into partial.
            float partial = 0.0f;
#ifdef _OPENMP
#pragma omp simd reduction(+ : partial)
#endif
            for (int n = 0; n < kernel_size_; ++n) {
              const int col = base_col + n;
              partial += in_row[static_cast<std::size_t>(col)] *
                         w_row[static_cast<std::size_t>(n)];
            }
            sum += partial;
          }
        }

        out[static_cast<std::size_t>(oc)][static_cast<std::size_t>(i)]
           [static_cast<std::size_t>(j)] = sum;
      }
    }
  }

  return out;
}
