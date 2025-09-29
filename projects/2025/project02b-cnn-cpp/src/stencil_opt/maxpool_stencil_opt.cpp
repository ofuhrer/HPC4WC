// maxpool_stencil_opt.cpp
/**
 * @file maxpool_stencil_opt.cpp
 * @brief Implementation of MaxPool2DStencilOpt forward passes.
 *
 * Performs 2D max pooling on 3D and 4D tensors. Parallelism is applied
 * across the batch dimension; inner pooling loops remain serial except
 * for SIMD reduction across kernel columns. Behavior matches the naive version.
 */

#ifdef _OPENMP
#include <omp.h>
#endif

#include "maxpool_stencil_opt.hpp"

MaxPool2DStencilOpt::MaxPool2DStencilOpt(int kernel_size, int stride)
    : kernel_size_(kernel_size), stride_(stride) {}

// [N × C × H × W] → [N × C × H_out × W_out]
// Pre-size output and parallelize across batch samples.
Tensor4D MaxPool2DStencilOpt::forward(const Tensor4D& input) const {
  const int N = static_cast<int>(input.size());
  if (N == 0) return {};

  Tensor4D output(static_cast<std::size_t>(N));

#ifdef _OPENMP
#pragma omp parallel for schedule(static) default(none) shared(input, output, N) if (N > 1)
#endif
  for (int n = 0; n < N; ++n) {
    output[static_cast<std::size_t>(n)] = forwardSingle(input[static_cast<std::size_t>(n)]);
  }

  return output;
}

// [C × H × W] → [C × H_out × W_out]
// Micro-opts: hoist indices, cache row refs, SIMD reduction over kernel cols.
Tensor3D MaxPool2DStencilOpt::forwardSingle(const Tensor3D& input) const {
  const int C = static_cast<int>(input.size());
  if (C == 0) return {};

  const int H = static_cast<int>(input[0].size());
  const int W = static_cast<int>(input[0][0].size());

  if ((H - kernel_size_) % stride_ != 0 || (W - kernel_size_) % stride_ != 0) {
    throw std::invalid_argument("MaxPool2DStencilOpt: input size not divisible by stride");
  }

  const int H_out = (H - kernel_size_) / stride_ + 1;
  const int W_out = (W - kernel_size_) / stride_ + 1;

  Tensor3D output(
      static_cast<std::size_t>(C),
      std::vector<std::vector<float>>(static_cast<std::size_t>(H_out),
                                      std::vector<float>(static_cast<std::size_t>(W_out), 0.0f)));

  for (int c = 0; c < C; ++c) {
    const auto& chan = input[static_cast<std::size_t>(c)];
    for (int i = 0; i < H_out; ++i) {
      const int row0 = i * stride_;
      auto& out_row = output[static_cast<std::size_t>(c)][static_cast<std::size_t>(i)];

      for (int j = 0; j < W_out; ++j) {
        const int col0 = j * stride_;

        float max_val = chan[static_cast<std::size_t>(row0)][static_cast<std::size_t>(col0)];

        for (int m = 0; m < kernel_size_; ++m) {
          const auto& in_row = chan[static_cast<std::size_t>(row0 + m)];

#ifdef _OPENMP
#pragma omp simd reduction(max : max_val)
#endif
          for (int n = 0; n < kernel_size_; ++n) {
            const float v = in_row[static_cast<std::size_t>(col0 + n)];
            if (v > max_val) max_val = v;
          }
        }

        out_row[static_cast<std::size_t>(j)] = max_val;
      }
    }
  }

  return output;
}
