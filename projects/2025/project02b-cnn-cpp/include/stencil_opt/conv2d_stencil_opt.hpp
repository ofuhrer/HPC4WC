#pragma once
// conv2d_stencil_opt.hpp
/**
 * @file conv2d_stencil_opt.hpp
 * @brief Explicit 3×3 stencil convolution (stride=1, pad=1) for the stencil-optimized CNN path.
 *
 * Assumptions (unchanged):
 *   - kernel_size = 3
 *   - stride = 1
 *   - padding = 1
 *   - typically in_channels = 1 for the stencil path
 *
 * Input:  [N × C × H × W]
 * Output: [N × C_out × H × W]
 */

#include <stdexcept>
#include <vector>
#include "tensors.hpp"

class Conv2DStencilOpt {
public:
  /**
   * @brief Construct a 2D convolution layer with fixed 3×3 kernel, stride=1, pad=1.
   * @param in_channels  Number of input channels (typically 1).
   * @param out_channels Number of output channels.
   * @param kernel_size  Must equal 3.
   * @param stride       Must equal 1.
   * @param padding      Must equal 1.
   * @throws std::invalid_argument if parameters differ from (3,1,1).
   */
  Conv2DStencilOpt(int in_channels, int out_channels, int kernel_size,
                   int stride = 1, int padding = 1);

  /**
   * @brief Forward pass on a batch.
   * @param batch Input tensor [N × C × H × W].
   * @return Output tensor [N × C_out × H × W].
   * @throws std::invalid_argument if input channel count mismatches in_channels_.
   */
  Tensor4D forward(const Tensor4D& batch) const;

private:
  int in_channels_;
  int out_channels_;
  int kernel_size_;
  int stride_;
  int padding_;

  // weights_[oc][ic][3][3]
  std::vector<Tensor3D> weights_;
  // biases_[oc]
  std::vector<float> biases_;

  /**
   * @brief Forward pass on a single image.
   * @param input Input tensor [C × H × W].
   * @return Output tensor [C_out × H × W].
   */
  Tensor3D forwardSingle(const Tensor3D& input) const;
};
