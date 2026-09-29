// cnn_naive.cpp
/**
 * @file cnn_naive.cpp
 * @brief Top level CNN forward using the baseline naive sublayers.
 *
 * Builds the MNIST style stack with two blocks of convolution, ReLU, and max pool,
 * then flattens to a feature vector and applies a fully connected layer. Execution
 * is single threaded with straightforward nested loops for flatten and final write.
 * Shapes follow [N][C][H][W] throughout. No changes to algorithm or numerics.
 */

#include "cnn_naive.hpp"

// Constructor wires up the MNIST-specific architecture without changing behavior.
CNNNaive::CNNNaive(int in_channels, int num_classes)
    : conv1_(in_channels, 8, 3, 1, 1),   // 1→8 channels, 3×3, stride=1, pad=1
      relu1_(),
      pool1_(2, 2),                      // 2×2 pooling, stride=2
      conv2_(8, 16, 3, 1, 1),            // 8→16 channels, 3×3, stride=1, pad=1
      relu2_(),
      pool2_(2, 2),                      // 2×2 pooling, stride=2
      fc_(16 * 7 * 7, num_classes)       // fully connected: 784 → num_classes
{}

// Forward pass through the entire CNN graph.
Tensor4D CNNNaive::forward(const Tensor4D& batch) const {
    // Block 1: Conv → ReLU → Pool
    Tensor4D x = conv1_.forward(batch);
    x = relu1_.forward(x);
    x = pool1_.forward(x);

    // Block 2: Conv → ReLU → Pool
    x = conv2_.forward(x);
    x = relu2_.forward(x);
    x = pool2_.forward(x);

    // Flatten [N × C × H × W] → [N × (C*H*W)] for the fully connected layer.
    const int N = static_cast<int>(x.size());
    const int C = static_cast<int>(x[0].size());
    const int H = static_cast<int>(x[0][0].size());
    const int W = static_cast<int>(x[0][0][0].size());

    Tensor2D flattened(N, std::vector<float>(C * H * W));
    for (int n = 0; n < N; ++n) {
        int idx = 0;
        for (int c = 0; c < C; ++c) {
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    flattened[n][idx++] = x[n][c][i][j];
                }
            }
        }
    }

    // Apply the fully connected layer to get [N × num_classes].
    Tensor2D fc_out = fc_.forward(flattened);

    // Reshape to [N][1][1][num_classes] to keep a 4D output convention.
    Tensor4D out(
        N,
        Tensor3D(1, std::vector<std::vector<float>>(1, std::vector<float>(fc_out[0].size())))
    );
    for (int n = 0; n < N; ++n) {
        out[n][0][0] = fc_out[n];
    }

    return out;
}
