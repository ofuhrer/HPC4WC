#include "cnn_cuda.hpp"

CNNCUDA::CNNCUDA(int in_channels, int num_classes)
  : conv1_(in_channels,  8, 3, 1, 1),
    relu1_(),
    pool1_(2,2),
    conv2_(8, 16, 3, 1, 1),
    relu2_(),
    pool2_(2,2),
    fc_(16 * 7 * 7, num_classes)
{}

Tensor4D CNNCUDA::forward(const Tensor4D& batch) const {
    
    Tensor4D x = conv1_.forward(batch);
    x = relu1_.forward(x);
    x = pool1_.forward(x);

    x = conv2_.forward(x);
    x = relu2_.forward(x);
    x = pool2_.forward(x);

    // flatten
    int N = x.size();
    int C = x[0].size();
    int H = x[0][0].size();
    int W = x[0][0][0].size();
    Tensor2D flat(N, std::vector<float>(C*H*W));
    for (int n=0; n<N; ++n) {
        int idx=0;
        for (int c=0; c<C; ++c)
        for (int i=0; i<H; ++i)
        for (int j=0; j<W; ++j)
            flat[n][idx++] = x[n][c][i][j];
    }

    Tensor2D out2 = fc_.forward(flat);

    // back to 4D
    Tensor4D out(N, Tensor3D(1, std::vector<std::vector<float>>(1, std::vector<float>(out2[0].size()))));
    for (int n=0; n<N; ++n)
        out[n][0][0] = out2[n];
    return out;
}
