#include "cnn_cuda.hpp"
#include <cnpy.h>
#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <algorithm>
#include <cstdlib>
#include <chrono>

int main(int argc, char** argv) {
    // Command‑line: [program] [N=10000] [B=100]
    const int total_samples = (argc > 1 ? std::atoi(argv[1]) : 10000);
    const int chunk_size    = (argc > 2 ? std::atoi(argv[2]) : 100);

    // Obtain world size and rank from Slurm
    int world_size = 1;
    int rank = 0;
    if (auto* env = std::getenv("SLURM_NTASKS")) {
        world_size = std::atoi(env);
    }
    if (auto* env = std::getenv("SLURM_PROCID")) {
        rank = std::atoi(env);
    }

    // Bind this process to one GPU
    cudaSetDevice(rank);

    // Compute per‑rank slice
    int per_rank = (total_samples + world_size - 1) / world_size;
    int start_idx = rank * per_rank;
    int end_idx   = std::min(total_samples, start_idx + per_rank);

    // Load MNIST images and labels once
    auto img_arr   = cnpy::npy_load("data/mnist/images.npy");
    auto lbl_arr   = cnpy::npy_load("data/mnist/labels.npy");
    uint8_t* raw    = img_arr.data<uint8_t>();
    uint8_t* labels = lbl_arr.data<uint8_t>();

    const int H = 28, W = 28, C = 1;
    CNNCUDA cnn;  // GPU CNN implementation

    std::cout << std::fixed << std::setprecision(3);

    // Process only this rank's slice
    auto start = std::chrono::high_resolution_clock::now();
    
    for (int offset = start_idx; offset < end_idx; offset += chunk_size) {
        int M = std::min(chunk_size, end_idx - offset);

        // Build Tensor4D [M][C][H][W]
        Tensor4D batch(M, Tensor3D(C, std::vector<std::vector<float>>(H, std::vector<float>(W))));
        for (int n = 0; n < M; ++n) {
            int global_n = offset + n;
            for (int i = 0; i < H; ++i) {
                for (int j = 0; j < W; ++j) {
                    int idx = global_n * H * W + i * W + j;
                    batch[n][0][i][j] = raw[idx] / 255.0f;
                }
            }
        }

        // Forward on GPU
        Tensor4D out = cnn.forward(batch);

        // Print results
        for (int n = 0; n < M; ++n) {
            int global_n = offset + n;
            int pred = 0;
            float maxv = out[n][0][0][0];
            for (int k = 1; k < 10; ++k) {
                if (out[n][0][0][k] > maxv) {
                    maxv = out[n][0][0][k];
                    pred = k;
                }
            }
            int truth = static_cast<int>(labels[global_n]);
            std::cout << "Sample " << global_n
                      << ": predicted=" << pred
                      << ", true=" << truth << "\n";
        }
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;

    std::cout << "Rank " << rank << ": processed " << (end_idx - start_idx) 
        << " samples in " << elapsed.count() << " seconds.\n";


    return 0;
}
