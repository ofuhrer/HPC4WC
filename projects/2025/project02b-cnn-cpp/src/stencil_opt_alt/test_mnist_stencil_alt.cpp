#include "tensors.hpp"
#include "cnn_stencil_alt.hpp"
#include <cnpy.h>
#include <iostream>
#include <vector>
#include <iomanip>
#include <cstdlib>
#include <chrono>
#include <algorithm>

// -------- helpers (consistent across all test programs) --------
inline int read_env_int(const char* key, int fallback) {
    if (const char* v = std::getenv(key)) {
        try { return std::max(1, std::stoi(v)); } catch (...) {}
    }
    return fallback;
}

struct ScopedComputeTimer {
    using Clock = std::chrono::high_resolution_clock;
    Clock::time_point t0;
    int samples;
    double* out_sec;
    explicit ScopedComputeTimer(int samples_, double* out_sec_=nullptr)
        : t0(Clock::now()), samples(samples_), out_sec(out_sec_) {}
    ~ScopedComputeTimer() {
        using F = std::chrono::duration<double>;
        double sec = std::chrono::duration_cast<F>(Clock::now() - t0).count();
        if (out_sec) *out_sec = sec;
        std::cout << "RESULT compute_time_sec=" << std::fixed << std::setprecision(6)
                  << sec << " samples=" << samples << "\n";
    }
};
// ----------------------------------------------------------------

int main() {
    // Config from environment
    const int H = 28, W = 28, C = 1;
    const int N_default = 10000;
    const int N = read_env_int("MNIST_SAMPLES", N_default);
    const bool QUIET = std::getenv("MNIST_QUIET") != nullptr;

    // Load MNIST images and labels
    auto arr = cnpy::npy_load("data/mnist/images.npy");
    uint8_t* raw_data = arr.data<uint8_t>();
    auto label_arr = cnpy::npy_load("data/mnist/labels.npy");
    uint8_t* labels = label_arr.data<uint8_t>();

    // Clamp samples to available data
    const size_t total_pixels = arr.num_bytes() / sizeof(uint8_t);
    const size_t max_samples_from_images = total_pixels / (H * W);
    const size_t max_samples_from_labels = label_arr.num_bytes() / sizeof(uint8_t);
    const int N_eff = static_cast<int>(std::min<size_t>(N, std::min(max_samples_from_images, max_samples_from_labels)));

    // Convert to Tensor4D: [N][1][28][28]
    Tensor4D batch(N_eff, Tensor3D(C, std::vector<std::vector<float>>(H, std::vector<float>(W))));
    for (int n = 0; n < N_eff; ++n) {
        const size_t base = static_cast<size_t>(n) * H * W;
        for (int i = 0; i < H; ++i) {
            const size_t row = base + static_cast<size_t>(i) * W;
            for (int j = 0; j < W; ++j) {
                batch[n][0][i][j] = static_cast<float>(raw_data[row + j]) / 255.0f;
            }
        }
    }

    // Run stencil-alt CNN and measure compute time only
    CNNStencil cnn;
    double compute_sec = 0.0;
    int correct = 0;

    {
        ScopedComputeTimer timer(N_eff, &compute_sec);
        Tensor4D output = cnn.forward(batch);

        // Argmax predictions and accuracy
        for (int n = 0; n < N_eff; ++n) {
            int predicted = 0;
            float max_val = output[n][0][0][0];
            for (int i = 1; i < 10; ++i) {
                if (output[n][0][0][i] > max_val) {
                    max_val = output[n][0][0][i];
                    predicted = i;
                }
            }
            const int true_label = static_cast<int>(labels[n]);
            if (!QUIET) {
                std::cout << "Sample " << n << ": predicted=" << predicted
                          << ", true=" << true_label << "\n";
            }
            if (predicted == true_label) ++correct;
        }
    }

    const double acc = (N_eff > 0) ? static_cast<double>(correct) / static_cast<double>(N_eff) : 0.0;
    std::cout << "RESULT accuracy=" << std::fixed << std::setprecision(6)
              << acc << " samples=" << N_eff << "\n";

    if (!QUIET) {
        std::cout << std::fixed << std::setprecision(3);
        std::cout << "Compute time: " << compute_sec << " s for " << N_eff << " samples\n";
        std::cout << "Accuracy: " << (100.0 * acc) << "%\n";
    }

    return 0;
}
