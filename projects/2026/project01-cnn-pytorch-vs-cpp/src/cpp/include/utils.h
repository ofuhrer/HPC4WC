#ifndef UTILS_H
#define UTILS_H

#include "cnn_internals.h"

#include <string>
#include <vector>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <algorithm>
#include <unistd.h>

// ---------------------------------------------------------
// TERMINAL COLOR CODES
// Empty strings when stdout is redirected (e.g. sbatch log files).
// ---------------------------------------------------------

namespace Color {
    inline const bool _tty       = (isatty(STDOUT_FILENO) != 0);
    inline const char* const RESET       = _tty ? "\033[0m"    : "";
    inline const char* const BOLD        = _tty ? "\033[1m"    : "";
    inline const char* const DIM         = _tty ? "\033[2m"    : "";
    inline const char* const RED         = _tty ? "\033[31m"   : "";
    inline const char* const GREEN       = _tty ? "\033[32m"   : "";
    inline const char* const YELLOW      = _tty ? "\033[33m"   : "";
    inline const char* const CYAN        = _tty ? "\033[36m"   : "";
    inline const char* const BOLD_GREEN  = _tty ? "\033[1;32m" : "";
    inline const char* const BOLD_CYAN   = _tty ? "\033[1;36m" : "";
    inline const char* const BOLD_YELLOW = _tty ? "\033[1;33m" : "";
    inline const char* const DIM_GREEN   = _tty ? "\033[2;32m" : "";
}

// ---------------------------------------------------------
// UTILITY FUNCTIONS
// ---------------------------------------------------------

// A binary file is loaded into a target vector.
inline void load_binary_weights(const std::string& filepath, std::vector<float>& target_vector) {
    std::ifstream file(filepath, std::ios::binary | std::ios::ate);
    if (!file.is_open())
        throw std::runtime_error("The file could not be opened: " + filepath);

    std::streamsize size = file.tellg();
    if (size != static_cast<std::streamsize>(target_vector.size() * sizeof(float)))
        throw std::runtime_error("File size mismatch for: " + filepath);

    file.seekg(0, std::ios::beg);
    if (file.read(reinterpret_cast<char*>(target_vector.data()), size))
        std::cout << Color::DIM_GREEN << "Successfully loaded: " << filepath << Color::RESET << "\n";
    else
        throw std::runtime_error("Read error: " + filepath);
}

// An integer is converted from Big-Endian to Little-Endian.
inline int reverse_int(int i) {
    unsigned char c1 = i & 255;
    unsigned char c2 = (i >> 8)  & 255;
    unsigned char c3 = (i >> 16) & 255;
    unsigned char c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

// The MNIST image file is parsed and loaded.
inline void load_mnist_images(const std::string& filepath, Tensor& images_tensor, int num_images_to_load) {
    std::ifstream file(filepath, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("The MNIST image file could not be opened: " + filepath +
                                 " — run 'python3 src/python/download_data.py' first.");

    int magic_number = 0, number_of_images = 0, n_rows = 0, n_cols = 0;
    file.read((char*)&magic_number,      sizeof(magic_number));      magic_number      = reverse_int(magic_number);
    file.read((char*)&number_of_images,  sizeof(number_of_images));  number_of_images  = reverse_int(number_of_images);
    file.read((char*)&n_rows,            sizeof(n_rows));             n_rows            = reverse_int(n_rows);
    file.read((char*)&n_cols,            sizeof(n_cols));             n_cols            = reverse_int(n_cols);

    int images_to_read = std::min(num_images_to_load, number_of_images);
    images_tensor = {std::vector<float>(images_to_read * n_rows * n_cols),
                     images_to_read, 1, n_rows, n_cols};

    for (int i = 0; i < images_to_read * n_rows * n_cols; ++i) {
        unsigned char pixel = 0;
        file.read((char*)&pixel, sizeof(pixel));
        images_tensor.data[i] = static_cast<float>(pixel) / 255.0f;
    }
}

// A vector of floats is written to a binary file.
inline void save_binary(const std::string& filepath, const std::vector<float>& v) {
    std::ofstream file(filepath, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("The file could not be opened for writing: " + filepath);
    file.write(reinterpret_cast<const char*>(v.data()),
               static_cast<std::streamsize>(v.size() * sizeof(float)));
}

// A single MNIST image is printed to the console.
inline void print_ascii_image(const Tensor& images_tensor, int batch_index) {
    int size      = images_tensor.height * images_tensor.width;
    int start_idx = batch_index * size;

    std::cout << Color::CYAN << "\nVisualizing image at index " << batch_index << ":" << Color::RESET << "\n";
    for (int r = 0; r < images_tensor.height; ++r) {
        for (int c = 0; c < images_tensor.width; ++c) {
            float pixel = images_tensor.data[start_idx + r * images_tensor.width + c];
            if      (pixel > 0.5f) std::cout << Color::BOLD << "##" << Color::RESET;
            else if (pixel > 0.2f) std::cout << Color::DIM  << ".." << Color::RESET;
            else                   std::cout << "  ";
        }
        std::cout << "\n";
    }
    std::cout << "\n";
}

// A tensor's checksum is computed by summing all its elements.
inline double checksum_tensor(const Tensor &t) {
    double s = 0.0;
    for (float v : t.data) s += (double)v;
    return s;
}

// ---------------------------------------------------------------
// Implementations registering and benchmarking utility functions.
// ---------------------------------------------------------------
inline bool contains_substr(const std::string &haystack, const std::string &needle) {
    return needle.empty() ? false : haystack.find(needle) != std::string::npos;
}

inline bool implementation_matches_filter(const std::string &name, const char *filter) {
    if (!filter || !*filter) return true;
    bool exact = std::getenv("CNN_BENCH_FILTER_EXACT") != nullptr;

    std::string f(filter);
    size_t pos = 0;
    while (pos <= f.size()) {
        size_t comma = f.find(',', pos);
        std::string token = f.substr(pos, comma == std::string::npos ? std::string::npos : comma - pos);
        size_t b = token.find_first_not_of(" \t");
        size_t e = token.find_last_not_of(" \t");
        if (b != std::string::npos) {
            std::string trimmed = token.substr(b, e - b + 1);
            if (exact ? (name == trimmed) : contains_substr(name, trimmed)) return true;
        }
        if (comma == std::string::npos) break;
        pos = comma + 1;
    }
    return false;
}

#endif // UTILS_H
