#pragma once

#include <algorithm>
#include <cstddef>
#include <fstream>

namespace benchmark {

inline double median(double* values, int count) {
    std::sort(values, values + count);
    const int middle = count / 2;

    if (count % 2 == 1) {
        return values[middle];
    }

    return 0.5 * (values[middle - 1] + values[middle]);
}

inline bool write_array(const char* path, const double* values, std::size_t count) {
    std::ofstream output(path, std::ios::binary);
    if (!output) {
        return false;
    }

    output.write(
        (const char*)values,
        (std::streamsize)(count * sizeof(double))
    );

    return bool(output);
}

}  // namespace benchmark
