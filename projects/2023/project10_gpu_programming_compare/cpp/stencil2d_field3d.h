#pragma once

#include <fstream>
#include <string.h>
#include <vector>

template <typename T>
class Field3D {
public:
    int x, y, z, h;
    std::vector<T> data;

public:
    Field3D(int x_, int y_, int z_, int h_, T value = 0)
        : x(x_), y(y_), z(z_), h(h_)
    {
        data = std::vector<T>((x + 2 * h) * (y + 2 * h) * z, value);
    }
    T& operator()(int i, int j, int k)
    {
        return data[(h + i) + (h + j) * (x + 2 * h) + k * (x + 2 * h) * (y + 2 * h)];
    }
    void writeToFile(const char* filename)
    {
        if (!strcmp(filename, ""))
            return;

        std::ofstream os;
        os.open(filename, std::ofstream::binary);

        int32_t w_three = 3;
        int32_t w_bits = 8 * sizeof(T);
        int32_t w_h = h;
        int32_t w_x = x + 2 * h;
        int32_t w_y = y + 2 * h;
        int32_t w_z = z;

        os.write(reinterpret_cast<const char*>(&w_three), sizeof(w_three));
        os.write(reinterpret_cast<const char*>(&w_bits), sizeof(w_bits));
        os.write(reinterpret_cast<const char*>(&w_h), sizeof(w_h));
        os.write(reinterpret_cast<const char*>(&w_x), sizeof(w_x));
        os.write(reinterpret_cast<const char*>(&w_y), sizeof(w_y));
        os.write(reinterpret_cast<const char*>(&w_z), sizeof(w_z));

        for (int k = 0; k < z; ++k) {
            for (int j = -h; j < y + h; ++j) {
                for (int i = -h; i < x + h; ++i) {
                    T* value = &operator()(i, j, k);
                    os.write(reinterpret_cast<const char*>(value), sizeof(T));
                }
            }
        }

        os.close();
    }
};

template <typename T>
void initialize(Field3D<T>& field)
{
    int x = field.x;
    int y = field.y;
    int z = field.z;
    int h = field.h;

    // Reproduce Fortran-Code
    int x1 = (x + 2 * h) / 4 - 1;
    int y1 = (y + 2 * h) / 4 - 1;
    int x2 = 3 * x / 4;
    int y2 = 3 * y / 4;

    for (int k = 0; k < z; ++k) {
        for (int j = y1; j < y2; ++j) {
            for (int i = x1; i < x2; ++i) {
                field(i, j, k) = 1;
            }
        }
    }
}

template <typename T>
void updateHalo(Field3D<T>& field)
{
    int x = field.x;
    int y = field.y;
    int z = field.z;
    int h = field.h;

    // Bottom edge
    for (int k = 0; k < z; ++k) {
        for (int j = -h; j < 0; ++j) {
            for (int i = -h; i < x + h; ++i) {
                field(i, j, k) = field(i, j + y, k);
            }
        }
    }

    // Top edge
    for (int k = 0; k < z; ++k) {
        for (int j = y; j < y + h; ++j) {
            for (int i = -h; i < x + h; ++i) {
                field(i, j, k) = field(i, j - y, k);
            }
        }
    }

    // Left edge
    for (int k = 0; k < z; ++k) {
        for (int j = -h; j < y + h; ++j) {
            for (int i = -h; i < 0; ++i) {
                field(i, j, k) = field(i + x, j, k);
            }
        }
    }

    // Right edge
    for (int k = 0; k < z; ++k) {
        for (int j = -h; j < y + h; ++j) {
            for (int i = x; i < x + h; ++i) {
                field(i, j, k) = field(i - x, j, k);
            }
        }
    }
}
