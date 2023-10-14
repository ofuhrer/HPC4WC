
#include "stencil2d_common.h"
#include "stencil2d_field3d.h"

template <typename T>
void applyDiffusion(Field3D<T>& field, T alpha, int numIter)
{
    int x = field.x;
    int y = field.y;
    int z = field.z;
    int h = field.h;

    Field3D<T> tmp(x, y, z, h);

    for (int iter = 0; iter < numIter; ++iter) {
        for (int k = 0; k < z; ++k) {
            for (int j = -1; j < y + 1; ++j) {
                for (int i = -1; i < x + 1; ++i) {
                    tmp(i, j, 0) = -4 * field(i, j, k)
                        + field(i - 1, j, k) + field(i + 1, j, k)
                        + field(i, j - 1, k) + field(i, j + 1, k);
                }
            }
            for (int j = 0; j < y; ++j) {
                for (int i = 0; i < x; ++i) {
                    T laplap = -4 * tmp(i, j, 0)
                        + tmp(i - 1, j, 0) + tmp(i + 1, j, 0)
                        + tmp(i, j - 1, 0) + tmp(i, j + 1, 0);
                    field(i, j, k) = field(i, j, k) - alpha * laplap;
                }
            }
        }
        updateHalo(field);
    }
}

void stencil2d_host(int x, int y, int z, int iter, const char* filename)
{
    Field3D<FloatType> field(x, y, z, const_h);
    initialize(field);
    updateHalo(field);
    applyDiffusion(field, const_alpha, iter);
    field.writeToFile(filename);
}
