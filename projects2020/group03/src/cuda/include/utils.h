#pragma once

#include <ostream>
#include <vector>
#include "common_types.h"

template <typename T>
class Storage3D {
public:
  Storage3D(int x, int y, int z, int nhalo, T value = 0)
      : xsize_(x + 2 * nhalo), ysize_(y + 2 * nhalo), zsize_(z), halosize_(nhalo),
        data_((x + 2 * nhalo) * (y + 2 * nhalo) * z, value) {}

  T& operator()(int i, int j, int k) { return data_[i + j * xsize_ + k * xsize_ * ysize_]; }

  void writeFile(std::ostream& os) {
    int32_t three = 3;
    int32_t sixtyfour = 8*sizeof(realType);
    int32_t writehalo = halosize_;
    int32_t writex = xsize_;
    int32_t writey = ysize_;
    int32_t writez = zsize_;

    os.write(reinterpret_cast<const char*>(&three), sizeof(three));
    os.write(reinterpret_cast<const char*>(&sixtyfour), sizeof(sixtyfour));
    os.write(reinterpret_cast<const char*>(&writehalo), sizeof(writehalo));
    os.write(reinterpret_cast<const char*>(&writex), sizeof(writex));
    os.write(reinterpret_cast<const char*>(&writey), sizeof(writey));
    os.write(reinterpret_cast<const char*>(&writez), sizeof(writez));
    for(std::size_t k = 0; k < zsize_; ++k) {
      for(std::size_t j = 0; j < ysize_; ++j) {
        for(std::size_t i = 0; i < xsize_; ++i) {
          os.write(reinterpret_cast<const char*>(&operator()(i, j, k)), sizeof(realType));
        }
      }
    }
  }

  void initialize() {
    for(std::size_t k = zsize_ / 4.0; k < 3 * zsize_ / 4.0; ++k) {
      for(std::size_t j = halosize_ + xsize_ / 4.; j < halosize_ + 3. / 4. * xsize_; ++j) {
        for(std::size_t i = halosize_ + xsize_ / 4.; i < halosize_ + 3. / 4. * xsize_; ++i) {
          operator()(i, j, k) = 1;
        }
      }
    }
  }

  std::size_t xMin() const { return halosize_; }
  std::size_t xMax() const { return xsize_ - halosize_; }
  std::size_t xSize() const { return xsize_; }
  std::size_t yMin() const { return halosize_; }
  std::size_t yMax() const { return ysize_ - halosize_; }
  std::size_t ySize() const { return ysize_; }
  std::size_t zMin() const { return 0; }
  std::size_t zMax() const { return zsize_; }

private:
  int32_t xsize_, ysize_, zsize_, halosize_;
  std::vector<T> data_;
};
