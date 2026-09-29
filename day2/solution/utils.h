#pragma once

#include <cstddef>
#include <cstdint>
#include <ostream>
#include <vector>

template <typename T> class Storage3D {
public:
  Storage3D(int x, int y, int z, int nhalo, T value = 0)
      : xsize_(static_cast<std::size_t>(x + 2 * nhalo)),
        ysize_(static_cast<std::size_t>(y + 2 * nhalo)),
        zsize_(static_cast<std::size_t>(z)),
        halosize_(static_cast<std::size_t>(nhalo)),
        data_(static_cast<std::size_t>(x + 2 * nhalo) *
                  static_cast<std::size_t>(y + 2 * nhalo) *
                  static_cast<std::size_t>(z + 2 * nhalo),
              value) {}

  T &operator()(std::size_t i, std::size_t j, std::size_t k) {
    return data_[i + j * xsize_ + k * xsize_ * ysize_];
  }

  const T &operator()(std::size_t i, std::size_t j, std::size_t k) const {
    return data_[i + j * xsize_ + k * xsize_ * ysize_];
  }

  void writeFile(std::ostream &os) const {
    int32_t three = 3;
    int32_t sixtyfour = 64;
    int32_t writehalo = static_cast<int32_t>(halosize_);
    int32_t writex = static_cast<int32_t>(xsize_);
    int32_t writey = static_cast<int32_t>(ysize_);
    int32_t writez = static_cast<int32_t>(zsize_);

    os.write(reinterpret_cast<const char *>(&three), sizeof(three));
    os.write(reinterpret_cast<const char *>(&sixtyfour), sizeof(sixtyfour));
    os.write(reinterpret_cast<const char *>(&writehalo), sizeof(writehalo));
    os.write(reinterpret_cast<const char *>(&writex), sizeof(writex));
    os.write(reinterpret_cast<const char *>(&writey), sizeof(writey));
    os.write(reinterpret_cast<const char *>(&writez), sizeof(writez));
    for (std::size_t k = 0; k < zsize_; ++k) {
      for (std::size_t j = 0; j < ysize_; ++j) {
        for (std::size_t i = 0; i < xsize_; ++i) {
          const T &value = operator()(i, j, k);
          os.write(reinterpret_cast<const char *>(&value), sizeof(value));
        }
      }
    }
  }

  void initialize() {
    const std::size_t xInterior = xMax() - xMin();
    const std::size_t yInterior = yMax() - yMin();

    for (std::size_t k = zsize_ / 4; k < 3 * zsize_ / 4; ++k) {
      for (std::size_t j = halosize_ + yInterior / 4;
           j < halosize_ + 3 * yInterior / 4; ++j) {
        for (std::size_t i = halosize_ + xInterior / 4;
             i < halosize_ + 3 * xInterior / 4; ++i) {
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
  std::size_t xsize_, ysize_, zsize_, halosize_;
  std::vector<T> data_;
};
