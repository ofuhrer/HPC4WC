// This code contains the Storage3D class in order to store a 3D field
// It is taken from HPC4WC/day2/ with minor changes
// In particular it fixes a bug in the initialize function that results in filling the 1s not exactly in the middle and adds a haloSize() member function

#pragma once

#include <ostream>
#include <vector>

template <typename T> class Storage3D {
public:
  Storage3D(int x, int y, int z, int nhalo, T value = 0)
      : xsize_(x + 2 * nhalo), ysize_(y + 2 * nhalo), zsize_(z),
        halosize_(nhalo),
        data_((x + 2 * nhalo) * (y + 2 * nhalo) * z, value) {}
    
  Storage3D()
      : xsize_(0), ysize_(0), zsize_(0), halosize_(0), data_() {}

  T &operator()(int i, int j, int k) {
    return data_[i + j * xsize_ + k * xsize_ * ysize_];
  }

  void writeFile(std::ostream &os) {
    int32_t three = 3;
    int32_t sixtyfour = 64;
    int32_t writehalo = halosize_;
    int32_t writex = xsize_ - 2*halosize_;
    int32_t writey = ysize_ - 2*halosize_;
    int32_t writez = zsize_;

    os.write(reinterpret_cast<const char *>(&three), sizeof(three));
    os.write(reinterpret_cast<const char *>(&sixtyfour), sizeof(sixtyfour));
    os.write(reinterpret_cast<const char *>(&writehalo), sizeof(writehalo));
    os.write(reinterpret_cast<const char *>(&writex), sizeof(writex));
    os.write(reinterpret_cast<const char *>(&writey), sizeof(writey));
    os.write(reinterpret_cast<const char *>(&writez), sizeof(writez));
    for (std::size_t k = 0; k < zsize_; ++k) {
      for (std::size_t j = halosize_; j < ysize_ - halosize_; ++j) {
        for (std::size_t i = halosize_; i < xsize_ - halosize_; ++i) {
          os.write(reinterpret_cast<const char *>(&operator()(i, j, k)),
                   sizeof(double));
        }
      }
    }
  }

  void initialize() {
    for (std::size_t k = zsize_ / 4.0; k < 3 * zsize_ / 4.0; ++k) {
      for (std::size_t j = xsize_ / 4.;
           j < 3. / 4. * xsize_; ++j) {
        for (std::size_t i = xsize_ / 4.;
             i < + 3. / 4. * xsize_; ++i) {
          operator()(i, j, k) = 1;
        }
      }
    }
  }

  const std::size_t haloSize() const { return halosize_; }
  const std::size_t xMin() const { return halosize_; }
  const std::size_t xMax() const { return xsize_ - halosize_; }
  const std::size_t xSize() const { return xsize_; }
  const std::size_t yMin() const { return halosize_; }
  const std::size_t yMax() const { return ysize_ - halosize_; }
  const std::size_t ySize() const { return ysize_; }
  const std::size_t zMin() const { return 0; }
  const std::size_t zMax() const { return zsize_; }
  const std::size_t data_size() const { return data_.size(); }
  const std::vector<T>& data() const { return data_; }
  std::vector<T>& data() { return data_; }


private:
  int32_t xsize_, ysize_, zsize_, halosize_;
  std::vector<T> data_;
};
