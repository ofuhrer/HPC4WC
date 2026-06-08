#pragma once

#include <ostream>
#include <vector>
#include <cmath>
#include <math.h>

// ===========================================================================
// Class to store data
// ===========================================================================
template <typename T> class Storage3D { // Template class for generic type T
public:
  Storage3D(int x, int y, int z, int nhalo, T value = 0)
      : xsize_(x + 2 * nhalo), ysize_(y + 2 * nhalo), zsize_(z),
        halosize_(nhalo),
        data_((x + 2 * nhalo) * (y + 2 * nhalo) * (z + 2 * nhalo), value) {}

  T &operator()(int i, int j, int k) {
    return data_[i + j * xsize_ + k * xsize_ * ysize_];
  }

  void writeFile(std::ostream &os) {
    int32_t three = 3;
    int32_t nbits = 8 * sizeof(T);
    int32_t writehalo = halosize_;
    int32_t writex = xsize_;
    int32_t writey = ysize_;
    int32_t writez = zsize_;

    os.write(reinterpret_cast<const char *>(&three), sizeof(three));
    os.write(reinterpret_cast<const char *>(&nbits), sizeof(nbits));
    os.write(reinterpret_cast<const char *>(&writehalo), sizeof(writehalo));
    os.write(reinterpret_cast<const char *>(&writex), sizeof(writex));
    os.write(reinterpret_cast<const char *>(&writey), sizeof(writey));
    os.write(reinterpret_cast<const char *>(&writez), sizeof(writez));
    for (std::size_t k = 0; k < zsize_; ++k) {
      for (std::size_t j = 0; j < ysize_; ++j) {
        for (std::size_t i = 0; i < xsize_; ++i) {
          os.write(reinterpret_cast<const char *>(&operator()(i, j, k)),
                   sizeof(T));    
        }
      }
    }
  }

  void readFile(std::istream &is) {
    int32_t three, nbits, writehalo, writex, writey, writez;
    is.read(reinterpret_cast<char *>(&three), sizeof(three));
    is.read(reinterpret_cast<char *>(&nbits), sizeof(nbits));
    is.read(reinterpret_cast<char *>(&writehalo), sizeof(writehalo));
    is.read(reinterpret_cast<char *>(&writex), sizeof(writex));
    is.read(reinterpret_cast<char *>(&writey), sizeof(writey));
    is.read(reinterpret_cast<char *>(&writez), sizeof(writez));

    halosize_ = writehalo;
    xsize_ = writex;
    ysize_ = writey;
    zsize_ = writez;

    data_.resize(xsize_ * ysize_ * zsize_);
    for (std::size_t k = 0; k < zsize_; ++k) {
      for (std::size_t j = 0; j < ysize_; ++j) {
        for (std::size_t i = 0; i < xsize_; ++i) {
          is.read(reinterpret_cast<char *>(&operator()(i, j, k)), sizeof(T));
        }
      }
    }
  }

// Initualize fields with shear flow
  void initialize(T amplitude = 100.0, int width = 50, int length = 50, const std::string& mode = "u") {
      for (std::size_t k = zMin(); k < zMax(); ++k) {
          for (std::size_t j = yMin(); j < yMax(); ++j) {
              for (std::size_t i = xMin(); i < xMax(); ++i) {
                    if (mode == "u" &&
                    j >= ysize_ / 2 - width / 2 && j <= ysize_ / 2 + width / 2 && 
                    i >= xsize_ / 2 - length / 2 && i <= xsize_ / 2 + length / 2) {
                      // operator()(i, j, k) = amplitude;
                      operator()(i, j, k) = amplitude * cos(j * M_PI / ysize_);
                    }
                    else {
                      operator()(i, j, k) =  0.5 - static_cast<T>(static_cast<double>(rand()) / RAND_MAX * 1.0);
                    }
              }
          }
      }
  }

  const std::size_t xMin() const { return halosize_; }            // size of the halo in x-direction
  const std::size_t xMax() const { return xsize_ - halosize_; }   // size of the halo + domain in x-direction
  const std::size_t xSize() const { return xsize_; }              // size of halo + domain + halo in x-direction
  const std::size_t yMin() const { return halosize_; }            // size of halo in y-direction
  const std::size_t yMax() const { return ysize_ - halosize_; }   // size of halo + domain in y-direction
  const std::size_t ySize() const { return ysize_; }              // size of halo + domain + halo in y-direction
  const std::size_t zMin() const { return 0; }                    // lowest level of z
  const std::size_t zMax() const { return zsize_; }               // highest level of z

private:
  int32_t xsize_, ysize_, zsize_, halosize_;
  std::vector<T> data_;
};
