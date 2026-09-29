#pragma once

#include <ostream>
#include <vector>
#ifdef __CUDACC__
#include <cuda_runtime.h>
#endif

// Template class for 3D storage with halo regions
// Supports both CPU and GPU memory management when compiled with CUDA
template <typename T> class Storage3D {
public:
  // Constructor: Initialize 3D storage with halo regions
  // PRE: x, y, z > 0 (interior dimensions), nhalo >= 0 (halo width)
  // POST: Creates storage of total size (x+2*nhalo) * (y+2*nhalo) * z
  //       All elements initialized to 'value'
  //       Device pointer initialized to nullptr if CUDA enabled
  Storage3D(int x, int y, int z, int nhalo, T value = 0)
      : xsize_(x + 2 * nhalo), ysize_(y + 2 * nhalo), zsize_(z),
        halosize_(nhalo),
        data_((x + 2 * nhalo) * (y + 2 * nhalo) * (z + 2 * nhalo), value)
#ifdef __CUDACC__
  , d_data_(nullptr)
#endif
  {}

  // Destructor: Clean up device memory if allocated
  // PRE: Object is being destroyed
  // POST: Device memory is freed if it was allocated
  ~Storage3D() {
#ifdef __CUDACC__
    if (d_data_) {
      cudaFree(d_data_);
    }
#endif
  }

  // Element access operator for 3D indexing
  // PRE: 0 <= i < xsize_, 0 <= j < ysize_, 0 <= k < zsize_
  // POST: Returns reference to element at position (i,j,k)
  T &operator()(int i, int j, int k) {
    return data_[i + j * xsize_ + k * xsize_ * ysize_];
  }

#ifdef __CUDACC__
  // Allocate memory on GPU device
  // PRE: Device memory not yet allocated (d_data_ == nullptr)
  // POST: Device memory allocated for total storage size
  //       d_data_ points to allocated device memory
  void allocateDevice() {
    size_t total_size = xsize_ * ysize_ * zsize_ * sizeof(T);
    cudaMalloc(&d_data_, total_size);
  }

  // Copy data from host to device
  // PRE: d_data_ points to valid device memory of correct size
  //      Host data is valid
  // POST: All host data copied to device memory
  void copyToDevice() {
    if (d_data_) {
      size_t total_size = xsize_ * ysize_ * zsize_ * sizeof(T);
      cudaMemcpy(d_data_, data_.data(), total_size, cudaMemcpyHostToDevice);
    }
  }

  // Copy data from device to host
  // PRE: d_data_ points to valid device memory with current data
  //      Host memory is allocated and valid
  // POST: All device data copied to host memory
  void copyFromDevice() {
    if (d_data_) {
      size_t total_size = xsize_ * ysize_ * zsize_ * sizeof(T);
      cudaMemcpy(data_.data(), d_data_, total_size, cudaMemcpyDeviceToHost);
    }
  }

  // Get device pointer for kernel launches
  // PRE: Device memory has been allocated
  // POST: Returns pointer to device memory
  T* deviceData() { return d_data_; }
#endif
  
  // Get host data pointer
  // PRE: Host data is valid
  // POST: Returns pointer to host memory
  T* data() { return data_.data(); }
  
  // Get total number of elements
  // PRE: Object is properly initialized
  // POST: Returns total size including halo regions
  size_t size() const { return xsize_ * ysize_ * zsize_; }

  // Write field data to binary file
  // PRE: os is a valid output stream opened in binary mode
  // POST: Writes header information (dimensions, halo size) followed by
  //       all field data in k-j-i order to the stream
  void writeFile(std::ostream &os) {
    int32_t three = 3;
    int32_t sixtyfour = 64;
    int32_t writehalo = halosize_;
    int32_t writex = xsize_;
    int32_t writey = ysize_;
    int32_t writez = zsize_;

    os.write(reinterpret_cast<const char *>(&three), sizeof(three));
    os.write(reinterpret_cast<const char *>(&sixtyfour), sizeof(sixtyfour));
    os.write(reinterpret_cast<const char *>(&writehalo), sizeof(writehalo));
    os.write(reinterpret_cast<const char *>(&writex), sizeof(writex));
    os.write(reinterpret_cast<const char *>(&writey), sizeof(writey));
    os.write(reinterpret_cast<const char *>(&writez), sizeof(writez));
    for (std::size_t k = 0; k < zsize_; ++k) {
      for (std::size_t j = 0; j < ysize_; ++j) {
        for (std::size_t i = 0; i < xsize_; ++i) {
          os.write(reinterpret_cast<const char *>(&operator()(i, j, k)),
                   sizeof(double));
        }
      }
    }
  }

  // Initialize field with a rectangular block pattern
  // PRE: Storage is properly allocated
  // POST: Central region (1/4 to 3/4 in each dimension) is set to 1,
  //       rest remains at initialization value (typically 0)
  void initialize() {
    for (std::size_t k = zsize_ / 4.0; k < 3 * zsize_ / 4.0; ++k) {
      for (std::size_t j = ysize_ / 4.;
           j < 3. / 4. * ysize_; ++j) {
        for (std::size_t i = xsize_ / 4.;
             i < 3. / 4. * xsize_; ++i) {
          operator()(i, j, k) = 1;
        }
      }
    }
  }

  // Accessor methods for interior region boundaries
  // PRE: Object is properly initialized
  // POST: Returns appropriate boundary indices
  std::size_t xMin() const { return halosize_; }      // First interior x index
  std::size_t xMax() const { return xsize_ - halosize_; } // Last interior x index + 1
  std::size_t xSize() const { return xsize_; }        // Total x size including halo
  std::size_t yMin() const { return halosize_; }      // First interior y index
  std::size_t yMax() const { return ysize_ - halosize_; } // Last interior y index + 1
  std::size_t ySize() const { return ysize_; }        // Total y size including halo
  std::size_t zMin() const { return 0; }              // First z index (no z halo)
  std::size_t zMax() const { return zsize_; }         // Last z index + 1

private:
  int32_t xsize_, ysize_, zsize_, halosize_;  // Dimensions and halo width
  std::vector<T> data_;                       // Host data storage
#ifdef __CUDACC__
  T* d_data_;  // Device pointer for GPU memory
#endif
};

// Update halo regions with periodic boundary conditions
// PRE: inField is a valid Storage3D object with properly allocated data
//      Field has halo regions of width >= 1
// POST: Halo regions are filled with periodic boundary conditions:
//       - Bottom halo copies from top interior
//       - Top halo copies from bottom interior  
//       - Left halo copies from right interior (includes corners)
//       - Right halo copies from left interior (includes corners)
//       Applied to all z-levels
void updateHalo(Storage3D<double> &inField) {
  const int xInterior = inField.xMax() - inField.xMin();
  const int yInterior = inField.yMax() - inField.yMin();
  const int zInterior = inField.zMax(); // Process all z-levels

  // bottom edge (without corners): map to top of interior
  for (std::size_t k = 0; k < zInterior; ++k) {
    for (std::size_t j = 0; j < inField.yMin(); ++j) {
      for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j + yInterior, k);
      }
    }
  }

  // top edge (without corners): map to bottom of interior  
  for (std::size_t k = 0; k < zInterior; ++k) {
    for (std::size_t j = inField.yMax(); j < inField.ySize(); ++j) {
      for (std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j - yInterior, k);
      }
    }
  }

  // left edge (including corners): map to right of interior
  for (std::size_t k = 0; k < zInterior; ++k) {
    for (std::size_t j = 0; j < inField.ySize(); ++j) {
      for (std::size_t i = 0; i < inField.xMin(); ++i) {
        inField(i, j, k) = inField(i + xInterior, j, k);
      }
    }
  }

  // right edge (including corners): map to left of interior
  for (std::size_t k = 0; k < zInterior; ++k) {
    for (std::size_t j = 0; j < inField.ySize(); ++j) {
      for (std::size_t i = inField.xMax(); i < inField.xSize(); ++i) {
        inField(i, j, k) = inField(i - xInterior, j, k);
      }
    }
  }
}