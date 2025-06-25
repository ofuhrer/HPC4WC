#include <cassert>
#include <chrono>
#include <fstream>
#include <iostream>
#include <omp.h>

#ifdef CRAYPAT
#include "pat_api.h"
#endif

namespace {

template <typename T>
void updateHalo(T* inField, int x, int y, int z, int halo) {
  const int xDimTotal = x + 2 * halo;
  const int yDimTotal = y + 2 * halo;
  const int zDimTotal = z + 2 * halo;

  const std::size_t xyDimTotal = static_cast<std::size_t>(xDimTotal) * yDimTotal;

  const int xMin = halo;
  const int xMax = x + halo;
  const int yMin = halo;
  const int yMax = y + halo;

  const int xInterior = x;
  const int yInterior = y;

  // Iterate over z-slices
  for (std::size_t k = halo; k < z + halo; ++k) {
    const std::size_t k_offset = k * xyDimTotal;

    // Update top halo (along y-dimension)
    for (std::size_t j = 0; j < yMin; ++j) {
      const std::size_t j_offset_dst = j * xDimTotal;
      const std::size_t j_offset_src = (j + yInterior) * xDimTotal;
      for (std::size_t i = 0; i < xDimTotal; ++i) {
        inField[k_offset + j_offset_dst + i] = inField[k_offset + j_offset_src + i];
      }
    }

    // Update bottom halo (along y-dimension)
    for (std::size_t j = yMax; j < yDimTotal; ++j) {
      const std::size_t j_offset_dst = j * xDimTotal;
      const std::size_t j_offset_src = (j - yInterior) * xDimTotal;
      for (std::size_t i = 0; i < xDimTotal; ++i) {
        inField[k_offset + j_offset_dst + i] = inField[k_offset + j_offset_src + i];
      }
    }

    // Update left/right halos (along x-dimension)
    for (std::size_t j = 0; j < yDimTotal; ++j) {
      const std::size_t kj_offset = k_offset + j * xDimTotal;
      for (std::size_t i = 0; i < xMin; ++i) {
        inField[kj_offset + i] = inField[kj_offset + (i + xInterior)];
      }
      for (std::size_t i = xMax; i < xDimTotal; ++i) {
        inField[kj_offset + i] = inField[kj_offset + (i - xInterior)];
      }
    }
  }
}

template <typename T>
void apply_diffusion(T* __restrict__ inField, T* __restrict__ outField,
                     T alpha, unsigned int numIter, int x, int y, int z,
                     int halo) {

  const int xDim = x + 2 * halo;
  const int yDim = y + 2 * halo;
  const int zDim = z + 2 * halo;
  const std::size_t xyDim = static_cast<std::size_t>(xDim) * yDim;
  const std::size_t tmp1_size = static_cast<std::size_t>(xDim) * yDim;

  const int xMin = halo;
  const int xMax = x + halo;
  const int yMin = halo;
  const int yMax = y + halo;
  const int zMin = halo;
  const int zMax = z + halo;

  T* tmp1Field = new T[tmp1_size];

  for (std::size_t iter = 0; iter < numIter; ++iter) {
    updateHalo<T>(inField, x, y, z, halo);

    for (std::size_t k = zMin; k < zMax; ++k) {
      const std::size_t k_offset = k * xyDim;
      for (std::size_t j = yMin - 1; j < yMax + 1; ++j) {
        const std::size_t j_offset = j * xDim;

        for (std::size_t i = xMin - 1; i < xMax + 1; ++i) {
          const std::size_t idx = k_offset + j_offset + i;
          tmp1Field[j_offset + i] =
              static_cast<T>(-4.0) * inField[idx]
              + inField[idx - 1]
              + inField[idx + 1]
              + inField[idx - xDim]
              + inField[idx + xDim];
        }
      }

      for (std::size_t j = yMin; j < yMax; ++j) {
        const std::size_t j_offset = j * xDim;
        const std::size_t j_offset_up = (j - 1) * xDim;
        const std::size_t j_offset_dn = (j + 1) * xDim;

        for (std::size_t i = xMin; i < xMax; ++i) {
          const std::size_t center_tmp = j_offset + i;
          const std::size_t out_idx = k_offset + j_offset + i;

          T laplap = static_cast<T>(-4.0) * tmp1Field[center_tmp]
              + tmp1Field[center_tmp - 1]
              + tmp1Field[center_tmp + 1]
              + tmp1Field[j_offset_up + i]
              + tmp1Field[j_offset_dn + i];

          T val = inField[out_idx] - alpha * laplap;

          if (iter == numIter - 1) {
            outField[out_idx] = val;
          } else {
            inField[out_idx] = val;
          }
        }
      }
    }
  }
  delete[] tmp1Field;
}

template <typename T>
void reportTime(int x, int y, int z, int nIter, T diff) {
  std::cout << "# ranks nx ny nz num_iter time\ndata = np.array( [ \\\n";
  int size = 1;
#pragma omp parallel
  {
#pragma omp master
    { size = omp_get_num_threads(); }
  }
    std::cout << "[ " << size << ", " << x << ", "
            << y << ", " << z << ", "
            << nIter << ", " << static_cast<double>(diff) << "],\n"; // Cast diff to double for printing
  std::cout << "] )" << std::endl;
}

template <typename T>
void initializeField(T* field, int xDim, int yDim, int zDim_total, int halo) {
  const int zInterior = zDim_total - 2 * halo;
  const std::size_t xyDim = static_cast<std::size_t>(xDim) * yDim;
  const std::size_t total_elements = static_cast<std::size_t>(xDim) * yDim * zDim_total;

  for (std::size_t i = 0; i < total_elements; ++i) {
        field[i] = static_cast<T>(0.0);
   }

  std::size_t k_start_storage3d = static_cast<std::size_t>(halo + zInterior / 4.0);
  std::size_t k_end_storage3d = static_cast<std::size_t>(halo + 3.0 * zInterior / 4.0);

  std::size_t j_start = static_cast<std::size_t>(halo + yDim / 4.0);
  std::size_t j_end = static_cast<std::size_t>(halo + 3.0 * yDim / 4.0);

  std::size_t i_start = static_cast<std::size_t>(halo + xDim / 4.0);
  std::size_t i_end = static_cast<std::size_t>(halo + 3.0 * xDim / 4.0);

  for (std::size_t k = k_start_storage3d; k < k_end_storage3d; ++k) {
      for (std::size_t j = j_start; j < j_end; ++j) {
          for (std::size_t i = i_start; i < i_end; ++i) {
              field[k * xyDim + j * xDim + i] =  static_cast<T>(1.0);
          }
      }
  }
}

template <typename T>
void writeFieldToFile(const T* field, int xDim, int yDim, int zDim, int halo, std::ofstream& fout) {

    constexpr int32_t three = 3;
    // Adjust if T is double (64) or float (32)
    constexpr int32_t sixtyfour = sizeof(T) == 8 ? 64 : 32;
    int32_t writehalo = halo;
    int32_t writex = xDim;
    int32_t writey = yDim;
    int32_t writez = zDim;

    fout.write(reinterpret_cast<const char *>(&three), sizeof(three));
    fout.write(reinterpret_cast<const char *>(&sixtyfour), sizeof(sixtyfour));
    fout.write(reinterpret_cast<const char *>(&writehalo), sizeof(writehalo));
    fout.write(reinterpret_cast<const char *>(&writex), sizeof(writex));
    fout.write(reinterpret_cast<const char *>(&writey), sizeof(writey));
    fout.write(reinterpret_cast<const char *>(&writez), sizeof(writez));

    const std::size_t total_elements = static_cast<std::size_t>(xDim) * yDim * zDim;
    fout.write(reinterpret_cast<const char*>(field), total_elements * sizeof(T));
}


} // namespace

int main(int argc, char const *argv[]) {
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif
  int x = atoi(argv[2]);
  int y = atoi(argv[4]);
  int z = atoi(argv[6]);
  int iter = atoi(argv[8]);
  int nHalo = 3;
  assert(x > 0 && y > 0 && z > 0 && iter > 0);

  // Define the precision type to use (e.g., float or double)
  using precision_t = float;

  int totalX, totalY, totalZ;
  totalX = x + 2 * nHalo;
  totalY = y + 2 * nHalo;
  totalZ = z + 2 * nHalo;

  std::size_t size = static_cast<std::size_t>(totalX) * totalY * totalZ;
  precision_t* input = new precision_t[size];
  precision_t* output = new precision_t[size];

  initializeField<precision_t>(input, totalX, totalY, totalZ, nHalo);
  initializeField<precision_t>(output, totalX, totalY, totalZ, nHalo);

  precision_t alpha = static_cast<precision_t>(1.0 / 32.0);

  std::ofstream fout;
  fout.open("in_field.dat", std::ios::binary | std::ofstream::trunc);
  writeFieldToFile<precision_t>(input, totalX, totalY, totalZ, nHalo, fout);
  fout.close();
#ifdef CRAYPAT
  PAT_record(PAT_STATE_ON);
#endif
  auto start = std::chrono::steady_clock::now();

  apply_diffusion<precision_t>(input, output, alpha, iter, x, y, z, nHalo);

  auto end = std::chrono::steady_clock::now();
#ifdef CRAYPAT
  PAT_record(PAT_STATE_OFF);
#endif
  updateHalo<precision_t>(output, x, y, z, nHalo);

  fout.open("out_field.dat", std::ios::binary | std::ofstream::trunc);
  writeFieldToFile<precision_t>(output, totalX, totalY, totalZ, nHalo, fout);
  fout.close();

  auto diff_chrono = end - start;
  double timeDiff =
      std::chrono::duration<double, std::milli>(diff_chrono).count() / 1000.;
  reportTime<precision_t>(x, y, z, iter, static_cast<precision_t>(timeDiff));

  delete[] input;
  delete[] output;

  return 0;
}
