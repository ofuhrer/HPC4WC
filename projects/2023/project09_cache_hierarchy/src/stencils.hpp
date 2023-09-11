#ifndef STENCILS_HPP
#define STENCILS_HPP

#include "cache_simulator.hpp"

#include <unsupported/Eigen/CXX11/Tensor>

// stringification
#define str(s) #s

// only column-major supported for Eigen tensors!
// column-major in this case means the following order in memory:
// a(0,0,0), a(1,0,0), ..., a(n-1,0,0), a(0,1,0), a(1,1,0), ..., a(n-1,1,0), ...
// ==> addresses increase by incrementing indices from left to right
using Arr = Eigen::Tensor<double, 3, Eigen::ColMajor>;
using Halo = std::pair<std::size_t, std::size_t>;

// cache simulator
// Intel i9-9900K stats:
// L1: 32768 bytes, 64 sets, 8-way associative, 64 bytes per cache line
// L2: 262144 bytes, 1024 sets, 4-way associative, 64 bytes per cache line
// L3: 16777216 bytes, 16384 sets, 16-way associative, 64 bytes per cache line
static Cache l1(64, 8, 64, 64, "L1");
static Cache l2(1024, 4, 64, 64, "L2");
static TwoLevelCache cache(&l1, &l2);

// stencil macros for cache simulator
#define one_point_stencil(b, a1)                                               \
  cache.access(&b);                                                            \
  cache.access(&a1);                                                           \
  b = a1
#define two_point_stencil(b, a1, a2)                                           \
  cache.access(&b);                                                            \
  cache.access(&a1);                                                           \
  cache.access(&a2);                                                           \
  b = a1 + a2
#define three_point_stencil(b, a1, a2, a3)                                     \
  cache.access(&b);                                                            \
  cache.access(&a1);                                                           \
  cache.access(&a2);                                                           \
  cache.access(&a3);                                                           \
  b = a1 + a2 + a3
#define five_point_stencil(b, a1, a2, a3, a4, a5)                              \
  cache.access(&b);                                                            \
  cache.access(&a1);                                                           \
  cache.access(&a2);                                                           \
  cache.access(&a3);                                                           \
  cache.access(&a4);                                                           \
  cache.access(&a5);                                                           \
  b = a1 + a2 + a3 + a4 + a5
#define seven_point_stencil(b, a1, a2, a3, a4, a5, a6, a7)                     \
  cache.access(&b);                                                            \
  cache.access(&a1);                                                           \
  cache.access(&a2);                                                           \
  cache.access(&a3);                                                           \
  cache.access(&a4);                                                           \
  cache.access(&a5);                                                           \
  cache.access(&a6);                                                           \
  cache.access(&a7);                                                           \
  b = a1 + a2 + a3 + a4 + a5 + a6 + a7

//-----------------------STENCIL TIME iteration template-----------------------

template <typename T>
inline static void
time_iteration_baseline(std::size_t N, const double &alpha, T stencil, Arr &a,
                        Arr &b, const Halo &halo_nx, const Halo &halo_ny,
                        const Halo &halo_nz) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  for (std::size_t n = 0; n < N; n++) {
    stencil(a, b, halo_nx, halo_ny, halo_nz);

    for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k) {
      for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; ++j) {
        for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
          cache.access(&b(i, j, k));
          cache.access(&a(i, j, k));
          b(i, j, k) = a(i, j, k) - alpha * b(i, j, k);
        }
      }
    }
    // omit swap for counting misses
    if (n < N - 1)
      std::swap(a, b);
  }
}

//-----------------------COPY STENCIL-----------------------

inline static void stencil_copy(Arr &a, Arr &b, const Halo &halo_nx,
                                const Halo &halo_ny, const Halo &halo_nz) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; ++j)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
        one_point_stencil(b(i, j, k), a(i, j, k));
      }
}

inline static void stencil_copy_blocked(Arr &a, Arr &b, const Halo &halo_nx,
                                        const Halo &halo_ny,
                                        const Halo &halo_nz,
                                        const std::size_t &block_nx,
                                        const std::size_t &block_ny) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(block_nx > 0 && block_ny > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; j += block_ny)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second;
           i += block_nx)
        for (std::size_t l = j;
             l < std::min(j + block_ny, ny - halo_ny.second - j); ++l)
          for (std::size_t m = i;
               m < std::min(i + block_nx, nx - halo_nx.second - i); ++m) {
            one_point_stencil(b(m, l, k), a(m, l, k));
          }
}

inline static void stencil_copy_time(std::size_t N, const double &alpha, Arr &a,
                                     Arr &b, const Halo &halo_nx,
                                     const Halo &halo_ny, const Halo &halo_nz,
                                     const std::size_t & /*block_nx*/,
                                     const std::size_t & /*block_ny*/,
                                     const std::size_t & /*block_nt*/) {

  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_nx.first > 0);
  // assert(block_nx > 0 && block_ny > 0 && block_nt > 0);

  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k) {
      for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; ++j) {
        for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
          one_point_stencil(b(i, j, k), a(i, j, k));
        }
      }
    }

    b = a - alpha * b;
    if (n < N - 1)
      std::swap(a, b);
  }
}

//-----------------------1D STENCIL in 1st dimension-----------------------
// b(i, j, k) = a(i, j, k) + a(i-1, j, k)

inline static void stencil_1D_i1(Arr &a, Arr &b, const Halo &halo_nx,
                                 const Halo &halo_ny, const Halo &halo_nz) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_nx.first > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; ++j)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
        two_point_stencil(b(i, j, k), a(i, j, k), a(i - 1, j, k));
      }
}

inline static void stencil_1D_i1_blocked(Arr &a, Arr &b, const Halo &halo_nx,
                                         const Halo &halo_ny,
                                         const Halo &halo_nz,
                                         const std::size_t &block_nx,
                                         const std::size_t &block_ny) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_nx.first > 0);
  assert(block_nx > 0 && block_ny > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; j += block_ny)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second;
           i += block_nx)
        for (std::size_t l = j; l < std::min(j + block_ny, ny - halo_ny.second);
             ++l)
          for (std::size_t m = i;
               m < std::min(i + block_nx, nx - halo_nx.second); ++m) {
            two_point_stencil(b(m, l, k), a(m, l, k), a(m - 1, l, k));
          }
}

inline static void stencil_1D_i1_time(std::size_t N, const double &alpha,
                                      Arr &a, Arr &b, const Halo &halo_nx,
                                      const Halo &halo_ny, const Halo &halo_nz,
                                      const std::size_t & /*block_nx*/,
                                      const std::size_t & /*block_ny*/,
                                      const std::size_t & /*block_nt*/) {

  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_nx.first > 0);
  // assert(block_nx > 0 && block_ny > 0 && block_nt > 0);

  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k) {
      for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; ++j) {
        for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
          two_point_stencil(b(i, j, k), a(i, j, k), a(i - 1, j, k));
        }
      }
    }

    b = a - alpha * b;
    if (n < N - 1)
      std::swap(a, b);
  }
}

//-----------------------1D STENCIL in 1st dimension-----------------------
// b(i, j, k) = a(i + 1, j, k) + a(i, j, k) + a(i-1, j, k)

inline static void stencil_1D_i2(Arr &a, Arr &b, const Halo &halo_nx,
                                 const Halo &halo_ny, const Halo &halo_nz) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_nx.first > 0 && halo_nx.second > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; ++j)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
        three_point_stencil(b(i, j, k), a(i + 1, j, k), a(i, j, k),
                            a(i - 1, j, k));
      }
}

inline static void stencil_1D_i2_blocked(Arr &a, Arr &b, const Halo &halo_nx,
                                         const Halo &halo_ny,
                                         const Halo &halo_nz,
                                         const std::size_t &block_nx,
                                         const std::size_t &block_ny) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_nx.first > 0 && halo_nx.second > 0);
  assert(block_nx > 0 && block_ny > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; j += block_ny)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second;
           i += block_nx)
        for (std::size_t l = j; l < std::min(j + block_ny, ny - halo_ny.second);
             ++l)
          for (std::size_t m = i;
               m < std::min(i + block_nx, nx - halo_nx.second); ++m) {
            three_point_stencil(b(m, l, k), a(m + 1, l, k), a(m, l, k),
                                a(m - 1, l, k));
          }
}

/* TODO
inline static void stencil_1D_i2_time(std::size_t N, const double &alpha,
                                      Arr &a, Arr &b, const Halo &halo_nx,
                                      const Halo &halo_ny, const Halo &halo_nz,
                                      const std::size_t &block_nx,
                                      const std::size_t &block_ny,
                                      const std::size_t &block_nt) {}
                                      */

//-----------------------1D STENCIL in 2nd dimension-----------------------
// b(i, j, k) = a(i, j, k) + a(i, j - 1, k)

inline static void stencil_1D_j1(Arr &a, Arr &b, const Halo &halo_nx,
                                 const Halo &halo_ny, const Halo &halo_nz) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_ny.first > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; ++j)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
        two_point_stencil(b(i, j, k), a(i, j, k), a(i, j - 1, k));
      }
}

inline static void stencil_1D_j1_blocked(Arr &a, Arr &b, const Halo &halo_nx,
                                         const Halo &halo_ny,
                                         const Halo &halo_nz,
                                         const std::size_t &block_nx,
                                         const std::size_t &block_ny) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_ny.first > 0);
  assert(block_nx > 0 && block_ny > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; j += block_ny)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second;
           i += block_nx)
        for (std::size_t l = j; l < std::min(j + block_ny, ny - halo_ny.second);
             ++l)
          for (std::size_t m = i;
               m < std::min(i + block_nx, nx - halo_nx.second); ++m) {
            two_point_stencil(b(m, l, k), a(m, l, k), a(m, l - 1, k));
          }
}

/* TODO
inline static void stencil_1D_j1_time(std::size_t N, const double &alpha,
                                      Arr &a, Arr &b, const Halo &halo_nx,
                                      const Halo &halo_ny, const Halo &halo_nz,
                                      const std::size_t &block_nx,
                                      const std::size_t &block_ny,
                                      const std::size_t &block_nt) {}
                                      */

//-----------------------1D STENCIL in 2nd dimension-----------------------
// b(i, j, k) = a(i, j + 1, k) + a(i, j, k) + a(i, j - 1, k);

inline static void stencil_1D_j2(Arr &a, Arr &b, const Halo &halo_nx,
                                 const Halo &halo_ny, const Halo &halo_nz) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_ny.first > 0 && halo_ny.second > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; ++j)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
        three_point_stencil(b(i, j, k), a(i, j + 1, k), a(i, j, k),
                            a(i, j - 1, k));
      }
}

inline static void stencil_1D_j2_blocked(Arr &a, Arr &b, const Halo &halo_nx,
                                         const Halo &halo_ny,
                                         const Halo &halo_nz,
                                         const std::size_t &block_nx,
                                         const std::size_t &block_ny) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_ny.first > 0 && halo_ny.second > 0);
  assert(block_nx > 0 && block_ny > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; j += block_ny)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second;
           i += block_nx)
        for (std::size_t l = j; l < std::min(j + block_ny, ny - halo_ny.second);
             ++l)
          for (std::size_t m = i;
               m < std::min(i + block_nx, nx - halo_nx.second); ++m) {
            three_point_stencil(b(m, l, k), a(m, l + 1, k), a(m, l, k),
                                a(m, l - 1, k));
          }
}

/* TODO
inline static void stencil_1D_j2_time(std::size_t N, const double &alpha,
                                      Arr &a, Arr &b, const Halo &halo_nx,
                                      const Halo &halo_ny, const Halo &halo_nz,
                                      const std::size_t &block_nx,
                                      const std::size_t &block_ny,
                                      const std::size_t &block_nt) {}
                                      */

//-----------------------2D STENCIL-----------------------
// b(i, j, k) = a(i, j, k) + a(i - 1, j, k) + a(i + 1, j, k) + a(i, j - 1, k) +
// a(i, j + 1, k);

inline static void stencil_2D(Arr &a, Arr &b, const Halo &halo_nx,
                              const Halo &halo_ny, const Halo &halo_nz) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_nx.first > 0 && halo_nx.second > 0);
  assert(halo_ny.first > 0 && halo_ny.second > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; ++j)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
        five_point_stencil(b(i, j, k), a(i, j, k), a(i - 1, j, k),
                           a(i + 1, j, k), a(i, j - 1, k), a(i, j + 1, k));
      }
}

inline static void stencil_2D_blocked(Arr &a, Arr &b, const Halo &halo_nx,
                                      const Halo &halo_ny, const Halo &halo_nz,
                                      const std::size_t &block_nx,
                                      const std::size_t &block_ny) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_nx.first > 0 && halo_nx.second > 0);
  assert(halo_ny.first > 0 && halo_ny.second > 0);
  assert(block_nx > 0 && block_ny > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; j += block_ny)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second;
           i += block_nx)
        for (std::size_t l = j; l < std::min(j + block_ny, ny - halo_ny.second);
             ++l)
          for (std::size_t m = i;
               m < std::min(i + block_nx, nx - halo_nx.second); ++m) {
            five_point_stencil(b(m, l, k), a(m, l, k), a(m - 1, l, k),
                               a(m + 1, l, k), a(m, l - 1, k), a(m, l + 1, k));
          }
}

inline static void stencil_2D_time(std::size_t N, const double &alpha, Arr &a,
                                   Arr &b, const Halo &halo_nx,
                                   const Halo &halo_ny, const Halo &halo_nz,
                                   const std::size_t & /*block_nx*/,
                                   const std::size_t & /*block_ny*/,
                                   const std::size_t & /*block_nt*/) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_nx.first > 0 && halo_nx.second > 0);
  assert(halo_ny.first > 0 && halo_ny.second > 0);
  // assert(block_nx > 0 && block_ny > 0);

  for (std::size_t n = 0; n < N; ++n) {
    for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k) {

      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
        five_point_stencil(
            b(i, halo_ny.first, k), a(i, halo_ny.first, k),
            a(i - 1, halo_ny.first, k), a(i + 1, halo_ny.first, k),
            a(i, halo_ny.first - 1, k), a(i, halo_ny.first + 1, k));
      }

      for (std::size_t j = halo_ny.first + 1; j < ny - halo_ny.second; ++j) {
        for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
          five_point_stencil(b(i, j, k), a(i, j, k), a(i - 1, j, k),
                             a(i + 1, j, k), a(i, j - 1, k), a(i, j + 1, k));

          cache.access(&b(i, j - 1, k));
          cache.access(&a(i, j - 1, k));

          b(i, j - 1, k) = a(i, j - 1, k) - alpha * b(i, j - 1, k);
        }
      }

      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
        cache.access(&b(i, ny - halo_ny.second - 1, k));
        cache.access(&a(i, ny - halo_ny.second - 1, k));

        b(i, ny - halo_ny.second - 1, k) =
            a(i, ny - halo_ny.second - 1, k) -
            alpha * b(i, ny - halo_ny.second - 1, k);
      }
    }
    if (n < N - 1)
      std::swap(a, b);
  }
}

inline static void stencil_3D(Arr &a, Arr &b, const Halo &halo_nx,
                              const Halo &halo_ny, const Halo &halo_nz) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_nx.first > 0 && halo_nx.second > 0);
  assert(halo_ny.first > 0 && halo_ny.second > 0);
  assert(halo_nz.first > 0 && halo_nz.second > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; ++j)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second; ++i) {
        seven_point_stencil(b(i, j, k), a(i, j, k), a(i - 1, j, k),
                            a(i + 1, j, k), a(i, j - 1, k), a(i, j + 1, k),
                            a(i, j, k - 1), a(i, j, k + 1));
      }
}

inline static void stencil_3D_blocked(Arr &a, Arr &b, const Halo &halo_nx,
                                      const Halo &halo_ny, const Halo &halo_nz,
                                      const std::size_t &block_nx,
                                      const std::size_t &block_ny) {
  const auto &d = a.dimensions();
  const std::size_t nx = d[0];
  const std::size_t ny = d[1];
  const std::size_t nz = d[2];

  assert(nx > 0 && ny > 0 && nz > 0);
  assert(halo_nx.first + halo_nx.second < nx);
  assert(halo_ny.first + halo_ny.second < ny);
  assert(halo_nz.first + halo_nz.second < nz);
  assert(halo_nx.first > 0 && halo_nx.second > 0);
  assert(halo_ny.first > 0 && halo_ny.second > 0);
  assert(halo_nz.first > 0 && halo_nz.second > 0);
  assert(block_nx > 0 && block_ny > 0);

  for (std::size_t k = halo_nz.first; k < nz - halo_nz.second; ++k)
    for (std::size_t j = halo_ny.first; j < ny - halo_ny.second; j += block_ny)
      for (std::size_t i = halo_nx.first; i < nx - halo_nx.second;
           i += block_nx)
        for (std::size_t l = j; l < std::min(j + block_ny, ny - halo_ny.second);
             ++l)
          for (std::size_t m = i;
               m < std::min(i + block_nx, nx - halo_nx.second); ++m) {
            seven_point_stencil(b(m, l, k), a(m, l, k), a(m - 1, l, k),
                                a(m + 1, l, k), a(m, l - 1, k), a(m, l + 1, k),
                                a(m, l, k - 1), a(m, l, k + 1));
          }
}

/* TODO
inline static void stencil_3D_time(std::size_t N, const double &alpha, Arr &a,
                                   Arr &b, const Halo &halo_nx,
                                   const Halo &halo_ny, const Halo &halo_nz,
                                   const std::size_t &block_nx,
                                   const std::size_t &block_ny,
                                   const std::size_t &block_nt) {}
                                   */

static const auto stencils = {
    std::make_pair(stencil_copy, str(stencil_copy)),
    std::make_pair(stencil_1D_i1, str(stencil_1D_i1)),
    std::make_pair(stencil_1D_i2, str(stencil_1D_i2)),
    std::make_pair(stencil_1D_j1, str(stencil_1D_j1)),
    std::make_pair(stencil_1D_j2, str(stencil_1D_j2)),
    std::make_pair(stencil_2D, str(stencil_2D)),
    std::make_pair(stencil_3D, str(stencil_3D)),
};

static const auto stencils_spatial_blocking = {
    std::make_pair(stencil_copy_blocked, str(stencil_copy_blocked)),
    std::make_pair(stencil_1D_i1_blocked, str(stencil_1D_i1_blocked)),
    std::make_pair(stencil_1D_i2_blocked, str(stencil_1D_i2_blocked)),
    std::make_pair(stencil_1D_j1_blocked, str(stencil_1D_j1_blocked)),
    std::make_pair(stencil_1D_j2_blocked, str(stencil_1D_j2_blocked)),
    std::make_pair(stencil_2D_blocked, str(stencil_2D_blocked)),
    std::make_pair(stencil_3D_blocked, str(stencil_3D_blocked)),
};

#endif
