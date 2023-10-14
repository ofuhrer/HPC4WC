#include "stencils.hpp"

#include <chrono>
#include <iostream>
#include <fstream>

using List = std::initializer_list<std::size_t>;

static const List nx_vals = {4, 6, 8, 16, 32, 64, 128};
static const List ny_vals = {4, 6, 8, 16, 32, 64, 128};
static const List nz_vals = {64};
static const Halo num_halo = std::make_pair(1, 1);
static const std::size_t block_nx = 16;
static const std::size_t block_ny = 16;

#ifndef CACHE_TRACKING
static constexpr std::size_t runs = 100;
#else
static constexpr std::size_t runs = 10;
#endif

#ifndef CACHE_TRACKING
static const std::string fname_end = ".csv";
#else
static const std::string fname_end = "_hr.csv";
#endif

static void benchmark_spatial_baseline() {
  for (const auto &s : stencils) {
    const auto func = s.first;
    const std::string name = s.second;
    const std::string fname = name + fname_end;
    std::ofstream of(fname);
    of << "nx,ny,nz,runtime,hr1,hr2,cmr1,cmr2" << std::endl;

    for (const auto &nz : nz_vals) {
      for (const auto &ny : ny_vals) {
        for (const auto &nx : nx_vals) {
          std::size_t num_halo_max = std::max(num_halo.first, num_halo.second);
          if (nx < num_halo_max + 2 || ny < num_halo_max + 2 ||
              nz < num_halo_max + 2)
            continue;

          Arr a(nx, ny, nz);
          a.setRandom();
          Arr b = a;

          bool first_run = true;
          TWO_LEVEL_HITS cold_miss_rates;

          // warmup
          constexpr double min_warmup_time = 1e-1; // in seconds
          auto start = std::chrono::high_resolution_clock::now();
          do {
            func(a, b, num_halo, num_halo, num_halo);
            if (first_run) {
              cold_miss_rates = cache.get_cold_miss_rates();
              first_run = false;
            }
            a.setRandom();
          } while (std::chrono::duration<double>(
                       std::chrono::high_resolution_clock::now() - start)
                       .count() < min_warmup_time);

          // benchmark
          double runtime = 0;
          cache.reset_statistics();
          for (std::size_t i = 0; i < runs; ++i) {
            start = std::chrono::high_resolution_clock::now();
            func(a, b, num_halo, num_halo, num_halo);
            const auto stop = std::chrono::high_resolution_clock::now();
            runtime += std::chrono::duration<double>(stop - start).count();
            a.setRandom();
          }
          runtime /= runs;
          const auto hitrates = cache.get_hit_rates();
          std::cout << runtime << std::endl;

          // write to file
          of << nx << "," << ny << "," << nz << "," << runtime << ","
             << hitrates.first << "," << hitrates.second << ","
             << cold_miss_rates.first << "," << cold_miss_rates.second
             << std::endl;
        }
      }
    }
    std::cout << "Saved " + fname << std::endl;
  }
}

static void benchmark_spatial_blocking() {
  for (const auto &s : stencils_spatial_blocking) {
    const auto func = s.first;
    const std::string name = s.second;
    const std::string fname = name + fname_end;
    std::ofstream of(fname);
    of << "nx,ny,nz,runtime,hr1,hr2,cmr1,cmr2" << std::endl;

    for (const auto &nz : nz_vals) {
      for (const auto &ny : ny_vals) {
        for (const auto &nx : nx_vals) {
          std::size_t num_halo_max = std::max(num_halo.first, num_halo.second);
          if (nx < num_halo_max + 2 || ny < num_halo_max + 2 ||
              nz < num_halo_max + 2)
            continue;

          Arr a(nx, ny, nz);
          a.setRandom();
          Arr b = a;

          bool first_run = true;
          TWO_LEVEL_HITS cold_miss_rates;

          // warmup
          constexpr double min_warmup_time = 1e-1; // in seconds
          auto start = std::chrono::high_resolution_clock::now();
          do {
            func(a, b, num_halo, num_halo, num_halo, block_nx, block_ny);
            if (first_run) {
              cold_miss_rates = cache.get_cold_miss_rates();
              first_run = false;
            }
            a.setRandom();
          } while (std::chrono::duration<double>(
                       std::chrono::high_resolution_clock::now() - start)
                       .count() < min_warmup_time);

          // benchmark
          double runtime = 0;
          cache.reset_statistics();
          for (std::size_t i = 0; i < runs; ++i) {
            start = std::chrono::high_resolution_clock::now();
            func(a, b, num_halo, num_halo, num_halo, block_nx, block_ny);
            const auto stop = std::chrono::high_resolution_clock::now();
            runtime += std::chrono::duration<double>(stop - start).count();
            a.setRandom();
          }
          runtime /= runs;
          const auto hitrates = cache.get_hit_rates();
          std::cout << runtime << std::endl;

          // write to file
          of << nx << "," << ny << "," << nz << "," << runtime << ","
             << hitrates.first << "," << hitrates.second << ","
             << cold_miss_rates.first << "," << cold_miss_rates.second
             << std::endl;
        }
      }
    }
    std::cout << "Saved " + fname << std::endl;
  }
}

static void benchmark_temporal_baseline() {
  const auto func = &stencil_2D;
  const std::string name = "stencil_2D_temporal_blocked_baseline";
  const std::string fname = name + ".csv";
  std::ofstream of(fname);
  of << "nx,ny,nz,runtime,hr1,hr2,cmr1,cmr2" << std::endl;

  for (const auto &nz : nz_vals) {
    for (const auto &ny : ny_vals) {
      for (const auto &nx : nx_vals) {
        std::size_t num_halo_max = std::max(num_halo.first, num_halo.second);
        if (nx < num_halo_max + 2 || ny < num_halo_max + 2 ||
            nz < num_halo_max + 2)
          continue;

        Arr a(nx, ny, nz);
        a.setRandom();
        Arr b = a;

        bool first_run = true;
        TWO_LEVEL_HITS cold_miss_rates;

        // warmup
        constexpr double min_warmup_time = 1e-1; // in seconds
        auto start = std::chrono::high_resolution_clock::now();
        do {
          time_iteration_baseline(16, 0.1, func, a, b, num_halo, num_halo,
                                  num_halo);
          if (first_run) {
            cold_miss_rates = cache.get_cold_miss_rates();
            first_run = false;
          }
          a.setRandom();
        } while (std::chrono::duration<double>(
                     std::chrono::high_resolution_clock::now() - start)
                     .count() < min_warmup_time);

        // benchmark
        constexpr std::size_t runs_ = 50;
        double runtime = 0;
        cache.reset_statistics();
        for (std::size_t i = 0; i < runs_; ++i) {
          start = std::chrono::high_resolution_clock::now();

          time_iteration_baseline(16, 0.1, func, a, b, num_halo, num_halo,
                                  num_halo);

          const auto stop = std::chrono::high_resolution_clock::now();
          runtime += std::chrono::duration<double>(stop - start).count();
          a.setRandom();
        }
        runtime /= runs_;
        const auto hitrates = cache.get_hit_rates();

        // write to file
        of << nx << "," << ny << "," << nz << "," << runtime << ","
           << hitrates.first << "," << hitrates.second << ","
           << cold_miss_rates.first << "," << cold_miss_rates.second
           << std::endl;

        std::cout << nx << "," << ny << "," << nz << "," << runtime << ","
                  << hitrates.first << "," << hitrates.second << ","
                  << cold_miss_rates.first << "," << cold_miss_rates.second
                  << std::endl;
      }
    }
  }
  std::cout << "Saved " + fname << std::endl;
}

static void benchmark_temporal_blocking() {

  const auto func = &stencil_2D_time;
  const std::string name = "stencil_2D_temporal_blocked";
  const std::string fname = name + ".csv";
  std::ofstream of(fname);
  of << "nx,ny,nz,runtime,hr1,hr2,cmr1,cmr2" << std::endl;

  for (const auto &nz : nz_vals) {
    for (const auto &ny : ny_vals) {
      for (const auto &nx : nx_vals) {
        std::size_t num_halo_max = std::max(num_halo.first, num_halo.second);
        if (nx < num_halo_max + 2 || ny < num_halo_max + 2 ||
            nz < num_halo_max + 2)
          continue;

        Arr a(nx, ny, nz);
        a.setRandom();
        Arr b = a;

        bool first_run = true;
        TWO_LEVEL_HITS cold_miss_rates;

        // warmup
        constexpr double min_warmup_time = 1e-1; // in seconds
        auto start = std::chrono::high_resolution_clock::now();
        do {
          func(16, 0.1, a, b, num_halo, num_halo, num_halo, 0, 0, 0);
          if (first_run) {
            cold_miss_rates = cache.get_cold_miss_rates();
            first_run = false;
          }
          a.setRandom();
        } while (std::chrono::duration<double>(
                     std::chrono::high_resolution_clock::now() - start)
                     .count() < min_warmup_time);

        // benchmark
        constexpr std::size_t runs_ = 50;
        double runtime = 0;
        cache.reset_statistics();
        for (std::size_t i = 0; i < runs_; ++i) {
          start = std::chrono::high_resolution_clock::now();

          func(16, 0.1, a, b, num_halo, num_halo, num_halo, 0, 0, 0);

          const auto stop = std::chrono::high_resolution_clock::now();
          runtime += std::chrono::duration<double>(stop - start).count();
          a.setRandom();
        }
        runtime /= runs_;
        const auto hitrates = cache.get_hit_rates();

        // write to file
        of << nx << "," << ny << "," << nz << "," << runtime << ","
           << hitrates.first << "," << hitrates.second << ","
           << cold_miss_rates.first << "," << cold_miss_rates.second
           << std::endl;

        std::cout << nx << "," << ny << "," << nz << "," << runtime << ","
                  << hitrates.first << "," << hitrates.second << ","
                  << cold_miss_rates.first << "," << cold_miss_rates.second
                  << std::endl;
      }
    }
  }
  std::cout << "Saved " + fname << std::endl;
}

static void benchmark_2d_asymmetric_blocking() {

  const List block_nx_list = {4, 8, 16, 32, 64, 128};
  const List block_ny_list = {4, 8, 16, 32, 64, 128};
  const std::size_t nx = 128;
  const std::size_t ny = 128;
  const std::size_t nz = 32;

  const auto func = &stencil_2D_blocked;
  const std::string name = "stencil_2D_asymmetric_blocked_hits";
  const std::string fname = name + ".csv";
  std::ofstream of(fname);
  of << "block_nx,block_ny,runtime,hr1,hr2,cmr1,cmr2" << std::endl;

  for (const auto &block_nx_ : block_nx_list) {
    for (const auto &block_ny_ : block_ny_list) {

      Arr a(nx, ny, nz);
      a.setRandom();
      Arr b = a;

      bool first_run = true;
      TWO_LEVEL_HITS cold_miss_rates;

      // warmup
      constexpr double min_warmup_time = 1e-1; // in seconds
      auto start = std::chrono::high_resolution_clock::now();
      do {
        func(a, b, num_halo, num_halo, num_halo, block_nx_, block_ny_);
        if (first_run) {
          cold_miss_rates = cache.get_cold_miss_rates();
          first_run = false;
        }
        a.setRandom();
      } while (std::chrono::duration<double>(
                   std::chrono::high_resolution_clock::now() - start)
                   .count() < min_warmup_time);

      // benchmark
      constexpr std::size_t runs_ = 10;
      double runtime = 0;
      cache.reset_statistics();
      for (std::size_t i = 0; i < runs_; ++i) {
        start = std::chrono::high_resolution_clock::now();

        func(a, b, num_halo, num_halo, num_halo, block_nx_, block_ny_);

        const auto stop = std::chrono::high_resolution_clock::now();
        runtime += std::chrono::duration<double>(stop - start).count();
        a.setRandom();
      }
      runtime /= runs_;
      const auto hitrates = cache.get_hit_rates();

      // write to file
      of << block_nx_ << "," << block_ny_ << "," << runtime << ","
         << hitrates.first << "," << hitrates.second << ","
         << cold_miss_rates.first << "," << cold_miss_rates.second << std::endl;

      std::cout << block_nx_ << "," << block_ny_ << "," << runtime << ","
                << hitrates.first << "," << hitrates.second << ","
                << cold_miss_rates.first << "," << cold_miss_rates.second
                << std::endl;
    }
  }
  std::cout << "Saved " + fname << std::endl;
}

int main() {
  benchmark_spatial_baseline();
  benchmark_spatial_blocking();
  return 0;
  // specialized experiments
  benchmark_temporal_baseline();
  benchmark_temporal_blocking();
  benchmark_2d_asymmetric_blocking();
  return 0;
}
