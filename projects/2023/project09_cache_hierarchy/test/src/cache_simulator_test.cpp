#include <array>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <gtest/gtest.h>
#include <unsupported/Eigen/CXX11/Tensor>

#include "cache_simulator.hpp"
#include "stencils.hpp"
#include "utils.hpp"

const static std::string PATH_TO_TEST_FILES = "../test/data/";

TEST(cache_simulator, cache_test_every_second) {
  Cache c(4, 1, sizeof(double) * 2, 64);
  std::vector<double> test_vec(c.cache_size_bytes() / sizeof(double));

  for (std::size_t i = 0; i < test_vec.size(); ++i) {
    c.access(test_vec.data() + i);
  }

  EXPECT_DOUBLE_EQ(c.get_hit_rate(), 1. / 2.);
}

TEST(cache_simulator, cache_test_fits_all) {
  Cache c(4, 1, sizeof(double) * 2, 64);
  std::vector<double> test_vec(c.cache_size_bytes() / sizeof(double));

  for (std::size_t i = 0; i < test_vec.size(); ++i) {
    c.access(test_vec.data() + i);
  }

  c.reset_statistics();
  for (std::size_t i = 0; i < test_vec.size(); ++i) {
    c.access(test_vec.data() + i);
  }

  EXPECT_DOUBLE_EQ(c.get_hit_rate(), 1.);
}

TEST(cache_simulator, cache_test_stride_two) {
  Cache c(4, 1, sizeof(double) * 2, 64);
  std::vector<double> test_vec(c.cache_size_bytes() / sizeof(double));

  for (std::size_t i = 0; i < test_vec.size(); i += 2) {
    c.access(test_vec.data() + i);
  }

  EXPECT_DOUBLE_EQ(c.get_hit_rate(), 0.);
}

TEST(cache_simulator, cache_test_associativity_1) {
  Cache c(4, 1, sizeof(double), 64);
  std::vector<double> test_vec(2 * c.cache_size_bytes() / sizeof(double));

  for (std::size_t i = 0; i < test_vec.size(); i += 1) {
    c.access(test_vec.data() + i);
  }

  for (std::size_t i = 0; i < test_vec.size(); i += 1) {
    c.access(test_vec.data() + i);
  }

  EXPECT_DOUBLE_EQ(c.get_hit_rate(), 0.);
}

TEST(cache_simulator, cache_test_associativity_2) {
  Cache c(4, 2, sizeof(double), 64);
  std::vector<double> test_vec(c.cache_size_bytes() / sizeof(double));

  for (std::size_t i = 0; i < test_vec.size(); i += 1) {
    c.access(test_vec.data() + i);
  }

  for (std::size_t i = 0; i < test_vec.size(); i += 1) {
    c.access(test_vec.data() + i);
  }

  EXPECT_DOUBLE_EQ(c.get_hit_rate(), 1. / 2.);
}

TEST(cache_simulator, cache_test_replacement_policy_LRU) {
  Cache c(1, 2, sizeof(double), 64, &CacheSet::policy_LRU);
  std::vector<double> test_vec(3 * c.cache_size_bytes() / sizeof(double));

  for (std::size_t i = 0; i < test_vec.size(); i += 2) {
    c.access(test_vec.data() + i);
  }

  EXPECT_FALSE(c.access(test_vec.data()));

  for (std::size_t i = 0; i < test_vec.size(); i += 2) {
    c.access(test_vec.data() + i);
  }

  EXPECT_TRUE(c.access(test_vec.data() + 2));
}

TEST(cache_simulator, cache_test_replacement_policy_LFU) {
  Cache c(1, 2, sizeof(double), 64, &CacheSet::policy_LFU);
  std::vector<double> test_vec(3 * c.cache_size_bytes() / sizeof(double));

  c.access(test_vec.data());

  for (std::size_t i = 0; i < test_vec.size(); i += 2) {
    c.access(test_vec.data() + i);
  }

  EXPECT_TRUE(c.access(test_vec.data()));
}

TEST(cache_simulator, cache_test_eigen_2d_tensor) {
  std::size_t dim = 2;
  Cache c(dim * dim, 1, sizeof(double), 64);

  std::vector<double> test_vec(dim * dim);
  Eigen::TensorMap<Eigen::Tensor<double, 2>> map(test_vec.data(), dim, dim);
  Eigen::Tensor<double, 2, Eigen::ColMajor> tensor = map;

  for (std::size_t i = 0; i < dim; i += 1) {
    for (std::size_t j = 0; j < dim; j += 1) {
      c.access(&tensor(i, j));
    }
  }

  for (std::size_t i = 0; i < dim; i += 1) {
    for (std::size_t j = 0; j < dim; j += 1) {
      c.access(&tensor(i, j));
    }
  }

  EXPECT_DOUBLE_EQ(c.get_hit_rate(), 4. / 8.);
}

TEST(cache_simulator, cache_test_multilevel) {
  Cache *const l1_ = new Cache(1, 2, sizeof(double), 64);
  Cache *const l2_ = new Cache(2, 2, sizeof(double), 64);

  TwoLevelCache c(l1_, l2_);

  std::vector<double> test_vec(l2_->cache_size_bytes() / sizeof(double));

  for (std::size_t i = 0; i < test_vec.size(); i += 1) {
    c.access(test_vec.data() + i);
  }

  for (std::size_t i = 0; i < test_vec.size(); i += 1) {
    c.access(test_vec.data() + i);
  }

  auto hit_rates = c.get_hit_rates();

  EXPECT_DOUBLE_EQ(hit_rates.first, 0.);
  EXPECT_DOUBLE_EQ(hit_rates.second, 1. / 2.);

  delete l1_;
  delete l2_;
}

TEST(cache_simulator, cache_test_eigen_3d_tensor) {
  std::size_t dim = 2;
  Cache c(dim * dim * dim / 2, 1, 2 * sizeof(double), 64);

  Eigen::Tensor<double, 3> tensor(dim, dim, dim);

  for (std::size_t i = 0; i < dim; i += 1) {
    for (std::size_t j = 0; j < dim; j += 1) {
      for (std::size_t k = 0; k < dim; k += 1) {
        // std::cout << "access at (" << i << "," << j << "," << k << ")" <<
        // std::endl;
        c.access(&tensor(k, j, i));
      }
    }
  }

  c.reset_statistics();

  for (std::size_t i = 0; i < dim; i += 1) {
    for (std::size_t j = 0; j < dim; j += 1) {
      for (std::size_t k = 0; k < dim; k += 1) {
        c.access(&tensor(i, j, k));
      }
    }
  }

  EXPECT_DOUBLE_EQ(c.get_hit_rate(), 1.);
}

TEST(cache_simulator, cache_test_cold_misses) {
  std::size_t dim = 2;
  Cache c(dim * dim * dim / 2, 1, sizeof(double), 64);

  Eigen::Tensor<double, 3> tensor(dim, dim, dim);

  for (std::size_t i = 0; i < dim; i += 1) {
    for (std::size_t j = 0; j < dim; j += 1) {
      for (std::size_t k = 0; k < dim; k += 1) {
        // std::cout << "access at (" << i << "," << j << "," << k << ")" <<
        // std::endl;
        c.access(&tensor(k, j, i));
      }
    }
  }

  for (std::size_t i = 0; i < dim; i += 1) {
    for (std::size_t j = 0; j < dim; j += 1) {
      for (std::size_t k = 0; k < dim; k += 1) {
        // std::cout << "access at (" << i << "," << j << "," << k << ")" <<
        // std::endl;
        c.access(&tensor(k, j, i));
      }
    }
  }

  EXPECT_DOUBLE_EQ(c.get_cold_miss_rate(), 0.5);
}

TEST(cache_simulator, cache_test_statistics) {
  Arr field_init;
  std::string output;
  DIMENSION dim = read_tensor_from_csv(
      PATH_TO_TEST_FILES + "test_case_stencil_copy_random_init.csv",
      field_init);

  testing::internal::CaptureStdout();
  l1.print_statistics();
  output = testing::internal::GetCapturedStdout();
  ASSERT_EQ(output.substr(0, 3), "L1 ");

  Arr field_out = field_init;
  time_iteration_baseline(16, 0.1, &stencil_2D, field_init, field_out,
                          std::make_pair(1, 1), std::make_pair(1, 1),
                          std::make_pair(0, 0));

  testing::internal::CaptureStdout();
  l1.print_statistics();
  output = testing::internal::GetCapturedStdout();
  ASSERT_EQ(output.substr(0, 3), "L1 ");
  l1.flush();

  testing::internal::CaptureStdout();
  l1.print_statistics();
  output = testing::internal::GetCapturedStdout();
  ASSERT_EQ(output.substr(0, 3), "L1 ");

  stencil_2D_time(16, 0.1, field_init, field_out, std::make_pair(1, 1),
                  std::make_pair(1, 1), std::make_pair(0, 0), 0, 0, 0);

  testing::internal::CaptureStdout();
  l1.print_statistics();
  output = testing::internal::GetCapturedStdout();
  ASSERT_EQ(output.substr(0, 3), "L1 ");
}

TEST(cache_simulator, stencil_copy) {
  Arr field_init;
  DIMENSION dim = read_tensor_from_csv(
      PATH_TO_TEST_FILES + "test_case_stencil_copy_random_init.csv",
      field_init);

  /*std::cout << "dims: " << dim[0] << ", " << dim[1] << ", " << dim[2]
            << std::endl;
  std::cout << "TENSOR SIZE: " << dim[0] * dim[1] * dim[2] * sizeof(double)
            << std::endl;
  std::cout << "WORKING-SET SIZE: "
            << 2 * dim[0] * dim[1] * dim[2] * sizeof(double) << std::endl;
  std::cout << "L1 SIZE: " << l1.cache_size_bytes() << std::endl;
  std::cout << "L2 SIZE: " << l2.cache_size_bytes() << std::endl;*/

  Arr field_out = field_init;
  stencil_copy(field_init, field_out, std::make_pair(0, 0),
               std::make_pair(0, 0), std::make_pair(0, 0));

  l1.flush();
  l2.flush();

  l1.reset_statistics();
  l2.reset_statistics();
  stencil_copy(field_init, field_out, std::make_pair(0, 0),
               std::make_pair(0, 0), std::make_pair(0, 0));

  l1.reset_statistics();
  l2.reset_statistics();
  stencil_copy_blocked(field_init, field_out, std::make_pair(0, 0),
                       std::make_pair(0, 0), std::make_pair(0, 0), 32, 32);

  // std::cout << "(blocked) L1 HIT-RATE: " << l1.get_hit_rate() << std::endl;
  // std::cout << "(blocked) L2 HIT-RATE: " << l2.get_hit_rate() << std::endl;

  ASSERT_LT(l1.get_hit_rate(), 0.9);
  ASSERT_GT(l1.get_hit_rate(), 0.8);
  ASSERT_EQ(l2.get_hit_rate(), 1);
}
