#include <array>
#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <unsupported/Eigen/CXX11/Tensor>
#include <gtest/gtest.h>

#include "stencils.hpp"
#include "utils.hpp"

const static std::string PATH_TO_TEST_FILES = "../test/data/";
const static std::size_t BLOCKING_SIZE = 13;
const static double EPS = 1e-10;

double difference(const Arr &field1, const Arr &field2) {
  /*std::cout << field1 << std::endl;
  std::cout << std::endl;
  std::cout << field2 << std::endl;
  std::cout << std::endl;
  std::cout << (field1-field2) << std::endl;*/
  Eigen::Tensor<double, 0> diff = (field1 - field2).abs().sum();
  return diff(0);
}

template <typename T>
std::pair<double, DIMENSION>
stencil_test(const std::string &test_case, T *stencil, const Halo &halo_nx,
             const Halo &halo_ny, const Halo &halo_nz,
             const std::size_t &block_nx, const std::size_t &block_ny) {
  Arr field_init;
  std::vector<std::size_t> dims = read_tensor_from_csv(
      PATH_TO_TEST_FILES + "test_case_" + test_case + "_init.csv", field_init);

  Arr field_out_calc = field_init;
  stencil(field_init, field_out_calc, halo_nx, halo_ny, halo_nz, block_nx,
          block_ny);

  Arr field_out;
  read_tensor_from_csv(
      PATH_TO_TEST_FILES + "test_case_" + test_case + "_out.csv", field_out);

  return std::make_pair(difference(field_out_calc, field_out), dims);
}

template <typename T>
std::pair<double, DIMENSION>
stencil_test(const std::string &test_case, T *stencil, const Halo &halo_nx,
             const Halo &halo_ny, const Halo &halo_nz) {

  Arr field_init;
  std::vector<std::size_t> dims = read_tensor_from_csv(
      PATH_TO_TEST_FILES + "test_case_" + test_case + "_init.csv", field_init);

  Arr field_out_calc = field_init;
  stencil(field_init, field_out_calc, halo_nx, halo_ny, halo_nz);

  Arr field_out;
  read_tensor_from_csv(
      PATH_TO_TEST_FILES + "test_case_" + test_case + "_out.csv", field_out);

  return std::make_pair(difference(field_out_calc, field_out), dims);
}

template <typename T>
std::pair<double, DIMENSION>
stencil_time_iteration_test(const std::string &test_case, const std::size_t &N,
                            T *stencil_time_iteration, const Halo &halo_nx,
                            const Halo &halo_ny, const Halo &halo_nz) {

  Arr field_init;
  std::vector<std::size_t> dims = read_tensor_from_csv(
      PATH_TO_TEST_FILES + "test_case_" + test_case + "_init.csv", field_init);

  Arr field_out_calc = field_init;
  stencil_time_iteration(N, 0.1, field_init, field_out_calc, halo_nx, halo_ny,
                         halo_nz, 0, 0, 0);

  Arr field_out;
  read_tensor_from_csv(PATH_TO_TEST_FILES + "test_case_" + test_case +
                           "_timesteps_" + std::to_string(N) + "_alpha_0.1.csv",
                       field_out);

  return std::make_pair(difference(field_out_calc, field_out), dims);
}

//------------------------------------------- copy

TEST(stencil_copy, stencil_copy_indicator) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_copy_inidcator", &stencil_copy,
                     std::make_pair(0, 0), std::make_pair(0, 0),
                     std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_test("stencil_copy_random", &stencil_copy, std::make_pair(0, 0),
                     std::make_pair(0, 0), std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

TEST(stencil_copy, stencil_copy_indicator_time_baseline) {
  std::string test_case = "stencil_copy_inidcator";
  std::size_t N = 16;

  Arr field_init;
  std::vector<std::size_t> dims = read_tensor_from_csv(
      PATH_TO_TEST_FILES + "test_case_" + test_case + "_init.csv", field_init);

  Arr field_out_calc = field_init;
  time_iteration_baseline(N, 0.1, &stencil_copy, field_init, field_out_calc,
                          std::make_pair(0, 0), std::make_pair(0, 0),
                          std::make_pair(0, 0));

  Arr field_out;
  DIMENSION dim = read_tensor_from_csv(PATH_TO_TEST_FILES + "test_case_" +
                                           test_case + "_timesteps_" +
                                           std::to_string(N) + "_alpha_0.1.csv",
                                       field_out);

  std::size_t dimension_sum = std::accumulate(dim.begin(), dim.end(), 0);
  EXPECT_TRUE(difference(field_out_calc, field_out) < EPS * dimension_sum);
}

TEST(stencil_copy, stencil_copy_blocked_indicator) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_copy_inidcator", &stencil_copy_blocked,
                     std::make_pair(0, 0), std::make_pair(0, 0),
                     std::make_pair(0, 0), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_test("stencil_copy_random", &stencil_copy_blocked,
                     std::make_pair(0, 0), std::make_pair(0, 0),
                     std::make_pair(0, 0), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

TEST(stencil_copy, stencil_copy_time) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_time_iteration_test("stencil_copy_inidcator", 16,
                                    &stencil_copy_time, std::make_pair(0, 0),
                                    std::make_pair(0, 0), std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_time_iteration_test("stencil_copy_random", 16,
                                    &stencil_copy_time, std::make_pair(0, 0),
                                    std::make_pair(0, 0), std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

//------------------------------------------- 1D i1

TEST(stencil_1D_i1, stencil_1D_i1) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_1D_i1_inidcator", &stencil_1D_i1,
                     std::make_pair(1, 0), std::make_pair(0, 0),
                     std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret =
      stencil_test("stencil_1D_i1_random", &stencil_1D_i1, std::make_pair(1, 0),
                   std::make_pair(0, 0), std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

TEST(stencil_1D_i1, stencil_1D_i1_blocked_indicator) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_1D_i1_inidcator", &stencil_1D_i1_blocked,
                     std::make_pair(1, 0), std::make_pair(0, 0),
                     std::make_pair(0, 0), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_test("stencil_1D_i1_random", &stencil_1D_i1_blocked,
                     std::make_pair(1, 0), std::make_pair(0, 0),
                     std::make_pair(0, 0), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

TEST(stencil_1D_i1, stencil_1D_i1_time) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_time_iteration_test("stencil_1D_i1_inidcator", 16,
                                    &stencil_1D_i1_time, std::make_pair(1, 0),
                                    std::make_pair(0, 0), std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_time_iteration_test("stencil_1D_i1_random", 16,
                                    &stencil_1D_i1_time, std::make_pair(1, 0),
                                    std::make_pair(0, 0), std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

//------------------------------------------- 1D i2

TEST(stencil_1D_i2, stencil_1D_i2_indicator) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_1D_i2_inidcator", &stencil_1D_i2,
                     std::make_pair(1, 1), std::make_pair(0, 0),
                     std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret =
      stencil_test("stencil_1D_i2_random", &stencil_1D_i2, std::make_pair(1, 1),
                   std::make_pair(0, 0), std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

TEST(stencil_1D_i2, stencil_1D_i2_blocked_indicator) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_1D_i2_inidcator", &stencil_1D_i2_blocked,
                     std::make_pair(1, 1), std::make_pair(0, 0),
                     std::make_pair(0, 0), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_test("stencil_1D_i2_random", &stencil_1D_i2_blocked,
                     std::make_pair(1, 1), std::make_pair(0, 0),
                     std::make_pair(0, 0), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

/*TEST(stencil_1D_i2, stencil_1D_i2_time) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_time_iteration_test("stencil_1D_i2_inidcator", 16,
&stencil_1D_i2_time, std::make_pair(1, 1), std::make_pair(0, 0),
                       std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_time_iteration_test("stencil_1D_i2_random", 16,
&stencil_1D_i2_time, std::make_pair(1, 1), std::make_pair(0, 0),
                       std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}*/

//------------------------------------------- 1D j1

TEST(stencil_1D_j1, stencil_1D_j1_indicator) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_1D_j1_inidcator", &stencil_1D_j1,
                     std::make_pair(0, 0), std::make_pair(1, 0),
                     std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret =
      stencil_test("stencil_1D_j1_random", &stencil_1D_j1, std::make_pair(0, 0),
                   std::make_pair(1, 0), std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

TEST(stencil_1D_j1, stencil_1D_j1_blocked_indicator) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_1D_j1_inidcator", &stencil_1D_j1_blocked,
                     std::make_pair(0, 0), std::make_pair(1, 0),
                     std::make_pair(0, 0), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_test("stencil_1D_j1_random", &stencil_1D_j1_blocked,
                     std::make_pair(0, 0), std::make_pair(1, 0),
                     std::make_pair(0, 0), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

/*TEST(stencil_1D_j1, stencil_1D_j1_time) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_time_iteration_test("stencil_1D_j1_inidcator", 16,
&stencil_1D_j1_time, std::make_pair(0, 0), std::make_pair(0, 0),
                       std::make_pair(1, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_time_iteration_test("stencil_1D_j1_random", 16,
&stencil_1D_j1_time, std::make_pair(0, 0), std::make_pair(0, 0),
                       std::make_pair(1, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}*/

//------------------------------------------- 1D j2

TEST(stencil_1D_j2, stencil_1D_j2_indicator) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_1D_j2_inidcator", &stencil_1D_j2,
                     std::make_pair(0, 0), std::make_pair(1, 1),
                     std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret =
      stencil_test("stencil_1D_j2_random", &stencil_1D_j2, std::make_pair(0, 0),
                   std::make_pair(1, 1), std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

TEST(stencil_1D_j2, stencil_1D_j2_blocked_indicator) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_1D_j2_inidcator", &stencil_1D_j2_blocked,
                     std::make_pair(0, 0), std::make_pair(1, 1),
                     std::make_pair(0, 0), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_test("stencil_1D_j2_random", &stencil_1D_j2_blocked,
                     std::make_pair(0, 0), std::make_pair(1, 1),
                     std::make_pair(0, 0), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

/*TEST(stencil_1D_j2, stencil_1D_j2_time) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_time_iteration_test("stencil_1D_j2_inidcator", 16,
&stencil_1D_j2_time, std::make_pair(0, 0), std::make_pair(0, 0),
                       std::make_pair(1, 1));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_time_iteration_test("stencil_1D_j2_random", 16,
&stencil_1D_j2_time, std::make_pair(0, 0), std::make_pair(0, 0),
                       std::make_pair(1, 1));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}*/

//------------------------------------------- 2D

TEST(stencil_2D, stencil_2D_indicator) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_2D_inidcator", &stencil_2D, std::make_pair(1, 1),
                     std::make_pair(1, 1), std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_test("stencil_2D_random", &stencil_2D, std::make_pair(1, 1),
                     std::make_pair(1, 1), std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

TEST(stencil_2D, stencil_2D_blocked_indicator) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_2D_inidcator", &stencil_2D_blocked,
                     std::make_pair(1, 1), std::make_pair(1, 1),
                     std::make_pair(0, 0), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_test("stencil_2D_random", &stencil_2D_blocked,
                     std::make_pair(1, 1), std::make_pair(1, 1),
                     std::make_pair(0, 0), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

TEST(stencil_2D, stencil_2D_time) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_time_iteration_test("stencil_2D_inidcator", 16,
                                    &stencil_2D_time, std::make_pair(1, 1),
                                    std::make_pair(1, 1), std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_time_iteration_test("stencil_2D_random", 16, &stencil_2D_time,
                                    std::make_pair(1, 1), std::make_pair(1, 1),
                                    std::make_pair(0, 0));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

//------------------------------------------- 3D

TEST(stencil_3D, stencil_3D_indicator) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_3D_inidcator", &stencil_3D, std::make_pair(1, 1),
                     std::make_pair(1, 1), std::make_pair(1, 1));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_test("stencil_3D_random", &stencil_3D, std::make_pair(1, 1),
                     std::make_pair(1, 1), std::make_pair(1, 1));
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}

TEST(stencil_3D, stencil_3D_blocked) {
  std::size_t dimension_sum;
  std::pair<double, DIMENSION> ret;

  ret = stencil_test("stencil_3D_inidcator", &stencil_3D_blocked,
                     std::make_pair(1, 1), std::make_pair(1, 1),
                     std::make_pair(1, 1), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);

  ret = stencil_test("stencil_3D_random", &stencil_3D_blocked,
                     std::make_pair(1, 1), std::make_pair(1, 1),
                     std::make_pair(1, 1), BLOCKING_SIZE, BLOCKING_SIZE);
  dimension_sum = std::accumulate(ret.second.begin(), ret.second.end(), 0);
  EXPECT_TRUE(ret.first < EPS * dimension_sum);
}
