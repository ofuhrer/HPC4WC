#ifndef UTILS_HPP
#define UTILS_HPP

#include "cache_simulator.hpp"

#include <unsupported/Eigen/CXX11/Tensor>

using DIMENSION = std::vector<std::size_t>;

// https://stackoverflow.com/questions/4654636/how-to-determine-if-a-string-is-a-number-with-c
bool is_number(const std::string &s) {
  std::string::const_iterator it = s.begin();
  while (it != s.end() && std::isdigit(*it)) ++it;
  return !s.empty() && it == s.end();
}

DIMENSION read_tensor_from_csv(const std::string &filename, Arr &tensor) {
  std::vector<std::size_t> dim(3);
  dim[2] = 1;

  std::ifstream file(filename);
  verify(!file.fail(), "file " + filename + " does not exist");
  std::string line;
  verify(getline(file, line), "file " + filename + " empty");

  std::string val;
  std::stringstream line_steam(line);
  std::size_t k = 0;
  while (getline(line_steam, val, ',')) {
    verify(is_number(val), "read dimension is nut a number: " + val);
    dim[k] = stoi(val);
    ++k;
  }

  // std::cout << "dims: " << dim[0] << ", " << dim[1] << ", " << dim[2] <<
  // std::endl;
  verify(k == 2 || k == 3, "expecting dimensions in first line = n1,n2,...");

  tensor = Arr(dim[0], dim[1], dim[2]);

  std::size_t i = 0;
  std::size_t j = 0;
  while (getline(file, line)) {
    verify(dim[0] > i,
           "there are more lines than the first dimension implying");

    std::string val_rows;
    std::stringstream line_steam_rows(line);

    j = 0;
    while (getline(line_steam_rows, val_rows, ',')) {
      verify(dim[2] > j / dim[1],
             "there are more rows than the third dimension implying");

      tensor(i, j % dim[1], j / dim[1]) = stod(val_rows);
      // std::cout << "(" << i+1 << "," << j%dim[1]+1 << "," << j/dim[1]+1 <<
      // ")= " << stod(val_rows) << std::endl;
      ++j;
    }
    ++i;
  }
  return dim;
}

#endif
