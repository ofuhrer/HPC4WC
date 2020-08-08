#include <iostream>
#include <omp.h>
#include "utils.h"
#include "update_halo.h"

void updateHalo(Storage3D<realType>& inField) {
  const int xInterior = inField.xMax() - inField.xMin();
  const int yInterior = inField.yMax() - inField.yMin();

#pragma omp parallel
  {
  // bottom edge (without corners)
#pragma omp for collapse(2) nowait
  for(std::size_t k = 0; k < inField.zMax(); ++k) {
    for(std::size_t j = 0; j < inField.yMin(); ++j) {
      for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j + yInterior, k);
      }
    }
  }

  // top edge (without corners)
#pragma omp for collapse(2) nowait
  for(std::size_t k = 0; k < inField.zMax(); ++k) {
    for(std::size_t j = inField.yMax(); j < inField.ySize(); ++j) {
      for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
        inField(i, j, k) = inField(i, j - yInterior, k);
      }
    }
  }

  // left edge (including corners)
#pragma omp for collapse(2) nowait
  for(std::size_t k = 0; k < inField.zMax(); ++k) {
    for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
      for(std::size_t i = 0; i < inField.xMin(); ++i) {
        inField(i, j, k) = inField(i + xInterior, j, k);
      }
    }
  }

  // right edge (including corners)
#pragma omp for collapse(2) nowait
  for(std::size_t k = 0; k < inField.zMax(); ++k) {
    for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
      for(std::size_t i = inField.xMax(); i < inField.xSize(); ++i) {
        inField(i, j, k) = inField(i - xInterior, j, k);
      }
    }
  }
  }
}
