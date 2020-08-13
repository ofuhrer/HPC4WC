#include <iostream>
#include <omp.h>
#include "utils.h"
#include "update_halo.h"

void apply_diffusion_fused(Storage3D<realType>& inField, Storage3D<realType>& outField, realType alpha,
                     unsigned numIter, int x, int y, int z, int halo) {

  Storage3D<realType> tmp1Field(x, y, z, halo);

  for(std::size_t iter = 0; iter < numIter; ++iter) {

    updateHalo(inField);
    realType a1 = -1. * alpha;
    realType a2 = -2. * alpha;
    realType a8 = 8. * alpha;
    realType a20 = 1. - 20. * alpha;

    for(std::size_t k = 0; k < inField.zMax(); ++k) {

      // apply the full diffusion
#pragma omp parallel for
      for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
        for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
          outField(i, j, k) = a1 * inField(i, j - 2, k) + a2 * inField(i - 1, j - 1, k) +
                              a8 * inField(i, j - 1, k) + a2 * inField(i + 1, j - 1, k) +
                              a1 * inField(i - 2, j, k) + a8 * inField(i - 1, j, k) +
                              a20 * inField(i, j, k) + a8 * inField(i + 1, j, k) +
                              a1 * inField(i + 2, j, k) + a2 * inField(i - 1, j + 1, k) +
                              a8 * inField(i, j + 1, k) + a2 * inField(i + 1, j + 1, k) +
                              a1 * inField(i, j + 2, k);
        }
      }

      // update the field
#pragma omp parallel for
      for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
        for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
          if(iter != numIter - 1) {
            inField(i, j, k) = outField(i, j, k);
          }
        }
      }
    }
  }
}

