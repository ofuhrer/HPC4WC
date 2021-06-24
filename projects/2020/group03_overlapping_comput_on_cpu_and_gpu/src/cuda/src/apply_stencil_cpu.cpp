#include <iostream>
#include <omp.h>
#include "utils.h"
#include "apply_stencil_cpu.h"

// Storage3D<realType> tmpField(x, y, z, halo);
void apply_stencil_cpu(Storage3D<realType>& inField,
                       Storage3D<realType>& outField,
                       Storage3D<realType>& tmpField,
                       realType const alpha,
                       unsigned const iter,
                       unsigned const numIter,
                       std::size_t const k0) {

  for(std::size_t k = k0; k < inField.zMax(); ++k) {

    // apply the initial laplacian
#pragma omp parallel
    {
#pragma omp for
      for(std::size_t j = inField.yMin() - 1; j < inField.yMax() + 1; ++j) {
        for(std::size_t i = inField.xMin() - 1; i < inField.xMax() + 1; ++i) {
          tmpField(i, j, 0) = -4.0 * inField(i, j, k) + inField(i - 1, j, k) +
            inField(i + 1, j, k) + inField(i, j - 1, k) + inField(i, j + 1, k);
        }
      }

      // apply the second laplacian
#pragma omp for
      for(std::size_t j = inField.yMin(); j < inField.yMax(); ++j) {
        for(std::size_t i = inField.xMin(); i < inField.xMax(); ++i) {
          realType const laplap = -4.0 * tmpField(i, j, 0) + tmpField(i - 1, j, 0) +
              tmpField(i + 1, j, 0) + tmpField(i, j - 1, 0) + tmpField(i, j + 1, 0);

          // and update the field
          if(iter == numIter - 1) {
            outField(i, j, k) = inField(i, j, k) - alpha * laplap;
          } else {
            inField(i, j, k) = inField(i, j, k) - alpha * laplap;
          }
        }
      }
    }

  } // k-loop
}
