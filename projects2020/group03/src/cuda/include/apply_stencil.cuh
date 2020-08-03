#pragma once
#include <iostream>
#include <cuda.h>

__global__
void apply_stencil(double const *infield,
                   double *outfield,
                   int const xMin,
                   int const xMax,
                   int const xSize,
                   int const yMin,
                   int const yMax,
                   int const ySize,
                   int const zMax,
                   double const alpha);
