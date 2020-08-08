#pragma once
#include <iostream>
#include <cuda.h>
#include "common_types.h"

__global__
void apply_stencil(realType const *infield,
                   realType *outfield,
                   int const xMin,
                   int const xMax,
                   int const xSize,
                   int const yMin,
                   int const yMax,
                   int const ySize,
                   int const zMax,
                   realType const alpha);
