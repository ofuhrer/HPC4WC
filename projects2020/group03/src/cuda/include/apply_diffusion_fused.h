#pragma once
#include <iostream>
#include "utils.h"
#include "common_types.h"

void apply_diffusion_fused(Storage3D<realType>& inField, Storage3D<realType>& outField, realType alpha,
                     unsigned numIter, int x, int y, int z, int halo);
