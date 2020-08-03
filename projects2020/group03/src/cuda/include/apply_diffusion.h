#pragma once
#include <iostream>
#include "utils.h"

void apply_diffusion(Storage3D<double>& inField, Storage3D<double>& outField, double alpha,
                     unsigned numIter, int x, int y, int z, int halo);
