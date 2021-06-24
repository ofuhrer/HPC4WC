#pragma once
#include <iostream>
#include "utils.h"

void apply_stencil_cpu(Storage3D<realType>& inField,
                       Storage3D<realType>& outField,
                       Storage3D<realType>& tmpField,
                       realType const alpha,
                       unsigned const iter,
                       unsigned const numIter,
                       std::size_t const k0 = 0);
