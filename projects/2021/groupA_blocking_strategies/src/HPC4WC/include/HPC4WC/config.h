#pragma once
#include <HPC4WC/field.h>

namespace HPC4WC {

/**
 * @brief Configuration class for blocking, OMP, MPI.
 */
class Config {
public:
    static Field::idx_t OPENMP_NUM_THREADS; ///< How many OpenMP threads the k loops should use, default 1.

    static Field::idx_t BLOCK_SIZE_I; ///< If BLOCK_I, how big the blocks should be in i direction, default 8.
    static Field::idx_t BLOCK_SIZE_J; ///< If BLOCK_J, how big the blocks should be in j direction, default 8.

    static bool BLOCK_I; ///< Whether to block in i direction (applied to the diffusion), default true.
    static bool BLOCK_J; ///< Whether to block in j direction (applied to the diffusion), default true.
};
}  // namespace HPC4WC
