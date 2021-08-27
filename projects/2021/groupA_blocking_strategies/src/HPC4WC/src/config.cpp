#include <HPC4WC/config.h>

namespace HPC4WC {
//bool Config::USE_OPENMP_ON_K = true;
Field::idx_t Config::OPENMP_NUM_THREADS = 1;

Field::idx_t Config::BLOCK_SIZE_I = 8;
Field::idx_t Config::BLOCK_SIZE_J = 8;

bool Config::BLOCK_I = true;
bool Config::BLOCK_J = true;
}  // namespace HPC4WC