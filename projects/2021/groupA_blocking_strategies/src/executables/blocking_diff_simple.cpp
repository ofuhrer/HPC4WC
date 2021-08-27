#include <HPC4WC/autotuner.h>

int main(int argc, char* argv[]) {
    using namespace HPC4WC;

    AutoTuner autotuner("diffusion2d", "--ni=128 --nj=128 --nk=3 --num-iterations=100 --block-i=true --block-j=true --openmp-num-threads=1", 10);

    autotuner.add_range_argument("blocking-size-i", {2, 4, 16, 64, 128});
    autotuner.add_range_argument("blocking-size-j", {2, 4, 16, 64, 128});

    autotuner.search();

    return 0;
}
