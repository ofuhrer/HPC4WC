#include <HPC4WC/autotuner.h>

int main(int argc, char* argv[]) {
    using namespace HPC4WC;

    AutoTuner autotuner("diffusion2d", "--ni=128 --nj=128 --nk=128 --num-iterations=1024 --block-i=false --block-j=false", 10);

    autotuner.add_range_argument("openmp-num-threads", 1, 12);

    autotuner.search();

    return 0;
}
