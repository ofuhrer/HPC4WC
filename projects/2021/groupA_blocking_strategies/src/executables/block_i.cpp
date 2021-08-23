#include <HPC4WC/autotuner.h>

int main(int argc, char* argv[]) {
    using namespace HPC4WC;

    AutoTuner autotuner("diffusion2d", "--ni=1000 --nj=200 --nk=100 --num-iterations=1024 --block-i=true --block-j=false --openmp-num-threads=1", 1);

    autotuner.add_range_argument("blocking-size-i", 5, 50, 5);
    autotuner.search();

    return 0;
}
