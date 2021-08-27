#include <HPC4WC/autotuner.h>
#include <HPC4WC/argsparser.h>

#include <iostream>

int main(int argc, char* argv[]) {
    using namespace HPC4WC;
    ArgsParser argsParser(argc, argv, false);

    std::string executable = "diffusion2d";
    std::string more_args = "";
    Field::idx_t iterations = 1;
    argsParser.add_argument(executable, "executable", "The file to execute and measure time for.");
    argsParser.add_argument(more_args, "exe-args", "More arguments passed to the executable.");
    argsParser.add_argument(iterations, "iterations", "How many times a single run should be measured (times will be averaged).");

    if (argsParser.help()) {
        argsParser.help(std::cout);
        return 0;
    }

    // Do not add .exe on windows, it will be found. This version now works on windows and linux!
    AutoTuner autotuner(executable.c_str(), more_args.c_str(), iterations);

    autotuner.add_range_argument("openmp-num-threads", 1, 2);
    autotuner.add_bool_argument("block-i");
    autotuner.add_bool_argument("block-j");
    autotuner.add_range_argument("blocking-size-i", 5, 15, 5);
    autotuner.add_range_argument("blocking-size-j", 5, 15, 5);

    autotuner.search();

    return 0;
}
