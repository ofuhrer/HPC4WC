#include <HPC4WC/argsparser.h>
#include <HPC4WC/boundary_condition.h>
#include <HPC4WC/config.h>
#include <HPC4WC/diffusion.h>
#include <HPC4WC/field.h>
#include <HPC4WC/initial_condition.h>
#include <HPC4WC/io.h>
#include <HPC4WC/timer.h>

#include <Eigen/Core>
#include <fstream>

int main(int argc, char* argv[]) {
    using namespace HPC4WC;

    ArgsParser argsParser(argc, argv);

    Field::idx_t ni = 128, nj = 128, nk = 3, num_halo = 2, num_timesteps = 100;
    argsParser.add_argument(ni, "ni", "Number of interiour points in i direction.");
    argsParser.add_argument(nj, "nj", "Number of interiour points in j direction.");
    argsParser.add_argument(nk, "nk", "Number of interiour points in k direction.");
    argsParser.add_argument(num_halo, "num_halo", "Number of halo points in i and j direction.");
    argsParser.add_argument(num_timesteps, "iterations", "Number of diffusion iterations.");

    if (argsParser.help()) {
        argsParser.help(std::cout);
        return 0;
    }

    Field f(ni, nj, nk, num_halo);

    CubeInitialCondition::apply(f);

    // std::ofstream initial_of("initial.mat");
    // IO::write(initial_of, f, nk / 2);

    auto timer = Timer();

    for (int t = 0; t < num_timesteps; t++) {
        PeriodicBoundaryConditions::apply(f);
        SimpleDiffusion::apply(f, 1. / 32.);
    }

    double time = timer.timeElapsed();
    std::cout << "Time elapsed: " << time << std::endl;

    // std::ofstream final_of("final.mat");
    // IO::write(final_of, f, nk / 2);

    return 0;
}
