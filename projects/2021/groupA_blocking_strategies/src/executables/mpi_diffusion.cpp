#include <HPC4WC/boundary_condition.h>
#include <HPC4WC/diffusion.h>
#include <HPC4WC/field.h>
#include <HPC4WC/initial_condition.h>
#include <HPC4WC/io.h>
#include <HPC4WC/partitioner.h>
#include <HPC4WC/timer.h>
#include <HPC4WC/argsparser.h>

#include <Eigen/Core>
#include <fstream>

int main(int argc, char* argv[]) {
    using namespace HPC4WC;

    ArgsParser argsParser(argc, argv);
    Field::idx_t ni = 64, nj = 64, nk = 64, num_halo = 2, num_timesteps = 100;
    argsParser.add_argument(ni, "ni", "Number of interiour points in i direction.");
    argsParser.add_argument(nj, "nj", "Number of interiour points in j direction.");
    argsParser.add_argument(nk, "nk", "Number of interiour points in k direction.");
    argsParser.add_argument(num_halo, "num_halo", "Number of halo points in i and j direction.");
    argsParser.add_argument(num_timesteps, "iterations", "Number of diffusion iterations.");

    Partitioner::init(argc, argv);
    Partitioner p(ni, nj, nk, num_halo);

    if (argsParser.help()) {
        if (p.rank() == 0) {
            argsParser.help(std::cout);
        }
        Partitioner::finalize();
        return 0;
    }

    if (p.rank() == 0) {
        FieldSPtr global_f = p.getGlobalField();
        CubeInitialCondition::apply(*global_f.get());
        std::ofstream initial_of("initial.txt");
        IO::write(initial_of, *global_f.get(), nk / 2);
    }

    p.scatter();

    auto timer = Timer();
    for (Field::idx_t t = 0; t < num_timesteps; t++) {
        p.applyPeriodicBoundaryConditions();
        Diffusion::apply(*p.getField().get(), 1. / 32.);
    }

    double time = timer.timeElapsed();
    std::cout << "Rank: " << p.rank() << ", Time elapsed: " << time << std::endl;

    p.gather();

    if (p.rank() == 0) {
        FieldSPtr global_f = p.getGlobalField();
        std::ofstream final_of("final.txt");
        IO::write(final_of, *global_f.get(), nk / 2);
    }

    Partitioner::finalize();

    return 0;
}
