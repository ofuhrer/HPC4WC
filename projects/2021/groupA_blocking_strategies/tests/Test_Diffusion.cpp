#include <HPC4WC/boundary_condition.h>
#include <HPC4WC/config.h>
#include <HPC4WC/diffusion.h>
#include <HPC4WC/initial_condition.h>
#include <HPC4WC/io.h>
#include <gtest/gtest.h>

TEST(Diffusion, Diffusion) {
    using namespace HPC4WC;
    Field f(1, 1, 1, 2, 0.);
    f(2, 2, 0) = 1.;
    Diffusion::apply(f, 0.5);

    Field f2(1, 1, 1, 2, 0.);
    f2(2, 2, 0) = -9.;

    // Make all halo rings 0
    DirichletBoundaryConditions::apply(f, 0.);
    DirichletBoundaryConditions::apply(f2, 0.);
    EXPECT_EQ(f, f2);
}

TEST(Diffusion, SimpleDiffusion) {
    using namespace HPC4WC;
    Field f(1, 1, 1, 2, 0.);
    f(2, 2, 0) = 1.;
    Config::BLOCK_SIZE_I = 1;
    Config::BLOCK_SIZE_J = 1;

    SimpleDiffusion::apply(f, 0.5);

    Field f2(1, 1, 1, 2, 0.);
    f2(2, 2, 0) = -9.;

    // Make all halo rings 0
    DirichletBoundaryConditions::apply(f, 0.);
    DirichletBoundaryConditions::apply(f2, 0.);
    EXPECT_EQ(f, f2);
}

TEST(Diffusion, SimpleDiffusionBlocking) {
    using namespace HPC4WC;
    Field f(10, 10, 3, 2);
    Field f2(10, 10, 3, 2);
    CubeInitialCondition::apply(f);
    CubeInitialCondition::apply(f2);

    Config::BLOCK_SIZE_I = 1;
    Config::BLOCK_SIZE_J = 1;
    SimpleDiffusion::apply(f);

    Config::BLOCK_SIZE_I = 5;
    Config::BLOCK_SIZE_J = 5;
    SimpleDiffusion::apply(f2);

    EXPECT_EQ(f, f2);
}

TEST(Diffusion, SimpleDiffusionBlockingErrors) {
    using namespace HPC4WC;
    Field f(10, 10, 3, 2);

    Config::BLOCK_SIZE_I = 5;
    Config::BLOCK_SIZE_J = 5;

    EXPECT_NO_THROW(SimpleDiffusion::apply(f));

    Config::BLOCK_SIZE_I = 10;
    EXPECT_NO_THROW(SimpleDiffusion::apply(f));

    Config::BLOCK_SIZE_J = 3;
    EXPECT_THROW(SimpleDiffusion::apply(f), std::logic_error);
}

TEST(Diffusion, Comparison) {
    using namespace HPC4WC;
    Field f1(10, 10, 10, 2);
    Field f2(10, 10, 10, 2);

    Config::BLOCK_SIZE_I = 5;
    Config::BLOCK_SIZE_J = 5;

    CubeInitialCondition::apply(f1);
    CubeInitialCondition::apply(f2);

    EXPECT_EQ(f1, f2);

    for (int i = 0; i < 100; i++) {
        Diffusion::apply(f1);
        SimpleDiffusion::apply(f2);

        EXPECT_EQ(f1, f2);
    }
}