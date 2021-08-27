#include <HPC4WC/initial_condition.h>
#include <gtest/gtest.h>

TEST(InitialCondition, CubeInitialCondition) {
    using namespace HPC4WC;

    Field f(3, 3, 3, 2);
    CubeInitialCondition::apply(f);

    Field f2(3, 3, 3, 2, 0.0);
    f2(1 + f2.num_halo(), 1 + f2.num_halo(), 1) = 1.;

    EXPECT_EQ(f, f2);
}

TEST(InitialCondition, DiagonalInitialCondition) {
    using namespace HPC4WC;

    Field f(3, 3, 3, 2);
    DiagonalInitialCondition::apply(f);

    Field f2(3, 3, 3, 2, 0.0);
    for (Field::idx_t k = 0; k < 3; k++) {
        f2(0 + f2.num_halo(), 0 + f2.num_halo(), k) = 1.;
        f2(2 + f2.num_halo(), 2 + f2.num_halo(), k) = 1.;
    }

    EXPECT_EQ(f, f2);
}

TEST(InitialCondition, XInitialCondition) {
    using namespace HPC4WC;

    Field f(5, 5, 3, 2);
    XInitialCondition::apply(f, 1);

    Field f2(5, 5, 3, 2, 0.0);
    for (Field::idx_t k = 0; k < 3; k++) {
        f2(0 + f2.num_halo(), 0 + f2.num_halo(), k) = 1.;
        f2(1 + f2.num_halo(), 1 + f2.num_halo(), k) = 1.;
        f2(2 + f2.num_halo(), 2 + f2.num_halo(), k) = 1.;
        f2(3 + f2.num_halo(), 3 + f2.num_halo(), k) = 1.;
        f2(4 + f2.num_halo(), 4 + f2.num_halo(), k) = 1.;
        f2(0 + f2.num_halo(), 4 + f2.num_halo(), k) = 1.;
        f2(1 + f2.num_halo(), 3 + f2.num_halo(), k) = 1.;
        f2(3 + f2.num_halo(), 1 + f2.num_halo(), k) = 1.;
        f2(4 + f2.num_halo(), 0 + f2.num_halo(), k) = 1.;
    }

    EXPECT_EQ(f, f2);
}