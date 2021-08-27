#include <HPC4WC/boundary_condition.h>
#include <HPC4WC/field.h>
#include <gtest/gtest.h>

TEST(BoundaryConditions, PeriodicI) {
    using namespace HPC4WC;
    Field f(2, 2, 1, 2);

    f(2, 2, 0) = 1.;

    PeriodicBoundaryConditions::apply(f, PeriodicBoundaryConditions::PERIODICITY::ONLY_I);
    EXPECT_DOUBLE_EQ(f(0, 2, 0), 1.);
    EXPECT_DOUBLE_EQ(f(1, 2, 0), 0.);
    EXPECT_DOUBLE_EQ(f(2, 2, 0), 1.);
    EXPECT_DOUBLE_EQ(f(3, 2, 0), 0.);
    EXPECT_DOUBLE_EQ(f(4, 2, 0), 1.);
    EXPECT_DOUBLE_EQ(f(5, 2, 0), 0.);

    EXPECT_DOUBLE_EQ(f(2, 0, 0), 0.);
    EXPECT_DOUBLE_EQ(f(2, 1, 0), 0.);
    EXPECT_DOUBLE_EQ(f(2, 2, 0), 1.);
    EXPECT_DOUBLE_EQ(f(2, 3, 0), 0.);
    EXPECT_DOUBLE_EQ(f(2, 4, 0), 0.);
    EXPECT_DOUBLE_EQ(f(2, 5, 0), 0.);
}

TEST(BoundaryConditions, PeriodicJ) {
    using namespace HPC4WC;
    Field f(2, 2, 1, 2);

    f(2, 2, 0) = 1.;

    PeriodicBoundaryConditions::apply(f, PeriodicBoundaryConditions::PERIODICITY::ONLY_J);
    EXPECT_DOUBLE_EQ(f(2, 0, 0), 1.);
    EXPECT_DOUBLE_EQ(f(2, 1, 0), 0.);
    EXPECT_DOUBLE_EQ(f(2, 2, 0), 1.);
    EXPECT_DOUBLE_EQ(f(2, 3, 0), 0.);
    EXPECT_DOUBLE_EQ(f(2, 4, 0), 1.);
    EXPECT_DOUBLE_EQ(f(2, 5, 0), 0.);

    EXPECT_DOUBLE_EQ(f(0, 2, 0), 0.);
    EXPECT_DOUBLE_EQ(f(1, 2, 0), 0.);
    EXPECT_DOUBLE_EQ(f(2, 2, 0), 1.);
    EXPECT_DOUBLE_EQ(f(3, 2, 0), 0.);
    EXPECT_DOUBLE_EQ(f(4, 2, 0), 0.);
    EXPECT_DOUBLE_EQ(f(5, 2, 0), 0.);
}

TEST(BoundaryConditions, PeriodicBoth) {
    using namespace HPC4WC;
    Field f(2, 2, 1, 2);

    f(2, 2, 0) = 1.;

    PeriodicBoundaryConditions::apply(f, PeriodicBoundaryConditions::PERIODICITY::BOTH);
    EXPECT_DOUBLE_EQ(f(2, 0, 0), 1.);
    EXPECT_DOUBLE_EQ(f(2, 1, 0), 0.);
    EXPECT_DOUBLE_EQ(f(2, 2, 0), 1.);
    EXPECT_DOUBLE_EQ(f(2, 3, 0), 0.);
    EXPECT_DOUBLE_EQ(f(2, 4, 0), 1.);
    EXPECT_DOUBLE_EQ(f(2, 5, 0), 0.);

    EXPECT_DOUBLE_EQ(f(2, 0, 0), 1.);
    EXPECT_DOUBLE_EQ(f(2, 1, 0), 0.);
    EXPECT_DOUBLE_EQ(f(2, 2, 0), 1.);
    EXPECT_DOUBLE_EQ(f(2, 3, 0), 0.);
    EXPECT_DOUBLE_EQ(f(2, 4, 0), 1.);
    EXPECT_DOUBLE_EQ(f(2, 5, 0), 0.);
}

TEST(BoundaryConditions, Dirichlet) {
    using namespace HPC4WC;
    Field f(2, 2, 1, 2, 1.);
    EXPECT_DOUBLE_EQ(f(0, 0, 0), 1.);

    DirichletBoundaryConditions::apply(f, 0.);
    EXPECT_DOUBLE_EQ(f(0, 0, 0), 0.);
    EXPECT_DOUBLE_EQ(f(1, 0, 0), 0.);
    EXPECT_DOUBLE_EQ(f(4, 0, 0), 0.);
    EXPECT_DOUBLE_EQ(f(5, 0, 0), 0.);
    EXPECT_DOUBLE_EQ(f(0, 1, 0), 0.);
    EXPECT_DOUBLE_EQ(f(1, 1, 0), 0.);
    EXPECT_DOUBLE_EQ(f(4, 1, 0), 0.);
    EXPECT_DOUBLE_EQ(f(5, 1, 0), 0.);

    // now check another halo with a different value
    DirichletBoundaryConditions::apply(f, 3.);
    EXPECT_DOUBLE_EQ(f(0, 0, 0), 3.);
    EXPECT_DOUBLE_EQ(f(0, 1, 0), 3.);
    EXPECT_DOUBLE_EQ(f(0, 4, 0), 3.);
    EXPECT_DOUBLE_EQ(f(0, 5, 0), 3.);
    EXPECT_DOUBLE_EQ(f(1, 0, 0), 3.);
    EXPECT_DOUBLE_EQ(f(1, 1, 0), 3.);
    EXPECT_DOUBLE_EQ(f(1, 4, 0), 3.);
    EXPECT_DOUBLE_EQ(f(1, 5, 0), 3.);
}