#include <HPC4WC/field.h>
#include <gtest/gtest.h>

TEST(Field, Constructor) {
    using namespace HPC4WC;
    Field f_small(1, 1, 1, 1);         // constant zero
    Field f_constant(2, 2, 2, 2, 3.);  // constant 3
    Field f2(2, 2, 2, 2);
    Field f_large(1000, 1000, 100, 2);  // large field

    EXPECT_THROW(Field f_err(-1, 1, 1, 1), std::logic_error);
    EXPECT_THROW(Field f_err2(1, 0, 1, 1), std::logic_error);
    EXPECT_THROW(Field f_err3(1, 1, 0, 1), std::logic_error);
    EXPECT_THROW(Field f_err4(1, 1, 1, -1), std::logic_error);
}

TEST(Field, AccessGood) {
    using namespace HPC4WC;
    Field f(2, 2, 2, 2, 3.);
    EXPECT_DOUBLE_EQ(f(0, 0, 0), 3);  // halo is also 3.
    EXPECT_DOUBLE_EQ(f(2, 2, 0), 3.);  // constant value

    f(1, 1, 0) = 4.;
    EXPECT_DOUBLE_EQ(f(1, 1, 1), 3.);
    EXPECT_DOUBLE_EQ(f(1, 1, 0), 4.);

    EXPECT_NO_THROW(f(5, 0, 0));
    EXPECT_NO_THROW(f(0, 5, 0));
    EXPECT_NO_THROW(f(0, 0, 1));
}

TEST(Field, AccessBad) {
    using namespace HPC4WC;
    Field f(2, 2, 2, 2, 3.);
    EXPECT_THROW(f(-1, 0, 0), std::out_of_range);
    EXPECT_THROW(f(6, 0, 0), std::out_of_range);
    EXPECT_THROW(f(0, -1, 0), std::out_of_range);
    EXPECT_THROW(f(0, 6, 0), std::out_of_range);
    EXPECT_THROW(f(0, 0, -1), std::out_of_range);
    EXPECT_THROW(f(0, 0, 2), std::out_of_range);
}

TEST(Field, SetFromField) {
    using namespace HPC4WC;
    Field f1(10, 10, 2, 2, 1.);
    Field f2(10, 10, 2, 2, 3.);
    Field f3(10, 10, 2, 1);  // different num_halo
    Field f4(9, 10, 2, 2);   // different i
    Field f5(10, 9, 2, 2);   // different j
    Field f6(10, 10, 3, 2);  // different k

    EXPECT_DOUBLE_EQ(f1(2, 2, 1), 1.);
    EXPECT_DOUBLE_EQ(f2(2, 2, 1), 3.);
    EXPECT_NO_THROW(f1.setFrom(f2));
    EXPECT_DOUBLE_EQ(f1(2, 2, 1), 3.);

    EXPECT_THROW(f1.setFrom(f3), std::logic_error);
    EXPECT_THROW(f1.setFrom(f4), std::logic_error);
    EXPECT_THROW(f1.setFrom(f5), std::logic_error);
    EXPECT_THROW(f1.setFrom(f6), std::logic_error);
}

TEST(Field, SetFromFieldPart) {
    using namespace HPC4WC;
    Field f1(10, 10, 2, 2, 1.);
    Eigen::MatrixXd ij_part = Eigen::MatrixXd::Constant(5, 5, 4.);

    EXPECT_NO_THROW(f1.setFrom(ij_part, 0, 0, 0));
    EXPECT_NO_THROW(f1.setFrom(ij_part, 8, 0, 0));
    EXPECT_THROW(f1.setFrom(ij_part, 9, 0, 0), std::out_of_range);
    EXPECT_THROW(f1.setFrom(ij_part, 0, 9, 0), std::out_of_range);
    EXPECT_THROW(f1.setFrom(ij_part, 0, 0, -1), std::out_of_range);
    EXPECT_THROW(f1.setFrom(ij_part, 0, 0, 2), std::out_of_range);
}

TEST(Field, SetFromFieldOffset) {
    using namespace HPC4WC;
    Field f1(10, 10, 2, 2, 1.);
    Field f2(3, 3, 2, 2, 2.);
    Field f3(3, 3, 3, 2);

    EXPECT_NO_THROW(f1.setFrom(f2, 0, 0));
    EXPECT_THROW(f1.setFrom(f2, -3, 0), std::out_of_range);
    EXPECT_THROW(f1.setFrom(f3, 0, 0), std::logic_error);

    EXPECT_DOUBLE_EQ(f1(2, 2, 0), 2.);
}

TEST(Field, Comparison) {
    using namespace HPC4WC;
    Field f1(10, 10, 2, 2, 1.);
    Field f2(10, 10, 2, 2, 1.);
    EXPECT_EQ(f1, f2);

    f1(2, 2, 0) = 1.1;
    EXPECT_NE(f1, f2);

    Field f3(11, 10, 2, 2, 1.);
    EXPECT_NE(f1, f3);
    EXPECT_NE(f3, f1);
    EXPECT_NE(f2, f3);
}