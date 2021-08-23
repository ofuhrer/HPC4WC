#include <HPC4WC/io.h>
#include <gtest/gtest.h>

TEST(IO, WriteFull) {
    using namespace HPC4WC;
    Field f1(1, 1, 1, 2, 0.0);
    std::stringstream ss1;
    IO::write(ss1, f1);

    EXPECT_EQ(ss1.str(), "0 \n");

    Field f2(2, 2, 2, 2, 1.);
    std::stringstream ss2;
    IO::write(ss2, f2);

    EXPECT_EQ(ss2.str(), "1 1 \n1 1 \n1 1 \n1 1 \n");
}

TEST(IO, WriteK) {
    using namespace HPC4WC;
    Field f1(1, 1, 1, 2, 0.0);
    std::stringstream ss1;
    IO::write(ss1, f1, 0);

    EXPECT_EQ(ss1.str(), "0 \n");
    EXPECT_THROW(IO::write(ss1, f1, 3), std::out_of_range);
}