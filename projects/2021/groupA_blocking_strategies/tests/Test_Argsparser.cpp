#include <HPC4WC/argsparser.h>
#include <gtest/gtest.h>

TEST(ArgsParser, Help) {
    using namespace HPC4WC;

    char* args1[] = {"binary", "--help", "--val=1"};
    char* args2[] = {"binary", "--help=false", "--val=1"};
    ArgsParser argsparser1(3, args1);
    ArgsParser argsparser2(3, args2);

    EXPECT_TRUE(argsparser1.help());
    EXPECT_FALSE(argsparser2.help());
}

TEST(ArgsParser, WriteHelp) {
    using namespace HPC4WC;

    char* args1[] = {"binary", "--help", "--val=1"};
    ArgsParser argsparser1(3, args1, false);
    std::stringstream ss;
    argsparser1.help(ss);
    // ss must at least contain a character now.
    EXPECT_LT(1, ss.str().size());
}

TEST(ArgsParser, DefaultArgs) {
    using namespace HPC4WC;
    char* args1[] = {"binary", "--help", "--val=1"};
    ArgsParser argsparser1(3, args1, false);
    ArgsParser argsparser2(3, args1, true);
    std::stringstream ss1;
    std::stringstream ss2;
    argsparser1.help(ss1);
    argsparser2.help(ss2);
    EXPECT_LT(ss1.str().size(), ss2.str().size());
}

TEST(ArgsParser, Arguments) {
    using namespace HPC4WC;

    char* args1[] = {"binary", "--help", "--val=1"};
    ArgsParser argsparser1(3, args1);

    Field::idx_t def_val = 0, val2 = 1;
    argsparser1.add_argument(def_val, "val", "Help for val.");
    argsparser1.add_argument(val2, "val2", "Help for val2.");
    EXPECT_EQ(def_val, 1);
    EXPECT_EQ(val2, 1);
}

TEST(ArgsParser, BoolArg) {
    using namespace HPC4WC;

    char* args[] = {"binary", "--val1=f", "--val2=1", "--val3"};
    ArgsParser argsparser(4, args);

    bool v1 = false, v2 = false, v3 = false;
    argsparser.add_argument(v1, "val1", "val1");
    argsparser.add_argument(v2, "val2", "val2");
    argsparser.add_argument(v3, "val3", "val3");

    EXPECT_FALSE(v1);
    EXPECT_TRUE(v2);
    EXPECT_TRUE(v3);
}