#include <gtest/gtest.h>

#include <c10/util/Flags.h>

// NOLINTNEXTLINE(misc-use-internal-linkage)
C10_DEFINE_bool(c10_flags_test_only_flag, true, "Only used in test.");

namespace c10_test {

TEST(FlagsTest, DefineAndAssign) {
  EXPECT_EQ(FLAGS_c10_flags_test_only_flag, true);
  FLAGS_c10_flags_test_only_flag = false;
  EXPECT_EQ(FLAGS_c10_flags_test_only_flag, false);
  FLAGS_c10_flags_test_only_flag = true;
  EXPECT_EQ(FLAGS_c10_flags_test_only_flag, true);
}

} // namespace c10_test
