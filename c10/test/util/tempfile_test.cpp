#include <c10/util/tempfile.h>
#include <gtest/gtest.h>
#include <sys/stat.h>
#include <optional>

static bool file_exists(const char* path) {
  struct stat st{};
  return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}
static bool directory_exists(const char* path) {
  struct stat st{};
  return stat(path, &st) == 0 && S_ISDIR(st.st_mode);
}

TEST(TempFileTest, MatchesExpectedPattern) {
  c10::TempFile file = c10::make_tempfile("test-pattern-");

  ASSERT_TRUE(file_exists(file.name.c_str()));
  ASSERT_NE(file.name.find("test-pattern-"), std::string::npos);
}

TEST(TempDirTest, tryMakeTempdir) {
  std::optional<c10::TempDir> tempdir = c10::make_tempdir("test-dir-");
  std::string tempdir_name = tempdir->name;

  // directory should exist while tempdir is alive
  ASSERT_TRUE(directory_exists(tempdir_name.c_str()));

  // directory should not exist after tempdir destroyed
  tempdir.reset();
  ASSERT_FALSE(directory_exists(tempdir_name.c_str()));
}
