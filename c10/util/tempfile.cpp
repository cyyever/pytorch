#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <c10/util/error.h>
#include <c10/util/tempfile.h>
#include <fmt/format.h>

#include <unistd.h>
#include <cerrno>

// Creates the filename pattern passed to and completed by `mkstemp`.
static std::string make_filename(std::string_view name_prefix) {
  // The filename argument to `mkstemp` needs "XXXXXX" at the end according to
  // http://pubs.opengroup.org/onlinepubs/009695399/functions/mkstemp.html
  constexpr const char* kRandomPattern = "XXXXXX";

  // We see if any of these environment variables is set and use their value, or
  // else default the temporary directory to `/tmp`.

  std::string tmp_directory = "/tmp";
  for (const char* variable : {"TMPDIR", "TMP", "TEMP", "TEMPDIR"}) {
    auto path_opt = c10::utils::get_env(variable);
    if (path_opt.has_value()) {
      tmp_directory = path_opt.value();
      break;
    }
  }
  return fmt::format("{}/{}{}", tmp_directory, name_prefix, kRandomPattern);
}

namespace c10 {
/// Attempts to return a temporary file or returns `nullopt` if an error
/// occurred.
std::optional<TempFile> try_make_tempfile(std::string_view name_prefix) {
  auto filename = make_filename(name_prefix);
  if (filename.empty()) {
    return std::nullopt;
  }
  const int fd = mkstemp(filename.data());
  if (fd == -1) {
    return std::nullopt;
  }
  return TempFile(std::move(filename), fd);
}

/// Like `try_make_tempfile`, but throws an exception if a temporary file could
/// not be returned.
TempFile make_tempfile(std::string_view name_prefix) {
  if (auto tempfile = try_make_tempfile(name_prefix)) {
    return std::move(*tempfile);
  }
  TORCH_CHECK(
      false, "Error generating temporary file: ", c10::utils::str_error(errno));
}

/// Attempts to return a temporary directory or returns `nullopt` if an error
/// occurred.
std::optional<TempDir> try_make_tempdir(std::string_view name_prefix) {
  auto filename = make_filename(name_prefix);
  const char* dirname = mkdtemp(filename.data());
  if (!dirname) {
    return std::nullopt;
  }
  return TempDir(dirname);
}


TempFile::~TempFile() {
  if (!name.empty()) {
    if (fd >= 0) {
      unlink(name.c_str());
      close(fd);
    }
  }
}

TempDir::~TempDir() {
  if (!name.empty()) {
    rmdir(name.c_str());
  }
}

/// Like `try_make_tempdir`, but throws an exception if a temporary directory
/// could not be returned.
TempDir make_tempdir(std::string_view name_prefix) {
  if (auto tempdir = try_make_tempdir(name_prefix)) {
    return std::move(*tempdir);
  }
  TORCH_CHECK(
      false,
      "Error generating temporary directory: ",
      c10::utils::str_error(errno));
}
} // namespace c10
