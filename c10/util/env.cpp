#include <c10/util/Exception.h>
#include <c10/util/env.h>
#include <cstdlib>
#include <mutex>
#include <shared_mutex>

namespace c10::utils {

static std::shared_mutex& get_env_mutex() {
  static std::shared_mutex env_mutex;
  return env_mutex;
}

// Set an environment variable.
void set_env(const char* name, const char* value, bool overwrite) {
  std::lock_guard lk(get_env_mutex());
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  auto err = setenv(name, value, static_cast<int>(overwrite));
  TORCH_INTERNAL_ASSERT(
      err == 0,
      "setenv failed for environment \"",
      name,
      "\", the error is: ",
      err);
  return;
}

// Remove an environment variable.
void unset_env(const char* name) {
  std::lock_guard lk(get_env_mutex());
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  auto err = unsetenv(name);
  TORCH_INTERNAL_ASSERT(
      err == 0,
      "unsetenv failed for environment \"",
      name,
      "\", the error is: ",
      err);
}

// Reads an environment variable and returns the content if it is set
std::optional<std::string> get_env(const char* name) noexcept {
  std::shared_lock lk(get_env_mutex());
  // NOLINTNEXTLINE(concurrency-mt-unsafe)
  auto envar = std::getenv(name);
  if (envar != nullptr) {
    return std::string(envar);
  }
  return std::nullopt;
}

// Checks an environment variable is set.
bool has_env(const char* name) noexcept {
  return get_env(name).has_value();
}

// Reads an environment variable and returns
// - optional<true>,              if set equal to "1"
// - optional<false>,             if set equal to "0"
// - nullopt,   otherwise
//
// NB:
// Issues a warning if the value of the environment variable is not 0 or 1.
std::optional<bool> check_env(const char* name) {
  auto env_opt = get_env(name);
  if (env_opt.has_value()) {
    if (env_opt == "0") {
      return false;
    }
    if (env_opt == "1") {
      return true;
    }
    TORCH_WARN(
        "Ignoring invalid value for boolean flag ",
        name,
        ": ",
        *env_opt,
        "valid values are 0 or 1.");
  }
  return std::nullopt;
}
} // namespace c10::utils
