#include <ATen/Context.h>

#include <map>
#include <string>

namespace at {

/// The build settings baked in by CMake, as shown by torch.__config__.show().
TORCH_API const std::map<std::string, std::string>& GetBuildOptions();

/// Returns a detailed string describing the configuration PyTorch.
TORCH_API std::string show_config();

TORCH_API std::string get_mkl_version();


TORCH_API std::string get_openmp_version();

TORCH_API std::string get_cxx_flags();

TORCH_API std::string get_cpu_capability();

} // namespace at
