#pragma once

#include <c10/core/Backend.h>
#include <c10/core/TensorOptions.h>
#include <string>
#include <utility>
#include <vector>

namespace torch::utils {

std::string options_to_string(const at::TensorOptions& options);
std::string type_to_string(at::Backend backend, at::ScalarType scalar_type);
at::TensorOptions options_from_string(const std::string& str);

// return a vector of all "declared" types, even those that weren't compiled
std::vector<std::pair<at::Backend, at::ScalarType>> all_declared_types();

// return python module name of backend, like torch.cuda, torch.foo
const char* backend_to_string(const at::Backend& backend);

} // namespace torch::utils
