#include <c10/macros/Macros.h>
#include <torch/headeronly/util/Float8_e4m3fnuz.h>

namespace c10 {

static_assert(
    std::is_standard_layout_v<Float8_e4m3fnuz>,
    "c10::Float8_e4m3fnuz must be standard layout.");

} // namespace c10
