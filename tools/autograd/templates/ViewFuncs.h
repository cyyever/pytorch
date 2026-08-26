#pragma once

// ${generated_comment}

#include <torch/library.h>
#include <torch/csrc/autograd/variable.h>
#include <c10/core/SymIntArrayRef.h>

$ops_headers

namespace torch::autograd::generated {

using at::Scalar;
using at::Tensor;
using at::IntArrayRef;
using at::ArrayRef;
using at::Type;
using at::ScalarType;
using std::optional;
using c10::fmap;

${view_func_declarations}

} // namespace torch::autograd::generated
