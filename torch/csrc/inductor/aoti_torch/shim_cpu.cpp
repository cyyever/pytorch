
#ifdef USE_DISTRIBUTED
#include <torch/csrc/distributed/c10d/Functional.hpp>
#endif
#include <torch/csrc/inductor/aoti_torch/c/shim_cpu.h>
#include <torch/csrc/inductor/aoti_torch/utils.h>

#ifndef AT_PER_OPERATOR_HEADERS
#include <ATen/CPUFunctions.h>
#else
#endif

using namespace torch::aot_inductor;


AOTITorchError aoti_torch_cpu__weight_int4pack_mm_cpu_tensor(
    AtenTensorHandle X,
    AtenTensorHandle w,
    AtenTensorHandle qGroupSize,
    AtenTensorHandle qScaleAndZeros,
    AtenTensorHandle* ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = at::native::_weight_int4pack_mm_cpu(
        *tensor_handle_to_tensor_pointer(X),
        *tensor_handle_to_tensor_pointer(w),
        tensor_handle_to_tensor_pointer(qGroupSize)->item<int64_t>(),
        *tensor_handle_to_tensor_pointer(qScaleAndZeros));
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

#ifdef USE_DISTRIBUTED
AOTITorchError aoti_torch_cpu__c10d_functional_all_reduce_(
    AtenTensorHandle inp,
    const char* reduce_op,
    const char* group_name,
    AtenTensorHandle* ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = c10d::all_reduce_(
        *tensor_handle_to_tensor_pointer(inp), reduce_op, group_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu__c10d_functional_all_reduce(
    AtenTensorHandle inp,
    const char* reduce_op,
    const char* group_name,
    AtenTensorHandle* ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = c10d::all_reduce(
        *tensor_handle_to_tensor_pointer(inp), reduce_op, group_name);
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}

AOTITorchError aoti_torch_cpu__c10d_functional_wait_tensor(
    AtenTensorHandle inp,
    AtenTensorHandle* ret0) {
  AOTI_TORCH_CONVERT_EXCEPTION_TO_ERROR_CODE({
    auto tmp_result = c10d::wait_tensor(*tensor_handle_to_tensor_pointer(inp));
    *ret0 = new_tensor_handle(std::move(tmp_result));
  });
}
#endif
