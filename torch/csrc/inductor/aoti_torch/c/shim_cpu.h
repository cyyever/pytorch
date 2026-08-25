#ifndef AOTI_TORCH_SHIM_CPU
#define AOTI_TORCH_SHIM_CPU

#include <ATen/Config.h>
#include <torch/csrc/inductor/aoti_torch/c/shim.h>

#ifdef __cplusplus
extern "C" {
#endif


AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__weight_int4pack_mm_cpu_tensor(
    AtenTensorHandle X,
    AtenTensorHandle w,
    AtenTensorHandle qGroupSize,
    AtenTensorHandle qScaleAndZeros,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__c10d_functional_all_reduce_(
    AtenTensorHandle inp,
    const char* reduce_op,
    const char* group_name,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__c10d_functional_all_reduce(
    AtenTensorHandle inp,
    const char* reduce_op,
    const char* group_name,
    AtenTensorHandle* ret0);

AOTI_TORCH_EXPORT AOTITorchError aoti_torch_cpu__c10d_functional_wait_tensor(
    AtenTensorHandle inp,
    AtenTensorHandle* ret0);

#ifdef __cplusplus
} // extern "C"
#endif
#endif // AOTI_TORCH_SHIM_CPU
