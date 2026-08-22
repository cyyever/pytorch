#include <ATen/native/quantized/IndexKernel.h>

namespace at::native {

DEFINE_DISPATCH(masked_fill_kernel_quantized_stub);
DEFINE_DISPATCH(index_put_kernel_quantized_stub);

} // namespace at::native
