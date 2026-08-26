#include <torch/csrc/inductor/aoti_torch/mkldnn_tensor.h>


namespace torch::aot_inductor {


void* data_ptr_from_mkldnn(at::Tensor* mkldnn_tensor) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}

at::Tensor mkldnn_tensor_from_data_ptr(
    void* data_ptr,
    at::IntArrayRef dims,
    at::ScalarType dtype,
    at::Device device,
    const uint8_t* opaque_metadata,
    int64_t opaque_metadata_size) {
  TORCH_CHECK(false, "MKL-DNN build is disabled");
}


} // namespace torch::aot_inductor
