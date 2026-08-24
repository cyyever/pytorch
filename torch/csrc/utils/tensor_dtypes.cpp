#include <torch/csrc/Dtype.h>
#include <torch/csrc/DynamicTypes.h>
#include <torch/csrc/Exceptions.h>
#include <torch/csrc/utils/object_ptr.h>
#include <torch/csrc/utils/tensor_dtypes.h>

namespace torch::utils {

void initializeDtypes() {
  auto torch_module = THPObjectPtr(PyImport_ImportModule("torch"));
  TORCH_CHECK_PYTHON(torch_module);

#define DEFINE_SCALAR_TYPE(_1, n) at::ScalarType::n,

  auto all_scalar_types = {
      AT_FORALL_SCALAR_TYPES_WITH_COMPLEX_AND_QINTS(DEFINE_SCALAR_TYPE)};

#undef DEFINE_SCALAR_TYPE

  // The quantized scalar types keep their enumerator values, which are part of
  // the serialization format, but this build has no quantization support at
  // all. Leaving them unbound means torch.qint8 and friends do not exist
  // rather than existing and failing somewhere deeper.
  auto is_quantized = [](at::ScalarType t) {
    switch (t) {
      case at::ScalarType::QInt8:
      case at::ScalarType::QUInt8:
      case at::ScalarType::QInt32:
      case at::ScalarType::QUInt4x2:
      case at::ScalarType::QUInt2x4:
        return true;
      default:
        return false;
    }
  };

  for (at::ScalarType scalarType : all_scalar_types) {
    if (is_quantized(scalarType)) {
      continue;
    }
    auto [primary_view, legacy_view] = c10::getDtypeNames(scalarType);
    std::string primary_name(primary_view);
    std::string legacy_name(legacy_view);
    THPObjectPtr dtype(THPDtype_New(scalarType, primary_name));
    torch::registerDtypeObject((THPDtype*)dtype.get(), scalarType);
    TORCH_CHECK_PYTHON(
        PyModule_AddObjectRef(
            torch_module.get(), primary_name.c_str(), dtype.get()) == 0);
    if (!legacy_name.empty()) {
      TORCH_CHECK_PYTHON(
          PyModule_AddObjectRef(
              torch_module.get(), legacy_name.c_str(), dtype.get()) == 0);
    }
  }
}

} // namespace torch::utils
