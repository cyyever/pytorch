#pragma once

#include <ATen/core/ivalue.h>
#include <c10/util/ArrayRef.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/unpickler.h>

namespace torch::jit {

/// Pickle an IValue by calling a function to handle writing the data.
///
/// `writer` is a function that takes in a pointer to a chunk of memory and its
/// size and consumes it.
///
/// See `jit::pickle` for more details.
TORCH_API void pickle(
    std::function<void(const char* data_start, size_t data_len)> writer,
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table = nullptr);

/// Save a `torch::IValue` in a format compatible with Python's `pickle` module
///
/// If present, `tensor_table` is a pointer to a table in which tensors that
/// are contained within `ivalue` are stored, and the bytes returned by the
/// pickler will only include references to these tensors in the table. This can
/// be used to keep the binary blob size small.
/// If not provided, tensors are stored in the same byte stream as the pickle
/// data, similar to `torch.save()` in eager Python.
///
/// Pickled values can be loaded in Python and C++:
/// \rst
/// .. code-block:: cpp
///
///  torch::IValue float_value(2.3);
///
///  // TODO: when tensors are stored in the pickle, delete this
///  std::vector<at::Tensor> tensor_table;
///  auto data = torch::jit::pickle(float_value, &tensor_table);
///
///  torch::IValue value = torch::jit::pickle_load(data);
///
/// .. code-block:: python
///
///   values = torch.load('data.pkl')
///   print(values)
///
/// \endrst
TORCH_API std::vector<char> pickle(
    const IValue& ivalue,
    std::vector<at::Tensor>* tensor_table = nullptr);

/// Save a `torch::IValue` in a format that can be loaded by both
/// `torch::pickle_load` in C++ and `torch.load` in Python.
TORCH_API std::vector<char> pickle_save(const IValue& ivalue);

/// Deserialize a `torch::IValue` from bytes produced by either
/// `torch::pickle_save` in C++ or `torch.save` in Python
TORCH_API IValue pickle_load(const std::vector<char>& data);

class VectorReader : public caffe2::serialize::ReadAdapterInterface {
 public:
  VectorReader(std::vector<char> data) : data_(std::move(data)) {}

  size_t size() const override {
    return data_.size();
  }

  size_t read(uint64_t pos, void* buf, size_t n, const char* what)
      const override;

 private:
  std::vector<char> data_;
};

class StringViewReader : public caffe2::serialize::ReadAdapterInterface {
 public:
  StringViewReader(std::string_view data) : data_(data) {}

  size_t size() const override {
    return data_.size();
  }

  size_t read(uint64_t pos, void* buf, size_t n, const char* what)
      const override;

 private:
  std::string_view data_;
};

/// Deserialize a `torch::IValue` from bytes produced by either
/// `torch::pickle_save` in C++ or `torch.save` in Python with custom object.
TORCH_API IValue pickle_load_obj(std::string_view data);

} // namespace torch::jit
