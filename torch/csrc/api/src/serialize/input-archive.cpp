#include <torch/serialize/input-archive.h>

#include <torch/types.h>
#include <torch/utils.h>

#include <c10/util/Exception.h>
#include <caffe2/serialize/file_adapter.h>
#include <caffe2/serialize/inline_container.h>
#include <caffe2/serialize/istream_adapter.h>
#include <caffe2/serialize/read_adapter_interface.h>
#include <torch/csrc/jit/serialization/import_read.h>

#include <istream>
#include <memory>
#include <string>
#include <utility>

namespace torch::serialize {

InputArchive::InputArchive()
    : dict_(c10::DictType::create(c10::StringType::get(), c10::AnyType::get())) {}

void InputArchive::read(const std::string& key, c10::IValue& ivalue) {
  TORCH_CHECK(
      try_read(key, ivalue),
      "No such serialized entry '",
      hierarchy_prefix_,
      key,
      "'");
}

bool InputArchive::try_read(const std::string& key, c10::IValue& ivalue) {
  if (!dict_.contains(key)) {
    return false;
  }
  ivalue = dict_.at(key);
  return true;
}

bool InputArchive::try_read(
    const std::string& key,
    Tensor& tensor,
    bool /*is_buffer*/) {
  if (!dict_.contains(key)) {
    return false;
  }
  auto iv = dict_.at(key);
  if (!iv.isTensor()) {
    return false;
  }
  auto read_tensor = std::move(iv).toTensor();
  if (tensor.defined()) {
    torch::NoGradGuard guard;
    if (tensor.device() != read_tensor.device()) {
      tensor.set_data(read_tensor);
    } else {
      tensor.set_(read_tensor);
    }
  } else {
    tensor = std::move(read_tensor);
  }
  return true;
}

void InputArchive::read(
    const std::string& key,
    Tensor& tensor,
    bool is_buffer) {
  TORCH_CHECK(
      try_read(key, tensor, is_buffer),
      "No such serialized tensor '",
      hierarchy_prefix_,
      key,
      "'");
}

bool InputArchive::try_read(const std::string& key, InputArchive& archive) {
  if (!dict_.contains(key)) {
    return false;
  }
  auto iv = dict_.at(key);
  if (!iv.isGenericDict()) {
    return false;
  }
  archive.dict_ = std::move(iv).toGenericDict();
  archive.hierarchy_prefix_ = hierarchy_prefix_ + key + ".";
  return true;
}

void InputArchive::read(const std::string& key, InputArchive& archive) {
  TORCH_CHECK(
      try_read(key, archive),
      "No such serialized submodule: '",
      hierarchy_prefix_,
      key,
      "'");
}

namespace {

c10::Dict<std::string, c10::IValue> readDataArchive(
    const std::shared_ptr<caffe2::serialize::ReadAdapterInterface>& rai,
    std::optional<torch::Device> device) {
  caffe2::serialize::PyTorchStreamReader reader(rai);
  c10::IValue ivalue = torch::jit::readArchiveAndTensors(
      /*archive_name=*/"data",
      /*pickle_prefix=*/"data/",
      /*tensor_prefix=*/"data/",
      /*type_resolver=*/std::nullopt,
      /*obj_loader=*/std::nullopt,
      device,
      reader);
  return std::move(ivalue).toGenericDict();
}

class RawDataAdapter : public caffe2::serialize::ReadAdapterInterface {
 public:
  RawDataAdapter(const char* data, size_t size) : data_(data), size_(size) {}
  size_t size() const override {
    return size_;
  }
  size_t read(
      uint64_t pos,
      void* buf,
      size_t n,
      [[maybe_unused]] const char* what = "") const override {
    if (pos >= size_) {
      return 0;
    }
    size_t nread = std::min(static_cast<size_t>(pos) + n, size_) - pos;
    memcpy(buf, data_ + pos, nread);
    return nread;
  }

 private:
  const char* data_;
  size_t size_;
};

class FuncDataAdapter : public caffe2::serialize::ReadAdapterInterface {
 public:
  FuncDataAdapter(
      const std::function<size_t(uint64_t, void*, size_t)>& read_func,
      const std::function<size_t(void)>& size_func)
      : read_func_(read_func), size_func_(size_func) {}
  size_t size() const override {
    return size_func_();
  }
  size_t read(
      uint64_t pos,
      void* buf,
      size_t n,
      [[maybe_unused]] const char* what = "") const override {
    return read_func_(pos, buf, n);
  }

 private:
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::function<size_t(uint64_t, void*, size_t)>& read_func_;
  // NOLINTNEXTLINE(cppcoreguidelines-avoid-const-or-ref-data-members)
  const std::function<size_t(void)>& size_func_;
};

} // namespace

void InputArchive::load_from(
    const std::string& filename,
    std::optional<torch::Device> device /*= std::nullopt*/) {
  dict_ = readDataArchive(
      std::make_shared<caffe2::serialize::FileAdapter>(filename), device);
}

void InputArchive::load_from(
    std::istream& stream,
    std::optional<torch::Device> device /*= std::nullopt*/) {
  dict_ = readDataArchive(
      std::make_shared<caffe2::serialize::IStreamAdapter>(&stream), device);
}

void InputArchive::load_from(
    const char* data,
    size_t size,
    std::optional<torch::Device> device /*= std::nullopt*/) {
  dict_ = readDataArchive(std::make_shared<RawDataAdapter>(data, size), device);
}

void InputArchive::load_from(
    const std::function<size_t(uint64_t, void*, size_t)>& read_func,
    const std::function<size_t(void)>& size_func,
    std::optional<torch::Device> device /*= std::nullopt*/) {
  dict_ = readDataArchive(
      std::make_shared<FuncDataAdapter>(read_func, size_func), device);
}

std::vector<std::string> InputArchive::keys() {
  std::vector<std::string> all_keys;
  all_keys.reserve(dict_.size());
  for (const auto& key : dict_.keys()) {
    all_keys.push_back(key);
  }
  return all_keys;
}

} // namespace torch::serialize
