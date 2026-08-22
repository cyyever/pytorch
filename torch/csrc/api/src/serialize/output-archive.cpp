#include <torch/serialize/output-archive.h>

#include <torch/types.h>

#include <c10/util/Exception.h>
#include <c10/util/Enumerate.h>
#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/serialization/pickler.h>
#include <torch/csrc/jit/serialization/pickler_helper.h>

#include <ostream>
#include <string>
#include <utility>

namespace torch::serialize {

namespace {

c10::impl::GenericDict newDict() {
  return c10::impl::GenericDict(
      c10::StringType::get(), c10::AnyType::get());
}

void writeArchive(
    const c10::IValue& value,
    caffe2::serialize::PyTorchStreamWriter& writer) {
  std::string data;
  std::vector<std::string> tensor_names;
  jit::Pickler pickler(
      [&](const char* buf, size_t size) { data.append(buf, size); },
      /*tensor_table=*/nullptr,
      /*type_renamer=*/nullptr,
      /*memoized_class_types=*/nullptr,
      [&](const at::Tensor&) {
        tensor_names.push_back(std::to_string(tensor_names.size()));
        return tensor_names.back();
      });
  pickler.protocol();
  pickler.pushIValue(value);
  pickler.stop();

  for (const auto [i, td] : c10::enumerate(pickler.tensorData())) {
    jit::WriteableTensorData writable_td = jit::getWriteableTensorData(td, /*to_cpu=*/false);
    writer.writeRecord(
        "data/" + tensor_names[i],
        writable_td.data(),
        writable_td.sizeInBytes());
  }
  writer.writeRecord("data/data.pkl", data.data(), data.size());
}

} // namespace

OutputArchive::OutputArchive() : dict_(newDict()) {}

void OutputArchive::write(const std::string& key, const c10::IValue& ivalue) {
  dict_.insert_or_assign(c10::IValue(key), ivalue);
}

void OutputArchive::write(
    const std::string& key,
    const Tensor& tensor,
    bool /*is_buffer*/) {
  dict_.insert_or_assign(c10::IValue(key), c10::IValue(tensor));
}

void OutputArchive::write(
    const std::string& key,
    OutputArchive& nested_archive) {
  dict_.insert_or_assign(c10::IValue(key), c10::IValue(nested_archive.dict_));
}

void OutputArchive::save_to(const std::string& filename) {
  caffe2::serialize::PyTorchStreamWriter writer(filename);
  writeArchive(dict_, writer);
  writer.writeEndOfFile();
}

void OutputArchive::save_to(std::ostream& stream) {
  save_to([&stream](const void* buf, size_t size) -> size_t {
    stream.write(static_cast<const char*>(buf), size);
    return !stream ? 0 : size;
  });
}

void OutputArchive::save_to(
    const std::function<size_t(const void*, size_t)>& func) {
  caffe2::serialize::PyTorchStreamWriter writer(func);
  writeArchive(dict_, writer);
  writer.writeEndOfFile();
}
} // namespace torch::serialize
