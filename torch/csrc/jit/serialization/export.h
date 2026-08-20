#pragma once

#include <caffe2/serialize/inline_container.h>
#include <torch/csrc/jit/api/module.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>
#include <ostream>

namespace torch::jit {

// Serializer for both oldstyle and unified format TorchScript serialization
class TORCH_API ScriptModuleSerializer {
 public:
  explicit ScriptModuleSerializer(
      caffe2::serialize::PyTorchStreamWriter& export_writer)
      : writer_(export_writer) {}

  void writeFiles(const std::string& code_dir);
  void serialize(const Module& module, const ExtraFilesMap& extra_files);
  void serialize_unified_format(Module& module, uint64_t script_module_id);
  SerializationStorageContext& storage_context();

  ~ScriptModuleSerializer() = default;

 private:
  void convertNamedType(const c10::NamedTypePtr& class_type);
  void convertTypes(const at::NamedTypePtr& root_type);
  void writeExtraFiles(const Module& module, const ExtraFilesMap& extra_files);
  void writeArchive(
      const IValue& value,
      const std::string& archive_name,
      const std::string& archive_dir,
      const std::string& tensor_dir,
      bool use_storage_context = false,
      bool skip_tensor_data = false);
  void updateSourceRangeTags(const SourceRangeRecords& ranges);

  caffe2::serialize::PyTorchStreamWriter& writer_;
  std::vector<at::IValue> constant_table_;

  std::unordered_set<c10::NamedTypePtr> converted_types_;
  PrintDepsTable class_deps_;
  TypeNameUniquer type_name_uniquer_;
  // qualifier, e.g. '__torch__.Bar' -> PythonPrint for the file that will be
  // created
  OrderedDict<std::string, PythonPrint> file_streams_;
  // Used to keep references of storages around during serialization to solve
  // for ABA memory reuse problem hit when storages are created/destroyed
  // during serialization process. Also used to coordinate sharing of storages
  // between Script and eager modules in torch.package.
  SerializationStorageContext storage_context_;

  // Uniquely identifies a SourceRange in a model, so that debug_pkl can store
  // <byte_offset, source_range_tag, source range> and a reader can recover the
  // source range for a given tag.
  SourceRangeTagMap source_range_tags_;
  int64_t current_source_range_tag_{0};
};

TORCH_API void ExportModule(
    const Module& module,
    std::ostream& out,
    const ExtraFilesMap& metadata = ExtraFilesMap());

TORCH_API void ExportModule(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& metadata = ExtraFilesMap());

TORCH_API void ExportModule(
    const Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func,
    const ExtraFilesMap& metadata = ExtraFilesMap());

// Write the bytes of a pickle archive and the tensors referenced inside that
// archive
TORCH_API void writeArchiveAndTensors(
    const std::string& archive_name,
    const char* pickle_bytes,
    size_t size,
    const std::vector<at::Tensor>& tensors,
    caffe2::serialize::PyTorchStreamWriter& out);

// Surrounding system can install an additional hook to produce extra files
// with metadata based on environment every time a module is serialized.
using ExportModuleExtraFilesHook = std::function<ExtraFilesMap(const Module&)>;
TORCH_API void SetExportModuleExtraFilesHook(ExportModuleExtraFilesHook hook);

} // namespace torch::jit
