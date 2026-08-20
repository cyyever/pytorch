#include <torch/csrc/jit/serialization/export.h>

#include <c10/util/Exception.h>
#include <torch/csrc/jit/api/function_impl.h>
#include <torch/csrc/jit/frontend/source_range.h>
#include <torch/csrc/jit/ir/attributes.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/type_hashing.h>
#include <torch/csrc/jit/passes/inliner.h>
#include <torch/csrc/jit/serialization/callstack_debug_info_serialization.h>
#include <torch/csrc/jit/serialization/import_export_helpers.h>
#include <torch/csrc/jit/serialization/pickle.h>
#include <torch/csrc/jit/serialization/python_print.h>
#include <torch/csrc/jit/serialization/source_range_serialization.h>
#include <torch/csrc/jit/serialization/type_name_uniquer.h>

#include <caffe2/serialize/inline_container.h>

#include <ATen/ATen.h>

#include <ATen/core/jit_type.h>
#include <ATen/core/qualified_name.h>
#include <cerrno>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace torch::jit {

namespace {

ExportModuleExtraFilesHook& GetExtraFilesHook() {
  static ExportModuleExtraFilesHook func = nullptr;
  return func;
}

/**
 * If the type is not NamedTuple, it will return default_type_str. If the type
 * is a NamedTuple, it will return a string with following structure to describe
 * the content in the NamedTuple: "qualified_named[ NamedTuple, [ [filed_name_1,
 * field_type_1], [filed_name_2, field_type_2]
 *   ]
 * ]"
 *  Example NamedTuple type:
 *  "__torch__.base_models.sparse_nn.pytorch_preproc_types.PreprocOutputType[
 *     NamedTuple, [
 *         [float_features, Tensor],
 *         [id_list_features, List[Tensor]],
 *         [label,  Tensor],
 *         [weight, Tensor],
 *         ]
 *     ]"
 *
 * @param compilation_unit Jit compilation unit to look up function schema.
 * @param type_ptr A type pointer and it can be possibly any type.
 * @param default_type_str The default string representation. The string can
 * either from type_ptr->str(), type_ptr->annotation_str(), or
 * type_ptr->repr_str(). In some cases, they could be different in different
 * scenario. For example, Tensor type can be "Tensor", "Tensor (inferred)" and
 * "Tensor[]", and we only want "Tensor". Leave it as part of arguments as the
 * default return, when type_ptr is not a NamedTuple.
 * @return string representation.
 */
} // namespace

void SetExportModuleExtraFilesHook(ExportModuleExtraFilesHook hook) {
  GetExtraFilesHook() = std::move(hook);
}

void ScriptModuleSerializer::serialize(
    const Module& module,
    const ExtraFilesMap& extra_files) {
  C10_LOG_API_USAGE_ONCE("torch.jit.save");
  writeExtraFiles(module, extra_files);
  // Serialize the model object
  writeArchive(
      module._ivalue(),
      /*archive_name=*/"data",
      /*archive_dir=*/"",
      /*tensor_dir=*/"data/");
  // Then we serialize all code info.
  convertTypes(module.type());
  writeFiles("code/");
  // The tensor constants from the code are written to a separate archive
  // so loading the code does not depend on loading the data
  std::vector<IValue> ivalue_constants(
      constant_table_.begin(), constant_table_.end());
  writeArchive(
      c10::ivalue::Tuple::create(ivalue_constants),
      /*archive_name=*/"constants",
      /*archive_dir=*/"",
      /*tensor_dir=*/"constants/");
  if (!module.retrieve_traced_inputs().empty()) {
    writeArchive(
        module.retrieve_traced_inputs(),
        /*archive_name=*/"traced_inputs",
        /*archive_dir=*/"",
        /*tensor_dir=*/"traced_inputs/",
        /*use_storage_context*/ false,
        /*skip_tensor_data*/ true);
  }
  // Acquires and sets minimum (dynamic) version
  for (auto& item : file_streams_) {
    writer_.setMinVersion(item.value().minVersion());
  }
}

void ScriptModuleSerializer::writeArchive(
    const IValue& value,
    const std::string& archive_name,
    const std::string& archive_dir,
    const std::string& tensor_dir,
    bool use_storage_context,
    bool skip_tensor_data) {
  std::vector<char> data;
  // Vector to capture the run-time class types during pickling the IValues
  std::vector<c10::ClassTypePtr> memoizedClassTypes;
  std::vector<std::string> tensor_names;
  // tensors that are already serialized in use_storage_context
  std::unordered_set<std::string> serialized_tensors;
  Pickler data_pickle(
      [&](const char* buf, size_t size) {
        data.insert(data.end(), buf, buf + size);
      },
      nullptr,
      [&](const c10::ClassTypePtr& t) {
        return type_name_uniquer_.getUniqueName(t);
      },
      &memoizedClassTypes,
      [&](const at::Tensor& tensor) {
        // returns a string to use in picker.cpp as storage obj key
        if (use_storage_context) {
          bool already_serialized =
              storage_context_.hasStorage(tensor.storage());
          std::string tensor_name =
              std::to_string(
                  storage_context_.getOrAddStorage(tensor.storage())) +
              ".storage";
          if (already_serialized) {
            // this case is hit when storage has been serialized already
            // from a torch.package context
            serialized_tensors.insert(tensor_name);
          }
          tensor_names.push_back(tensor_name);
        } else {
          tensor_names.push_back(std::to_string(tensor_names.size()));
        }
        return tensor_names.back();
      });
  data_pickle.protocol();
  data_pickle.pushIValue(value);
  data_pickle.stop();
  // write out tensor data
  size_t i = 0;

  TORCH_INTERNAL_ASSERT(tensor_names.size() == data_pickle.tensorData().size());

  for (const auto& td : data_pickle.tensorData()) {
    const std::string& tensor_name = tensor_names[i++];
    if (td.is_meta() || skip_tensor_data) {
      writer_.writeRecord(tensor_dir + tensor_name, nullptr, 0);
      continue;
    }
    WriteableTensorData writable_td = getWriteableTensorData(td);
    if (use_storage_context && serialized_tensors.contains(tensor_name)) {
      // storage has been serialized already, skip
      continue;
    }
    writer_.writeRecord(
        tensor_dir + tensor_name,
        writable_td.data(),
        writable_td.sizeInBytes());
  }

  std::string fname = archive_dir + archive_name + ".pkl";
  writer_.writeRecord(fname, data.data(), data.size());

  // serialize all the captured run-time class types
  for (const c10::ClassTypePtr& wroteType : memoizedClassTypes) {
    convertNamedType(wroteType);
  }
}

void ScriptModuleSerializer::writeExtraFiles(
    const Module& module,
    const ExtraFilesMap& extra_files) {
  // Write out extra files.
  for (const auto& kv : extra_files) {
    const std::string key = "extra/" + kv.first;
    writer_.writeRecord(key, kv.second.data(), kv.second.size());
  }
  auto hook = GetExtraFilesHook();
  if (hook) {
    ExtraFilesMap hook_files = hook(module);
    for (const auto& kv : hook_files) {
      // Checks if the hooked file is already written in extra files,
      //   if so, skips it and warns
      if (extra_files.contains(kv.first)) {
        TORCH_WARN_ONCE(
            "An extra files hook attempted to write ",
            kv.first,
            " but ",
            "this is already written in extra files and so will be skipped. ",
            "This warning will only appear once per process.");
        continue;
      }
      const std::string key = "extra/" + kv.first;
      writer_.writeRecord(key, kv.second.data(), kv.second.size());
    }
  }
}

void ScriptModuleSerializer::updateSourceRangeTags(
    const SourceRangeRecords& ranges) {
  for (const auto& range : ranges) {
    if (!source_range_tags_.contains(range.range)) {
      source_range_tags_[range.range] = current_source_range_tag_;
      current_source_range_tag_++;
    }
  }
}

void ScriptModuleSerializer::convertTypes(const at::NamedTypePtr& root_type) {
  class_deps_.add(root_type);
  for (size_t i = 0; i < class_deps_.size(); ++i) {
    // note: convertNameType may extend class_deps_, so re-checking .size() is
    // necessary
    convertNamedType(class_deps_[i]);
  }
}

void ScriptModuleSerializer::writeFiles(const std::string& code_dir) {
  current_source_range_tag_ = 0;
  // Mapping of filename => src. We need this because multiple classes may go
  // in the same file (e.g. foo.bar.Baz and foo.bar.Qux)
  for (auto& item : file_streams_) {
    const std::string filename = qualifierToArchivePath(item.key(), code_dir);

    std::string src = item.value().str();

    // Only compress these records if they're not tiny.
    // The cpu cost of generating zip datastructs and compressing isn't
    // well-spent for very small records.
    static constexpr size_t kMinToCompress = 200;

    writer_.writeRecord(
        filename,
        src.c_str(),
        src.size(),
        src.size() > kMinToCompress /*compress*/);

    // Write out the debug information
    std::string debugFilename = filename + ".debug_pkl";
    SourceRangePickler source_range_pickler;
    updateSourceRangeTags(item.value().ranges());
    auto range_data =
        source_range_pickler.pickle(item.value().ranges(), source_range_tags_);
    writer_.writeRecord(
        debugFilename,
        range_data.data(),
        range_data.size(),
        range_data.size() > kMinToCompress /*compress*/);
  }
}

namespace {

std::optional<std::string> type_printer(
    const c10::Type& type,
    torch::jit::TypeNameUniquer& type_name_uniquer) {
  if (auto dyn = type.castRaw<c10::DynamicType>()) {
    return dyn->fallback()->annotation_str(
        [&](auto&& t) { return type_printer(t, type_name_uniquer); });
  }
  auto namedType = type.cast<c10::NamedType>();
  if (namedType && namedType->name()) {
    return type_name_uniquer.getUniqueName(namedType).qualifiedName();
  }
  return std::nullopt;
}

} // namespace

void ScriptModuleSerializer::convertNamedType(
    const c10::NamedTypePtr& class_type) {
  if (converted_types_.contains(class_type)) {
    return;
  }
  converted_types_.insert(class_type);
  auto qualname = type_name_uniquer_.getUniqueName(class_type);
  std::string qualifier = qualname.prefix();
  PythonPrint* pp = file_streams_.find(qualifier);

  if (!pp) {
    pp = &file_streams_.insert(
        std::move(qualifier),
        PythonPrint(
            constant_table_,
            class_deps_,
            [&](const c10::Type& t) {
              return type_printer(t, type_name_uniquer_);
            },
            /*enforce_importable=*/true));
  }
  pp->printNamedType(class_type);
}

void ScriptModuleSerializer::serialize_unified_format(
    Module& module,
    uint64_t script_module_id) {
  const std::string archive_dir =
      ".data/ts_code/" + std::to_string(script_module_id) + "/";

  // Serialize the model object
  writeArchive(
      module._ivalue(),
      "data",
      archive_dir,
      /*tensor_dir=*/".data/",
      /*use_storage_context=*/true);
  // Then we serialize all code info.
  convertTypes(module.type());
  // The tensor constants from the code are written to a separate archive
  // so loading the code does not depend on loading the data
  std::vector<IValue> ivalue_constants(
      constant_table_.begin(), constant_table_.end());
  writeArchive(
      c10::ivalue::Tuple::create(ivalue_constants),
      "constants",
      archive_dir,
      /*tensor_dir=*/".data/",
      /*use_storage_context=*/true);

  // Note: writeFiles() call needs to be made in addition to calling this
  // function to have the code actually saved (tensors are saved)
}

SerializationStorageContext& ScriptModuleSerializer::storage_context() {
  return storage_context_;
}

void ExportModule(
    const Module& module,
    std::ostream& out,
    const ExtraFilesMap& extra_files) {
  auto writer_func = [&](const void* buf, size_t nbytes) -> size_t {
    out.write(
        static_cast<const char*>(buf), static_cast<std::streamsize>(nbytes));
    return !out ? 0 : nbytes;
  };
  ExportModule(module, writer_func, extra_files);
}

void ExportModule(
    const Module& module,
    const std::string& filename,
    const ExtraFilesMap& extra_files) {
  // the zip archive needs to know the filepath
  caffe2::serialize::PyTorchStreamWriter writer(filename);
  ScriptModuleSerializer serializer(writer);
  serializer.serialize(module, extra_files);
}

void ExportModule(
    const Module& module,
    const std::function<size_t(const void*, size_t)>& writer_func,
    const ExtraFilesMap& extra_files) {
  caffe2::serialize::PyTorchStreamWriter writer(writer_func);
  ScriptModuleSerializer serializer(writer);
  serializer.serialize(module, extra_files);
}

} // namespace torch::jit
