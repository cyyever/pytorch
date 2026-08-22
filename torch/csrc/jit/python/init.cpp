#include <torch/csrc/utils/pybind.h>
#include <torch/csrc/utils/python_arg_parser.h>
#include <torch/csrc/utils/schema_info.h>

#include <ATen/core/operator_name.h>
#include <c10/core/SymNodeImpl.h>
#include <torch/csrc/jit/frontend/schema_type_parser.h>
#include <torch/csrc/jit/frontend/tracer.h>
#include <torch/csrc/jit/python/init.h>
#include <torch/csrc/jit/python/opaque_obj.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/jit/python/python_custom_class.h>
#include <torch/csrc/jit/python/utf8_decoding_ignore.h>
#include <torch/csrc/jit/runtime/graph_executor.h>
#include <torch/csrc/jit/runtime/jit_exception.h>
#include <torch/csrc/jit/runtime/operator.h>
#include <torch/csrc/jit/serialization/storage_context.h>
#include <torch/csrc/utils/cpp_stacktraces.h>

#include <c10/macros/Export.h>
#include <c10/util/irange.h>
#include <caffe2/serialize/inline_container.h>

#include <pybind11/cast.h>
#include <pybind11/functional.h>
#include <pybind11/operators.h>

#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <utility>

namespace torch::jit {

using c10::AliasInfo;
using c10::Argument;
using c10::FunctionSchema;
using c10::SchemaArgType;
using c10::SchemaArgument;
using c10::SymNode;
using caffe2::serialize::PyTorchStreamReader;
using caffe2::serialize::PyTorchStreamWriter;
using torch::utils::SchemaInfo;

namespace {

static bool opAllowsNumbersAsTensors(c10::Symbol symbol) {
  return symbol.is_prims() || symbol.is_nvprims() ||
      (symbol.is_aten() &&
       torch::should_allow_numbers_as_tensors(symbol.toUnqualString()));
}

std::optional<IValue> toTypeInferredIValueOptional(py::handle input) {
  // Errors need to be caught here because toTypeInferredIValue errors out
  // on various object types, but we want it to work with all types.
  try {
    return toTypeInferredIValue(input);
  } catch (const c10::Error&) {
    return std::nullopt;
  }
}

} // anonymous namespace

void initJITBindings(PyObject* module) {
  auto m = py::handle(module).cast<py::module>();

  // This is a static object, so we must leak the Python object
  // "release()" is used here to preserve 1 refcount on the
  // object, preventing it from ever being de-allocated by CPython.
  static py::handle exc =
      py::exception<JITException>(m, "JITException").release();

  py::register_exception_translator([](std::exception_ptr p) {
    try {
      if (p) {
        std::rethrow_exception(p);
      }
    } catch (const JITException& e) {
      // special handling of JITException, to set its python class name and msg
      py::gil_scoped_acquire acquire;
      const auto& className = e.getPythonClassName();
      const auto& originalMsg = e.getOriginalMsg();
      JITException::setCaughtOriginalMsg(originalMsg.value_or(""));
      JITException::setCaughtPythonClassName(className.value_or(""));
      PyErr_SetString(exc.ptr(), e.what());
    }
  });

  m.def(
      "_get_caught_jit_exception_class_name",
      JITException::getCaughtPythonClassName);
  m.def(
      "_get_caught_jit_exception_original_msg",
      JITException::getCaughtOriginalMsg);

  m.def("_jit_init", []() { return true; })
      // Tracing is always off; kept as a None shim because Python code
      // (e.g. torch.distributions) calls torch._C._get_tracing_state().
      .def("_get_tracing_state", []() { return py::none(); })
      .def(
          "_jit_set_tracer_state_warn",
          [](bool new_warn) {
            jit::tracer::getTracerStateWarnMode() = new_warn;
          })
      .def(
          "_jit_get_tracer_state_warn",
          []() { return jit::tracer::getTracerStateWarnMode().load(); })
      .def(
          "_jit_set_profiling_mode",
          [](bool profiling_flag) {
            bool oldState = getProfilingMode();
            getProfilingMode() = profiling_flag;
            return oldState;
          })
      .def(
          "_jit_set_profiling_executor",
          [](bool profiling_flag) {
            bool oldState = getExecutorMode();
            getExecutorMode() = profiling_flag;
            return oldState;
          })
      .def(
          "_jit_set_num_profiled_runs",
          [](size_t num) {
            size_t old_num = getNumProfiledRuns();
            getNumProfiledRuns() = num;
            return old_num;
          })
      .def("_jit_get_num_profiled_runs", [] {
        // pybind can't automatically bind to atomic size_t
        return static_cast<size_t>(getNumProfiledRuns());
      })
      .def(
          "_jit_set_fusion_strategy",
          [](const std::vector<std::pair<std::string, size_t>>& strategy) {
            FusionStrategy vec_conv;
            for (const auto& pair : strategy) {
              if (pair.first == "STATIC") {
                vec_conv.emplace_back(FusionBehavior::STATIC, pair.second);
              } else if (pair.first == "DYNAMIC") {
                vec_conv.emplace_back(FusionBehavior::DYNAMIC, pair.second);
              } else {
                throw py::value_error(
                    "FusionBehavior only supported 'STATIC' or 'DYNAMIC', got: " +
                    pair.first);
              }
            }
            auto old_strategy = getFusionStrategy();
            setFusionStrategy(vec_conv);
            return fmap(
                old_strategy,
                [](std::pair<FusionBehavior, size_t> behav) {
                  return std::pair<std::string, size_t>(
                      behav.first == FusionBehavior::STATIC ? "STATIC"
                                                            : "DYNAMIC",
                      behav.second);
                });
          })
      .def(
          "_jit_set_inline_everything_mode",
          [](bool enabled) { getInlineEverythingMode() = enabled; })
      .def("_jit_get_inline_everything_mode", []() {
        return getInlineEverythingMode();
      })
      .def(
          "_storage_id",
          [](const at::Tensor& ten) -> int64_t {
            return reinterpret_cast<int64_t>(
                ten.storage().unsafeGetStorageImpl());
          })
      .def(
          "_jit_set_utf8_decoding_ignore",
          &setUTF8DecodingIgnore);

  py::class_<PyTorchStreamWriter>(m, "PyTorchFileWriter")
      .def(
          py::init<std::string, bool, uint64_t>(),
          py::arg("file_name"),
          py::arg("compute_crc32") = true,
          py::arg("storage_alignment") = 64)
      .def(
          py::init([](const py::object& buffer,
                      bool compute_crc32 = true,
                      uint64_t storage_alignment = 64) {
            auto writer_func = [=](const void* data, size_t size) {
              // Writing an empty file is a noop
              if (size == 0) {
                return size;
              }
              py::gil_scoped_acquire acquire;
              if (!data) {
                // See [Note: write_record_metadata]
                buffer.attr("seek")(
                    size, py::module::import("os").attr("SEEK_CUR"));
              } else {
                auto memory_view = py::memoryview::from_memory(
                    reinterpret_cast<const char*>(data), size);
                buffer.attr("write")(std::move(memory_view));
              }
              return size;
            };
            return std::make_unique<PyTorchStreamWriter>(
                std::move(writer_func), compute_crc32, storage_alignment);
          }),
          py::arg("buffer"),
          py::arg("compute_crc32") = true,
          py::arg("storage_alignment") = 64)
      .def(
          py::init<
              const std::function<size_t(const void*, size_t)>&,
              bool,
              uint64_t>(),
          py::arg("writer_func"),
          py::arg("compute_crc32") = true,
          py::arg("storage_alignment") = 64)
      // [Note: write_record_metadata]
      // The write_record_metadata function is intended to write metadata (i.e.
      // the zipfile header and end of central directory record) for a file
      // while reserving nbytes of space for the file for the bytes of the
      // actual file to be added in later. This functionality is achieved by
      // defining `m_pWrite` to seek instead of write if the buffer passed is a
      // nullptr. This has implications on CRC-32 which will not be written at
      // write_record_metadata time, and will not be combined with the hash in
      // combined_uncomp_crc32_. We define this in `m_pWrite` rather than
      // extending the interface of miniz to have an `m_pSeek` since different
      // versions of miniz are used in fbcode/oss.
      .def(
          "write_record_metadata",
          [](PyTorchStreamWriter& self, const std::string& name, size_t size) {
            return self.writeRecord(name, nullptr, size);
          })
      .def(
          "write_record",
          [](PyTorchStreamWriter& self,
             const std::string& name,
             const char* data,
             size_t size) {
            // Since we don't know where the data come from, we cannot
            // release the GIL in this overload
            return self.writeRecord(name, data, size);
          })
      .def(
          "write_record",
          [](PyTorchStreamWriter& self,
             const std::string& name,
             py::bytes data,
             size_t size) {
            // It is not clear from the doc but according to CPython own code,
            // it is ok to use the result of PyBytes_AsString without the GIL
            // being held
            const char* data_str = PyBytes_AsString(data.ptr());
            py::gil_scoped_release release;
            return self.writeRecord(name, data_str, size);
          })
      .def(
          "write_record",
          [](PyTorchStreamWriter& self,
             const std::string& name,
             const c10::Storage& data,
             size_t size) {
            // Reading Tensor data is always ok without the GIL held
            py::gil_scoped_release release;
            return self.writeRecord(
                name, reinterpret_cast<const char*>(data.data()), size);
          })
      .def(
          "write_record",
          [](PyTorchStreamWriter& self,
             const std::string& name,
             uintptr_t data,
             size_t size) {
            TORCH_WARN_ONCE(
                "write_record(): Passing Storage by data pointer is deprecated and will be an error in ",
                "the future, please pass the Storage object instead.");
            return self.writeRecord(
                name, reinterpret_cast<const char*>(data), size);
          })
      .def("write_end_of_file", &PyTorchStreamWriter::writeEndOfFile)
      .def("set_min_version", &PyTorchStreamWriter::setMinVersion)
      .def("archive_name", &PyTorchStreamWriter::archiveName)
      .def("serialization_id", &PyTorchStreamWriter::serializationId)
      .def(
          "get_all_written_records",
          &PyTorchStreamWriter::getAllWrittenRecords);

  // This allows PyTorchStreamReader to read from a Python buffer. It requires
  // that the buffer implement `seek()`, `tell()`, and `read()`.
  class BufferAdapter : public caffe2::serialize::ReadAdapterInterface {
   public:
    BufferAdapter(const py::object& buffer) : buffer_(buffer) {
      // Jump to the end of the buffer to get its size
      auto current = buffer.attr("tell")();
      start_offset_ = py::cast<size_t>(current);
      buffer.attr("seek")(0, py::module::import("os").attr("SEEK_END"));
      size_ = py::cast<size_t>(buffer.attr("tell")()) - start_offset_;
      buffer.attr("seek")(current);
      // If we can read directly into a buffer, do that instead of an extra copy
      // NOLINTNEXTLINE(cppcoreguidelines-prefer-member-initializer)
      use_readinto_ = py::hasattr(buffer, "readinto");
    }

    size_t size() const override {
      return size_;
    }

    THPObjectPtr getMemview(void* buf, size_t n) const {
      THPObjectPtr memview(PyMemoryView_FromMemory(
          reinterpret_cast<char*>(buf), n, PyBUF_WRITE));
      TORCH_CHECK_PYTHON(memview);
      return memview;
    }

    size_t read(uint64_t pos, void* buf, size_t n, const char* what)
        const override {
      // Seek to desired position (NB: this has to be a Py_ssize_t or Python
      // throws a weird error)
      Py_ssize_t absolute_pos = start_offset_ + pos;
      buffer_.attr("seek")(absolute_pos);

      if (use_readinto_) {
        auto memview = getMemview(buf, n);
        auto res =
            PyObject_CallMethod(buffer_.ptr(), "readinto", "O", memview.get());
        if (res) {
          int64_t i = static_cast<int64_t>(PyLong_AsLongLong(res));
          Py_DECREF(res);
          if (i > 0) {
            return i;
          }
        }
      }

      // Read bytes into `buf` from the buffer
      std::string bytes = py::cast<std::string>(buffer_.attr("read")(n));
      std::copy(
          bytes.data(),
          bytes.data() + bytes.size(),
          reinterpret_cast<char*>(buf));
      return bytes.size();
    }

    py::object buffer_;
    size_t size_;
    size_t start_offset_;
    bool use_readinto_{};
  };

  py::class_<PyTorchStreamReader, std::shared_ptr<PyTorchStreamReader>>(
      m, "PyTorchFileReader")
      .def(py::init<std::string>())
      .def(py::init([](const py::object& buffer) {
        auto adapter = std::make_unique<BufferAdapter>(buffer);
        return std::make_shared<PyTorchStreamReader>(std::move(adapter));
      }))
      .def(
          "get_record",
          [](PyTorchStreamReader& self, const std::string& key) {
            auto [data, size] = self.getRecord(key);
            return py::bytes(reinterpret_cast<const char*>(data.get()), size);
          })
      .def(
          "has_record",
          [](PyTorchStreamReader& self, const std::string& key) {
            return self.hasRecord(key);
          })
      .def(
          "get_storage_from_record",
          [](PyTorchStreamReader& self,
             const std::string& key,
             size_t numel,
             py::object data_type_obj) {
            auto [data, size] = self.getRecord(key);
            auto scalar_type =
                reinterpret_cast<THPDtype*>(data_type_obj.ptr())->scalar_type;

            TORCH_CHECK(
                size == numel * elementSize(scalar_type),
                "record size (",
                size,
                " bytes) does not match expected size (",
                numel * elementSize(scalar_type),
                " bytes = ",
                numel,
                " elements * ",
                elementSize(scalar_type),
                " bytes/element) for dtype ",
                scalar_type);

            c10::Storage storage(
                c10::Storage::use_byte_size_t(),
                numel * elementSize(scalar_type),
                std::move(data),
                /*allocator=*/nullptr,
                /*resizable=*/false);
            auto ptr =
                c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
                    std::move(storage),
                    at::DispatchKeySet(),
                    at::CPU(scalar_type).typeMeta());
            return at::Tensor(std::move(ptr));
          })
      .def("serialization_id", &PyTorchStreamReader::serializationId)
      .def(
          "get_all_records",
          [](PyTorchStreamReader& self) { return self.getAllRecords(); })
      .def(
          "get_record_offset",
          [](PyTorchStreamReader& self, const std::string& key) {
            return self.getRecordOffset(key);
          })
      .def(
          "get_record_header_offset",
          [](PyTorchStreamReader& self, const std::string& key) {
            return self.getRecordHeaderOffset(key);
          })
      .def(
          "get_record_offset_no_read",
          [](PyTorchStreamReader& self,
             size_t zipfile_header_offset,
             const std::string& filename,
             size_t size,
             uint64_t storage_alignment) {
            return self.getRecordOffsetNoRead(
                zipfile_header_offset, filename, size, storage_alignment);
          })
      .def(
          "get_record_size",
          [](PyTorchStreamReader& self, const std::string& key) {
            return self.getRecordSize(key);
          });

  // Used by torch.Package to coordinate deserialization of storages across
  // ScriptModules and eager modules
  py::class_<
      DeserializationStorageContext,
      std::shared_ptr<DeserializationStorageContext>>(
      m, "DeserializationStorageContext")
      .def(py::init<>())
      .def(
          "get_storage",
          [](DeserializationStorageContext& self,
             const std::string& name,
             py::object data_type_obj) {
            c10::Storage storage = self.getStorage(name);
            auto scalar_type =
                reinterpret_cast<THPDtype*>(data_type_obj.ptr())->scalar_type;
            auto ptr =
                c10::make_intrusive<at::TensorImpl, at::UndefinedTensorImpl>(
                    std::move(storage),
                    at::DispatchKeySet(),
                    at::CPU(scalar_type).typeMeta());

            return at::Tensor(std::move(ptr));
          })
      .def(
          "add_storage",
          [](DeserializationStorageContext& self,
             const std::string& name,
             const at::Tensor& tensor) {
            return self.addStorage(name, tensor.storage());
          })
      .def("has_storage", &DeserializationStorageContext::hasStorage);

  m.def(
      "_get_schema",
      [](const std::string& op_name, const std::string& overload_name) {
        try {
          auto symbol = Symbol::fromQualString(op_name);
          auto operations = getAllOperatorsFor(symbol);
          for (const auto& op : operations) {
            if (op->schema().overload_name() == overload_name) {
              return op->schema();
            }
          }
          throw std::runtime_error("Found no matching schema");
        } catch (const c10::Error& e) {
          auto msg = torch::get_cpp_stacktraces_enabled()
              ? e.what()
              : e.what_without_backtrace();
          throw std::runtime_error(msg);
        }
      });

  m.def(
      "_get_operation_overload",
      [](const std::string& op_name,
         const std::string& overload_name) -> std::optional<py::tuple> {
        try {
          auto symbol = Symbol::fromQualString(op_name);
          auto operations = getAllOperatorsFor(symbol);
          bool allow_numbers_as_tensors = opAllowsNumbersAsTensors(symbol);
          for (const auto& op : operations) {
            if (op->schema().overload_name() == overload_name) {
              auto func = py::cpp_function(
                  [op, symbol, allow_numbers_as_tensors](
                      const py::args& args, const py::kwargs& kwargs) {
                    ToIValueAllowNumbersAsTensors g(allow_numbers_as_tensors);
                    return _get_operation_for_overload_or_packet(
                        op, symbol, args, kwargs, /*is_overload*/ true);
                  });
              auto func_dk =
                  py::cpp_function([op, symbol, allow_numbers_as_tensors](
                                       c10::DispatchKey dk_,
                                       const py::args& args,
                                       const py::kwargs& kwargs) {
                    ToIValueAllowNumbersAsTensors g(allow_numbers_as_tensors);
                    return _get_operation_for_overload_or_packet(
                        op, symbol, args, kwargs, /*is_overload*/ true, dk_);
                  });
              return py::make_tuple(
                  std::move(func),
                  std::move(func_dk),
                  py::cast(op->getTags().vec()));
            }
          }
          return std::nullopt;
        } catch (const c10::Error& e) {
          auto msg = torch::get_cpp_stacktraces_enabled()
              ? e.what()
              : e.what_without_backtrace();
          throw std::runtime_error(msg);
        }
      });

  m.def(
      "_check_schema_allow_fake_script_object",
      [](const FunctionSchema& schema,
         const py::args& args,
         const py::kwargs& kwargs) {
        // checkSchemaAllowFakeScriptObject will throw runtime error if there is
        // a schema mismatch. Otherwise, it returns true.
        return checkSchemaAllowFakeScriptObject(schema, args, kwargs);
      });

  m.def(
      "_jit_resolve_packet",
      [](const char* op_name, const py::args& args, const py::kwargs& kwargs) {
        try {
          auto symbol = Symbol::fromQualString(op_name);
          bool allow_numbers_as_tensors = opAllowsNumbersAsTensors(symbol);
          ToIValueAllowNumbersAsTensors g(allow_numbers_as_tensors);
          const auto overloads = getAllSortedOperatorsFor(symbol);
          auto opWithStack = getOpWithStack(overloads, args, kwargs);
          std::shared_ptr<Operator> overload =
              std::move(std::get<0>(opWithStack));
          auto result = overload->schema().overload_name();
          if (result.empty()) {
            result = "default";
          }
          return result;
        } catch (const c10::Error& e) {
          auto msg = torch::get_cpp_stacktraces_enabled()
              ? e.what()
              : e.what_without_backtrace();
          throw std::runtime_error(msg);
        }
      });

  m.def(
      "_jit_get_operation",
      [](const std::string& op_name) -> py::tuple {
        try {
          auto symbol = Symbol::fromQualString(op_name);
          const auto sortedOps = getAllSortedOperatorsFor(symbol);
          if (sortedOps.empty()) {
            // No such operator
            return py::make_tuple(py::none(), py::none());
          }

          std::ostringstream docstring;
          docstring << "Automatically bound operator '" << op_name
                    << "' with schema(s):\n";

          for (const auto& op : sortedOps) {
            docstring << "  " << op->schema() << '\n';
          }

          py::list overload_names;
          for (const auto& op : sortedOps) {
            overload_names.append(py::str(op->schema().overload_name()));
          }

          bool allow_numbers_as_tensors = opAllowsNumbersAsTensors(symbol);

          auto func = py::cpp_function(
              [sortedOps, symbol, allow_numbers_as_tensors](
                  const py::args& args, const py::kwargs& kwargs) {
                ToIValueAllowNumbersAsTensors g(allow_numbers_as_tensors);
                return _get_operation_for_overload_or_packet(
                    sortedOps, symbol, args, kwargs, false);
              },
              py::name(symbol.toUnqualString()),
              py::doc(std::move(docstring).str().c_str()));
          return py::make_tuple(func, overload_names);
        } catch (const c10::Error& e) {
          auto msg = torch::get_cpp_stacktraces_enabled()
              ? e.what()
              : e.what_without_backtrace();
          throw std::runtime_error(msg);
        }
      },
      py::arg("qualified_name"));

  m.def(
      "_maybe_call_torch_function_for_op_packet",
      [](py::handle op_overload_packet,
         const py::args& args,
         const py::kwargs& kwargs) -> py::tuple {
        py::list ns_method =
            op_overload_packet.attr("_qualified_op_name").attr("split")("::");
        auto res = _maybe_handle_torch_function(
            py::cast<std::string>(ns_method[0]),
            py::cast<std::string>(ns_method[1]),
            "",
            false,
            args,
            kwargs);
        if (res) {
          return py::make_tuple(true, *res);
        } else {
          return py::make_tuple(false, py::none());
        }
      });

  m.def(
      "parse_schema",
      &parseSchema,
      py::arg("schema"),
      py::arg("allow_typevars") = true);
  m.def(
      "_register_opaque_type",
      [](const std::string& type_name) {
        torch::jit::registerOpaqueType(type_name);
      },
      R"doc(Registers a type name to be treated as an opaque type (PyObject) in schema parsing.)doc");
  m.def(
      "_is_opaque_type_registered",
      [](const std::string& type_name) -> bool {
        return torch::jit::isRegisteredOpaqueType(type_name);
      },
      R"doc(Checks if a type name is registered as an opaque type.)doc");
  m.def(
      "_unregister_opaque_type",
      [](const std::string& type_name) {
        torch::jit::unregisterOpaqueType(type_name);
      },
      R"doc(Unregisters a type name from the opaque type registry.)doc");

  py::enum_<SchemaArgType>(m, "_SchemaArgType")
      .value("input", SchemaArgType::input)
      .value("output", SchemaArgType::output);
  py::class_<SchemaArgument>(m, "_SchemaArgument")
      .def(py::init<SchemaArgType, size_t>())
      .def_readwrite("type", &SchemaArgument::type)
      .def_readwrite("index", &SchemaArgument::index);
  py::class_<SchemaInfo>(m, "_SchemaInfo")
      .def(py::init<FunctionSchema>())
      .def("is_mutable", [](SchemaInfo& self) { return self.is_mutable(); })
      .def(
          "is_mutable",
          [](SchemaInfo& self, const SchemaArgument& argument) {
            return self.is_mutable(argument);
          })
      .def(
          "has_argument",
          [](SchemaInfo& self, const std::string& name) {
            return self.has_argument(name);
          })
      .def(
          "is_mutable",
          [](SchemaInfo& self, const std::string& name) {
            return self.is_mutable(name);
          })
      .def(
          "may_alias",
          [](SchemaInfo& self,
             const SchemaArgument& lhs,
             const SchemaArgument& rhs) { return self.may_alias(lhs, rhs); })
      .def(
          "may_contain_alias",
          [](SchemaInfo& self,
             const SchemaArgument& lhs,
             const SchemaArgument& rhs) {
            return self.may_contain_alias(lhs, rhs);
          })
      .def(
          "add_argument_value",
          [](SchemaInfo& self,
             const std::string& name,
             const py::object& value) {
            std::optional<IValue> i_value = toTypeInferredIValueOptional(value);
            if (i_value) {
              // For normalization purposes there is an inconsistency within
              // torch.fx that turns all arguments named "self" into "input".
              // Thus this check ensures that those arguments are checked
              // correctly.
              if (name == "input" && !self.hasInputArgumentNamed("input")) {
                self.addArgumentValue("self", *i_value);
              } else {
                self.addArgumentValue(name, *i_value);
              }
            }
          })
      .def("add_argument_values", [](SchemaInfo& self, const py::dict& values) {
        std::unordered_map<std::string, IValue> value_map;
        for (const auto& key_pair : values) {
          IValue key = toTypeInferredIValue(key_pair.first);
          TORCH_INTERNAL_ASSERT(
              key.isString(),
              "Add argument value keys types should be strings.");
          std::optional<IValue> value =
              toTypeInferredIValueOptional(key_pair.second);
          if (value) {
            // For normalization purposes there is an inconsistency within
            // torch.fx that
            // turns all arguments named "self" into "input". Thus this check
            // ensures that those arguments are checked correctly.
            if (key.toStringRef() == "input" &&
                !self.hasInputArgumentNamed("input")) {
              self.addArgumentValue("self", *value);
            } else {
              value_map[key.toStringRef()] = *value;
            }
          }
        }
        self.addArgumentValues(value_map);
      });

  py::class_<FunctionSchema>(m, "FunctionSchema")
      .def(py::init<
           std::string,
           std::string,
           std::vector<Argument>,
           std::vector<Argument>,
           bool,
           bool>())
      .def_property_readonly("name", &FunctionSchema::name)
      .def_property_readonly("overload_name", &FunctionSchema::overload_name)
      .def_property_readonly("arguments", &FunctionSchema::arguments)
      .def_property_readonly("returns", &FunctionSchema::returns)
      .def(
          "_is_view_op",
          [](const FunctionSchema& self) -> bool {
            for (const auto& arg : self.arguments()) {
              if (arg.alias_info() && !arg.alias_info()->isWrite()) {
                return true;
              }
            }
            return false;
          })
      .def(
          "is_backward_compatible_with",
          [](const FunctionSchema& self, const FunctionSchema& old_schema) {
            return self.isBackwardCompatibleWith(old_schema);
          })
      .def(
          "check_forward_compatible_with",
          [](const FunctionSchema& self, const FunctionSchema& old_schema) {
            std::ostringstream out;
            auto result = self.isForwardCompatibleWith(old_schema, out);
            return std::make_pair(result, std::move(out).str());
          })
      .def(
          "__eq__",
          [](const FunctionSchema& self, const FunctionSchema& other) {
            return self == other;
          })
      .def(
          "__hash__",
          [](const FunctionSchema& self) {
            return std::hash<FunctionSchema>{}(self);
          })
      .def(
          "__str__",
          [](const FunctionSchema& self) {
            std::stringstream ss;
            ss << self;
            return std::move(ss).str();
          })
      .def(
          "__repr__",
          [](const FunctionSchema& self) {
            std::stringstream ss;
            ss << self;
            return std::move(ss).str();
          })
      .def(py::pickle(
          [](const FunctionSchema& self) { // __getstate__
            std::stringstream ss;
            ss << self;
            return py::str(std::move(ss).str());
          },
          [](const py::str& schema) { // __setstate__, note: no `self` argument
            return parseSchema(schema);
          }))
      .def_property_readonly("is_mutable", [](const FunctionSchema& self) {
        return self.is_mutable();
      });
  py::class_<Argument>(m, "Argument")
      .def(py::init<
           std::string,
           const TypePtr&,
           std::optional<int32_t>,
           std::optional<IValue>,
           bool,
           std::optional<AliasInfo>>())
      .def_property_readonly("name", &Argument::name)
      .def_property_readonly("type", &Argument::type)
      .def_property_readonly("real_type", &Argument::real_type)
      .def_property_readonly(
          "N",
          [](const Argument& self) -> py::object {
            return (self.N()) ? py::cast(*self.N()) : py::none();
          })
      .def_property_readonly(
          "default_value",
          [](const Argument& self) -> py::object {
            if (!self.default_value()) {
              return py::none();
            }
            IValue v = *self.default_value();
            return toPyObject(std::move(v));
          })
      .def(
          "has_default_value",
          [](const Argument& self) -> py::bool_ {
            return self.default_value().has_value();
          })
      .def_property_readonly(
          "alias_info", [](const Argument& self) { return self.alias_info(); })
      .def_property_readonly(
          "is_write",
          [](const Argument& self) {
            if (self.alias_info() == nullptr) {
              return false;
            }
            return self.alias_info()->isWrite();
          })
      .def_property_readonly(
          "is_out", [](const Argument& self) { return self.is_out(); })
      .def_property_readonly("kwarg_only", [](const Argument& self) -> bool {
        return self.kwarg_only();
      });
  py::class_<AliasInfo>(m, "_AliasInfo")
      .def(py::init<bool, std::set<std::string>, std::set<std::string>>())
      .def_property_readonly(
          "is_write", [](const AliasInfo& self) { return self.isWrite(); })
      .def_property_readonly(
          "before_set",
          [](const AliasInfo& self) {
            std::set<py::str> before_set_python;
            for (const auto& set : self.beforeSets()) {
              before_set_python.insert(py::str(set.toUnqualString()));
            }
            return before_set_python;
          })
      .def_property_readonly("after_set", [](const AliasInfo& self) {
        std::set<py::str> after_set_python;
        for (const auto& set : self.afterSets()) {
          after_set_python.insert(py::str(set.toUnqualString()));
        }
        return after_set_python;
      });
  m.def("_jit_get_all_schemas", []() {
    const std::vector<std::shared_ptr<Operator>>& operations =
        getAllOperators();
    return fmap(operations, [](const std::shared_ptr<Operator>& op) {
      return op->schema();
    });
  });
  m.def("_jit_get_custom_class_schemas", customClassSchemasForBCCheck);
  m.def("_jit_get_schemas_for_operator", [](const std::string& qualified_name) {
    auto symbol = Symbol::fromQualString(qualified_name);
    const auto& operations = getAllOperatorsFor(symbol);
    return fmap(operations, [](const std::shared_ptr<Operator>& op) {
      return op->schema();
    });
  });

  py::class_<PythonFutureWrapper, std::shared_ptr<PythonFutureWrapper>>(
      m, "Future")
      .def(py::init([](std::vector<c10::Device> devices = {}) {
        return std::make_shared<PythonFutureWrapper>(
            c10::make_intrusive<c10::ivalue::Future>(
                PyObjectType::get(), std::move(devices)));
      }))
      .def(
          "done",
          // Intentionally not releasing GIL
          &PythonFutureWrapper::done)
      .def(
          "value",
          &PythonFutureWrapper::value,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "wait",
          &PythonFutureWrapper::wait,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "then",
          &PythonFutureWrapper::then,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "add_done_callback",
          &PythonFutureWrapper::add_done_callback,
          py::call_guard<py::gil_scoped_release>())
      .def(
          "set_result",
          // Intentionally not releasing GIL
          &PythonFutureWrapper::markCompleted)
      .def(
          "_set_unwrap_func",
          // Intentionally not releasing GIL as this just does an assign
          [](PythonFutureWrapper& self, py::function unwrapFunc) {
            auto functionGuard =
                std::make_shared<torch::jit::PythonFunctionGuard>(
                    std::move(unwrapFunc));

            std::function<void(py::object)> pf =
                [functionGuard(std::move(functionGuard))](
                    const py::object& inp) {
                  return functionGuard->func_(inp);
                };
            self.unwrap_func = std::move(pf);
          })
      .def(
          py::pickle(
              /* __getstate__ */
              [](const PythonFutureWrapper& /* unused */) -> py::tuple {
                TORCH_CHECK(false, "Can not pickle torch.futures.Future");
              },
              /* __setstate__ */
              [](const py::tuple& /* unused */) -> std::nullptr_t {
                TORCH_CHECK(false, "Can not unpickle torch.futures.Future");
              }),
          py::call_guard<py::gil_scoped_release>());

  py::class_<PythonAwaitWrapper, std::shared_ptr<PythonAwaitWrapper>>(
      m, "_Await")
      .def(
          "wait",
          &PythonAwaitWrapper::wait,
          py::call_guard<py::gil_scoped_release>())
      .def("fn", &PythonAwaitWrapper::fn)
      .def("args", &PythonAwaitWrapper::args)
      .def("type", &PythonAwaitWrapper::type)
      .def("is_nowait", &PythonAwaitWrapper::is_nowait)
      .def(
          "__getattr__",
          [](PythonAwaitWrapper& self, const std::string& name) -> py::object {
            // In eager mode allow Await[W] to be used as W, redirecting getattr
            // to the result of delayed function.
            return py::getattr(self.wait(), name.c_str(), py::none());
          })
      .def(
          py::pickle(
              /* __getstate__ */
              [](const PythonAwaitWrapper& /* unused */) -> py::tuple {
                TORCH_CHECK(false, "Can not pickle torch.jit._Await");
              },
              /* __setstate__ */
              [](const py::tuple& /* unused */) -> std::nullptr_t {
                TORCH_CHECK(false, "Can not unpickle torch.jit._Await");
              }),
          py::call_guard<py::gil_scoped_release>());

  m.def("_is_alias_of", [](const py::object& self, const py::object& other) {
    std::optional<IValue> self_value = toTypeInferredIValueOptional(self);
    std::optional<IValue> other_value = toTypeInferredIValueOptional(other);

    // Only return true if we are certain that self and other are aliasing.
    if (!self_value || !other_value) {
      return false;
    }
    return self_value->isAliasOf(*other_value);
  });
  m.def("_overlaps", [](const py::object& self, const py::object& other) {
    std::optional<IValue> self_value = toTypeInferredIValueOptional(self);
    std::optional<IValue> other_value = toTypeInferredIValueOptional(other);

    // Only return true if we are certain that self and other are overlapping.
    if (!self_value || !other_value) {
      return false;
    }
    return self_value->overlaps(*other_value);
  });
  m.def("_awaitable", [](const py::args& args, const py::kwargs& kwargs) {
    AT_ASSERT(!args.empty());
    py::tuple args_tup(args.size() - 1);
    for (const auto i : c10::irange(1, args.size())) {
      args_tup[i - 1] = args[i];
    }
    return std::make_shared<PythonAwaitWrapper>(
        py::cast<py::function>(args[0]), std::move(args_tup));
  });
  m.def("_awaitable_nowait", [](py::handle input) {
    return std::make_shared<PythonAwaitWrapper>(input);
  });
  m.def(
      "_awaitable_wait", [](const std::shared_ptr<PythonAwaitWrapper>& py_aw) {
        TORCH_CHECK(py_aw, "Await can't be None");
        return py_aw->wait();
      });
  m.def(
      "_collect_all",
      [](const std::vector<std::shared_ptr<jit::PythonFutureWrapper>>& futures)
          -> std::shared_ptr<jit::PythonFutureWrapper> {
        auto typePtr = futures.empty() || futures[0] == nullptr
            ? AnyType::get()
            : futures[0]->fut->elementType();
        c10::List<c10::intrusive_ptr<c10::ivalue::Future>> asList(
            c10::FutureType::create(typePtr));
        asList.reserve(futures.size());
        for (const auto& f : futures) {
          TORCH_CHECK(f, "Future can't be None");
          asList.push_back(f->fut);
        }
        return std::make_shared<jit::PythonFutureWrapper>(
            c10::collectAll(asList),
            /* unwrap_func */ [futures](const py::object& /*unused*/) {
              // Throw errors when calling wait() on the returned Future if
              // any of the original futures would throw.
              for (auto& fut : futures) {
                fut->wait();
              }
            });
      },
      py::call_guard<py::gil_scoped_release>());

#if defined(C10_SUPPORTS_FATAL_SIGNAL_HANDLERS)
  m.def("_set_print_stack_traces_on_fatal_signal", [](bool print) {
    c10::FatalSignalHandler::getInstance().setPrintStackTracesOnFatalSignal(
        print);
  });
#endif // defined(C10_SUPPORTS_SIGNAL_HANDLER)

  initPythonCustomClassBindings(module);
}

} // namespace torch::jit
