#include <torch/csrc/jit/python/python_types.h>

#include <ATen/core/jit_type.h>
#include <torch/csrc/jit/api/method.h>
#include <torch/csrc/jit/api/object.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <pybind11/pybind11.h>
#include <torch/csrc/jit/python/pybind_utils.h>
#include <torch/csrc/utils/pybind.h>
#include <torch/python.h>

#include <sstream>
#include <string>
#include <vector>

namespace py = pybind11;

namespace torch::jit {

// The c10::Type hierarchy is still what FunctionSchema arguments are made of,
// so these bindings must outlive TorchScript: torch.ops, torch.library and
// dynamo all inspect Argument.type from Python.
void initPythonTypeBindings(PyObject* module_) {
  auto m = py::handle(module_).cast<py::module>();

  // FileCheck is a test-only string matcher, independent of TorchScript apart
  // from the graph overloads, which are dropped here along with the IR.
  py::class_<testing::FileCheck>(m, "FileCheck")
      .def(py::init<>())
      .def("check", &testing::FileCheck::check)
      .def("check_not", &testing::FileCheck::check_not)
      .def("check_same", &testing::FileCheck::check_same)
      .def("check_next", &testing::FileCheck::check_next)
      .def("check_dag", &testing::FileCheck::check_dag)
      .def(
          "check_source_highlighted",
          &testing::FileCheck::check_source_highlighted)
      .def("check_regex", &testing::FileCheck::check_regex)
      .def(
          "check_count",
          [](testing::FileCheck& f,
             const std::string& str,
             size_t count,
             bool exactly) { return f.check_count(str, count, exactly); },
          "Check Count",
          py::arg("str"),
          py::arg("count"),
          py::arg("exactly") = false)
      .def(
          "run",
          [](testing::FileCheck& f, const std::string& str) {
            return f.run(str);
          })
      .def(
          "run",
          [](testing::FileCheck& f,
             const std::string& input,
             const std::string& output) { return f.run(input, output); },
          "Run",
          py::arg("checks_file"),
          py::arg("test_file"));

  // torchbind custom class instances come back from C++ as
  // torch::jit::Object (toPyObject does py::cast(Object(...))), so the type
  // must be registered or any op returning one fails to convert. This is how
  // profiler::_record_function_enter_new returns its handle, which is what
  // every optimizer step and record_function block goes through.
  // tryToInferType's result type; _jit_try_infer_type below returns it and the
  // HOP schema generator reads .type() off it.
  py::class_<InferredType, std::shared_ptr<InferredType>>(m, "InferredType")
      .def(py::init([](std::shared_ptr<Type> type) {
        return std::make_shared<InferredType>(std::move(type));
      }))
      .def(py::init([](std::string reason) {
        return std::make_shared<InferredType>(std::move(reason));
      }))
      .def(
          "type",
          [](const std::shared_ptr<InferredType>& self) {
            return self->type();
          })
      .def(
          "success",
          [](const std::shared_ptr<InferredType>& self) {
            return self->success();
          })
      .def("reason", [](const std::shared_ptr<InferredType>& self) {
        return self->reason();
      });

  py::class_<Object>(m, "ScriptObject")
      .def("_type", [](Object& o) { return o.type(); })
      .def(
          "_method_names",
          [](Object& self) {
            std::vector<std::string> names;
            for (const auto* method : self.type()->methods()) {
              names.push_back(method->name());
            }
            return names;
          })
      .def(
          "_method_schema",
          [](Object& self, const std::string& name) -> py::object {
            if (const auto* fn = self.type()->findMethod(name)) {
              return py::cast(fn->getSchema());
            }
            return py::none();
          })
      .def(
          "getattr",
          [](Object& self, const std::string& name) {
            return toPyObject(self.attr(name));
          })
      .def(
          "hasattr",
          [](Object& self, const std::string& name) {
            return self.hasattr(name);
          })
      .def(
          "setattr",
          [](Object& self, const std::string& name, py::object value) {
            self.setattr(name, toIValue(std::move(value), self.attr(name).type()));
          })
      .def(
          "__getattr__",
          [](Object& self, const std::string& name) -> py::object {
            if (name == "__qualname__") {
              return py::cast(self.type()->name()->name());
            }
            // Bind the method into a callable rather than returning a Method:
            // Method has no pybind type here and does not need one.
            if (auto* fn = self.type()->findMethod(name)) {
              auto ivalue = self._ivalue();
              return py::cpp_function(
                  [ivalue, fn](const py::args& args, const py::kwargs& kwargs) {
                    Method method(ivalue, fn);
                    return invokeScriptMethodFromPython(
                        method, tuple_slice(args), kwargs);
                  });
            }
            try {
              return toPyObject(self.attr(name));
            } catch (const ObjectAttributeError& err) {
              PyErr_SetString(PyExc_AttributeError, err.what());
              throw py::error_already_set();
            }
          });

  using ::c10::Type;
  py::class_<Type, TypePtr>(m, "Type")
      .def("__repr__", [](Type& t) { return t.annotation_str(); })
      .def(
          "str",
          [](Type& t) {
            std::ostringstream s;
            s << t;
            return std::move(s).str();
          })
      .def(
          "containedTypes",
          [](Type& self) { return self.containedTypes().vec(); })
      .def("kind", [](const Type& t) { return typeKindToString(t.kind()); })
      .def(
          "__eq__",
          [](const TypePtr& self, const TypePtr& other) {
            if (!other) {
              return false;
            }
            return *self == *other;
          })
      .def(
          "isSubtypeOf",
          [](const TypePtr& self, const TypePtr& other) {
            if (!other) {
              return false;
            }
            return self->isSubtypeOf(other);
          })
      .def_property_readonly(
          "annotation_str", [](const std::shared_ptr<Type>& self) {
            return self->annotation_str();
          });

  py::class_<AnyType, Type, AnyTypePtr>(m, "AnyType")
      .def_static("get", &AnyType::get);
  py::class_<NumberType, Type, NumberTypePtr>(m, "NumberType")
      .def_static("get", &NumberType::get);
  py::class_<IntType, Type, IntTypePtr>(m, "IntType")
      .def_static("get", &IntType::get);
  py::class_<SymIntType, Type, SymIntTypePtr>(m, "SymIntType")
      .def_static("get", &SymIntType::get);
  py::class_<SymBoolType, Type, SymBoolTypePtr>(m, "SymBoolType")
      .def_static("get", &SymBoolType::get);
  py::class_<FloatType, Type, FloatTypePtr>(m, "FloatType")
      .def_static("get", &FloatType::get);
  py::class_<ComplexType, Type, ComplexTypePtr>(m, "ComplexType")
      .def_static("get", &ComplexType::get);
  py::class_<TensorType, Type, TensorTypePtr>(m, "TensorType")
      .def_static("get", &TensorType::get)
      .def_static("getInferred", &TensorType::getInferred)
      .def_static("create_from_tensor", [](const at::Tensor& t) {
        return TensorType::create(t);
      });
  py::class_<BoolType, Type, BoolTypePtr>(m, "BoolType")
      .def_static("get", &BoolType::get);
  py::class_<StringType, Type, StringTypePtr>(m, "StringType")
      .def_static("get", &StringType::get);
  py::class_<DeviceObjType, Type, DeviceObjTypePtr>(m, "DeviceObjType")
      .def_static("get", &DeviceObjType::get);
  // TODO(antoniojkim): Add GeneratorType to the public API once its been added
  //                    to the public documentation
  py::class_<GeneratorType, Type, GeneratorTypePtr>(m, "_GeneratorType")
      .def_static("get", &GeneratorType::get);
  py::class_<StreamObjType, Type, StreamObjTypePtr>(m, "StreamObjType")
      .def_static("get", &StreamObjType::get);
  py::class_<PyObjectType, Type, PyObjectTypePtr>(m, "PyObjectType")
      .def_static("get", &PyObjectType::get);
  py::class_<NoneType, Type, NoneTypePtr>(m, "NoneType")
      .def_static("get", &NoneType::get);

  py::class_<TupleType, Type, TupleTypePtr>(m, "TupleType")
      .def(py::init([](std::vector<TypePtr> a) {
        return TupleType::create(std::move(a));
      }))
      .def("elements", [](TupleType& self) {
        std::vector<TypePtr> types;
        for (const auto& type : self.elements()) {
          types.push_back(type);
        }
        return types;
      });
  py::class_<UnionType, Type, UnionTypePtr>(m, "UnionType")
      .def(py::init(
          [](const std::vector<TypePtr>& a) { return UnionType::create(a); }));
  py::class_<ListType, Type, ListTypePtr>(m, "ListType")
      .def(py::init([](const TypePtr& a) { return ListType::create(a); }))
      .def_static("ofInts", &ListType::ofInts)
      .def_static("ofTensors", &ListType::ofTensors)
      .def_static("ofFloats", &ListType::ofFloats)
      .def_static("ofComplexDoubles", &ListType::ofComplexDoubles)
      .def_static("ofBools", &ListType::ofBools)
      .def_static("ofStrings", &ListType::ofStrings)
      .def("getElementType", &ListType::getElementType);
  py::class_<DictType, Type, DictTypePtr>(m, "DictType")
      .def(py::init([](TypePtr key, TypePtr value) {
        return DictType::create(std::move(key), std::move(value));
      }))
      .def("getKeyType", &DictType::getKeyType)
      .def("getValueType", &DictType::getValueType);
  py::class_<OptionalType, Type, OptionalTypePtr>(m, "OptionalType")
      .def(py::init([](const TypePtr& a) { return OptionalType::create(a); }))
      .def_static("ofTensor", &OptionalType::ofTensor)
      .def("getElementType", &OptionalType::getElementType);
  py::class_<RRefType, Type, RRefTypePtr>(m, "RRefType")
      .def(py::init([](TypePtr a) { return RRefType::create(std::move(a)); }))
      .def("getElementType", &RRefType::getElementType);

  py::class_<FutureType, Type, FutureTypePtr>(m, "FutureType")
      .def(py::init([](TypePtr a) { return FutureType::create(std::move(a)); }))
      .def("getElementType", &FutureType::getElementType);

  py::class_<AwaitType, Type, AwaitTypePtr>(m, "AwaitType")
      .def(py::init([](TypePtr a) { return AwaitType::create(std::move(a)); }))
      .def("getElementType", &AwaitType::getElementType);

  py::class_<ClassType, Type, ClassTypePtr>(m, "ClassType")
      .def("name", [](ClassType& self) { return self.name()->name(); })
      .def(
          "qualified_name",
          [](ClassType& self) { return self.name()->qualifiedName(); })
      .def("method_names", [](ClassType& self) {
        std::vector<std::string> method_names;
        for (const auto* method : self.methods()) {
          method_names.push_back(method->name());
        }
        return method_names;
      });

}

} // namespace torch::jit
