#pragma once

#include <ATen/core/alias_info.h>
#include <ATen/core/enum_type.h>
#include <ATen/core/class_type.h>
#include <ATen/core/function_schema.h>
#include <ATen/core/jit_type.h>
#include <c10/macros/Macros.h>
#include <torch/csrc/jit/frontend/lexer.h>

#include <functional>

namespace torch::jit {

using TypePtr = c10::TypePtr;

// Re-export the c10 type vocabulary into torch::jit. Upstream these names
// reach this header transitively; the trims cut that path, so the parsers
// declare them here.
using ::c10::Argument;
using ::c10::FunctionSchema;
using ::c10::Symbol;
using ::c10::IValue;
#define C10_USING(T) using ::c10::T;
C10_FORALL_TYPES(C10_USING)
#undef C10_USING
#define C10_USING(T) using ::c10::T##Ptr;
C10_FORALL_TYPES(C10_USING)
#undef C10_USING
using ::c10::Type;
using ::c10::TypePtr;
using ::c10::getTypePtr;
using ::c10::MatchTypeReturn;
using ::c10::TypeKind;
using ::c10::fmap;

TORCH_API void registerOpaqueType(const std::string& type_name);
TORCH_API void unregisterOpaqueType(const std::string& type_name);
TORCH_API bool isRegisteredOpaqueType(const std::string& type_name);

struct TORCH_API SchemaTypeParser {
  TypePtr parseBaseType();
  std::optional<c10::AliasInfo> parseAliasAnnotation();
  std::pair<TypePtr, std::optional<c10::AliasInfo>> parseType();
  std::tuple</*fake*/ TypePtr, /*real*/ TypePtr, std::optional<c10::AliasInfo>>
  parseFakeAndRealType();
  std::optional<at::ScalarType> parseTensorDType(const std::string& dtype);
  TypePtr parseRefinedTensor();

  SchemaTypeParser(
      Lexer& L,
      bool parse_complete_tensor_types,
      bool allow_typevars)
      : complete_tensor_types(parse_complete_tensor_types),
        L(L),
        allow_typevars_(allow_typevars) {}

 private:
  std::optional<bool> tryToParseRequiresGrad();
  std::optional<c10::Device> tryToParseDeviceType();
  void parseList(
      int begin,
      int sep,
      int end,
      std::function_ref<void()> callback);

  bool complete_tensor_types;
  Lexer& L;
  size_t next_id = 0;
  bool allow_typevars_;
};
} // namespace torch::jit
