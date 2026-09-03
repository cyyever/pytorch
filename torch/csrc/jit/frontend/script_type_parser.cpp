#include <torch/csrc/jit/frontend/script_type_parser.h>

#include <ATen/core/type_factory.h>
#include <torch/csrc/jit/frontend/parser.h>
#include <torch/custom_class.h>

namespace torch::jit {

// These used to arrive through jit/ir/ir.h, which this file no longer needs.
using ::c10::Argument;
using ::c10::AwaitType;
using ::c10::DictType;
using ::c10::FutureType;
using ::c10::IntType;
using ::c10::ListType;
using ::c10::OptionalType;
using ::c10::RRefType;
using ::c10::FunctionSchema;
using ::c10::IValue;
using ::c10::TensorType;
using ::c10::TupleType;
using ::c10::TypePtr;
using ::c10::UnionType;

namespace {

bool isTorch(const Expr& expr) {
  return expr.kind() == TK_VAR && Var(expr).name().name() == "torch";
}

std::string collectQualname(const Select& select) {
  Expr base = select.value();
  if (base.kind() == TK_VAR) {
    return Var(base).name().name() + "." + select.selector().name();
  }
  std::string basename = collectQualname(Select(base));
  return basename + "." + select.selector().name();
}

const std::unordered_map<std::string, c10::TypePtr>& string_to_type_lut() {
  return c10::DefaultTypeFactory::basePythonTypes();
}

} // namespace

TypePtr ScriptTypeParser::subscriptToType(
    const std::string& typeName,
    const Subscript& subscript) const {
  if (typeName == "Tuple" || typeName == "tuple") {
    if (subscript.subscript_exprs().size() == 1 &&
        subscript.subscript_exprs()[0].kind() == TK_TUPLE_LITERAL) {
      // `typing.Tuple` special cases syntax for empty tuple annotations,
      // i.e. `typing.Tuple[()]`. Allow for parsing an empty tuple literal
      // here. See https://docs.python.org/3/library/typing.html#typing.Tuple
      auto tup_literal = TupleLiteral(subscript.subscript_exprs()[0]);
      if (!tup_literal.inputs().empty()) {
        throw(
            ErrorReport(tup_literal.range())
            << "Tuple literal in Tuple type annotation must not "
            << "have any elements!");
      }
      return TupleType::create({});
    }
    std::vector<TypePtr> subscript_expr_types;
    subscript_expr_types.reserve(subscript.subscript_exprs().size());
    for (auto expr : subscript.subscript_exprs()) {
      subscript_expr_types.emplace_back(parseTypeFromExprImpl(expr));
    }
    return TupleType::create(subscript_expr_types);
  } else if (typeName == "List" || typeName == "list") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type =
        parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
    return ListType::create(elem_type);

  } else if (typeName == "Optional") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type =
        parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
    return OptionalType::create(elem_type);

  } else if (typeName == "Union") {
    std::vector<TypePtr> subscript_expr_types;
    subscript_expr_types.reserve(subscript.subscript_exprs().size());
    for (auto expr : subscript.subscript_exprs()) {
      subscript_expr_types.emplace_back(parseTypeFromExprImpl(expr));
    }
    return UnionType::create(subscript_expr_types);
  } else if (typeName == "Future" || typeName == "torch.jit.Future") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type =
        parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
    return FutureType::create(elem_type);
  } else if (typeName == "Await" || typeName == "torch.jit._Await") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type =
        parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
    return AwaitType::create(elem_type);
  } else if (typeName == "RRef") {
    if (subscript.subscript_exprs().size() != 1) {
      throw ErrorReport(subscript)
          << " expected exactly one element type but found "
          << subscript.subscript_exprs().size();
    }
    auto elem_type =
        parseTypeFromExprImpl(*subscript.subscript_exprs().begin());
    return RRefType::create(elem_type);
  } else if (typeName == "Dict" || typeName == "dict") {
    if (subscript.subscript_exprs().size() != 2) {
      throw ErrorReport(subscript)
          << " expected exactly 2 element types but found "
          << subscript.subscript_exprs().size();
    }
    auto key_type = parseTypeFromExprImpl(subscript.subscript_exprs()[0]);
    auto value_type = parseTypeFromExprImpl(subscript.subscript_exprs()[1]);
    return DictType::create(key_type, value_type);
  } else {
    throw ErrorReport(subscript.range())
        << "Unknown type constructor " << typeName;
  }
}

std::optional<std::pair<TypePtr, int32_t>> ScriptTypeParser::parseBroadcastList(
    const Expr& expr) const {
  // Alias torch.nn._common_types._size_?_t to BroadcastingList?[int]
  if (expr.kind() == TK_VAR) {
    auto var = Var(expr);
    auto& name = var.name().name();
    constexpr auto _size_prefix = "_size_";
    constexpr auto _size_suffix = "_t";
    constexpr auto _size_n_len = 9; // strlen("_size_X_t")
    constexpr auto _size_prefix_len = 6; // strlen("_size_");
    if (name.starts_with(_size_prefix) && name.length() == _size_n_len &&
        name.find(_size_suffix) == _size_prefix_len + 1 &&
        ::isdigit(name[_size_prefix_len])) {
      int n = name[_size_prefix_len] - '0';
      return std::pair<TypePtr, int32_t>(ListType::create(IntType::get()), n);
    }
  }

  if (expr.kind() != TK_SUBSCRIPT)
    return std::nullopt;
  auto subscript = Subscript(expr);
  if (subscript.value().kind() != TK_VAR)
    return std::nullopt;
  auto var = Var(subscript.value());
  auto subscript_exprs = subscript.subscript_exprs();

  // handle the case where the BroadcastingList is wrapped in an Optional type
  if (var.name().name() == "Optional") {
    auto broadcast_list = parseBroadcastList(subscript_exprs[0]);
    if (broadcast_list) {
      TypePtr opt_type = OptionalType::create(broadcast_list->first);
      return std::pair<TypePtr, int32_t>(opt_type, broadcast_list->second);
    } else {
      return std::nullopt;
    }
  } else if (!var.name().name().starts_with("BroadcastingList")) {
    return std::nullopt;
  }

  if (subscript_exprs.size() != 1)
    throw ErrorReport(subscript.subscript_exprs().range())
        << "BroadcastingList/Optional[BroadcastingList] "
           "must be subscripted with a type";

  auto typ = subscript_exprs[0];
  auto len = var.name().name().substr(strlen("BroadcastingList"));

  if (typ.kind() != TK_VAR)
    throw ErrorReport(subscript.value().range())
        << "Subscripted type must be a type identifier";

  auto value_name = Var(typ).name().name();
  if (value_name != "float" && value_name != "int")
    throw ErrorReport(subscript.value().range())
        << "Broadcastable lists only supported for int or float";

  auto elem_ptr = string_to_type_lut().find(value_name);
  AT_ASSERT(elem_ptr != string_to_type_lut().end());
  TypePtr list_ptr = ListType::create(elem_ptr->second);

  const char* len_c = len.c_str();
  char* end = nullptr;
  size_t len_v = strtoull(len_c, &end, 10);
  if (end != len_c + len.size()) {
    throw(
        ErrorReport(subscript.subscript_exprs().range())
        << "subscript of Broadcastable list must be a positive integer");
  }
  return std::pair<TypePtr, int32_t>(list_ptr, len_v);
}

// gets the base type name given namespaces where the types live
// turns torch.Tensor -> Tensor, X -> X
std::optional<std::string> ScriptTypeParser::parseBaseTypeName(
    const Expr& expr) const {
  switch (expr.kind()) {
    case TK_VAR: {
      return Var(expr).name().name();
    }
    case TK_NONE: {
      return "None";
    }
    case TK_NONE_TYPE: {
      return "NoneType";
    }
    case '.': {
      auto select = Select(expr);
      const std::string& name = select.selector().name();
      // Special case for torch.Tensor and its' subclasses
      const std::unordered_set<std::string> tensor_subtypes = {
          "Tensor",
          "LongTensor",
          "FloatTensor",
          "DoubleTensor",
          "IntTensor",
          "ShortTensor",
          "HalfTensor",
          "CharTensor",
          "ByteTensor",
          "BoolTensor"};
      if (isTorch(select.value()) && tensor_subtypes.count(name) == 1) {
        return name;
      } else {
        // Otherwise, it's a fully qualified class name
        return collectQualname(select);
      }
    } break;
  }
  return std::nullopt;
}

TypePtr ScriptTypeParser::parseTypeFromExpr(const Expr& expr) const {
  if (expr.kind() == '|') {
    auto converted = pep604union_to_union(expr);
    return parseTypeFromExpr(converted);
  }
  return parseTypeFromExprImpl(expr);
}

TypePtr ScriptTypeParser::parseTypeFromExprImpl(const Expr& expr) const {
  if (expr.kind() == '|') {
    auto converted = pep604union_to_union(expr);
    return parseTypeFromExprImpl(converted);
  }
  if (expr.kind() == TK_SUBSCRIPT) {
    auto subscript = Subscript(expr);
    auto value_name = parseBaseTypeName(subscript.value());
    if (!value_name) {
      throw ErrorReport(subscript.value().range())
          << "Subscripted type must be a type identifier";
    }
    return subscriptToType(*value_name, subscript);

  } else if (expr.kind() == TK_STRINGLITERAL) {
    const auto& type_name = StringLiteral(expr).text();

    // Check if the type is a custom class. This is done by checking
    // if type_name starts with "torch.classes."
    if (type_name.starts_with("torch.classes.")) {
      auto custom_class_type = getCustomClass("__torch__." + type_name);
      return custom_class_type;
    }

    // `torch.cuda.Stream` and `torch.cuda.Event` are aliased as
    // custom classes of type torch.classes.cuda.Stream and
    // torch.classes.cuda.Event respectively. Return the respective
    // custom class types for these two cases.
    if (type_name.starts_with("torch.cuda.Stream")) {
      auto custom_class_type =
          getCustomClass("__torch__.torch.classes.cuda.Stream");
      return custom_class_type;
    }

    if (type_name.starts_with("torch.cuda.Event")) {
      auto custom_class_type =
          getCustomClass("__torch__.torch.classes.cuda.Event");
      return custom_class_type;
    }

    throw ErrorReport(expr) << "Unknown type name '" << type_name << '\'';
  } else if (auto name = parseBaseTypeName(expr)) {
    auto itr = string_to_type_lut().find(*name);
    if (itr != string_to_type_lut().end()) {
      return itr->second;
    }
    if (auto custom_class_type = getCustomClass(*name)) {
      return custom_class_type;
    }

    throw ErrorReport(expr) << "Unknown type name '" << *name << '\'';
  }
  throw ErrorReport(expr.range())
      << "Expression of type " << kindToString(expr.kind())
      << " cannot be used in a type expression";
}

TypePtr ScriptTypeParser::parseType(const std::string& str) {
  Parser p(std::make_shared<Source>(str));
  return parseTypeFromExpr(p.parseExp());
}

} // namespace torch::jit
