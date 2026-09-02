#pragma once
#include <torch/csrc/jit/frontend/error_report.h>
#include <torch/csrc/jit/frontend/strtod.h>
#include <torch/csrc/jit/frontend/tree.h>

#include <c10/util/complex.h>
#include <functional>
#include <iostream>
#include <string>
#include <utility>

namespace torch::jit {

// clang-format off
// TreeView provides a statically-typed way to traverse the tree, which should
// be formed according to the grammar below.
//
// A few notes on types and their aliases:
// - List<T> is really a Tree with kind TK_LIST and elements as subtrees
// - Maybe<T> is really a Tree with kind TK_OPTION that has 0 or 1 subtree of type T
// - Builtin types are: Ident (TK_IDENT), String (TK_STRING)
//
// Expr  = TernaryIf(Expr cond, Expr true_expr, Expr false_expr)        TK_IF_EXPR
//       | BinOp(Expr lhs, Expr rhs)
//       |     And                                                      TK_AND
//       |     Or                                                       TK_OR
//       |     Lt                                                       '<'
//       |     Gt                                                       '>'
//       |     Eq                                                       TK_EQ
//       |     Le                                                       TK_LE
//       |     Ge                                                       TK_GE
//       |     Ne                                                       TK_NE
//       |     Is                                                       TK_IS
//       |     IsNot                                                    TK_ISNOT
//       |     Add                                                      '+'
//       |     Sub                                                      '-'
//       |     Mul                                                      '*'
//       |     Div                                                      '/'
//       |     Mod                                                      '%'
//       |     MatMult                                                  '@'
//       |     Pow                                                      TK_POW
//       | UnaryOp(Expr expr)
//       |     Not                                                      TK_NOT
//       |     USub                                                     '-'
//       | Const(String value)                                          TK_CONST
//       -- NB: x.name(y) is desugared into name(x, y)
//       | Apply(Ident name, List<Expr> args, List<Attribute> kwargs)   TK_APPLY
//       | Select(Expr value, Ident selector)                           '.'
//       | Subscript(Expr value, List<Expr> subscript_exprs)            TK_SUBSCRIPT
//       | SliceExpr(Maybe<Expr> start, Maybe<Expr> end)                TK_SLICE_EXPR
//       | Var(Ident name)                                              TK_VAR
//       | ListLiteral(List<Expr> inputs)                               TK_LIST_LITERAL
//       | TupleLiteral(List<Expr> inputs)                              TK_TUPLE_LITERAL
// -- NB: only allowed expressions are Const or List(Const)
//        (List as a value, not type constructor)
// Attribute = Attribute(Ident name, Expr value)                        TK_ATTRIBUTE
//

// Each subclass of TreeView should provide:
// 1. Constructor that takes a TreeRef, and checks that it's of the right type.
// 2. Accessors that get underlying information out of the object. If they
//    return subtrees, they should wrap them in appropriate views too.
// 3. Static method 'create' that creates the underlying TreeRef object
//    for every TreeRef kind that has a TreeView, the parser always uses
//    (e.g.) Ident::create rather than Compound::Create, this means that
//    changes to the structure of Ident are always made right here rather
//    than both in the parser and in this code.
// XXX: these structs should have no fields to prevent slicing when passing by value
// clang-format on
struct TreeView {
  explicit TreeView(TreeRef tree) : tree_(std::move(tree)) {}
  TreeRef tree() const {
    return tree_;
  }
  const SourceRange& range() const {
    return tree_->range();
  }
  operator TreeRef() const {
    return tree_;
  }
  const TreeRef& get() const {
    return tree_;
  }
  int kind() const {
    return tree_->kind();
  }
  void dump() const {
    std::cout << tree_;
  }

 protected:
  const TreeRef& subtree(size_t i) const {
    return tree_->trees().at(i);
  }
  // NOLINTNEXTLINE(cppcoreguidelines-non-private-member-variables-in-classes)
  TreeRef tree_;
};

template <typename T>
struct ListIterator {
  ListIterator(TreeList::const_iterator it) : it(it) {}
  bool operator!=(const ListIterator& rhs) const {
    return it != rhs.it;
  }
  bool operator==(const ListIterator& rhs) const {
    return it == rhs.it;
  }
  T operator*() const {
    return T(*it);
  }
  ListIterator& operator+=(std::ptrdiff_t n) {
    it += n;
    return *this;
  }
  ListIterator& operator++() {
    ++it;
    return *this;
  }
  ListIterator& operator--() {
    --it;
    return *this;
  }

 private:
  TreeList::const_iterator it;
};

template <typename T>
struct List : public TreeView {
  using iterator = ListIterator<T>;
  using const_iterator = ListIterator<T>;

  List(const TreeRef& tree) : TreeView(tree) {
    tree->match(TK_LIST);
    // Iterate over list to temporarily instantiate Ts that will check the type
    for (const T& elem : *this) {
      (void)elem; // silence unused warning
    }
  }
  iterator begin() const {
    return iterator(tree_->trees().begin());
  }
  iterator end() const {
    return iterator(tree_->trees().end());
  }
  bool empty() const {
    return tree_->trees().begin() == tree_->trees().end();
  }
  T operator[](size_t i) const {
    return T(subtree(i));
  }
  TreeRef map(const std::function<TreeRef(const T&)>& fn) {
    return tree_->map([&](TreeRef v) { return fn(T(v)); });
  }
  static List create(const SourceRange& range, const std::vector<T>& subtrees) {
    TreeList type_erased_sub{subtrees.begin(), subtrees.end()};
    return List(Compound::create(TK_LIST, range, std::move(type_erased_sub)));
  }
  static List unsafeCreate(const SourceRange& range, TreeList&& subtrees) {
    return List(Compound::create(TK_LIST, range, std::move(subtrees)));
  }
  size_t size() const {
    return tree_->trees().size();
  }
};

template <typename T>
struct Maybe : public TreeView {
  explicit Maybe(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_OPTION);
    if (tree_->trees().size() > 1)
      throw(ErrorReport(tree) << "Maybe trees can have at most one subtree");
  }
  /* implicit */ Maybe(const T& tree) : TreeView(tree) {}
  bool present() const {
    return tree_->trees().size() > 0;
  }
  T get() const {
    return T(tree_->trees().at(0));
  }
  TreeRef map(const std::function<TreeRef(const T&)>& fn) {
    return tree_->map([&](TreeRef v) { return fn(T(v)); });
  }
  static Maybe<T> create(const SourceRange& range) {
    return Maybe<T>(Compound::create(TK_OPTION, range, {}));
  }
  static Maybe<T> create(const SourceRange& range, const T& value) {
    return Maybe<T>(Compound::create(TK_OPTION, range, {value}));
  }
};

struct Ident : public TreeView {
  explicit Ident(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_IDENT);
  }
  const std::string& name() const {
    return subtree(0)->stringValue();
  }
  static Ident create(const SourceRange& range, std::string name) {
    return Ident(
        Compound::create(TK_IDENT, range, {String::create(std::move(name))}));
  }
};

////////////////////////////////////////////////////////////////////////////////
// Base types (production LHS)
////////////////////////////////////////////////////////////////////////////////

struct Expr : public TreeView {
  explicit Expr(const TreeRef& tree) : TreeView(tree) {
    switch (tree->kind()) {
      case TK_IF_EXPR:
      case TK_AND:
      case TK_OR:
      case '<':
      case '>':
      case TK_IS:
      case TK_ISNOT:
      case TK_EQ:
      case TK_LE:
      case TK_GE:
      case TK_NE:
      case '+':
      case '-':
      case TK_UNARY_MINUS:
      case '~':
      case '*':
      case TK_STARRED:
      case '/':
      case '%':
      case TK_NOT:
      case TK_CONST:
      case TK_STRINGLITERAL:
      case TK_TRUE:
      case TK_FALSE:
      case TK_NONE:
      case TK_NONE_TYPE:
      case TK_CAST:
      case TK_APPLY:
      case '.':
      case TK_SUBSCRIPT:
      case TK_SLICE_EXPR:
      case TK_VAR:
      case TK_LIST_LITERAL:
      case TK_TUPLE_LITERAL:
      case TK_DICT_LITERAL:
      case '@':
      case TK_POW:
      case TK_LSHIFT:
      case TK_RSHIFT:
      case TK_FLOOR_DIV:
      case '&':
      case '^':
      case '|':
      case TK_LIST_COMP:
      case TK_DICT_COMP:
      case TK_DOTS:
      case TK_IN:
      case TK_WITH_ITEM:
        return;
      default:
        throw(
            ErrorReport(tree)
            << kindToString(tree->kind()) << " is not a valid Expr");
    }
  }
};

////////////////////////////////////////////////////////////////////////////////
// Helper nodes (mostly for function arguments)
////////////////////////////////////////////////////////////////////////////////

struct Attribute : public TreeView {
  explicit Attribute(const TreeRef& tree) : TreeView(tree) {
    tree_->match(TK_ATTRIBUTE);
  }
  Ident name() const {
    return Ident(subtree(0));
  }
  Expr value() const {
    return Expr(subtree(1));
  }
  static Attribute create(
      const SourceRange& range,
      const Ident& name,
      const TreeRef& value) {
    return Attribute(Compound::create(TK_ATTRIBUTE, range, {name, value}));
  }
};

////////////////////////////////////////////////////////////////////////////////
// Top level definitions
////////////////////////////////////////////////////////////////////////////////

// Property represents a named attribute combined with a getter and setter
// method to access and mutate that attribute.
////////////////////////////////////////////////////////////////////////////////
// Statements
////////////////////////////////////////////////////////////////////////////////

// TODO: supports only single comprehension for now
struct ListComp : public Expr {
  explicit ListComp(const TreeRef& tree) : Expr(tree) {
    tree->match(TK_LIST_COMP);
  }
  Expr elt() const {
    return Expr(subtree(0));
  }
  Expr target() const {
    return Expr(subtree(1));
  }
  Expr iter() const {
    return Expr(subtree(2));
  }
  // TODO: no ifs for now
  static ListComp create(
      const SourceRange& range,
      const Expr& elt,
      const Expr& target,
      const Expr& iter) {
    return ListComp(Compound::create(TK_LIST_COMP, range, {elt, target, iter}));
  }
};

// TODO: supports only single comprehension for now
struct DictComp : public Expr {
  explicit DictComp(const TreeRef& tree) : Expr(tree) {
    tree->match(TK_DICT_COMP);
  }
  Expr key() const {
    return Expr(subtree(0));
  }
  Expr value() const {
    return Expr(subtree(1));
  }
  Expr target() const {
    return Expr(subtree(2));
  }
  Expr iter() const {
    return Expr(subtree(3));
  }
  // TODO: no ifs for now
  static DictComp create(
      const SourceRange& range,
      const Expr& key,
      const Expr& value,
      const Expr& target,
      const Expr& iter) {
    return DictComp(
        Compound::create(TK_DICT_COMP, range, {key, value, target, iter}));
  }
};

// Augmented assignment, like "foo += bar"
struct Dots : public Expr {
  explicit Dots(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_DOTS);
  }
  static Dots create(const SourceRange& range) {
    return Dots(Compound::create(TK_DOTS, range, {}));
  }
};

////////////////////////////////////////////////////////////////////////////////
// Expressions
////////////////////////////////////////////////////////////////////////////////

struct Const : public Expr {
  explicit Const(const TreeRef& tree) : Expr(tree) {
    tree_->matchNumSubtrees(TK_CONST, 1);
  }
  bool isFloatingPoint() const {
    if (isComplex())
      return false;

    bool is_inf = subtree(0)->stringValue() == "inf";
    return is_inf ||
        subtree(0)->stringValue().find_first_of(".eE") != std::string::npos;
  }
  bool isIntegral() const {
    return !isFloatingPoint() && !isComplex();
  }
  bool isComplex() const {
    return subtree(0)->stringValue().find_first_of('j') != std::string::npos;
  }
  int64_t asIntegral() const {
    try {
      return std::stoll(subtree(0)->stringValue(), nullptr, 0);
    } catch (const std::out_of_range&) {
      throw(
          ErrorReport(range()) << "Integral constant out of range "
                                  "(must fit in a signed 64 bit integer)");
    }
  }
  double asFloatingPoint() const {
    // We can't pass in nullptr as the dummy pointer gets dereferenced by
    // some strtod_c() implementations.
    char* dummy = nullptr;
    return torch::jit::strtod_c(subtree(0)->stringValue().c_str(), &dummy);
  }
  c10::complex<double> asComplex() const {
    char* dummy = nullptr;
    auto str = subtree(0)->stringValue();
    // Complex numbers (a+bj, where a is non-zero) are parsed as an addition
    // between float/int a and a complex number "bj". When a is 0, a complex
    // number bj is created as above. So, while parsing the string, we don't
    // have to worry about the real component of the complex number.
    auto imag =
        torch::jit::strtod_c(str.substr(0, str.size() - 1).c_str(), &dummy);
    return c10::complex<double>(0, imag);
  }
  const std::string& text() const {
    return subtree(0)->stringValue();
  }
  static Const create(const SourceRange& range, const std::string& value) {
    return Const(Compound::create(TK_CONST, range, {String::create(value)}));
  }
};

struct StringLiteral : public Expr {
  explicit StringLiteral(const TreeRef& tree) : Expr(tree) {
    tree_->matchNumSubtrees(TK_STRINGLITERAL, 1);
  }
  const std::string& text() const {
    return subtree(0)->stringValue();
  }
  static StringLiteral create(
      const SourceRange& range,
      const std::string& value) {
    return StringLiteral(
        Compound::create(TK_STRINGLITERAL, range, {String::create(value)}));
  }
};

struct Apply : public Expr {
  explicit Apply(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_APPLY);
  }
  Expr callee() const {
    return Expr(subtree(0));
  }
  List<Expr> inputs() const {
    return List<Expr>(subtree(1));
  }
  List<Attribute> attributes() const {
    return List<Attribute>(subtree(2));
  }
  static Apply create(
      const SourceRange& range,
      const Expr& callee,
      const List<Expr>& inputs,
      const List<Attribute>& attributes) {
    return Apply(
        Compound::create(TK_APPLY, range, {callee, inputs, attributes}));
  }
};

struct Select : public Expr {
  explicit Select(const TreeRef& tree) : Expr(tree) {
    tree_->match('.');
  }
  Expr value() const {
    return Expr(subtree(0));
  }
  Ident selector() const {
    return Ident(subtree(1));
  }
  static Select create(
      const SourceRange& range,
      const Expr& value,
      const Ident& selector) {
    return Select(Compound::create('.', range, {value, selector}));
  }
};

struct SliceExpr : public Expr {
  explicit SliceExpr(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_SLICE_EXPR);
  }
  Maybe<Expr> start() const {
    return Maybe<Expr>(subtree(0));
  }
  Maybe<Expr> end() const {
    return Maybe<Expr>(subtree(1));
  }
  Maybe<Expr> step() const {
    return Maybe<Expr>(subtree(2));
  }
  Expr startOr(int64_t alternative) const {
    const auto startOption = start();
    return startOption.present() ? startOption.get() : createInt(alternative);
  }
  Expr endOr(int64_t alternative) const {
    const auto endOption = end();
    return endOption.present() ? endOption.get() : createInt(alternative);
  }
  Expr stepOr(int64_t alternative) const {
    const auto stepOption = step();
    return stepOption.present() ? stepOption.get() : createInt(alternative);
  }
  static SliceExpr create(
      const SourceRange& range,
      const Maybe<Expr>& start,
      const Maybe<Expr>& end,
      const Maybe<Expr>& step) {
    return SliceExpr(
        Compound::create(TK_SLICE_EXPR, range, {start, end, step}));
  }

 private:
  Expr createInt(int64_t value) const {
    return Expr(Const::create(range(), std::to_string(value)));
  }
};

struct Subscript : public Expr {
  explicit Subscript(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_SUBSCRIPT);
  }
  Expr value() const {
    return Expr(subtree(0));
  }
  List<Expr> subscript_exprs() const {
    return List<Expr>(subtree(1));
  }
  static Subscript create(
      const SourceRange& range,
      const Expr& value,
      const List<Expr>& subscript_exprs) {
    auto whole_range = SourceRange(
        range.source(), range.start(), subscript_exprs.range().end() + 1);
    return Subscript(
        Compound::create(TK_SUBSCRIPT, whole_range, {value, subscript_exprs}));
  }
};

struct Var : public Expr {
  explicit Var(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_VAR);
  }
  Ident name() const {
    return Ident(subtree(0));
  }
  static Var create(const SourceRange& range, const Ident& name) {
    return Var(Compound::create(TK_VAR, range, {name}));
  }
};

// WithItem represents an item using with a WithStmt.
// With represents a with statement consisting of a list of with items and a
// body of statements.
struct ListLiteral : public Expr {
  explicit ListLiteral(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_LIST_LITERAL);
  }
  List<Expr> inputs() const {
    return subtree(0);
  }
  static ListLiteral create(
      const SourceRange& range,
      const List<Expr>& inputs) {
    return ListLiteral(Compound::create(TK_LIST_LITERAL, range, {inputs}));
  }
};

struct TupleLiteral : public Expr {
  explicit TupleLiteral(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_TUPLE_LITERAL);
  }
  List<Expr> inputs() const {
    return subtree(0);
  }
  static TupleLiteral create(
      const SourceRange& range,
      const List<Expr>& inputs) {
    return TupleLiteral(Compound::create(TK_TUPLE_LITERAL, range, {inputs}));
  }
};

struct DictLiteral : public Expr {
  explicit DictLiteral(const TreeRef& tree) : Expr(tree) {
    tree_->match(TK_DICT_LITERAL);
  }
  List<Expr> key_inputs() const {
    return subtree(0);
  }
  List<Expr> value_inputs() const {
    return subtree(1);
  }
  static DictLiteral create(
      const SourceRange& range,
      const List<Expr>& keys,
      const List<Expr>& values) {
    return DictLiteral(
        Compound::create(TK_DICT_LITERAL, range, {keys, values}));
  }
};

/*
 * NOTE: transforming PEP 604 union into equivalent union type
 *
 * NOTE: Union[int, float] parses into:
 * <EXPR> expr:(subscript
 *  (variable (ident Union))
 *  (list
 *    (variable (ident int))
 *    (variable (ident float))))
 * <KIND> subscript
 *
 * NOTE: (int | float) parses into:
 * <EXPR> expr:(|
 *  (variable (ident int))
 *  (variable (ident float)))
 * <KIND> |
 */

struct BinOp : public Expr {
  explicit BinOp(const TreeRef& tree) : Expr(tree) {
    switch (tree->kind()) {
      case TK_AND:
      case TK_OR:
      case '<':
      case '>':
      case TK_IS:
      case TK_ISNOT:
      case TK_EQ:
      case TK_LE:
      case TK_GE:
      case TK_NE:
      case '+':
      case '*':
      case '/':
      case '-':
      case '@':
      case TK_POW:
      case TK_LSHIFT:
      case TK_RSHIFT:
      case '%':
      case '&':
      case '^':
      case '|':
      case TK_FLOOR_DIV:
      case TK_IN:
        if (tree->trees().size() != 2)
          throw(
              ErrorReport(tree)
              << "BinOp expected 2 subtrees, found " << tree->trees().size());
        return;
      default:
        throw(
            ErrorReport(tree)
            << kindToString(tree->kind()) << " is not a valid BinOp");
    }
  }
  Expr lhs() const {
    return Expr(subtree(0));
  }
  Expr rhs() const {
    return Expr(subtree(1));
  }
  static BinOp create(
      const SourceRange& range,
      int kind,
      const Expr& lhs,
      const Expr& rhs) {
    return BinOp(Compound::create(kind, range, {lhs, rhs}));
  }
};

inline void _flatten_pep604_union(
    const torch::jit::Expr& node,
    std::vector<torch::jit::Expr>* result) {
  // flatten possibly nested union expressions like (int | (float | str))
  // into a flat list of expressions like [int, float, str]
  if (node.kind() == '|') {
    auto as_binop = torch::jit::BinOp(node);
    _flatten_pep604_union(as_binop.lhs(), result);
    _flatten_pep604_union(as_binop.rhs(), result);
  } else {
    result->push_back(node);
  }
}

inline std::vector<Expr> get_pep604_union_members(const Expr& node) {
  std::vector<Expr> result;
  _flatten_pep604_union(node, &result);
  return result;
}

// Flattens a PEP 604 union into a classical union.
// For example, ((x | y) | z) is transformed into Union[x, y, z].
inline Expr pep604union_to_union(const Expr& expr) {
  // noop if not a pep604 union
  if (expr.kind() != '|')
    return expr;

  // In order to support unions with more than 2 operands ((x|y)|z), we need to
  // recursively flatten the tree of | expressions.
  auto members = get_pep604_union_members(expr);
  auto synthesised_union = Subscript::create(
      expr.range(),
      Var::create(expr.range(), Ident::create(expr.range(), "Union")),
      List<Expr>::create(expr.range(), members));
  return synthesised_union;
}

} // namespace torch::jit

namespace std {

template <typename T>
struct iterator_traits<torch::jit::ListIterator<T>>
    : std::iterator_traits<torch::jit::TreeList::const_iterator> {};

} // namespace std
