#include <torch/csrc/jit/frontend/parser.h>

#include <torch/csrc/jit/frontend/lexer.h>
#include <torch/csrc/jit/frontend/parse_string_literal.h>
#include <torch/csrc/jit/frontend/tree.h>
#include <torch/csrc/jit/frontend/tree_views.h>
#include <optional>

namespace torch::jit {

struct ParserImpl {
  explicit ParserImpl(const std::shared_ptr<Source>& source)
      : L(source), shared(sharedParserData()) {}

  Ident parseIdent() {
    auto t = L.expect(TK_IDENT);
    // whenever we parse something that has a TreeView type we always
    // use its create method so that the accessors and the constructor
    // of the Compound tree are in the same place.
    return Ident::create(t.range, t.text());
  }
  TreeRef createApply(const Expr& expr) {
    TreeList attributes;
    auto range = L.cur().range;
    TreeList inputs;
    parseArguments(inputs, attributes);
    return Apply::create(
        range,
        expr,
        List<Expr>(makeList(range, std::move(inputs))),
        List<Attribute>(makeList(range, std::move(attributes))));
  }

  static bool followsTuple(int kind) {
    switch (kind) {
      case TK_PLUS_EQ:
      case TK_MINUS_EQ:
      case TK_TIMES_EQ:
      case TK_DIV_EQ:
      case TK_MOD_EQ:
      case TK_BIT_OR_EQ:
      case TK_BIT_AND_EQ:
      case TK_BIT_XOR_EQ:
      case TK_LSHIFT_EQ:
      case TK_RSHIFT_EQ:
      case TK_POW_EQ:
      case TK_NEWLINE:
      case '=':
      case ')':
        return true;
      default:
        return false;
    }
  }

  // exp | expr, | expr, expr, ...
  Expr parseExpOrExpTuple() {
    auto prefix = parseExp();
    if (L.cur().kind == ',') {
      std::vector<Expr> exprs = {prefix};
      while (L.nextIf(',')) {
        if (followsTuple(L.cur().kind))
          break;
        exprs.push_back(parseExp());
      }
      auto list = List<Expr>::create(prefix.range(), exprs);
      prefix = TupleLiteral::create(list.range(), list);
    }
    return prefix;
  }
  // things like a 1.0 or a(4) that are not unary/binary expressions
  // and have higher precedence than all of them
  TreeRef parseBaseExp() {
    TreeRef prefix;
    switch (L.cur().kind) {
      case TK_NUMBER: {
        prefix = parseConst();
      } break;
      case TK_TRUE:
      case TK_FALSE:
      case TK_NONE:
      case TK_NONE_TYPE: {
        auto k = L.cur().kind;
        auto r = L.cur().range;
        prefix = create_compound(k, r, {});
        L.next();
      } break;
      case '(': {
        L.next();
        if (L.nextIf(')')) {
          /// here we have the empty tuple case
          std::vector<Expr> vecExpr;
          List<Expr> listExpr = List<Expr>::create(L.cur().range, vecExpr);
          prefix = TupleLiteral::create(L.cur().range, listExpr);
          break;
        }
        prefix = parseExpOrExpTuple();
        L.expect(')');
      } break;
      case '[': {
        auto list = parseList('[', ',', ']', &ParserImpl::parseExp);

        if (list.size() == 1 && (*list.begin()).kind() == TK_LIST_COMP) {
          prefix = *list.begin();
        } else {
          for (auto se : list) {
            if (se.kind() == TK_LIST_COMP) {
              throw ErrorReport(list.range())
                  << " expected a single list comprehension within '[' , ']'";
            }
          }
          prefix = ListLiteral::create(list.range(), List<Expr>(list));
        }

      } break;
      case '{': {
        L.next();
        // If we have a dict literal, `keys` and `values` will store the keys
        // and values used in the object's construction. EDGE CASE: We have a
        // dict comprehension, so we'll get the first element of the dict
        // comprehension in `keys` and a list comprehension in `values`.
        // For example, `{i : chr(i + 65) for i in range(4)}` would give us
        // `i` in `keys` and `chr(i + 65) for i in range(4)` in `values`.
        // The optimal way of handling this case is to simply splice the new
        // dict comprehension together from the existing list comprehension.
        // Splicing prevents breaking changes to our API and does not require
        // the use of global variables.
        std::vector<Expr> keys;
        std::vector<Expr> values;
        auto range = L.cur().range;
        if (L.cur().kind != '}') {
          do {
            keys.push_back(parseExp());
            L.expect(':');
            values.push_back(parseExp());
          } while (L.nextIf(','));
        }
        L.expect('}');
        if (keys.size() == 1 && (*values.begin()).kind() == TK_LIST_COMP) {
          ListComp lc(*values.begin());
          prefix = DictComp::create(
              range, *keys.begin(), lc.elt(), lc.target(), lc.iter());
        } else {
          prefix = DictLiteral::create(
              range,
              List<Expr>::create(range, keys),
              List<Expr>::create(range, values));
        }
      } break;
      case TK_STRINGLITERAL: {
        prefix = parseConcatenatedStringLiterals();
      } break;
      case TK_ELLIPSIS:
      case TK_DOTS: {
        prefix = Dots::create(L.cur().range);
        L.next();
      } break;
      default: {
        Ident name = parseIdent();
        prefix = Var::create(name.range(), name);
      } break;
    }
    while (true) {
      if (L.nextIf('.')) {
        const auto name = parseIdent();
        prefix = Select::create(name.range(), Expr(prefix), Ident(name));
      } else if (L.cur().kind == '(') {
        prefix = createApply(Expr(prefix));
      } else if (L.cur().kind == '[') {
        prefix = parseSubscript(prefix);
      } else {
        break;
      }
    }
    return prefix;
  }
  TreeRef parseTrinary(
      TreeRef true_branch,
      const SourceRange& range,
      int binary_prec) {
    auto cond = parseExp();
    L.expect(TK_ELSE);
    auto false_branch = parseExp(binary_prec);
    return create_compound(
        TK_IF_EXPR, range, {cond, std::move(true_branch), false_branch});
  }
  // parse the longest expression whose binary operators have
  // precedence strictly greater than 'precedence'
  // precedence == 0 will parse _all_ expressions
  // this is the core loop of 'top-down precedence parsing'
  Expr parseExp() {
    return parseExp(0);
  }
  Expr parseExp(int precedence) {
    TreeRef prefix;
    int unary_prec = 0;
    if (shared.isUnary(L.cur().kind, &unary_prec)) {
      auto kind = L.cur().kind;
      auto pos = L.cur().range;
      L.next();
      auto unary_kind = kind == '*' ? TK_STARRED
          : kind == '-'             ? TK_UNARY_MINUS
                                    : kind;
      auto subexp = parseExp(unary_prec);
      // fold '-' into constant numbers, so that attributes can accept
      // things like -1
      if (unary_kind == TK_UNARY_MINUS && subexp.kind() == TK_CONST) {
        prefix = Const::create(subexp.range(), "-" + Const(subexp).text());
      } else {
        prefix = create_compound(unary_kind, pos, {subexp});
      }
    } else {
      prefix = parseBaseExp();
    }
    int binary_prec = 0;
    while (shared.isBinary(L.cur().kind, &binary_prec)) {
      if (binary_prec <= precedence) // not allowed to parse something which is
        // not greater than 'precedence'
        break;

      int kind = L.cur().kind;
      auto pos = L.cur().range;
      L.next();
      if (shared.isRightAssociative(kind))
        binary_prec--;

      if (kind == TK_NOTIN) {
        // NB: `not in` is just `not( in )`, so we don't introduce new tree view
        // but just make it a nested call in our tree view structure
        prefix = create_compound(TK_IN, pos, {prefix, parseExp(binary_prec)});
        prefix = create_compound(TK_NOT, pos, {prefix});
        continue;
      }

      // special case for trinary operator
      if (kind == TK_IF) {
        prefix = parseTrinary(prefix, pos, binary_prec);
        continue;
      }

      if (kind == TK_FOR) {
        // TK_FOR targets should only parse exprs prec greater than 4, which
        // only includes subset of Exprs that are supposed to be on the LHS
        // according to the python grammar
        // https://docs.python.org/3/reference/grammar.html
        auto target = parseLHSExp();
        L.expect(TK_IN);
        auto iter = parseExp();
        prefix = ListComp::create(pos, Expr(prefix), target, iter);
        continue;
      }

      prefix = create_compound(kind, pos, {prefix, parseExp(binary_prec)});
    }
    return Expr(prefix);
  }

  void parseSequence(
      int begin,
      int sep,
      int end,
      const std::function<void()>& parse) {
    if (begin != TK_NOTHING) {
      L.expect(begin);
    }
    while (end != L.cur().kind) {
      parse();
      if (!L.nextIf(sep)) {
        if (end != TK_NOTHING) {
          L.expect(end);
        }
        return;
      }
    }
    L.expect(end);
  }
  template <typename T>
  List<T> parseList(int begin, int sep, int end, T (ParserImpl::*parse)()) {
    auto r = L.cur().range;
    std::vector<T> elements;
    parseSequence(
        begin, sep, end, [&] { elements.emplace_back((this->*parse)()); });
    return List<T>::create(r, elements);
  }

  Const parseConst() {
    auto range = L.cur().range;
    auto t = L.expect(TK_NUMBER);
    return Const::create(t.range, t.text());
  }

  StringLiteral parseConcatenatedStringLiterals() {
    auto range = L.cur().range;
    std::string ss;
    while (L.cur().kind == TK_STRINGLITERAL) {
      auto literal_range = L.cur().range;
      ss.append(parseStringLiteral(literal_range, L.next().text()));
    }
    return StringLiteral::create(range, ss);
  }

  Expr parseAttributeValue() {
    return parseExp();
  }

  void parseArguments(TreeList& inputs, TreeList& attributes) {
    parseSequence('(', ',', ')', [&] {
      if (L.cur().kind == TK_IDENT && L.lookahead().kind == '=') {
        auto ident = parseIdent();
        L.expect('=');
        auto v = parseAttributeValue();
        attributes.push_back(Attribute::create(ident.range(), Ident(ident), v));
      } else {
        inputs.push_back(parseExp());
      }
    });
  }

  // parse LHS acceptable exprs, which only includes subset of Exprs that prec
  // is greater than 4 according to the python grammar
  Expr parseLHSExp() {
    return parseExp(4);
  }

  // Parse expr's of the form [a:], [:b], [a:b], [:] and all variations with
  // "::"
  Expr parseSubscriptExp() {
    TreeRef first, second, third;
    auto range = L.cur().range;
    if (L.cur().kind != ':') {
      first = parseExp();
    }
    if (L.nextIf(':')) {
      if (L.cur().kind != ',' && L.cur().kind != ']' && L.cur().kind != ':') {
        second = parseExp();
      }
      if (L.nextIf(':')) {
        if (L.cur().kind != ',' && L.cur().kind != ']') {
          third = parseExp();
        }
      }
      auto maybe_first = first ? Maybe<Expr>::create(range, Expr(first))
                               : Maybe<Expr>::create(range);
      auto maybe_second = second ? Maybe<Expr>::create(range, Expr(second))
                                 : Maybe<Expr>::create(range);
      auto maybe_third = third ? Maybe<Expr>::create(range, Expr(third))
                               : Maybe<Expr>::create(range);
      return SliceExpr::create(range, maybe_first, maybe_second, maybe_third);
    } else {
      return Expr(first);
    }
  }

  TreeRef parseSubscript(const TreeRef& value) {
    const auto range = L.cur().range;

    auto subscript_exprs =
        parseList('[', ',', ']', &ParserImpl::parseSubscriptExp);

    const auto whole_range =
        SourceRange(range.source(), range.start(), L.cur().range.start());
    return Subscript::create(whole_range, Expr(value), subscript_exprs);
  }





  // 'first' has already been parsed since expressions can exist
  // alone on a line:
  // first[,other,lhs] = rhs










  Lexer& lexer() {
    return L;
  }

 private:
  // short helpers to create nodes
  TreeRef create_compound(
      int kind,
      const SourceRange& range,
      TreeList&& trees) {
    return Compound::create(kind, range, std::move(trees));
  }
  TreeRef makeList(const SourceRange& range, TreeList&& trees) {
    return create_compound(TK_LIST, range, std::move(trees));
  }
  Lexer L;
  SharedParserData& shared;
};

Parser::Parser(const std::shared_ptr<Source>& src)
    : pImpl(new ParserImpl(src)) {}

Parser::~Parser() = default;

Lexer& Parser::lexer() {
  return pImpl->lexer();
}
Expr Parser::parseExp() {
  return pImpl->parseExp();
}

} // namespace torch::jit
