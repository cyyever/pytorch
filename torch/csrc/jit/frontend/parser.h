#pragma once
#include <torch/csrc/Export.h>
#include <torch/csrc/jit/frontend/tree.h>
#include <torch/csrc/jit/frontend/tree_views.h>
#include <memory>

namespace torch::jit {

struct ParserImpl;
struct Lexer;


struct TORCH_API Parser {
  explicit Parser(const std::shared_ptr<Source>& src);
  Expr parseExp();
  Lexer& lexer();
  ~Parser();

 private:
  std::unique_ptr<ParserImpl> pImpl;
};

} // namespace torch::jit
