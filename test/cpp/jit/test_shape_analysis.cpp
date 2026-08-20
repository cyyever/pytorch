#include <gtest/gtest.h>

#include <ATen/ATen.h>
#include <ATen/core/interned_strings.h>
#include <c10/util/Exception.h>
#include <c10/util/Optional.h>
#include <test/cpp/jit/test_utils.h>
#include <torch/csrc/jit/ir/ir.h>
#include <torch/csrc/jit/ir/ir_views.h>
#include <torch/csrc/jit/ir/irparser.h>
#include <torch/csrc/jit/passes/constant_propagation.h>
#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_cache.h>
#include <torch/csrc/jit/passes/utils/subgraph_utils.h>
#include <torch/csrc/jit/runtime/graph_iterator.h>
#include <torch/csrc/jit/runtime/interpreter.h>
#include <torch/csrc/jit/testing/file_check.h>
#include <torch/cuda.h>
#include <unordered_map>

namespace torch {
namespace jit {

namespace {

std::optional<int64_t> sym_dim = std::nullopt;

// NOLINTNEXTLINE(bugprone-easily-swappable-parameters)
void assertShapeEqual(c10::SymbolicShape& a, c10::SymbolicShape& e) {
  auto a_canonical = CanonicalizedSymbolicShape(a);
  auto e_canonical = CanonicalizedSymbolicShape(e);
  EXPECT_EQ(a_canonical, e_canonical);
}

void assertShapeEqual(
    std::optional<std::vector<c10::SymbolicShape>>& actual,
    std::vector<std::optional<int64_t>> expected) {
  ASSERT_TRUE(actual.has_value());
  ASSERT_EQ(actual->size(), 1);

  auto symb_expected = c10::SymbolicShape(expected);
  assertShapeEqual(actual->at(0), symb_expected);
}

const FunctionSchema* getSchema(const char* name) {
  return &(getOperatorForLiteral(name)->schema());
}
} // namespace

TEST(ShapeAnalysisTest, SymbolicShapeAPI) {
  // Figure out how to fetch a function schema

  // Ask someone else how to create a function schema / operator in C++
  auto schema = getSchema(
      "aten::sub.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor");

  c10::IValue const_size_1 = std::vector<int64_t>{64, 56, 56};
  c10::IValue const_size_2 = std::vector<int64_t>{1, 56, 56};

  // Check vector initializer list syntax
  c10::SymbolicShape ss_concrete =
      std::vector<std::optional<int64_t>>{1, 56, 56};
  c10::SymbolicShape ss1 = std::vector<std::optional<int64_t>>{sym_dim, 56, 56};
  c10::SymbolicShape ss2 =
      std::vector<std::optional<int64_t>>{64, sym_dim, sym_dim};
  c10::SymbolicShape ss3 =
      std::vector<std::optional<int64_t>>{sym_dim, sym_dim, sym_dim, sym_dim};

  auto res = calculateSymbolicShapesOnOp(
      schema, std::vector<SSAInput>{const_size_1, const_size_1});
  assertShapeEqual(res, {64, 56, 56});

  res = calculateSymbolicShapesOnOp(
      schema, std::vector<SSAInput>{const_size_1, const_size_2});
  assertShapeEqual(res, {64, 56, 56});

  res = calculateSymbolicShapesOnOp(
      schema, std::vector<SSAInput>{const_size_1, ss1});
  assertShapeEqual(res, {64, 56, 56});

  res = calculateSymbolicShapesOnOp(
      schema, std::vector<SSAInput>{const_size_2, ss1});
  assertShapeEqual(res, {sym_dim, 56, 56});

  res = calculateSymbolicShapesOnOp(
      schema, std::vector<SSAInput>{ss_concrete, ss2});
  assertShapeEqual(res, {64, 56, 56});

  res = calculateSymbolicShapesOnOp(schema, std::vector<SSAInput>{ss2, ss3});
  assertShapeEqual(res, {sym_dim, 64, sym_dim, sym_dim});
}

TEST(ShapeAnalysisTest, BoundedSymbolicShapes) {
  auto schema = getSchema("aten::nonzero(Tensor self) -> (Tensor)");

  // Test that we generate symbolic shapes for the output of a nonzero op
  c10::IValue const_size_1 = std::vector<int64_t>{5, 10};
  auto res =
      calculateSymbolicShapesOnOp(schema, std::vector<SSAInput>{const_size_1});
  assertShapeEqual(res, {sym_dim, 2});

  // Test that nonzero can also create concrete shapes
  c10::IValue const_size_2 = std::vector<int64_t>({1, 0});
  res =
      calculateSymbolicShapesOnOp(schema, std::vector<SSAInput>{const_size_2});
  assertShapeEqual(res, {0, 2});
}

TEST(ShapeAnalysisTest, SymbolicShapeCaching) {
  clear_shape_cache();
  auto schema = getSchema("aten::mm(Tensor self, Tensor mat2) -> Tensor");

  c10::IValue const_size_1 = std::vector<int64_t>{64, 56};
  c10::IValue const_size_2 = std::vector<int64_t>{64, 56};
  c10::IValue const_size_3 = std::vector<int64_t>{64, 20};

  c10::SymbolicShape ss1 = c10::SymbolicShape({sym_dim, 64});
  c10::SymbolicShape ss2 = c10::SymbolicShape({sym_dim, 64});
  c10::SymbolicShape ss3 = c10::SymbolicShape({sym_dim, sym_dim});

  auto res = calculateSymbolicShapesOnOp(schema, {ss1, const_size_1});
  assertShapeEqual(res, {sym_dim, 56});
  auto res1_val = res->at(0);

  // The exact same arguments should return the exact same result
  res = calculateSymbolicShapesOnOp(schema, {ss1, const_size_1});
  auto res2_val = res->at(0);
  EXPECT_EQ(res1_val, res2_val);
  EXPECT_EQ(get_shape_cache_size(), 1);

  // Same shape but different symbols should return same shape
  // but different symbolic indices
  res = calculateSymbolicShapesOnOp(schema, {ss2, const_size_2});
  auto res3_val = res->at(0);

  assertShapeEqual(res3_val, res2_val);
  EXPECT_NE(res3_val, res2_val);
  EXPECT_EQ(get_shape_cache_size(), 1);

  // Different concrete shape should be cached separately
  res = calculateSymbolicShapesOnOp(schema, {ss1, const_size_3});
  assertShapeEqual(res, {sym_dim, 20});
  EXPECT_EQ(get_shape_cache_size(), 2);

  res = calculateSymbolicShapesOnOp(schema, {ss3, const_size_3});
  assertShapeEqual(res, {sym_dim, 20});
  EXPECT_EQ(get_shape_cache_size(), 3);

  res = calculateSymbolicShapesOnOp(schema, {ss3, ss3});
  assertShapeEqual(res, {sym_dim, sym_dim});
  EXPECT_EQ(get_shape_cache_size(), 4);
}

TEST(ShapeAnalysisTest, ShapeCacheMultipleFns) {
  clear_shape_cache();

  auto squeeze_op =
      getSchema("aten::squeeze.dim(Tensor(a) self, int dim) -> Tensor(a)");
  auto mul_tensor =
      getSchema("aten::mul.Tensor(Tensor self, Tensor other) -> Tensor");
  auto mul_scalar =
      getSchema("aten::mul.Scalar(Tensor self, Scalar other) -> Tensor");
  auto div_tensor =
      getSchema("aten::div.Tensor(Tensor self, Tensor other) -> Tensor");
  auto matmul = getSchema("aten::mm(Tensor self, Tensor mat2) -> Tensor");

  c10::IValue const_int = 1;

  c10::SymbolicShape ss1 = c10::SymbolicShape({sym_dim, 64});

  auto res = calculateSymbolicShapesOnOp(squeeze_op, {ss1, const_int});
  assertShapeEqual(res, {sym_dim, 64});

  // Show that cache can handle multiple functions
  res = calculateSymbolicShapesOnOp(mul_scalar, {ss1, const_int});
  assertShapeEqual(res, {sym_dim, 64});
  EXPECT_EQ(get_shape_cache_size(), 2);

  res = calculateSymbolicShapesOnOp(mul_tensor, {ss1, ss1});
  assertShapeEqual(res, {sym_dim, 64});
  EXPECT_EQ(get_shape_cache_size(), 3);

  // Even when the expected outcome is the same, should not collide
  res = calculateSymbolicShapesOnOp(div_tensor, {ss1, ss1});
  assertShapeEqual(res, {sym_dim, 64});
  EXPECT_EQ(get_shape_cache_size(), 4);

  // Don't lose cached objects
  res = calculateSymbolicShapesOnOp(mul_scalar, {ss1, const_int});
  assertShapeEqual(res, {sym_dim, 64});
  EXPECT_EQ(get_shape_cache_size(), 4);

  res = calculateSymbolicShapesOnOp(matmul, {ss1, ss1});
  // SSA can infer that sym_dim is 64 as both tensors
  // use the same sym_dim
  assertShapeEqual(res, {64, 64});
  EXPECT_EQ(get_shape_cache_size(), 5);
}

TEST(ShapeAnalysisTest, TestShapeMultipleReturns) {
  clear_shape_cache();

  auto max_dim_op = getSchema(
      "aten::max.dim(Tensor self, int dim, bool keepdim=False) -> (Tensor values, Tensor indices)");
  c10::IValue const_int = 1;
  c10::IValue false_ival = false;

  c10::SymbolicShape ss1 = c10::SymbolicShape({sym_dim, 64});
  c10::SymbolicShape ss2 = c10::SymbolicShape({sym_dim, 64});

  auto res =
      calculateSymbolicShapesOnOp(max_dim_op, {ss1, const_int, false_ival});
  c10::SymbolicShape expected_res =
      c10::SymbolicShape(std::vector<std::optional<int64_t>>{sym_dim});
  assertShapeEqual(res->at(0), expected_res);
  // res0 and res1 should share the same symbolic symbol
  EXPECT_EQ(res->at(0), res->at(1));

  // Also test that the shape cache also returns consistent result shapes
  res = calculateSymbolicShapesOnOp(max_dim_op, {ss2, const_int, false_ival});
  assertShapeEqual(res->at(0), expected_res);
  EXPECT_EQ(res->at(0), res->at(1));
  EXPECT_EQ(get_shape_cache_size(), 1);
}
} // namespace jit
} // namespace torch
