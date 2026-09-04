#include <gtest/gtest.h>
#include <torch/all.h>
#include <ATen/core/jit_type.h>

namespace c10 {

TEST(TypeCustomPrinter, Basic) {
  TypePrinter printer =
      [](const Type& t) -> std::optional<std::string> {
    if (auto tensorType = t.cast<TensorType>()) {
      return "CustomTensor";
    }
    return std::nullopt;
  };

  // Tensor types should be rewritten
  torch::Tensor iv = torch::rand({2, 3});
  const auto type = TensorType::create(iv);
  EXPECT_EQ(type->annotation_str(), "Tensor");
  EXPECT_EQ(type->annotation_str(printer), "CustomTensor");

  // Unrelated types should not be affected
  const auto intType = IntType::get();
  EXPECT_EQ(intType->annotation_str(printer), intType->annotation_str());
}

TEST(TypeCustomPrinter, ContainedTypes) {
  TypePrinter printer =
      [](const Type& t) -> std::optional<std::string> {
    if (auto tensorType = t.cast<TensorType>()) {
      return "CustomTensor";
    }
    return std::nullopt;
  };
  torch::Tensor iv = torch::rand({2, 3});
  const auto type = TensorType::create(iv);

  // Contained types should work
  const auto tupleType = TupleType::create({type, IntType::get(), type});
  EXPECT_EQ(tupleType->annotation_str(), "Tuple[Tensor, int, Tensor]");
  EXPECT_EQ(
      tupleType->annotation_str(printer), "Tuple[CustomTensor, int, CustomTensor]");
  const auto dictType = DictType::create(IntType::get(), type);
  EXPECT_EQ(dictType->annotation_str(printer), "Dict[int, CustomTensor]");
  const auto listType = ListType::create(tupleType);
  EXPECT_EQ(
      listType->annotation_str(printer),
      "List[Tuple[CustomTensor, int, CustomTensor]]");
}

TEST(TypeCustomPrinter, NamedTuples) {
  TypePrinter printer =
      [](const Type& t) -> std::optional<std::string> {
    if (auto tupleType = t.cast<TupleType>()) {
      // Rewrite only NamedTuples
      if (tupleType->name()) {
        return "Rewritten";
      }
    }
    return std::nullopt;
  };
  torch::Tensor iv = torch::rand({2, 3});
  const auto type = TensorType::create(iv);

  std::vector<std::string> field_names = {"foo", "bar"};
  const auto namedTupleType = TupleType::createNamed(
      "my.named.tuple", field_names, {type, IntType::get()});
  EXPECT_EQ(namedTupleType->annotation_str(printer), "Rewritten");

  // Put it inside another tuple, should still work
  const auto outerTupleType = TupleType::create({IntType::get(), namedTupleType});
  EXPECT_EQ(outerTupleType->annotation_str(printer), "Tuple[int, Rewritten]");
}






TEST(TypeEquality, TupleEquality) {
  // Tuples should be structurally typed
  auto type = TupleType::create({IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
  auto type2 = TupleType::create({IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});

  EXPECT_EQ(*type, *type2);
}

TEST(TypeEquality, NamedTupleEquality) {
  // Named tuples should compare equal if they share a name and field names
  std::vector<std::string> fields = {"a", "b", "c", "d"};
  std::vector<std::string> otherFields = {"wow", "so", "very", "different"};
  auto type = TupleType::createNamed(
      "MyNamedTuple",
      fields,
      {IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
  auto type2 = TupleType::createNamed(
      "MyNamedTuple",
      fields,
      {IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
  EXPECT_EQ(*type, *type2);

  auto differentName = TupleType::createNamed(
      "WowSoDifferent",
      fields,
      {IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
  EXPECT_NE(*type, *differentName);

  auto differentField = TupleType::createNamed(
      "MyNamedTuple",
      otherFields,
      {IntType::get(), TensorType::get(), FloatType::get(), ComplexType::get()});
  EXPECT_NE(*type, *differentField);
}
} // namespace c10
