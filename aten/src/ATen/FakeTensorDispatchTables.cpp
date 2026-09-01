#include <ATen/FakeTensorDispatchTables.h>

#include <c10/util/Exception.h>
#include <c10/util/LeftRight.h>

#include <unordered_set>

namespace at::impl {
namespace {

struct FakeDispatchTables {
  std::unordered_set<c10::OperatorName> decomp;
  std::unordered_set<c10::OperatorName> meta;
  std::unordered_set<c10::OperatorName> op_impl;
  std::unordered_set<c10::OperatorName> prim_meta;
};

c10::LeftRight<FakeDispatchTables>& fakeDispatchTables() {
  static c10::LeftRight<FakeDispatchTables> tables;
  return tables;
}

std::unordered_set<c10::OperatorName>& setForCategory(
    FakeDispatchTables& t,
    FakeDispatchCategory category) {
  switch (category) {
    case FakeDispatchCategory::Decomp:
      return t.decomp;
    case FakeDispatchCategory::Meta:
      return t.meta;
    case FakeDispatchCategory::OpImpl:
      return t.op_impl;
    case FakeDispatchCategory::PrimMeta:
      return t.prim_meta;
  }
  TORCH_INTERNAL_ASSERT(false, "unknown FakeDispatchCategory");
}

} // namespace

void fakeDispatchTableAdd(
    FakeDispatchCategory category,
    const c10::OperatorName& name) {
  fakeDispatchTables().write(
      [&](FakeDispatchTables& t) { setForCategory(t, category).insert(name); });
}

void fakeDispatchTableRemove(
    FakeDispatchCategory category,
    const c10::OperatorName& name) {
  fakeDispatchTables().write(
      [&](FakeDispatchTables& t) { setForCategory(t, category).erase(name); });
}

} // namespace at::impl
