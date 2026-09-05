#include <ATen/core/List.h>

#include <algorithm>

namespace c10::detail {
bool operator==(const ListImpl& lhs, const ListImpl& rhs) {
  // see: [container equality]
  return *lhs.elementType == *rhs.elementType &&
      std::ranges::equal(lhs.list, rhs.list, _fastEqualsForContainer);
}

ListImpl::ListImpl(list_type list_, TypePtr elementType_)
  : list(std::move(list_))
  , elementType(std::move(elementType_)) {}
} // namespace c10::detail
