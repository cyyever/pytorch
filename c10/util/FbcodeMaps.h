#ifndef C10_UTIL_FBCODEMAPS_H_
#define C10_UTIL_FBCODEMAPS_H_

// Map typedefs kept as a single place to swap the hash containers used by
// nativert and the picklers. fbcode used to point them at folly's F14 maps;
// this fork does not build there, so they are the standard ones.

#include <unordered_map>
#include <unordered_set>

namespace c10 {
template <typename Key, typename Value>
using FastMap = std::unordered_map<Key, Value>;
template <typename Key>
using FastSet = std::unordered_set<Key>;
} // namespace c10

#endif // C10_UTIL_FBCODEMAPS_H_
