#include <torch/csrc/jit/passes/symbolic_shape_analysis.h>
#include <torch/csrc/jit/passes/symbolic_shape_cache.h>

#include <list>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <utility>

// SHAPE CACHING CODE

namespace torch::jit {
namespace {
using CanonicalArg = std::variant<CanonicalizedSymbolicShape, IValue>;
using CanonicalArgVec = std::vector<CanonicalArg>;
using CanonicalRet = std::vector<CanonicalizedSymbolicShape>;
using ShapeCacheKey = std::tuple<c10::OperatorName, CanonicalArgVec>;

CanonicalArgVec cannonicalizeVec(
    const std::vector<SSAInput>& arg_vec,
    std::unordered_map<int64_t, int64_t>& ss_map,
    bool deep_copy = true) {
  CanonicalArgVec canonical_args;
  canonical_args.reserve(arg_vec.size());
  for (auto& arg : arg_vec) {
    if (const IValue* iv = std::get_if<IValue>(&arg)) {
      if (deep_copy) {
        canonical_args.emplace_back(iv->deepcopy());
      } else {
        canonical_args.emplace_back(*iv);
      }
    } else {
      auto& ss = std::get<at::SymbolicShape>(arg);
      canonical_args.emplace_back(CanonicalizedSymbolicShape(ss, ss_map));
    }
  }
  return canonical_args;
}

std::vector<CanonicalizedSymbolicShape> cannonicalizeVec(
    const std::vector<at::SymbolicShape>& ret_vec,
    std::unordered_map<int64_t, int64_t>& ss_map) {
  std::vector<CanonicalizedSymbolicShape> canonical_rets;
  canonical_rets.reserve(ret_vec.size());
  for (auto& ss : ret_vec) {
    canonical_rets.emplace_back(ss, ss_map);
  }
  return canonical_rets;
}

struct ArgumentsHasher {
  size_t operator()(const ShapeCacheKey& cacheKey) const {
    // TODO: ignore arguments that are not used in shape function (not needed
    // initially)
    auto& op_name = std::get<0>(cacheKey);
    auto& arg_vec = std::get<1>(cacheKey);

    size_t hash_val = c10::hash<c10::OperatorName>()(op_name);

    hash_val = at::hash_combine(std::hash<size_t>{}(arg_vec.size()), hash_val);
    for (const CanonicalArg& arg : arg_vec) {
      size_t cur_arg = 0;
      if (const IValue* ival = std::get_if<IValue>(&arg)) {
        // IValue doesn't hash List (as Python doesn't), so we will do a custom
        // list hash
        if (ival->isList()) {
          TORCH_INTERNAL_ASSERT(ival->isIntList(), "Unexpected Args in List");
          cur_arg = ival->toListRef().size();
          for (const IValue& elem_ival : ival->toListRef()) {
            cur_arg = at::hash_combine(cur_arg, IValue::hash(elem_ival));
          }
        } else {
          cur_arg = IValue::hash(ival);
        }
      } else {
        cur_arg = std::get<CanonicalizedSymbolicShape>(arg).hash();
      }
      hash_val = at::hash_combine(hash_val, cur_arg);
    }
    return hash_val;
  }
};

// LRU cache keyed by op name plus canonicalized arguments. The map keys on a
// pointer into the list so the (potentially large) key is stored only once.
class ShapeCache {
 public:
  explicit ShapeCache(size_t max_size) : max_size_(max_size) {}

  void Add(ShapeCacheKey key, std::shared_ptr<CanonicalRet> value) {
    std::lock_guard<std::mutex> g(lock_);
    element_list_.emplace_front(std::move(key), std::move(value));
    auto it = element_list_.begin();
    auto [pos, inserted] = element_map_.emplace(&it->first, it);
    if (!inserted) {
      element_list_.erase(it);
      element_list_.splice(element_list_.begin(), element_list_, pos->second);
    } else if (element_list_.size() > max_size_) {
      element_map_.erase(&element_list_.back().first);
      element_list_.pop_back();
    }
  }

  std::shared_ptr<CanonicalRet> Get(const ShapeCacheKey& key) {
    std::lock_guard<std::mutex> g(lock_);
    auto it = element_map_.find(&key);
    if (it == element_map_.end()) {
      return nullptr;
    }
    element_list_.splice(element_list_.begin(), element_list_, it->second);
    return it->second->second;
  }

  void Clear() {
    std::lock_guard<std::mutex> g(lock_);
    element_map_.clear();
    element_list_.clear();
  }

  size_t Numel() const {
    std::lock_guard<std::mutex> g(lock_);
    return element_map_.size();
  }

 private:
  using Element = std::pair<ShapeCacheKey, std::shared_ptr<CanonicalRet>>;
  using ElementList = std::list<Element>;

  struct KeyPtrHasher {
    size_t operator()(const ShapeCacheKey* key) const {
      return ArgumentsHasher{}(*key);
    }
  };

  struct KeyPtrEqual {
    bool operator()(const ShapeCacheKey* a, const ShapeCacheKey* b) const {
      return *a == *b;
    }
  };

  mutable std::mutex lock_;
  size_t max_size_;
  ElementList element_list_;
  std::unordered_map<
      const ShapeCacheKey*,
      ElementList::iterator,
      KeyPtrHasher,
      KeyPtrEqual>
      element_map_;
};

constexpr size_t kShapeCacheSize = 1024;
ShapeCache shapeCache(kShapeCacheSize);

ShapeCacheKey get_cache_key(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec,
    std::unordered_map<int64_t, int64_t>& ss_map,
    bool deep_copy = true) {
  CanonicalArgVec canonical_args = cannonicalizeVec(arg_vec, ss_map, deep_copy);
  return std::make_tuple(schema->operator_name(), std::move(canonical_args));
}

} // namespace

TORCH_API void cache_shape_function(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec,
    const std::vector<at::SymbolicShape>& ret_vec) {
  // TODO: compare perf using std::vector<std::tuple<int64_t, int64_t>>
  auto ss_map = std::unordered_map<int64_t, int64_t>();
  auto cache_key = get_cache_key(schema, arg_vec, ss_map, /* deep_copy */ true);
  auto can_ret_vec = std::make_shared<std::vector<CanonicalizedSymbolicShape>>(
      cannonicalizeVec(ret_vec, ss_map));
  shapeCache.Add(std::move(cache_key), std::move(can_ret_vec));
}

TORCH_API std::optional<std::vector<at::SymbolicShape>>
get_cached_shape_function(
    const FunctionSchema* schema,
    const std::vector<SSAInput>& arg_vec) {
  // TODO: compare perf using std::vector<std::tuple<int64_t, int64_t>> for both
  // ss_map and inverse_ss_map
  auto ss_map = std::unordered_map<int64_t, int64_t>();
  auto cache_key =
      get_cache_key(schema, arg_vec, ss_map, /* deep_copy */ false);
  auto cached_ret_vec = shapeCache.Get(cache_key);
  if (cached_ret_vec == nullptr) {
    return std::nullopt;
  }
  // Decanonicalize the return values
  auto inverse_ss_map = std::unordered_map<int64_t, int64_t>();
  for (auto& ss_val : ss_map) {
    inverse_ss_map[ss_val.second] = ss_val.first;
  }
  std::vector<at::SymbolicShape> ret_vec;
  for (auto& css : *cached_ret_vec) {
    ret_vec.emplace_back(css.toSymbolicShape(inverse_ss_map));
  }
  return ret_vec;
}

// Function only to access the cache, used for testing
TORCH_API void clear_shape_cache() {
  shapeCache.Clear();
}

TORCH_API size_t get_shape_cache_size() {
  return shapeCache.Numel();
}

void CanonicalizedSymbolicShape::init(
    const c10::SymbolicShape& orig_shape,
    std::unordered_map<int64_t, int64_t>& ss_map) {
  auto sizes = orig_shape.sizes();
  if (!sizes) {
    values_ = std::nullopt;
    return;
  }
  values_ = std::vector<int64_t>();
  int64_t cur_symbolic_index = -static_cast<int64_t>(ss_map.size()) - 1;
  for (auto& cur_shape : *sizes) {
    if (cur_shape.is_static()) {
      values_->push_back(cur_shape.static_size());
    } else {
      // Check for aliasing
      auto it = ss_map.find(cur_shape.value());

      if (it == ss_map.end()) {
        values_->push_back(cur_symbolic_index);
        ss_map.insert({cur_shape.value(), cur_symbolic_index});
        cur_symbolic_index--;
      } else {
        values_->push_back(it->second);
      }
    }
  }
}

c10::SymbolicShape CanonicalizedSymbolicShape::toSymbolicShape(
    std::unordered_map<int64_t, int64_t>& inverse_ss_map) const {
  if (!values_.has_value()) {
    return c10::SymbolicShape();
  }
  std::vector<at::ShapeSymbol> sizes;
  for (long long cur_val : *values_) {
    if (cur_val >= 0) {
      sizes.push_back(at::ShapeSymbol::fromStaticSize(cur_val));
      continue;
    }
    auto res = inverse_ss_map.find(cur_val);
    if (res != inverse_ss_map.end()) {
      sizes.push_back(at::ShapeSymbol::fromStaticSize(res->second));
    } else {
      auto new_symbol = at::ShapeSymbol::newSymbol();
      inverse_ss_map.insert({cur_val, new_symbol.value()});
      sizes.push_back(new_symbol);
    }
  }
  return c10::SymbolicShape(std::move(sizes));
}

size_t CanonicalizedSymbolicShape::hash() const {
  if (!values_.has_value()) {
    return 0x8cc80c80; // random value to prevent hash collisions
  }
  return c10::hash<std::vector<int64_t>>()(values_.value());
}

bool operator==(
    const CanonicalizedSymbolicShape& a,
    const CanonicalizedSymbolicShape& b) {
  return a.values_ == b.values_;
}
} // namespace torch::jit
