#pragma once

#include <ATen/core/ivalue.h>

namespace torch::jit {

// Coordinates sharing of storages between models during deserialization.
class TORCH_API DeserializationStorageContext {
 public:
  explicit DeserializationStorageContext() = default;
  DeserializationStorageContext operator=(
      const DeserializationStorageContext&) = delete;
  DeserializationStorageContext(const DeserializationStorageContext&) = delete;

  void addStorage(std::string name, c10::Storage storage) {
    TORCH_INTERNAL_ASSERT(!hasStorage(name));
    name_storage_map_.emplace(std::move(name), std::move(storage));
  }

  bool hasStorage(const std::string& name) {
    return name_storage_map_.contains(name);
  }

  c10::Storage getStorage(const std::string& name) {
    TORCH_INTERNAL_ASSERT(hasStorage(name));
    return name_storage_map_.find(name)->second;
  }
  ~DeserializationStorageContext() = default;

 private:
  std::unordered_map<std::string, c10::Storage> name_storage_map_;
};

} // namespace torch::jit
