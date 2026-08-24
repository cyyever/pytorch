#pragma once

#include <ATen/core/class_type.h>
#include <ATen/core/functional.h>
#include <ATen/core/ivalue.h>
#include <sstream>
#include <utility>

namespace torch::jit {

using ObjectPtr = c10::intrusive_ptr<c10::ivalue::Object>;

// Throw this in C++ land if `attr` fails. This will be converted to a Python
// AttributeError by the Python binding code
class ObjectAttributeError : public std::runtime_error {
 public:
  ObjectAttributeError(const std::string& what) : std::runtime_error(what) {}
};

// Thin owning wrapper around c10::ivalue::Object, used by torchbind
// (torch::class_) custom classes.
struct TORCH_API Object {
  Object() = default;
  Object(const Object&) = default;
  Object& operator=(const Object&) = default;
  Object(Object&&) noexcept = default;
  Object& operator=(Object&&) noexcept = default;
  Object(ObjectPtr _ivalue) : _ivalue_(std::move(_ivalue)) {}

  ObjectPtr _ivalue() const {
    TORCH_INTERNAL_ASSERT(_ivalue_);
    return _ivalue_;
  }

  c10::ClassTypePtr type() const {
    return _ivalue()->type();
  }

  void setattr(const std::string& name, c10::IValue v) {
    if (_ivalue()->type()->hasConstant(name)) {
      TORCH_CHECK(
          false,
          "Can't set constant '",
          name,
          "' which has value:",
          _ivalue()->type()->getConstant(name));
    } else if (auto slot = _ivalue()->type()->findAttributeSlot(name)) {
      const c10::TypePtr& expected = _ivalue()->type()->getAttribute(*slot);
      TORCH_CHECK(
          v.type()->isSubtypeOf(*expected),
          "Expected a value of type '",
          expected->repr_str(),
          "' for field '",
          name,
          "', but found '",
          v.type()->repr_str(),
          "'");
      _ivalue()->setSlot(*slot, std::move(v));
    } else {
      TORCH_CHECK(false, "Module has no attribute '", name, "'");
    }
  }

  c10::IValue attr(const std::string& name) const {
    if (auto r = _ivalue()->type()->findAttributeSlot(name)) {
      return _ivalue()->getSlot(*r);
    }
    if (auto r = _ivalue()->type()->findConstantSlot(name)) {
      return _ivalue()->type()->getConstant(*r);
    }
    std::stringstream err;
    err << _ivalue()->type()->repr_str() << " does not have a field with name '"
        << name.c_str() << '\'';
    throw ObjectAttributeError(std::move(err).str());
  }

  c10::IValue attr(const std::string& name, c10::IValue or_else) const {
    if (auto r = _ivalue()->type()->findAttributeSlot(name)) {
      return _ivalue()->getSlot(*r);
    }
    if (auto r = _ivalue()->type()->findConstantSlot(name)) {
      return _ivalue()->type()->getConstant(*r);
    }
    return or_else;
  }

  bool hasattr(const std::string& name) const {
    return _ivalue()->type()->hasAttribute(name) ||
        _ivalue()->type()->hasConstant(name);
  }

  size_t num_slots() const {
    return _ivalue()->slots().size();
  }

 private:
  mutable ObjectPtr _ivalue_;
};

} // namespace torch::jit
