#pragma once

#include <c10/macros/Macros.h>

/**
 * c10::ThreadLocal<T> wraps a thread_local value with smart pointer semantics.
 *
 * Convenient macro C10_DEFINE_TLS_static is available.
 * To define static TLS variable of type std::string, do the following
 * ```
 *  C10_DEFINE_TLS_static(std::string, str_tls_);
 *  ///////
 *  {
 *    *str_tls_ = "abc";
 *    assert(str_tls_->length(), 3);
 *  }
 * ```
 *
 * (see c10/test/util/ThreadLocal_test.cpp for more examples)
 */
namespace c10 {

/**
 * @brief Default thread_local implementation.
 * To be used with composite types that provide default ctor.
 */
template <typename Type>
class ThreadLocal {
 public:
  using Accessor = Type* (*)();
  explicit ThreadLocal(Accessor accessor) : accessor_(accessor) {}

  ThreadLocal(const ThreadLocal&) = delete;
  ThreadLocal(ThreadLocal&&) noexcept = default;
  ThreadLocal& operator=(const ThreadLocal&) = delete;
  ThreadLocal& operator=(ThreadLocal&&) noexcept = default;
  ~ThreadLocal() = default;

  Type& get() {
    return *accessor_();
  }

  Type& operator*() {
    return get();
  }

  Type* operator->() {
    return &get();
  }

 private:
  Accessor accessor_;
};

} // namespace c10

#define C10_DEFINE_TLS_static(Type, Name)     \
  static ::c10::ThreadLocal<Type> Name([]() { \
    static thread_local Type var;             \
    return &var;                              \
  })

#define C10_DECLARE_TLS_class_static(Class, Type, Name) \
  static ::c10::ThreadLocal<Type> Name

#define C10_DEFINE_TLS_class_static(Class, Type, Name) \
  ::c10::ThreadLocal<Type> Class::Name([]() {          \
    static thread_local Type var;                      \
    return &var;                                       \
  })
