#pragma once

#include <c10/core/SymBool.h>
#include <c10/core/SymFloat.h>
#include <c10/core/SymNodeImpl.h>
#include <c10/macros/Export.h>
#include <c10/macros/Macros.h>
#include <c10/util/Exception.h>

#include <algorithm>
#include <concepts>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <optional>
#include <ostream>
#include <type_traits>

namespace c10 {

class SymInt;

namespace detail {

// Scalars reach SymInt arithmetic in whatever type the caller happened to have.
// Constrain on the two categories instead of listing them: everything that
// converts to int64_t goes through SymInt, floating point through SymFloat.
// Unscoped enums are in the integral set because the by-type overloads this
// replaces accepted them through promotion; scoped enums do not convert and
// were rejected then too.
// SymInt holds an int64_t and SymFloat a double, so operands wider than those
// are excluded: they would truncate inside the constructor with nothing said,
// where the by-type overloads this replaces made them a hard error. Both cases
// are reachable -- __int128 satisfies std::integral under the gnu dialects this
// builds with, and long double satisfies std::floating_point.
template <typename T>
concept sym_int_operand =
    (std::integral<T> || (std::is_enum_v<T> && std::convertible_to<T, int64_t>)) &&
    sizeof(T) <= sizeof(int64_t);

template <typename T>
concept sym_operand =
    sym_int_operand<T> || (std::floating_point<T> && sizeof(T) <= sizeof(double));

// Too wide to hold. These are not simply left out of sym_operand: SymInt's
// implicit constructor would then take them through the SymInt member
// operators instead, truncating with nothing said. The operators below delete
// them so the call is an error, which is what the by-type overloads this
// replaces produced -- by accident, through ambiguity, but produced.
template <typename T>
concept sym_too_wide = (std::integral<T> && sizeof(T) > sizeof(int64_t)) ||
    (std::floating_point<T> && sizeof(T) > sizeof(double));

template <typename T>
using sym_result_t = std::conditional_t<std::is_floating_point_v<T>, SymFloat, SymInt>;

// Returns a reference in the identity case, so that reaching the SymInt
// overload of an operator does not bump a refcount. Defined below, where
// SymInt is complete.
template <typename R>
decltype(auto) sym_promote(const SymInt& a);

} // namespace detail

// SymInt represents either a regular int64_t, or a symbolic integer
// (represented in a type erased way as SymNode).  The intention is for SymInt
// to represent symbolic sizes that arise when doing shape computation in
// operator kernels. This allows for tracing through programs without baking in
// concrete sizes into kernel calls.
//
// SymInt has an API equivalent to int64_t.  In particular, it is a value type.
// Internally, SymInt is represented in a clever packed way, so that it only
// occupies one word of space; but morally, it is a union between an int64_t
// and an intrusive pointer to SymNodeImpl.
//
// Invariant: the referenced SymNodeImpl is guaranteed to be a SymNode where
// is_int() returns true

class C10_API SymInt {
 public:
  enum Unchecked {
    UNCHECKED,
  };

  /*implicit*/ SymInt(int64_t d) : data_(d) {
    if (is_heap_allocated()) {
      // Large negative number, heap allocate it
      promote_to_negative();
    }
  }
  SymInt() : data_(0) {}
  SymInt(SymNode n);

  // unchecked c-tor accepting raw `data_`
  // One appropriate use for this is when you are constructing a symint
  // in a situation where you know it is non-negative (or, if it is negative,
  // the negative value is -1; i.e., not user controlled)
  SymInt(Unchecked /*unused*/, int64_t d) : data_(d) {}

  SymInt(const SymInt& s) : data_(s.data_) {
    if (s.is_heap_allocated()) {
      c10::raw::intrusive_ptr::incref(s.toSymNodeImplUnowned());
    }
  }
  SymInt(SymInt&& s) noexcept : data_(s.data_) {
    s.data_ = 0;
  }

  SymInt& operator=(const SymInt& s) {
    if (this != &s) {
      release_();
      data_ = s.data_;
      if (s.is_heap_allocated()) {
        c10::raw::intrusive_ptr::incref(s.toSymNodeImplUnowned());
      }
    }
    return *this;
  }
  SymInt& operator=(SymInt&& s) noexcept {
    if (this != &s) {
      release_(); // release the current SymNode if any
      data_ = s.data_;
      if (s.is_heap_allocated())
        s.data_ = 0;
    };
    return *this;
  }

  SymNodeImpl* toSymNodeImplUnowned() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(is_heap_allocated());
    uint64_t unextended_bits = static_cast<uint64_t>(data_) & ~MASK;
    uint64_t sign_bit_mask = 1ULL << (62 - 1);
    // https://stackoverflow.com/questions/42534749/signed-extension-from-24-bit-to-32-bit-in-c
    uint64_t extended_bits = (unextended_bits ^ sign_bit_mask) - sign_bit_mask;
    return static_cast<SymNodeImpl*>(
        // NOLINTNEXTLINE(performance-no-int-to-ptr, bugprone*)
        reinterpret_cast<void*>(static_cast<uintptr_t>(extended_bits)));
  }

  void release_() {
    if (is_heap_allocated()) {
      SymNode::reclaim(toSymNodeImplUnowned()); // steal
    }
  }

  SymNodeImpl* release() && {
    TORCH_INTERNAL_ASSERT(is_heap_allocated());
    auto* r = toSymNodeImplUnowned();
    data_ = 0; // transfer ownership
    return r;
  }

  // Only valid if is_heap_allocated()
  SymNode toSymNode() const;

  // Guaranteed to return a SymNode, wrapping using base if necessary
  SymNode wrap_node(const SymNode& base) const;

  ~SymInt() {
    release_();
  }

  // Require the int to be non-symbolic, and if it is symbolic raise an
  // error.  This is safe to use for C++ code that doesn't work for symbolic
  // shapes, and you don't have time to fix it immediately, as if we
  // try to trigger the path in C++ you'll appropriately get an error
  int64_t expect_int() const {
    if (auto r = maybe_as_int()) {
      return *r;
    }
    TORCH_CHECK_ALWAYS_SHOW_CPP_STACKTRACE(
        false, "when unpacking SymInt, expected int but got ", *this);
  }

  // Test if we have a hint for this int (e.g., guard_int would work).
  // Most of the time this is true; it is only false when you have
  // an unbacked SymInt.
  bool has_hint() const;

  // Insert a guard for the int to be its concrete value, and then return
  // that value.  This operation always works, even if the int is symbolic,
  // so long as we know what the underlying value is (e.g., this won't work
  // if you call it on the size of nonzero output).  Don't blindly put this
  // everywhere; you can cause overspecialization of PyTorch programs with
  // this method.
  //
  // It should be called as guard_int(__FILE__, __LINE__).  The file and line
  // number can be used to diagnose overspecialization.
  int64_t guard_int(const char* file, int64_t line) const;

  // Distinguish actual symbolic values from constants stored on the heap
  bool is_symbolic() const {
    return is_heap_allocated() &&
        !toSymNodeImplUnowned()->constant_int().has_value();
  }

  C10_ALWAYS_INLINE bool is_heap_allocated() const {
    return !check_range(data_);
  }

  SymInt operator+(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymInt(*ma + *mb);
      }
    }
    return operator_add_slow_path(sci);
  }

  SymInt operator-(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymInt(*ma - *mb);
      }
    }
    return operator_sub_slow_path(sci);
  }

  SymInt operator*(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymInt(*ma * *mb);
      }
    }
    return operator_mul_slow_path(sci);
  }

  SymInt operator/(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymInt(*ma / *mb);
      }
    }
    return operator_div_slow_path(sci);
  }

  SymInt operator%(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymInt(*ma % *mb);
      }
    }
    return operator_mod_slow_path(sci);
  }

  void operator*=(const SymInt& sci) {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        *this = SymInt(*ma * *mb);
        return;
      }
    }
    operator_imul_slow_path(sci);
  }

  void operator+=(const SymInt& sci) {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        *this = SymInt(*ma + *mb);
        return;
      }
    }
    operator_iadd_slow_path(sci);
  }

  void operator/=(const SymInt& sci) {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        *this = SymInt(*ma / *mb);
        return;
      }
    }
    operator_idiv_slow_path(sci);
  }

  SymInt clone() const;

  SymBool sym_eq(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymBool(*ma == *mb);
      }
    }
    return sym_eq_slow_path(sci);
  }

  SymBool sym_ne(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymBool(*ma != *mb);
      }
    }
    return sym_ne_slow_path(sci);
  }

  SymBool sym_lt(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymBool(*ma < *mb);
      }
    }
    return sym_lt_slow_path(sci);
  }

  SymBool sym_le(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymBool(*ma <= *mb);
      }
    }
    return sym_le_slow_path(sci);
  }

  SymBool sym_gt(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymBool(*ma > *mb);
      }
    }
    return sym_gt_slow_path(sci);
  }

  SymBool sym_ge(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymBool(*ma >= *mb);
      }
    }
    return sym_ge_slow_path(sci);
  }

  bool operator==(const SymInt& o) const {
    return sym_eq(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator!=(const SymInt& o) const {
    return sym_ne(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator<(const SymInt& o) const {
    return sym_lt(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator<=(const SymInt& o) const {
    return sym_le(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator>(const SymInt& o) const {
    return sym_gt(o).guard_bool(__FILE__, __LINE__);
  }
  bool operator>=(const SymInt& o) const {
    return sym_ge(o).guard_bool(__FILE__, __LINE__);
  }

  // Mixed SymInt/scalar operators, as hidden friends so that only ADL finds
  // them. At namespace scope they would also be candidates for expressions
  // with no SymInt in them: in `int == some_unscoped_enum` the template binds
  // the enum exactly while the built-in operator only promotes it, and the int
  // reaches SymInt through the implicit constructor above, so neither candidate
  // wins and the comparison is ambiguous.
#define C10_SYMINT_FRIEND_OP(op, RetTy, Operand)          \
  template <detail::Operand T>                            \
  friend RetTy operator op(const SymInt& a, T b) {        \
    using R = detail::sym_result_t<T>;                    \
    return detail::sym_promote<R>(a) op R(b);             \
  }                                                       \
  template <detail::Operand T>                            \
  friend RetTy operator op(T a, const SymInt& b) {        \
    using R = detail::sym_result_t<T>;                    \
    return R(a) op detail::sym_promote<R>(b);             \
  }                                                       \
  template <detail::sym_too_wide T>                       \
  friend RetTy operator op(const SymInt& a, T b) = delete; \
  template <detail::sym_too_wide T>                       \
  friend RetTy operator op(T a, const SymInt& b) = delete;

  C10_SYMINT_FRIEND_OP(+, detail::sym_result_t<T>, sym_operand)
  C10_SYMINT_FRIEND_OP(-, detail::sym_result_t<T>, sym_operand)
  C10_SYMINT_FRIEND_OP(*, detail::sym_result_t<T>, sym_operand)
  C10_SYMINT_FRIEND_OP(/, detail::sym_result_t<T>, sym_operand)
  C10_SYMINT_FRIEND_OP(%, SymInt, sym_int_operand)

  C10_SYMINT_FRIEND_OP(==, bool, sym_operand)
  C10_SYMINT_FRIEND_OP(!=, bool, sym_operand)
  C10_SYMINT_FRIEND_OP(<, bool, sym_operand)
  C10_SYMINT_FRIEND_OP(<=, bool, sym_operand)
  C10_SYMINT_FRIEND_OP(>, bool, sym_operand)
  C10_SYMINT_FRIEND_OP(>=, bool, sym_operand)

#undef C10_SYMINT_FRIEND_OP

  SymInt min(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymInt(std::min(*ma, *mb));
      }
    }
    return min_slow_path(sci);
  }

  SymInt max(const SymInt& sci) const {
    if (auto ma = maybe_as_int()) {
      if (auto mb = sci.maybe_as_int()) {
        return SymInt(std::max(*ma, *mb));
      }
    }
    return max_slow_path(sci);
  }

  // If both are symbolic, this checks if
  // they share the same node.
  // If both are not symbolic this just checks normal equality.
  bool is_same(const SymInt& other) const;

  operator SymFloat() const;

  void unsafe_set_data(size_t nbytes) {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!is_heap_allocated());
    data_ = static_cast<int64_t>(nbytes);
  }

  // Don't use this.  Prefer maybe_as_int instead
  int64_t as_int_unchecked() const {
    TORCH_INTERNAL_ASSERT_DEBUG_ONLY(!is_heap_allocated());
    return data_;
  }

  std::optional<int64_t> maybe_as_int() const {
    if (!is_heap_allocated()) {
      return data_;
    }
    return maybe_as_int_slow_path();
  }

  // Return whether the integer is directly coercible to a SymInt
  // without requiring heap allocation.  You don't need to use this
  // to check if you can pass an integer to SymInt; this is guaranteed
  // to work (it just might heap allocate!)
  static bool check_range(int64_t i) {
    return i > MAX_UNREPRESENTABLE_INT;
  }

  // Return the min representable integer as a SymInt without
  // heap allocation.  For quantities that count bytes (or larger),
  // this is still much larger than you need, so you may consider
  // using this as a more efficient version of MIN_INT
  static constexpr int64_t min_representable_int() {
    return MAX_UNREPRESENTABLE_INT + 1;
  }

 private:
  void promote_to_negative();
  SymInt operator_add_slow_path(const SymInt& sci) const;
  SymInt operator_sub_slow_path(const SymInt& sci) const;
  SymInt operator_mul_slow_path(const SymInt& sci) const;
  SymInt operator_div_slow_path(const SymInt& sci) const;
  SymInt operator_mod_slow_path(const SymInt& sci) const;
  void operator_imul_slow_path(const SymInt& sci);
  void operator_iadd_slow_path(const SymInt& sci);
  void operator_idiv_slow_path(const SymInt& sci);
  SymBool sym_eq_slow_path(const SymInt& sci) const;
  SymBool sym_ne_slow_path(const SymInt& sci) const;
  SymBool sym_lt_slow_path(const SymInt& sci) const;
  SymBool sym_le_slow_path(const SymInt& sci) const;
  SymBool sym_gt_slow_path(const SymInt& sci) const;
  SymBool sym_ge_slow_path(const SymInt& sci) const;

  SymInt min_slow_path(const SymInt& sci) const;
  SymInt max_slow_path(const SymInt& sci) const;

  std::optional<int64_t> maybe_as_int_slow_path() const;

  // Constraints on the internal representation:
  //
  // - Should represent positive and small negative ints
  // - No conversion necessary for operations on ints
  // - Must represent valid 64-bit pointers
  // - Is symbolic test should be FAST (two arithmetic instructions is too
  // much).
  //   This code being a hotpath is based on Strobelight profiles of
  //   is_heap_allocated().  FB only: https://fburl.com/strobelight/5l50ncxd
  //   (you will need to change the time window).
  //
  // So, the scheme is to reserve large negative numbers (assuming
  // two's complement):
  //
  // - 0b0.... means we are a positive int
  // - 0b11... means we are a small negative int
  // - 0b10... means we are are a pointer. This means that
  //           [-2^63, -2^62-1] are not representable as ints.
  //           We don't actually need all of this space as on x86_64
  //           as the top 16bits aren't used for anything
  static constexpr uint64_t MASK = 1ULL << 63 | 1ULL << 62 | 1ULL << 61;
  static constexpr uint64_t IS_SYM = 1ULL << 63 | 1ULL << 61;
  // We must manually translate the bit pattern test into a greater
  // than test because compiler doesn't figure it out:
  // https://godbolt.org/z/356aferaW
  static constexpr int64_t MAX_UNREPRESENTABLE_INT =
      -1LL & static_cast<int64_t>(~(1ULL << 62));
  int64_t data_;
};

namespace detail {

template <typename R>
decltype(auto) sym_promote(const SymInt& a) {
  if constexpr (std::is_same_v<R, SymInt>) {
    return (a);
  } else {
    return SymFloat(a);
  }
}

} // namespace detail

/// Sum of a list of SymInt; accumulates into the c10::SymInt expression
template <typename C>
  requires std::is_same_v<typename C::value_type, c10::SymInt>
inline c10::SymInt multiply_integers(const C& container) {
  return std::accumulate(
      container.begin(),
      container.end(),
      c10::SymInt(1),
      [](const c10::SymInt& a, const c10::SymInt& b) { return a * b; });
}

template <typename Iter>
  requires std::is_same_v<
      typename std::iterator_traits<Iter>::value_type,
      c10::SymInt>
inline c10::SymInt multiply_integers(Iter begin, Iter end) {
  return std::accumulate(
      begin,
      end,
      c10::SymInt(1),
      [](const c10::SymInt& a, const c10::SymInt& b) { return a * b; });
}


C10_API std::ostream& operator<<(std::ostream& os, const SymInt& s);
C10_API SymInt operator-(const SymInt& s);

inline bool sym_eq(int64_t a, int64_t b) {
  return a == b;
}

inline SymBool sym_eq(const SymInt& a, const SymInt& b) {
  return a.sym_eq(b);
}

inline bool sym_ne(int64_t a, int64_t b) {
  return a != b;
}

inline SymBool sym_ne(const SymInt& a, const SymInt& b) {
  return a.sym_ne(b);
}

inline bool sym_lt(int64_t a, int64_t b) {
  return a < b;
}

inline SymBool sym_lt(const SymInt& a, const SymInt& b) {
  return a.sym_lt(b);
}

inline bool sym_le(int64_t a, int64_t b) {
  return a <= b;
}

inline SymBool sym_le(const SymInt& a, const SymInt& b) {
  return a.sym_le(b);
}

inline bool sym_gt(int64_t a, int64_t b) {
  return a > b;
}

inline SymBool sym_gt(const SymInt& a, const SymInt& b) {
  return a.sym_gt(b);
}

inline bool sym_ge(int64_t a, int64_t b) {
  return a >= b;
}

inline SymBool sym_ge(const SymInt& a, const SymInt& b) {
  return a.sym_ge(b);
}

} // namespace c10

#include <limits>

namespace std {

template <>
class numeric_limits<c10::SymInt> {
 public:
  static constexpr bool is_specialized = true;

  static constexpr int64_t max() noexcept {
    return std::numeric_limits<int64_t>::max();
  }

  static constexpr int64_t min() noexcept {
    return std::numeric_limits<int64_t>::min();
  }

  static constexpr bool is_signed = true;
  static constexpr bool is_integer = true;
};

} // namespace std
