#pragma once

/**
 * This file provides portable macros for marking declarations
 * as deprecated.  You should generally use C10_DEPRECATED,
 * except when marking 'using' declarations as deprecated,
 * in which case you should use C10_DEFINE_DEPRECATED_USING
 * (due to portability concerns).
 */

// Sample usage:
//
//    C10_DEPRECATED void bad_func();
//    struct C10_DEPRECATED BadStruct {
//      ...
//    };

// NB: __cplusplus doesn't work for MSVC, so for now MSVC always uses
// the "__declspec(deprecated)" implementation and not the C++14
// "[[deprecated]]" attribute. We tried enabling "[[deprecated]]" for C++14 on
// MSVC, but ran into issues with some older MSVC versions.
#if (defined(__cplusplus) && __cplusplus >= 201402L)
#define C10_DEPRECATED [[deprecated]]
#define C10_DEPRECATED_MESSAGE(message) [[deprecated(message)]]
#elif defined(__GNUC__)
#define C10_DEPRECATED __attribute__((deprecated))
// TODO Is there some way to implement this?
#define C10_DEPRECATED_MESSAGE(message) __attribute__((deprecated))

#else
#warning "You need to implement C10_DEPRECATED for this compiler"
#define C10_DEPRECATED
#endif

// Sample usage:
//
//    C10_DEFINE_DEPRECATED_USING(BadType, int)
//
//   which is the portable version of
//
//    using BadType [[deprecated]] = int;

// technically [[deprecated]] syntax is from c++14 standard, but it works in
// many compilers.
#if defined(__has_cpp_attribute)
#if __has_cpp_attribute(deprecated) && !defined(__CUDACC__)
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName [[deprecated]] = TypeThingy;
#endif
#endif


#if !defined(C10_DEFINE_DEPRECATED_USING) && defined(__GNUC__)
// nvcc has a bug where it doesn't understand __attribute__((deprecated))
// declarations even when the host compiler supports it. We'll only use this gcc
// attribute when not cuda, and when using a GCC compiler that doesn't support
// the c++14 syntax we checked for above (available in __GNUC__ >= 5)
#if !defined(__CUDACC__)
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName __attribute__((deprecated)) = TypeThingy;
#else
// using cuda + gcc < 5, neither deprecated syntax is available so turning off.
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName = TypeThingy;
#endif
#endif

#if !defined(C10_DEFINE_DEPRECATED_USING)
#warning "You need to implement C10_DEFINE_DEPRECATED_USING for this compiler"
#define C10_DEFINE_DEPRECATED_USING
#endif
