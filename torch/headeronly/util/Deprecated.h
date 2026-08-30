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

#define C10_DEPRECATED [[deprecated]]
#define C10_DEPRECATED_MESSAGE(message) [[deprecated(message)]]

// Sample usage:
//
//    C10_DEFINE_DEPRECATED_USING(BadType, int)
//
//   which is the portable version of
//
//    using BadType [[deprecated]] = int;

// nvcc does not understand a deprecation attribute on a using-declaration, so
// the alias is defined without one there.
#if defined(__CUDACC__)
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName = TypeThingy;
#else
#define C10_DEFINE_DEPRECATED_USING(TypeName, TypeThingy) \
  using TypeName [[deprecated]] = TypeThingy;
#endif
