#ifndef C10_UTIL_FLAGS_H_
#define C10_UTIL_FLAGS_H_

/* Commandline flags support for C10.
 *
 * A lightweight commandline flags tool for c10; the gflags backend this
 * used to be able to route to is gone.
 *
 * To define a flag foo of type bool default to true, do the following in the
 * *global* namespace:
 *     C10_DEFINE_bool(foo, true, "An example.");
 *
 * To use it in another .cc file, you can use C10_DECLARE_* as follows:
 *     C10_DECLARE_bool(foo);
 *
 * In both cases, you can then access the flag via FLAGS_foo.
 *
 * Note about Python users / devs: flags are parsed from a C++ function,
 * usually in a native binary's main function. As Python does not have a
 * modifiable main function, it is usually difficult to change the flags after
 * Python starts. Hence, it is recommended that one sets the default value of
 * the flags to one that's acceptable in general - that will allow Python to
 * run without wrong flags.
 */

#include <c10/macros/Export.h>
#include <string>

#include <c10/util/Registry.h>

namespace c10 {
/**
 * Sets the usage message when a commandline tool is called with "--help".
 */
C10_API void SetUsageMessage(const std::string& str);

/**
 * Returns the usage message for the commandline tool set by SetUsageMessage.
 */
C10_API const char* UsageMessage();

/**
 * Parses the commandline flags.
 *
 * This command parses all the commandline arguments passed in via pargc
 * and argv. Once it is finished, partc and argv will contain the remaining
 * commandline args that c10 does not deal with. Note that following
 * convention, argv[0] contains the binary name and is not parsed.
 */
C10_API bool ParseCommandLineFlags(int* pargc, char*** pargv);

/**
 * Checks if the commandline flags has already been passed.
 */
C10_API bool CommandLineFlagsHasBeenParsed();

} // namespace c10

////////////////////////////////////////////////////////////////////////////////
// The macros below let one declare (C10_DECLARE) or define (C10_DEFINE) flags:
// C10_{DECLARE,DEFINE}_{int,int64,double,bool,string}
////////////////////////////////////////////////////////////////////////////////

namespace c10 {

class C10_API C10FlagParser {
 public:
  bool success() {
    return success_;
  }

 protected:
  template <typename T>
  bool Parse(const std::string& content, T* value);
  bool success_{false};
};

C10_DECLARE_REGISTRY(C10FlagsRegistry, C10FlagParser, const std::string&);

} // namespace c10

// The macros are defined outside the c10 namespace. In your code, you should
// write the C10_DEFINE_* and C10_DECLARE_* macros outside any namespace
// as well.

#define C10_DEFINE_typed_var(type, name, default_value, help_str)       \
  C10_EXPORT type FLAGS_##name = default_value;                         \
  namespace c10 {                                                       \
  namespace {                                                           \
  class C10FlagParser_##name : public C10FlagParser {                   \
   public:                                                              \
    explicit C10FlagParser_##name(const std::string& content) {         \
      success_ = C10FlagParser::Parse<type>(content, &FLAGS_##name);    \
    }                                                                   \
  };                                                                    \
  RegistererC10FlagsRegistry g_C10FlagsRegistry_##name(                 \
      #name,                                                            \
      C10FlagsRegistry(),                                               \
      RegistererC10FlagsRegistry::DefaultCreator<C10FlagParser_##name>, \
      "(" #type ", default " #default_value ") " help_str);             \
  }                                                                     \
  }

#define C10_DEFINE_int(name, default_value, help_str) \
  C10_DEFINE_typed_var(int, name, default_value, help_str)
#define C10_DEFINE_int32(name, default_value, help_str) \
  C10_DEFINE_int(name, default_value, help_str)
#define C10_DEFINE_int64(name, default_value, help_str) \
  C10_DEFINE_typed_var(int64_t, name, default_value, help_str)
#define C10_DEFINE_double(name, default_value, help_str) \
  C10_DEFINE_typed_var(double, name, default_value, help_str)
#define C10_DEFINE_bool(name, default_value, help_str) \
  C10_DEFINE_typed_var(bool, name, default_value, help_str)
#define C10_DEFINE_string(name, default_value, help_str) \
  C10_DEFINE_typed_var(std::string, name, default_value, help_str)

// DECLARE_typed_var should be used in header files and in the global namespace.
#define C10_DECLARE_typed_var(type, name) C10_API extern type FLAGS_##name

#define C10_DECLARE_int(name) C10_DECLARE_typed_var(int, name)
#define C10_DECLARE_int32(name) C10_DECLARE_int(name)
#define C10_DECLARE_int64(name) C10_DECLARE_typed_var(int64_t, name)
#define C10_DECLARE_double(name) C10_DECLARE_typed_var(double, name)
#define C10_DECLARE_bool(name) C10_DECLARE_typed_var(bool, name)
#define C10_DECLARE_string(name) C10_DECLARE_typed_var(std::string, name)

#define TORCH_DECLARE_typed_var(type, name) TORCH_API extern type FLAGS_##name

#define TORCH_DECLARE_int(name) TORCH_DECLARE_typed_var(int, name)
#define TORCH_DECLARE_int32(name) TORCH_DECLARE_int(name)
#define TORCH_DECLARE_int64(name) TORCH_DECLARE_typed_var(int64_t, name)
#define TORCH_DECLARE_double(name) TORCH_DECLARE_typed_var(double, name)
#define TORCH_DECLARE_bool(name) TORCH_DECLARE_typed_var(bool, name)
#define TORCH_DECLARE_string(name) TORCH_DECLARE_typed_var(std::string, name)

#endif // C10_UTIL_FLAGS_H_
