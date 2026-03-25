load("@rules_cc//cc:defs.bzl", "cc_library")

# This macro provides for generating both "sleef<foo>" and
# "sleefdet<foo>" libraries for a given set of code. The difference is
# that the "det" libraries get compiled with "-DDETERMINISTIC=1".

_SLEEF_INCLUDES = ["src/arch", "src/common", "src/libm"]

def sleef_cc_library(name, copts, includes = [], **kwargs):
    cc_library(
        name = name,
        copts = copts,
        includes = includes + _SLEEF_INCLUDES,
        **kwargs
    )

    prefix = "sleef"
    if not name.startswith(prefix):
        fail("name {} does not start with {}".format(repr(name), repr(prefix)))

    cc_library(
        name = name.replace(prefix, prefix + "det", 1),
        copts = copts + ["-DDETERMINISTIC=1"],
        includes = includes + _SLEEF_INCLUDES,
        **kwargs
    )
