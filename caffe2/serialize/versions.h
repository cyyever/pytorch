#pragma once
#include <cstdint>


namespace caffe2::serialize {

constexpr uint64_t kMinSupportedFileFormatVersion = 0x1L;

constexpr uint64_t kMaxSupportedFileFormatVersion = 0xAL;

// Versions (i.e. why was the version number bumped?)

// Note [Dynamic Versions and torch.jit.save vs. torch.save]
//
// Our versioning scheme has a "produced file format version" which
// describes how an archive is to be read. The version written in an archive
// is at least this current produced file format version, but may be greater
// if it includes certain symbols. We refer to these conditional versions
// as "dynamic," since they are identified at runtime.
//
// Dynamic versioning is useful when an operator's semantics are updated.
// When using torch.jit.save we want those semantics to be preserved. If
// we bumped the produced file format version on every change, however,
// then older versions of PyTorch couldn't read even simple archives, like
// a single tensor, from newer versions of PyTorch. Instead, we
// assign dynamic versions to these changes that override the
// produced file format version as needed. That is, when the semantics
// of torch.div changed it was assigned dynamic version 4, and when
// torch.jit.saving modules that use torch.div those archives also have
// (at least) version 4. This prevents earlier versions of PyTorch
// from accidentally performing the wrong kind of division. Modules
// that don't use torch.div or other operators with dynamic versions
// can write the produced file format version, and these programs will
// run as expected on earlier versions of PyTorch.
//
// While torch.jit.save attempts to preserve operator semantics,
// torch.save does not. torch.save is analogous to pickling Python, so
// a function that uses torch.div will have different behavior if torch.saved
// and torch.loaded across PyTorch versions. From a technical perspective,
// torch.save ignores dynamic versioning.

// 1. Initial version
// 2. Removed op_version_set version numbers
// 3. Added type tags to pickle serialization of container types
// 4. (Dynamic) Stopped integer division using torch.div
//      (a versioned symbol preserves the historic behavior of versions 1--3)
// 5. (Dynamic) Stops torch.full inferring a floating point dtype
//      when given bool or integer fill values.
// 6. Write version string to `./data/version` instead of `version`.

// [12/15/2021]
// kProducedFileFormatVersion is set to 7 from 3 due to a different
// interpretation of what file format version is.
// Whenever there is new upgrader introduced,
// this number should be bumped.
// The reasons that version is bumped in the past:
//     1. aten::div is changed at version 4
//     2. aten::full is changed at version 5
//     3. torch.package uses version 6
//     4. Introduce new upgrader design and set the version number to 7
//        mark this change
// --------------------------------------------------
// We describe new operator version bump reasons here:
// 1) [01/24/2022]
//     We bump the version number to 8 to update aten::linspace
//     and aten::linspace.out to error out when steps is not
//     provided. (see: https://github.com/pytorch/pytorch/issues/55951)
// 2) [01/30/2022]
//     Bump the version number to 9 to update aten::logspace and
//     and aten::logspace.out to error out when steps is not
//     provided. (see: https://github.com/pytorch/pytorch/issues/55951)
// 3) [02/11/2022]
//     Bump the version number to 10 to update aten::gelu and
//     and aten::gelu.out to support the new approximate kwarg.
//     (see: https://github.com/pytorch/pytorch/pull/61439)
constexpr uint64_t kProducedFileFormatVersion = 0xAL;

// Absolute minimum version we will write packages. This
// means that every package from now on will always be
// greater than this number.
constexpr uint64_t kMinProducedFileFormatVersion = 0x3L;


} // namespace caffe2::serialize
