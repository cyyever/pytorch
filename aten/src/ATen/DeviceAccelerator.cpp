#include <ATen/Context.h>
#include <ATen/core/CachingHostAllocator.h>
#include <ATen/DeviceAccelerator.h>
#include <c10/core/impl/VirtualGuardImpl.h>

namespace at::accelerator {

std::optional<c10::DeviceType> getAccelerator(bool checked) {
  // Check the compile-time backends. They are mutually exclusive, so finding
  // two of them built into one binary is a build error, not a choice to make.
  std::optional<c10::DeviceType> device_type = std::nullopt;

#define DETECT_AND_ASSIGN_ACCELERATOR_COMP(device_name) \
  if (at::detail::get##device_name##Hooks().isBuilt()) {  \
    TORCH_CHECK(                                         \
        !device_type.has_value(),                        \
        "Cannot have both " #device_name " and ",             \
        device_type.value(), ".");                       \
    device_type = k##device_name;                        \
  }

  DETECT_AND_ASSIGN_ACCELERATOR_COMP(CUDA)
  DETECT_AND_ASSIGN_ACCELERATOR_COMP(XPU)
  DETECT_AND_ASSIGN_ACCELERATOR_COMP(HIP)
  DETECT_AND_ASSIGN_ACCELERATOR_COMP(MPS)

  // A PrivateUse1 backend registers itself at run time rather than being built
  // in, so it is asked for last: an out-of-tree backend loaded next to a
  // built-in accelerator does not displace it. The check is on hook
  // registration alone and so initializes no device and poisons no fork.
  if (!device_type.has_value() && is_privateuse1_backend_registered()) {
    device_type = kPrivateUse1;
  }

  if (checked) {
    TORCH_CHECK(
        device_type, "Cannot access accelerator device when none is available.")
  }
  return device_type;

#undef DETECT_AND_ASSIGN_ACCELERATOR_COMP
}

bool isAccelerator(c10::DeviceType device_type) {
  switch (device_type) {
    case at::kCUDA:
    case at::kXPU:
    case at::kHIP:
    case at::kMPS:
    case at::kPrivateUse1:
      return true;
    default:
      return false;
  }
}

// NOLINTBEGIN(bugprone-unchecked-optional-access)
c10::DeviceIndex deviceCount() {
  const auto device_type = getAccelerator(false);
  if (!device_type.has_value()) {
    return static_cast<c10::DeviceIndex>(0);
  }
  c10::impl::VirtualGuardImpl impl(device_type.value());
  return impl.deviceCount();
}

void setDeviceIndex(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  impl.setDevice({device_type, device_index});
}

c10::DeviceIndex getDeviceIndex() {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.getDevice().index();
}

void setCurrentStream(c10::Stream stream) {
  const auto device_type = getAccelerator(true).value();
  TORCH_CHECK(
      device_type == stream.device_type(),
      "stream's device type ",
      c10::DeviceTypeName(stream.device_type()),
      " doesn't match the current accelerator ",
      c10::DeviceTypeName(device_type));
  c10::impl::VirtualGuardImpl impl(device_type);
  impl.exchangeStream(stream);
}

c10::Stream getCurrentStream(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.getStream({device_type, device_index});
}

void synchronizeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  // impl.synchronizeDevice should can be safely called from any device
  impl.synchronizeDevice(device_index);
}

c10::DeviceIndex exchangeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.exchangeDevice({device_type, device_index}).index();
}

c10::DeviceIndex maybeExchangeDevice(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  // Avoid creating a new context if the context for the given device_index
  // is not initialized.
  impl.uncheckedSetDevice({device_type, device_index});
  return impl.getDevice().index();
}

c10::DeviceCapability getDeviceCapability(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  c10::impl::VirtualGuardImpl impl(device_type);
  return impl.getDeviceCapability({device_type, device_index});
}

void emptyHostCache() {
  const auto device_type = getAccelerator(true).value();
  at::getHostAllocator(device_type)->empty_cache();
}

const at::Generator& getDefaultGenerator(c10::DeviceIndex device_index) {
  const auto device_type = getAccelerator(true).value();
  return at::globalContext()
      .getAcceleratorHooksInterface(device_type)
      .getDefaultGenerator(device_index);
}
// NOLINTEND(bugprone-unchecked-optional-access)

} // namespace at::accelerator
