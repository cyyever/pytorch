//  Copyright © 2022 Apple Inc.

#include <ATen/mps/MPSAllocatorInterface.h>
#include <ATen/mps/MPSDevice.h>
#include <ATen/mps/MPSStream.h>
#include <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

namespace at::mps {

namespace {
bool is_os_version_at_least(int major, int minor) {
  @autoreleasepool {
    NSProcessInfo* processInfo = [[NSProcessInfo new] autorelease];
    return
        [processInfo isOperatingSystemAtLeastVersion:{.majorVersion = major, .minorVersion = minor, .patchVersion = 0}];
  }
}
} // namespace

MPSDevice* MPSDevice::getInstance() {
  static MPSDevice mps_device;
  return &mps_device;
}

MPSDevice::~MPSDevice() {
  [_mtl_device release];
  _mtl_device = nil;
}

MPSDevice::MPSDevice() : _mtl_device(nil) {
  // macOS 27.0 is the minimum supported version: the shaders are built for
  // Metal 4 against a 27.0 deployment target, and every workaround the backend
  // carried for older releases has been dropped, so refuse to bring the device
  // up below that.
  if (!is_os_version_at_least(27, 0)) {
    TORCH_WARN("MPS backend requires macOS 27.0 or newer, disabling it");
    return;
  }

  NSArray* devices = [MTLCopyAllDevices() autorelease];
  for (unsigned long i = 0; i < [devices count]; i++) {
    id<MTLDevice> device = devices[i];
    // is_apple_family_or_newer cannot be used here: it reaches through the
    // singleton that this constructor is still building.
    if (![device supportsFamily:static_cast<MTLGPUFamily>(AppleGPUFamily::APPLE_8_PLUS)]) {
      // M1 (Apple7) and older are not supported, and neither is anything that
      // reports no Apple family at all, such as a virtualised device.
      TORCH_WARN("Skipping device ", [[device name] UTF8String], " that is older than Apple8 (M2)");
      continue;
    }
    _mtl_device = [device retain];
    break;
  }
  TORCH_INTERNAL_ASSERT_DEBUG_ONLY(_mtl_device);
}

std::string MPSDevice::getName() const {
  @autoreleasepool {
    return [[_mtl_device name] UTF8String];
  }
}

unsigned MPSDevice::getCoreCount() const {
  io_iterator_t iterator = 0;
  io_registry_entry_t entry = 0;
  int core_count = 0;
  auto matchingDict = IOServiceMatching("AGXAccelerator");
  TORCH_INTERNAL_ASSERT(matchingDict, "Failed to create matching dict");
  const auto status = IOServiceGetMatchingServices(kIOMainPortDefault, matchingDict, &iterator);
  TORCH_INTERNAL_ASSERT(status == KERN_SUCCESS);
  while ((entry = IOIteratorNext(iterator)) != 0) {
    auto property = IORegistryEntryCreateCFProperty(entry, CFSTR("gpu-core-count"), kCFAllocatorDefault, 0);
    auto found = CFNumberGetValue(static_cast<CFNumberRef>(property), kCFNumberIntType, &core_count);
    CFRelease(property);
    IOObjectRelease(entry);
    if (found) {
      break;
    }
  }
  IOObjectRelease(iterator);
  return core_count;
}

at::Allocator* GetMPSAllocator() {
  return getIMPSAllocator();
}
bool is_available() {
  return MPSDevice::getInstance()->device() != nil;
}

bool is_apple_family_or_newer(AppleGPUFamily family) {
  // some ops which are on MPSGraph behave differently between GPU families
  auto mtl_family = static_cast<MTLGPUFamily>(family);
  return [MPSDevice::getInstance()->device() supportsFamily:mtl_family];
}

bool has_mpp() {
  // MetalPerformancePrimitives matmul2d (cooperative tensors) needs macOS
  // 26.2+, which is below this backend's floor.
  return true;
}

} // namespace at::mps
