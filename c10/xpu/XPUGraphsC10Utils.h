#pragma once

#include <c10/xpu/XPUStream.h>
#include <iostream>

// XPU Graphs utils used by c10 and aten.
using namespace sycl::ext::oneapi::experimental;
namespace c10::xpu {

static_assert(
    int8_t(queue_state::executing) == 0,
    "unexpected int(queue_state::executing) value");
static_assert(
    int8_t(queue_state::recording) == 1,
    "unexpected int(queue_state::recording) value");

enum class CaptureStatus : int8_t {
  Executing = int8_t(queue_state::executing),
  Recording = int8_t(queue_state::recording)
};

inline std::ostream& operator<<(std::ostream& os, CaptureStatus status) {
  switch (status) {
    case CaptureStatus::Executing:
      os << "Executing";
      break;
    case CaptureStatus::Recording:
      os << "Recording";
      break;
    default:
      TORCH_INTERNAL_ASSERT(
          false, "Unknown XPU graph CaptureStatus", int(status));
  }
  return os;
}

inline CaptureStatus currentStreamCaptureStatusMayInitCtx() {
  auto state = c10::xpu::getCurrentXPUStream().queue().ext_oneapi_get_state();
  return CaptureStatus(state);
}

inline void addStreamToCurrentCaptureIfCapturing(const XPUStream& stream) {
  auto capture_stream = c10::xpu::getCurrentXPUStream(stream.device_index());
  auto& capture_queue = capture_stream.queue();
  if (capture_queue.ext_oneapi_get_state() == queue_state::recording) {
    auto graph = capture_queue.ext_oneapi_get_graph();
    graph.begin_recording(stream.queue());
  }
}

} // namespace c10::xpu
