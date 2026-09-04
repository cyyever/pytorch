#ifdef USE_C10D_GLOO
#include <torch/csrc/distributed/c10d/ProcessGroupGloo.hpp>
#include <torch/csrc/distributed/c10d/gloo/ProcessGroupGlooDetail.hpp>
#include <utility>

namespace c10d {

class AsyncAllreduceCUDAHostWork : public AsyncAllreduceWork {
 public:
  AsyncAllreduceCUDAHostWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      ReduceOp reduceOp,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : AsyncAllreduceWork(
            context,
            inputs,
            std::move(reduceOp),
            tag,
            seq,
            timeout) {
    initializeStreamsEvents(inputs, streams, events);

    // Kick off copy from CUDA tensors to pinned CPU tensors.
    tmp.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      tmp.push_back(pinnedLike(inputs[i]).copy_(inputs[i], true));
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      streams[i].synchronize();
    }

    // Run allreduce on host side tensors.
    allreduce(tmp);

    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(tmp[i], /* non_blocking */ true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmp;
  std::vector<c10::Stream> streams;
  std::vector<c10::Event> events;
};

class AsyncSparseAllreduceCUDAWork : public AsyncSparseAllreduceWork {
 public:
  AsyncSparseAllreduceCUDAWork(
      const std::shared_ptr<gloo::Context>& context,
      std::vector<at::Tensor>& inputs,
      uint32_t tag,
      uint64_t seq,
      std::chrono::milliseconds timeout)
      : AsyncSparseAllreduceWork(context, inputs, tag, seq, timeout) {
    initializeStreamsEvents(inputs, streams, events);

    // Kick off copy from CUDA tensors to CPU tensors.
    // Note that both coalescing the sparse tensor and copying it to CPU
    // memory must be performed asynchronously, or we block the caller.
    tmp.reserve(inputs.size());
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      tmp.push_back(
          inputs[i].coalesce().to(at::DeviceType::CPU, /*non_blocking=*/true));
    }
  }

  void run() override {
    // Synchronize with copy operations.
    for (const auto i : c10::irange(inputs.size())) {
      streams[i].synchronize();
    }

    // Run allreduce on host side tensors.
    auto output = allreduce(tmp);

    // Kick off copy back to the CUDA tensors.
    c10::OptionalStreamGuard guard;
    for (const auto i : c10::irange(inputs.size())) {
      guard.reset_stream(streams[i]);
      inputs[i].copy_(output, /*non_blocking=*/true);
      events[i].record(streams[i]);
    }
  }

  void synchronize() override {
    // Synchronize with the copy back to CUDA tensors.
    for (const auto i : c10::irange(inputs.size())) {
      c10::Device device = inputs[i].device();
      events[i].block(
          c10::impl::VirtualGuardImpl(device.type()).getStream(device));
    }
  }

  std::vector<at::Tensor> tmp;
  std::vector<c10::Stream> streams;
  std::vector<c10::Event> events;
};

static c10::intrusive_ptr<ProcessGroupGloo::AsyncWork> makeAllreduceCUDAWork(
    std::shared_ptr<gloo::Context> context,
    std::vector<at::Tensor>& inputs,
    ReduceOp reduceOp,
    uint32_t tag,
    uint64_t seq,
    std::chrono::milliseconds timeout) {
  auto layout = inputs[0].layout();

  if (layout == c10::kStrided) {
    // Staging through host memory. gloo's own CUDA algorithms are not built:
    // they are only reachable with GPUDirect, and CUDA collectives go through
    // NCCL here.
    return c10::make_intrusive<AsyncAllreduceCUDAHostWork>(
        std::move(context), inputs, reduceOp, tag, seq, timeout);
  } else if (layout == c10::kSparse) {
    return c10::make_intrusive<AsyncSparseAllreduceCUDAWork>(
        std::move(context), inputs, tag, seq, timeout);
  } else {
    TORCH_CHECK(false, "ProcessGroupGloo::allreduce: unsupported layout");
  }
}

C10_REGISTER_TYPED_CREATOR(
    GlooAllreduceRegistry,
    at::kCUDA,
    makeAllreduceCUDAWork)
} // namespace c10d

#endif // USE_C10D_GLOO
