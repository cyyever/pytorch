#pragma once

#if USE_NCCL

#include <nccl.h>
#include <torch/csrc/cuda/nccl.h>

#define NCCL_HAS_SYMMEM_SUPPORT

#if !defined(USE_ROCM)
#define NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#include <nccl_device.h>
#endif

// Host-side device-communicator setup (ncclDevCommCreate with
// ncclDevCommRequirements / NCCL_DEV_COMM_REQUIREMENTS_INITIALIZER), the
// one-sided API and device-side reduce-copy all landed by NCCL 2.29.7, below
// the 2.30 floor NCCLUtils.hpp asserts, so only the device-side support itself
// is still in question.
#ifdef NCCL_HAS_SYMMEM_DEVICE_SUPPORT
#define NCCL_HAS_DEVCOMM
#define NCCL_HAS_ONE_SIDED_API
#define NCCL_DEVICE_HAS_REDUCE_COPY
#endif

// Host-side CFT (Compute Fabric Transport) logical-endpoint queries:
// ncclGetPeerDeviceLeInfo / ncclGetMultimemDeviceLeInfo. They resolve a window
// offset into the `(leId, leOffset)` pair that the device-side `ncclCft`
// put/get/red family consumes, so a custom kernel can drive CFT without
// building a ncclDevComm. The LEs only exist if the communicator was created
// with `ncclConfig_t::hostCftMode` enabled (see NCCL_HAS_HOST_CFT_MODE).
#if defined(NCCL_HAS_SYMMEM_DEVICE_SUPPORT) && \
    NCCL_VERSION_CODE >= NCCL_VERSION(2, 31, 2)
#define NCCL_HAS_HOST_CFT
#endif
#endif // USE_NCCL
