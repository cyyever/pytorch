// SPDX-License-Identifier: MIT
// Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

#include <aiter_hip_common.h>
#include <fmha_bwd.hpp>
#include <mha_bwd.h>

namespace aiter {

float mha_bwd(mha_bwd_args a, const ck_tile::stream_config& s) {
  if (a.is_group_mode &&
      (a.seqstart_q_ptr == nullptr || a.seqstart_k_ptr == nullptr)) {
    AITER_LOG_ERROR(
        "mha_bwd: group mode requires seqstart_q_ptr and seqstart_k_ptr");
    return -1;
  }

  if (a.is_group_mode && !a.pinned_host_alloc) {
    AITER_LOG_ERROR(
        "mha_bwd: group mode requires pinned_host_alloc callback");
    return -1;
  }

  const fmha_bwd_traits traits{
      a.seqlen_q,
      a.seqlen_k,
      a.batch,
      a.max_seqlen_q,
      a.max_seqlen_k,
      a.hdim_q,
      a.hdim_v,
      a.nhead_q,
      a.nhead_k,
      a.data_type,
      a.is_group_mode,
      static_cast<mask_enum>(a.mask_type),
      static_cast<bias_enum>(a.bias_type),
      a.has_dbias,
      a.has_dropout,
      a.is_store_randval,
      a.is_deterministic,
  };

  fmha_bwd_launcher launcher(traits);
  void* workspace_ptr =
      a.workspace_alloc(launcher.workspace_size, /*zero_init=*/false);
  launcher.prepare_workspace_async(
      workspace_ptr,
      static_cast<const int*>(a.seqstart_q_ptr),
      static_cast<const int*>(a.seqstart_k_ptr),
      s,
      a.pinned_host_alloc);

  fmha_bwd_args ck_args{
      /* q_ptr              */ a.q_ptr,
      /* k_ptr              */ a.k_ptr,
      /* v_ptr              */ a.v_ptr,
      /* bias_ptr           */ a.bias_ptr,
      /* o_ptr              */ a.o_ptr,
      /* lse_ptr            */ a.lse_ptr,
      /* do_ptr             */ a.do_ptr,
      /* d_ptr              */ a.d_ptr,
      /* rand_val_ptr       */ a.rand_val_ptr,
      /* dq_ptr             */ a.dq_ptr,
      /* dk_ptr             */ a.dk_ptr,
      /* dv_ptr             */ a.dv_ptr,
      /* dbias_ptr          */ a.dbias_ptr,
      /* workspace_ptr      */ workspace_ptr,
      /* sink_ptr           */ a.sink_ptr,
      /* d_sink_ptr         */ a.d_sink_ptr,

      /* seqstart_q_ptr     */ a.seqstart_q_ptr,
      /* seqstart_k_ptr     */ a.seqstart_k_ptr,
      /* seqlen_q_ptr       */ a.seqlen_q_ptr,
      /* seqlen_k_ptr       */ a.seqlen_k_ptr,
      /* cu_seqlen_q_ptr    */ a.cu_seqlen_q_ptr,
      /* cu_seqlen_k_ptr    */ a.cu_seqlen_k_ptr,

      /* seqlen_q           */ a.seqlen_q,
      /* seqlen_k           */ a.seqlen_k,
      /* batch              */ a.batch,
      /* max_seqlen_q       */ a.max_seqlen_q,
      /* max_seqlen_k       */ a.max_seqlen_k,
      /* hdim_q             */ a.hdim_q,
      /* hdim_v             */ a.hdim_v,
      /* nhead_q            */ a.nhead_q,
      /* nhead_k            */ a.nhead_k,
      /* scale              */ a.scale,

      /* stride_q           */ a.stride_q,
      /* stride_k           */ a.stride_k,
      /* stride_v           */ a.stride_v,
      /* stride_bias        */ a.stride_bias,
      /* stride_o           */ a.stride_o,
      /* stride_randval     */ a.stride_randval,
      /* stride_do          */ a.stride_do,
      /* stride_dq          */ a.stride_dq,
      /* stride_dk          */ a.stride_dk,
      /* stride_dv          */ a.stride_dv,
      /* stride_dbias       */ a.stride_dbias,

      /* nhead_stride_q     */ a.nhead_stride_q,
      /* nhead_stride_k     */ a.nhead_stride_k,
      /* nhead_stride_v     */ a.nhead_stride_v,
      /* nhead_stride_bias  */ a.nhead_stride_bias,
      /* nhead_stride_o     */ a.nhead_stride_o,
      /* nhead_stride_randval*/ a.nhead_stride_randval,
      /* nhead_stride_do    */ a.nhead_stride_do,
      /* nhead_stride_lsed  */ a.nhead_stride_lsed,
      /* nhead_stride_dq    */ a.nhead_stride_dq,
      /* nhead_stride_dk    */ a.nhead_stride_dk,
      /* nhead_stride_dv    */ a.nhead_stride_dv,
      /* nhead_stride_dbias */ a.nhead_stride_dbias,

      /* batch_stride_q     */ a.batch_stride_q,
      /* batch_stride_k     */ a.batch_stride_k,
      /* batch_stride_v     */ a.batch_stride_v,
      /* batch_stride_bias  */ a.batch_stride_bias,
      /* batch_stride_o     */ a.batch_stride_o,
      /* batch_stride_randval*/ a.batch_stride_randval,
      /* batch_stride_do    */ a.batch_stride_do,
      /* batch_stride_lsed  */ a.batch_stride_lsed,
      /* batch_stride_dq    */ a.batch_stride_dq,
      /* batch_stride_dk    */ a.batch_stride_dk,
      /* batch_stride_dv    */ a.batch_stride_dv,
      /* batch_stride_dbias */ a.batch_stride_dbias,

      /* window_size_left   */ a.window_size_left,
      /* window_size_right  */ a.window_size_right,
      /* mask_type          */ a.mask_type,
      /* p_drop             */ a.p_drop,
      /* p_undrop           */ a.p_undrop,
      /* drop_seed_offset   */ a.drop_seed_offset,
  };

  return launcher.run(ck_args, s);
}

} // namespace aiter
