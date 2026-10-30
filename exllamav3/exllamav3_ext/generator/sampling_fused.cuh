#pragma once

#include <ATen/Tensor.h>

#define FUSED_SAMPLER_MAX_BLOCKS 128

// Histogram buffer geometry for mode 3 (top-K/top-P): per row, a coarse and a refinement
// histogram (u64 mass[BUCKETS] + u32 count[BUCKETS] each) plus a control block
#define FUSED_SAMPLER_HIST_BUCKETS 1024
#define FUSED_SAMPLER_HIST_STRIDE (2 * FUSED_SAMPLER_HIST_BUCKETS * 12 + 64)

// Filter flag bits for mode 3
#define FUSED_SAMPLER_F_TOPK 1
#define FUSED_SAMPLER_F_TOPP 2
#define FUSED_SAMPLER_F_MINP 4

void fused_sampler
(
    const at::Tensor& logits,
    const c10::optional<at::Tensor>& logit_mask,
    const c10::optional<at::Tensor>& logit_bitmask,
    at::Tensor& out,
    at::Tensor& workspace,
    int size,
    float inv_temp,
    float minp_log,
    uint32_t random,
    int mode,
    int filters,
    int top_k,
    float top_p,
    float inv_temp_filter,
    const c10::optional<at::Tensor>& histogram
);

void apply_logit_bitmask
(
    const at::Tensor& logits_in,
    at::Tensor& logits_out,
    const at::Tensor& bitmask
);

// DADDy partials and diagnostic counters per row.
#define DADDY_SAMPLER_PARTIAL_FLOATS 26
// Counter columns: direct proposals, proposal fallback draws, all-full fallbacks.
#define DADDY_SAMPLER_COUNTS 3
#define DADDY_SAMPLER_API_VERSION 2

void daddy_sampler
(
    const at::Tensor& logits,
    const c10::optional<at::Tensor>& logit_mask,
    const c10::optional<at::Tensor>& logit_bitmask,
    at::Tensor& out,
    at::Tensor& workspace,
    int size,
    float inv_temp,
    float minp_log,
    uint32_t random,
    int mode,
    int filters,
    int top_k,
    float top_p,
    float inv_temp_filter,
    const c10::optional<at::Tensor>& histogram,
    const at::Tensor& daddy_workspace,
    const at::Tensor& balances,
    const at::Tensor& counts,
    const c10::optional<at::Tensor>& draft,
    double overdraft,
    double boost, double fixed_repulsor, double dynamic_repulsor
);
