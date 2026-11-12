#pragma once

#include "core/device_class.h"
#include "core/tensor.h"
#include "ops/generic/kernel_facts.h"

#include <cuda_runtime.h>

#include <cstdint>
#include <optional>

namespace ninferno::ops::detail {

// The text form: one warp per head, three heads per block (launch.cu). Its grid is
// tokens x ceil((Q + K heads) / 3) blocks of three warps.
inline constexpr int kTextHeadsPerBlock = 3;

// Resident blocks an SM holds of one geometry's text kernel instance on the current device (the
// compiled kernel's occupancy: registers and shared memory included); 0 for a geometry the kernel
// is not instantiated for. The running GPU's answer
// (cudaOccupancyMaxActiveBlocksPerMultiprocessor): the bench's check of the committed facts, never
// an input of a decision.
int text_kernel_resident_ctas(std::int32_t query_heads, std::int32_t key_heads);

// The committed facts (ops/generic/kernel_facts.h: registers, static shared memory) of one
// geometry's text kernel in the engine a board of `device`'s capability runs. nullopt on every
// architecture today: the kernel-facts snapshots hold the register-MMA kernels only
// (tools/kernel_facts), not rmsnorm_rope_d256_text_kernel. The law below then refuses (0) rather
// than read the device's thread and slot limits alone, which overstate the resident warps wherever
// the kernel's registers or shared memory bind.
std::optional<generic::kernel_facts::KernelFacts>
text_kernel_facts(const DeviceClass& device, std::int32_t query_heads, std::int32_t key_heads);

// Resident blocks an SM holds of a text kernel with these facts on `device`: the CUDA occupancy
// calculator over the facts and the device's limits (core/kernel_occupancy.h: threads, CTA slots,
// registers, shared memory), launched at three warps and no dynamic shared memory. No GPU; 0 where
// a fact is unknown or the kernel does not fit.
int text_kernel_facts_resident_ctas(const DeviceClass& device,
                                    const generic::kernel_facts::KernelFacts& facts);

// The launch-amortisation law with an explicit chain excess in SM cycles
// (rmsnorm_rope_text_fused_max_tokens passes the pending Fitted constant once calibrated), the
// resident warps from the geometry's committed facts (text_kernel_facts): nullopt where none are
// committed for the board's engine, the generation has no launch node, the geometry is not
// instantiated or the excess is not positive; 0 is "never fuse".
std::optional<std::int32_t> text_fused_max_tokens_with_excess(const DeviceClass& device,
                                                              std::int32_t query_heads,
                                                              std::int32_t key_heads,
                                                              double excess);

// The same law with the kernel's facts given (what text_fused_max_tokens_with_excess reads once
// the snapshots carry the text kernel; a test passes its own).
std::optional<std::int32_t>
text_fused_max_tokens_with_facts(const DeviceClass& device, std::int32_t query_heads,
                                 std::int32_t key_heads,
                                 const generic::kernel_facts::KernelFacts& facts, double excess);

void rmsnorm_rope_pair_launch(const Tensor& positions, const Tensor& q_norm_weight,
                              const Tensor& k_norm_weight, Tensor& q, Tensor& k,
                              std::int32_t tokens, cudaStream_t stream);

void rmsnorm_rope_single_launch(const Tensor& positions, const Tensor& norm_weight, Tensor& x,
                                std::int32_t tokens, cudaStream_t stream);

void rmsnorm_rope_text_launch(const Tensor& positions, const Tensor& q_norm_weight,
                              const Tensor& k_norm_weight, const Tensor& q_in, const Tensor& k_in,
                              Tensor& q_out, Tensor& k_out, std::int32_t tokens,
                              cudaStream_t stream);

} // namespace ninferno::ops::detail
