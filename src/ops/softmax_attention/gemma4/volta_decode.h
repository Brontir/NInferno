#pragma once

#include "ops/generic/kernel_facts.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <functional>

namespace ninferno::ops {
struct Gemma4AttentionArgs;
} // namespace ninferno::ops

namespace ninferno::ops::detail {

// The Volta / Turing SIMT decode partials (volta_decode.cu): whether a decode call of `rows` query
// rows per KV head (T * G) at this head width takes them on this device (no native BF16 MMA,
// D = 64..512, rows * D <= 4096); the plan, the partials and the combine are the split-K route's.
[[nodiscard]] bool gemma4_volta_decode_takes(std::int32_t head_dim, std::int32_t rows);

// The CTAs of the instance a call of `rows` query rows per KV head takes that the device keeps
// resident at once: the decode plan sizes its splits to one such wave.
[[nodiscard]] int gemma4_volta_decode_resident_ctas(std::int32_t head_dim, std::int32_t rows);

// The CTAs of a Gemma 4 attention kernel one SM of device_class() holds (#830), by the occupancy
// calculator (core/kernel_occupancy.h) over the kernel's committed facts (kernel_facts.h, the
// gemma4 section of the snapshot of the kernels the board runs: sass_arch), never the host's
// kernels: a substituted board answers as the real one would, and a host without a GPU answers.
// Where no snapshot holds the kernel, a real device asks its own kernel (`live`: a build or a
// board the snapshots do not cover, the device the call runs on); a substituted board throws, so
// a plan of it keeps its formula estimate rather than the host GPU's answer. At least 1.
[[nodiscard]] int gemma4_resident_ctas_per_sm(const generic::kernel_facts::Gemma4KernelKey& key,
                                              int threads, std::size_t dynamic_shared_bytes,
                                              const std::function<int()>& live);

// The keys a split of this route takes are rounded to multiples of this many (the combine derives
// the same split boundaries from it). A CTA walks its keys one after another, one warp a scheduler
// on the few SMs that hold work (GV100, Gemma 4 12B sliding layer at ~60 keys: 2 splits x 8 KV
// heads = 16 CTAs, each ~45k cycles for its 32 keys: P V 24.5k, the scores 12k, the softmax 6k;
// loading the keys in batches changed nothing). Splits rounded to 8 keys put four times the CTAs
// of 32-key splits on a short context; on a long one the plan's split count stands (the resident
// wave; a sliding layer at most window / kGeoDecodeBc), each split taking len / splits keys
// rounded up to 8.
inline constexpr std::int32_t kGemma4VoltaDecodeBc = 8;

// grid: (splits, Hkv * batch rows of the launch); bc: kGemma4VoltaDecodeBc.
void gemma4_volta_decode_launch(const Gemma4AttentionArgs& args, std::int32_t head_dim,
                                std::int32_t bc, dim3 grid, cudaStream_t stream);

} // namespace ninferno::ops::detail
