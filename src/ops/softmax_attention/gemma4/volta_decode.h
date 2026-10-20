#pragma once

#include <cuda_runtime.h>

#include <cstdint>

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

// grid: (splits, Hkv * batch rows of the launch); bc: the split-K route's key tile (the combine
// derives the same split boundaries from it).
void gemma4_volta_decode_launch(const Gemma4AttentionArgs& args, std::int32_t head_dim,
                                std::int32_t bc, dim3 grid, cudaStream_t stream);

} // namespace ninferno::ops::detail
