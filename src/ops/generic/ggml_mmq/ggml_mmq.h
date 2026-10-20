#pragma once

// ggml's integer tensor-core matrix kernel (third_party/ggml-quants, ggml-cuda/mmq.cuh) for the
// ggml block formats as NInferno stores them: blocks unchanged (GgmlBlocks: Q4_K, Q5_K) or split
// into a code, a high and a scale plane with each block's bytes unchanged (GgmlPlanes, whose
// loaders read the planes, mmq-load-tiles.cuh). The ggml headers define CUDA_CHECK and friends, so
// only this directory's translation units include them; everything here is raw pointers.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace ninferno::ops::ggml_mmq {

enum class Type : std::uint8_t {
    Q4_K,
    Q5_K,
    Q6_K,
    IQ4_XS,
    IQ3_XXS,
    IQ3_S,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    IQ4_NL,
    Q2_K,
    Q3_K,
    IQ2_XXS,
    IQ2_XS,
    IQ2_S,
    IQ1_S,
    IQ1_M // the vector kernel only (matrix_supported)
};

// Whether ggml's matrix kernel reads the format (every Type but IQ1_M).
[[nodiscard]] constexpr bool matrix_supported(Type type) { return type != Type::IQ1_M; }

// The weight rows of one product: row r's blocks start at codes + r * blocks_per_row * code_bytes
// (and likewise in the high and scale planes, null for blocks unchanged).
struct Planes {
    const void* codes  = nullptr;
    const void* high   = nullptr;
    const void* scales = nullptr;
};

// Whether the device runs the kernel: ggml's int8 tensor-core forms (sm_75 and up). The device
// is device_class()'s, so a selector test's substitute class answers for its own card.
[[nodiscard]] bool device_supported();
// A product's rows are whole 128-row tiles; K whole 256-value steps of the kernel's K loop
// (MMQ_ITER_K), so a hidden size of 5376 (Gemma 4 31B), 2816 or 3840 takes the kernel too.
inline constexpr int kRowMultiple = 128;
inline constexpr int kKMultiple   = 256;

// The activation: ggml's q8_1 tiles for `columns` tokens of K values (with ggml's over-read slack).
[[nodiscard]] std::size_t activation_bytes(int k, int columns);
// Formats of one activation layout read one quantized activation.
[[nodiscard]] int activation_layout(Type type);
// x is BF16, token c at x + c * ldx.
void quantize_activation(Type type, const __nv_bfloat16* x, int ldx, int k, int columns, void* out,
                         cudaStream_t stream);

[[nodiscard]] std::size_t fixup_bound(int columns);

// out[c * out_column_stride + r] = sum_k W[r, k] x[k, c] in FP32 for rows [0, rows).
void product(Type type, const Planes& weight, int blocks_per_row, int rows, int k,
             const void* activation, int columns, float* out, std::int64_t out_column_stride,
             void* fixup, cudaStream_t stream);

// ggml's vector kernel (ggml_vec.cu, after ninfer-all's): 1..kVecMaxColumns tokens with dp4a on
// a planar q8_1 activation, two output rows a warp, the epilogue in the kernel. A token's sums do
// not depend on the other tokens of the call.
inline constexpr int kVecMaxColumns  = 8;
inline constexpr int kVecMaxSegments = 4;

// The output of a vector product: local row r goes to the segment s holding it (rows
// [segment_end[s-1], segment_end[s])), token c at out[s] + c * ld[s] (BF16).
// StoreF32 writes v to the FP32 plane `f32` instead (column stride f32_ld); GateProduct stores
// act(gate) * v, `gate` the FP32 plane of the same rows (column stride gate_ld) that a StoreF32
// product of the gate weight wrote: the SwiGLU / GeGLU of a gate and an up weight of two formats.
struct VecOut {
    enum class Epilogue : std::uint8_t { Store, AddResidual, SwiGlu, StoreF32, GateProduct };
    Epilogue epilogue                = Epilogue::Store;
    int gelu_gate                    = 0; // SwiGlu, GateProduct: GELU's tanh form for the gate
    int relu2                        = 0; // Store: relu(v)^2
    void* out[kVecMaxSegments]       = {};
    int segment_end[kVecMaxSegments] = {};
    int ld[kVecMaxSegments]          = {};
    int segments                     = 1;
    float* f32                       = nullptr;
    int f32_ld                       = 0;
    const float* gate                = nullptr;
    int gate_ld                      = 0;
};

// Whether the format has a vector decoder.
[[nodiscard]] bool vec_supported(Type type);
// The planar activation of `columns` tokens of K values (K whole 256-value blocks).
[[nodiscard]] std::size_t vec_activation_bytes(int k, int columns);
void vec_quantize_activation(const __nv_bfloat16* x, int ldx, int k, int columns, void* out,
                             cudaStream_t stream);
// rows output rows of `weight` (SwiGlu: the gate rows; `up` the up rows) against the activation.
void vec_product(Type type, const Planes& weight, const Planes& up, int blocks_per_row, int rows,
                 int k, const void* activation, int columns, const VecOut& out,
                 cudaStream_t stream);

} // namespace ninferno::ops::ggml_mmq
