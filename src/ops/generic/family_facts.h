#pragma once

// The tile, step and stage facts of the generic path's kernel families, one constexpr record per
// family (#619 step 3): the one spelling the price's Counted constants are read from
// (route_cost_constants.h) and each family's kernel template is static_asserted against
// (generic_gemm.cuh, generic_tiled.cuh, generic_staged.cuh, generic_w4a8.cuh, generic_i8.cu,
// ggml_mmq/ggml_mmq.cu), so the price never restates a kernel by hand: a K step, a stage or a tile
// that changes in the kernel and not here stops that kernel's build. route_resources hands the
// record of the route's family back with the kernel it describes (KernelResources::family).
//
// Geometry only. What ptxas made of a kernel (registers, stack, shared bytes) is the kernel-facts
// snapshots' (kernel_facts.h) or the driver's, per architecture; it is not here.
//
// Plain C++: no CUDA type, includable from host code and from .cu alike.

namespace ninferno::ops::detail::generic {

// A field a family has no such fact for is 0.
struct FamilyFacts {
    int mma_k             = 0; // K of one tensor MMA of the K loop
    int step_k            = 0; // K values one step of the K loop covers
    int stage_k           = 0; // K values one shared-memory pipeline stage holds
    int stage_row_bytes   = 0; // shared bytes a stage holds per weight row of the tile
    int stage_token_bytes = 0; // shared bytes a stage holds per token of the tile
    int stages            = 0; // pipeline stages, where the family fixes them
    int producer_stages   = 0; // the same, of its producer-warp form
    int tile_rows         = 0; // weight rows of every tile, where the family fixes them
    int column_step       = 0; // the granularity of a tile's token columns

    friend constexpr bool operator==(const FamilyFacts&, const FamilyFacts&) = default;
};

// RegisterMma (generic_gemm.cuh rmma_kernel): a K step is four lanes' runs of kRun values, J =
// kRun / 4 MMAs of m16n8k16 (Turing composes the k16 from two k8). step_k is the step of the
// 32-value-run decoders (every ggml block, the K-quants, the codebooks, U4Zp, MXFP4); a decoder
// of a shorter run steps four times its run (rmma_step_values).
inline constexpr FamilyFacts kRegisterMmaFacts{.mma_k = 16, .step_k = 4 * 32};

// TiledMma (generic_tiled.cuh): a stage is a K tile of 64 values; the activation is staged as
// BF16, the weight as its codec's raw chunks (TiledShared::raw, kRawChunks x 16 bytes a row: the
// codec's, so no row bytes here) and decoded once per tile.
inline constexpr FamilyFacts kTiledMmaFacts{.stage_k = 64, .stage_token_bytes = 64 * 2};

// StagedMma (generic_staged.cuh): a stage is a K tile of 64 BF16 values of every weight row and
// token (StagedShared).
inline constexpr FamilyFacts kStagedMmaFacts{
    .stage_k = 64, .stage_row_bytes = 64 * 2, .stage_token_bytes = 64 * 2};

// W4a8Mma (generic_w4a8.cuh): a step is one K tile of four 32-value blocks, decoded to int8 codes
// in shared memory with two floats per block (scale and min, or scale and sum) for every row and
// token; two buffers, three stages in the producer-warp form (w4a8_ws_kernel).
inline constexpr FamilyFacts kW4a8MmaFacts{.step_k            = 4 * 32,
                                           .stage_k           = 4 * 32,
                                           .stage_row_bytes   = 4 * 32 + 4 * 2 * 4,
                                           .stage_token_bytes = 4 * 32 + 4 * 2 * 4,
                                           .stages            = 2,
                                           .producer_stages   = 3};

// I8Mma (generic_i8.cu I8Tile, generic_i8_schedules.h): a stage is one 64-value activation scale
// group: per token the int8 codes, an FP32 scale and an int32 sum; per row the low nibbles (a Q5 /
// Q6 tile adds its high plane, i8_high_bytes).
inline constexpr FamilyFacts kI8MmaFacts{
    .stage_k = 64, .stage_row_bytes = 64 / 2, .stage_token_bytes = 64 + 4 + 4};

// GgmlMmq (third_party ggml mul_mat_q): a step is MMQ_ITER_K values; every tile is I = 128 weight
// rows (each NVIDIA config table of ggml's) by J token columns, J a multiple of 8.
inline constexpr FamilyFacts kGgmlMmqFacts{.step_k = 256, .tile_rows = 128, .column_step = 8};

// The other families (R8, the FP4 tiles, ggml's vector kernel, scalar) export nothing yet: no
// price reads a counted fact of theirs.

} // namespace ninferno::ops::detail::generic
