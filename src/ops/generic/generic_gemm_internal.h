#pragma once

// Private launch entry points of the generic projection path, one per compiled family and format.

#include "ops/generic/generic_gemm.h"
#include "ops/generic/generic_variant_rules.h"
#include "ops/generic/kernel_facts.h"

#include "core/cuda_stream.h"

#include <cstddef>
#include <cstdint>
#include <optional>

namespace ninferno::ops::detail::generic {

// A launch described instead of run (#298, rmma_resources): while a KernelResources is installed on
// this thread, the register-MMA launch (launch_rmma_kernel) and the R8 launch (r8_launch_tokens)
// record the kernel they would run and its geometry there and launch nothing.
KernelResources* kernel_describe_target() noexcept;
class ScopedKernelDescribe {
public:
    explicit ScopedKernelDescribe(KernelResources& out) noexcept;
    ~ScopedKernelDescribe();
    ScopedKernelDescribe(const ScopedKernelDescribe&)            = delete;
    ScopedKernelDescribe& operator=(const ScopedKernelDescribe&) = delete;

private:
    KernelResources* previous_;
};

// A kernel's committed facts (kernel_facts.h, #612) as the register-MMA launch describes it. The
// block limit is the launch bounds' (Schedule::kThreads); the versions are the build's
// architecture. Host code (rmma_resources, generic_rmma_describe.cpp) describes from the facts the
// same way. Not noexcept: the K loop's counts (KernelResources::loop) copy a string, which may
// allocate; both callers already throw (rmma_resources, the launch's describe mode).
inline void describe_from_facts(KernelResources& out, const kernel_facts::KernelFacts& f,
                                unsigned grid_x, unsigned grid_y, unsigned grid_z, int threads,
                                std::size_t dynamic_smem, bool latency) {
    out.registers             = f.registers;
    out.local_bytes           = f.spill_bytes();
    out.static_smem           = f.static_shared_bytes;
    out.dynamic_smem          = dynamic_smem;
    out.threads               = threads;
    out.max_threads_per_block = threads;
    out.ptx_version           = NINFERNO_BUILD_ARCH;
    out.binary_version        = NINFERNO_BUILD_ARCH;
    out.grid_x                = grid_x;
    out.grid_y                = grid_y;
    out.grid_z                = grid_z;
    out.latency_variant       = latency;
    out.loop                  = f.loop;
}

#ifndef NINFERNO_HOST_ONLY // the launch describes a kernel; the selector does not
inline void describe_kernel(KernelResources& out, const cudaFuncAttributes& a, dim3 grid,
                            int threads, std::size_t dynamic_smem, bool latency) noexcept {
    out.registers             = a.numRegs;
    out.local_bytes           = a.localSizeBytes;
    out.static_smem           = a.sharedSizeBytes;
    out.dynamic_smem          = dynamic_smem;
    out.threads               = threads;
    out.max_threads_per_block = a.maxThreadsPerBlock;
    out.ptx_version           = a.ptxVersion;
    out.binary_version        = a.binaryVersion;
    out.grid_x                = grid.x;
    out.grid_y                = grid.y;
    out.grid_z                = grid.z;
    out.latency_variant       = latency;
}

inline void describe_kernel(KernelResources& out, const kernel_facts::KernelFacts& f, dim3 grid,
                            int threads, std::size_t dynamic_smem, bool latency) {
    describe_from_facts(out, f, grid.x, grid.y, grid.z, threads, dynamic_smem, latency);
}
#endif

// rmma_resources by running the register-MMA launch in describe mode (generic_describe.cpp): the
// selector's own rmma_resources reads the facts without a launch, and ninferno_rmma_describe_test
// holds the two equal.
[[nodiscard]] std::optional<KernelResources>
rmma_resources_by_launch(int schedule, const Weight& weight, Epilogue epilogue, std::int32_t tokens,
                         std::int32_t rows = 0);

// The register-MMA launch of `codec` (one of the launch_rmma_* below): run_route's, and
// rmma_resources_by_launch's described one.
void launch_rmma_codec(WeightCodec codec, int schedule, Epilogue epilogue, const GemmArgs& args,
                       cudaStream_t stream);

// out [K, T] BF16 = x [K, T] (leading dimension ldx) gathered by K int32 column indices.
void launch_gather_k(const void* x, int ldx, const void* order, int k, int tokens, void* out,
                     cudaStream_t stream);
void launch_rmma_bf16(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_fp8(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_nvfp4(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_q4(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_q5(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_q6(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_q3g128(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_q3g64(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_q8(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_mxfp4(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_fp8_row_f32(int schedule, Epilogue epilogue, const GemmArgs& args,
                             cudaStream_t stream);
void launch_rmma_fp8_b128(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_u4zp(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);
void launch_rmma_u4zp_g32(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_u4zp_g64(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q4k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q5k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q6k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q3k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q2k(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq4xs(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq3xxs(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq3s(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q40(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q41(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q50(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q51(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq4nl(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq2xxs(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq2xs(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq2s(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq1s(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_iq1m(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_tq10(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_tq20(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q10(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);
void launch_rmma_ggml_q20(int schedule, Epilogue epilogue, const GemmArgs& args,
                          cudaStream_t stream);

void launch_scalar(WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                   cudaStream_t stream);

// The DRAM bytes the device's bandwidth asks of each SM per cycle (MachineFacts: bandwidth over
// SMs times the reference clock); 0 when unknown. Once per device.
[[nodiscard]] double dram_bytes_per_sm_cycle();

// The DRAM bytes per SM cycle from which the K-quant register MMA of one token tile takes its
// latency variant (rmma_latency_pays, generic_gemm_launch.cuh, where the measurements are), for
// SwiGLU and for one row set.
inline constexpr double kRmmaLatencySwiGluBytes = 4.5;
inline constexpr double kRmmaLatencyOneSetBytes = 5.5;



// Volta's warp-per-row GEMV of the ggml Q4_K / Q5_K / Q6_K, Q8 G32 and BF16 formats at T <= 4
// (generic_volta_gemv.cu), in place of a lane-local register-MMA call; x in 16-byte words (ldx a
// multiple of 8, x on 16 bytes). run_route and rmma_resources both ask this one predicate.
bool volta_gemv_takes(WeightCodec codec, const GemmArgs& args);
void launch_volta_gemv(WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                       cudaStream_t stream);

// Volta's tensor-core route of the speculative-verify widths (generic_volta_hmma.cu): Q4_K / Q5_K
// / Q6_K and Q8 G32 at kVoltaHmmaMinTokens..16 tokens, in place of a lane-local register-MMA call
// (and of ggml's vector kernel, volta_hmma_route). From one token: a lane-local call of one token
// ran Volta's FP32 decode GEMV (generic_volta_gemv.cu) and the same token in a wider call the HMMA
// kernel, so an A16 projection's bytes depended on its call's width (an MoE's A16 shared expert at
// T = 1 + 3 against T = 4, test_moe); a call of one token keeps R8 or the vector kernel where an
// int8 route takes it (volta_hmma_route), as at two to four.
inline constexpr std::int32_t kVoltaHmmaMinTokens = 1;
inline constexpr std::int32_t kVoltaHmmaMaxTokens = 16;
bool volta_hmma_codec(WeightCodec codec);
// The K multiple a row needs: whole superblocks, or for Q8 G32 whole pairs of groups.
int volta_hmma_k_multiple(WeightCodec codec);
bool volta_hmma_takes(WeightCodec codec, const GemmArgs& args);
void launch_volta_hmma(WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                       cudaStream_t stream);

// Tiled A16 (generic_tiled.cu); the codec must be one tiled_supports() admits.
void launch_tiled(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                  cudaStream_t stream);

// Volta prefill on CUTLASS's sm_70 GEMM over weight rows decoded once to FP16
// (generic_volta_cutlass.cu; built for sm_70 alone). launch_volta_cutlass returns false, having
// launched nothing, when the build lacks it, the GEMM refuses the shape or the workspace left
// cannot hold one chunk; the tiled route then takes the call. volta_cutlass_compiled and the sizes
// below are host code (generic_volta_cutlass_sizes.cpp, with the selector), so a host-only build
// answers them as the device build does.
bool volta_cutlass_compiled() noexcept;
std::size_t volta_cutlass_workspace_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens,
                                          bool swiglu);
// The bytes of the chunk launch_volta_cutlass takes from `free_bytes` of workspace (halved until
// it fits); 0 when it would launch nothing for want of room.
std::size_t volta_cutlass_chunk_bytes(std::int32_t n, std::int32_t k, std::int32_t tokens,
                                      bool swiglu, std::size_t free_bytes);
// `free_bytes`: the workspace the chunk is sized to (the call's aligned free space, as
// volta_cutlass_chunk_bytes takes it).
bool launch_volta_cutlass(WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                          WorkspaceArena& workspace, std::size_t free_bytes, cudaStream_t stream);

// Staged register MMA (generic_staged.cu): BF16 weights only.
void launch_staged(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);

// NVFP4 A16 in shared-memory activation tiles (generic_nvfp4_tiled.cu), summing in the order of
// the register MMA with WarpsK = classes (1, 4 or 8).
void launch_nvfp4_tiled(int classes, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream);

// NVFP4 W4A4: quantize the activation into `codes`/`scales` (row-major, [T,K/2] and [T,K/16]),
// then run the schedule.
void launch_w4a4(int schedule, Epilogue epilogue, const GemmArgs& args, float input_divisor,
                 std::uint8_t* codes, std::uint8_t* scales, cudaStream_t stream);

// The W4A8 activation planes' row length: K rounded up to whole 128-value tiles (w4a8_act_ld).
inline int w4a8_ld(int k) { return (k + 127) / 128 * 128; }

// W4A8 (generic_w4a8.cu): quantize the activation into `codes` [T,ld] int8, `scales` [T,ld/32]
// (FP32 amax/127) and `sums` [T,ld/32] (the scale times the codes' sum; ld = w4a8_ld(K), the tail
// blocks zero), then run the schedule. The
// codec must be one w4a8_supports() admits. token_major: the grid order of the tiles
// (W4a8Activation::token_major).
void launch_w4a8(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                 std::int8_t* codes, float* scales, float* sums, bool token_major,
                 cudaStream_t stream);
// The I8 route: quantize x per 64 values into [padded T][K] codes (k-permuted), [K/64][padded T]
// scales and sums, then the tile (generic_i8.cu).
void launch_i8(int schedule, int bits, Epilogue epilogue, const GemmArgs& args, std::int8_t* codes,
               float* scales, int* sums, cudaStream_t stream);
// The W4A8 activation quantizer alone (generic_w4a8.cu): args.tokens x args.k values into codes,
// scales and sums, rows of `ld` values (0: args.k; the blocks past args.k zero).
void launch_w4a8_quantize(const GemmArgs& args, std::int8_t* codes, float* scales, float* sums,
                          cudaStream_t stream, int ld = 0);
// The register MMA's group-sums pre-pass (generic_group_sums.cu, #619): for each of args.tokens
// tokens, the sum of each group of args.group_sums_values BF16 values of x into `out` (rows of
// args.group_sums_ld floats; the groups past args.k zero), in one fixed order (a pairwise tree
// in index order, rmma_group_sum). K must be whole groups (refused otherwise). A described launch
// (kernel_describe_target) runs nothing.
void launch_rmma_group_sums(const GemmArgs& args, float* out, cudaStream_t stream);
// R8 (generic_r8.cu): W8A8 decode for T <= 8, the activation quantized inside the kernel, or, with
// `pre`, read from a pre-quantized activation (verify widths, r8_verify_supports formats). The
// codec must be one r8_supports() admits; the iq3/iq2/ggml4/kquant entries return false for
// other codecs.
void launch_r8(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
               cudaStream_t stream, W4a8Activation pre = {});
bool launch_r8_iq3(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                   cudaStream_t stream, W4a8Activation pre);
bool launch_r8_iq2(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                   cudaStream_t stream, W4a8Activation pre);
bool launch_r8_ggml4(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                     cudaStream_t stream, W4a8Activation pre);
bool launch_r8_kquant(int schedule, WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                      cudaStream_t stream, W4a8Activation pre);

// Family::GgmlMmq's epilogue: the FP32 plane [T][plane_rows] (SwiGlu: gate rows, then up rows from
// args.rows) to the call's BF16 segments.
void launch_mmq_epilogue(Epilogue epilogue, const GemmArgs& args, const float* plane,
                         int plane_rows, cudaStream_t stream);

} // namespace ninferno::ops::detail::generic
