#pragma once

// Generic W8A8 decode (the R8 family): weights whose 32-value blocks are int8 codes times a scale,
// plus a min (Q4_1, Q5_1, Q4_K, Q5_K) or with a second scale for values 16..31 (IQ2_XS, IQ2_S,
// Q6_K, Q3_K; Q2_K also a min per 16 values) (q8_g32, the ggml 4/5-bit blocks, the K-quants
// Q2_K..Q6_K and the codebook i-quants, through their W4A8 codecs) against activations quantized
// to int8 per 32 values with the W4A8 quantizer's arithmetic, for the few tokens of decode and
// speculative verify (T <= 8). It is llama.cpp's
// MMVQ with Q8_1 activations: a lane takes whole 32-value blocks of a row, eight dp4a per block
// and token, and the block's int32 sum becomes FP32 times the weight's and the token's scales.
//
// Every CTA quantizes the whole activation [T, K] into shared memory before its first block (the
// first weight blocks are already in flight), so the call needs no workspace and no second
// launch. A warp owns kRows rows (SwiGLU: gate and up rows), its lanes stride K by whole blocks,
// and one butterfly reduction ends it. A token's result depends on nothing but its own
// activation: the bytes do not change with T.

#include "core/pdl.cuh"
#include "ops/generic/generic_w4a8.cuh"

#include <cuda_bf16.h>

#include <algorithm>
#include <cstdint>
#include <stdexcept>

namespace ninferno::ops::detail::generic {

template <int Warps, int Rows>
struct R8Schedule {
    static_assert(Warps > 0 && Warps <= 16 && Rows > 0 && Rows <= 8);
    static constexpr int kWarps      = Warps;
    static constexpr int kRows       = Rows; // rows per warp
    static constexpr int kThreads    = Warps * 32;
    static constexpr int kRowsPerCta = Warps * Rows;
};

// Shared bytes of the quantized activation for NT token slots and K values: the two 16-byte
// halves of every block in two planes (a warp's lanes read consecutive 16-byte words: no bank
// conflict), one FP32 scale per block and, for a codec with a min, the block's FP32 sum.
__host__ __device__ constexpr std::int64_t r8_smem_bytes(int nt, std::int64_t k, bool sums) {
    return static_cast<std::int64_t>(nt) * (k / 32) * (sums ? 40 : 36);
}

// NINFERNO-SPECIAL(arch:sm75): before sm_80 r8_kernel stages a codebook codec's grid in shared
// memory once per CTA and its lookups read it there (SharedGrid): from the global table a warp's
// random words took up to 32 L1 wavefronts each (RTX 2080 Ti, r8_schedule_bench on Qwen3.8 27B
// UD-IQ2_XXS's shapes: IQ1_S / IQ1_M / IQ2_* 160..230 -> 220..320 GB/s, decode 30.1 -> 38.9
// tok/s). The RTX 5080 reads them from L1 as fast (the same shapes -4..+4 %, Gemma 4 31B IQ3_XXS
// decode -1.6 % staged), so later GPUs keep the global reads and the shared bytes.
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ < 800
#    define NINFERNO_R8_STAGED_GRID 1
#else
#    define NINFERNO_R8_STAGED_GRID 0
#endif

// The host's side of the same rule: whether this device's R8 launches stage the grid
// (generic_gemm.cpp: no native BF16 MMA, sm_80 on has it).
[[nodiscard]] bool r8_stage_grid();

// Whether this device's R8 launches (in-CTA quantization) take a resident grid: no more CTAs than
// the SMs hold at once, each looping over row blocks (generic_gemm.cpp).
[[nodiscard]] bool r8_resident_grid();

// The staged grid starts at the next 8-byte boundary after the activation (whose bytes are a
// multiple of 4): at most this many bytes of padding.
inline constexpr std::size_t r8_grid_pad = 4;

// A codebook codec's grid words: those r8_kernel stages (GridWords / SharedGrid).
template <class C>
__host__ __device__ constexpr int r8_grid_words() {
    if constexpr (requires { C::kGridWords; }) {
        return C::kGridWords;
    } else {
        return 0;
    }
}

__device__ __forceinline__ int dp4a_words(const uint4& w, const uint4& x, int acc) {
    acc = __dp4a(static_cast<int>(w.x), static_cast<int>(x.x), acc);
    acc = __dp4a(static_cast<int>(w.y), static_cast<int>(x.y), acc);
    acc = __dp4a(static_cast<int>(w.z), static_cast<int>(x.z), acc);
    acc = __dp4a(static_cast<int>(w.w), static_cast<int>(x.w), acc);
    return acc;
}

// kPre: the activation arrives quantized (codes [T, K], scales and sums [T, K/32], the W4A8
// quantizer's output, the same arithmetic as the in-CTA quantization) and is read from global
// memory; otherwise every CTA quantizes all T tokens into shared memory first.
template <class C, class S, int NT, Epilogue E, bool kPre = false>
__global__ __launch_bounds__(S::kThreads) void r8_kernel(GemmArgs a, W4a8Activation act) {
    constexpr bool kSwiGlu  = E == Epilogue::SwiGlu;
    constexpr int kSets     = kSwiGlu ? 2 : 1; // gate rows, then the matching up rows
    constexpr int kRows     = S::kRows;
    constexpr bool kHalves  = w4a8_halves<C>();
    constexpr bool kHalfMin = w4a8_half_min<C>();
    // A lane's unit is a group of kSpan*kStride blocks: kStride loads of kSpan blocks each (load q
    // holds blocks q + kStride*s), consecutive groups on consecutive lanes.
    constexpr int kSpan   = w4a8_span<C>();
    constexpr int kStride = w4a8_stride<C>();
    constexpr int kGroup  = kSpan * kStride;
    static_assert(4 % kGroup == 0, "a 128-value K tile holds whole groups");
    using Raw = typename C::Raw;
    extern __shared__ __align__(16) unsigned char r8_shared[];

    // Call::release_next: the next call (a joined one) may launch once every CTA has started. Not
    // with kPre: the next call's quantizer may take the workspace this kernel still reads. Under
    // GemmArgs::pdl_prefetch that waits until after this CTA's own wait below: a joined call reads
    // x without waiting, and x is the previous kernel's.
    if constexpr (E == Epilogue::Store && !kPre) {
        if (a.release_next != 0 && a.pdl_prefetch == 0 && threadIdx.x == 0) {
            pdl::trigger_dependents();
        }
    }

    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int blocks = a.k / 32;
    const int groups = blocks / kGroup;
    // Row blocks: blockIdx.x, then every gridDim.x-th after it (r8_launch_tokens may launch fewer
    // CTAs than row blocks, as many as stay resident: each CTA then quantizes the activation once
    // for all of its row blocks). A row's sums do not depend on which CTA takes it.
    const int row_blocks = (a.rows + S::kRowsPerCta - 1) / S::kRowsPerCta;
    int row_block        = static_cast<int>(blockIdx.x);
    int local0           = row_block * S::kRowsPerCta + warp * kRows;
    auto* x_lo           = reinterpret_cast<uint4*>(r8_shared);
    auto* x_hi           = x_lo + NT * blocks;
    auto* x_scale        = reinterpret_cast<float*>(x_hi + NT * blocks);
    auto* x_sum          = x_scale + NT * blocks; // codecs with a min: scale times the codes' sum
    // A codebook codec's grid, after the activation (none under kPre), staged once per CTA: the
    // lookups then read shared memory, not random lines of the global table.
    constexpr int kGridWords = NINFERNO_R8_STAGED_GRID ? r8_grid_words<C>() : 0;
    // 8-byte aligned for SharedGrid::pair (r8_grid_pad).
    auto* grid_words = reinterpret_cast<std::uint32_t*>(
        r8_shared +
        (kPre ? 0 : (r8_smem_bytes(NT, static_cast<std::int64_t>(a.k), C::kMin) + 7) & ~7LL));
    if constexpr (kGridWords > 0) {
        for (int i = static_cast<int>(threadIdx.x); i < kGridWords; i += S::kThreads) {
            grid_words[i] = __ldg(C::grid() + i);
        }
        if constexpr (kPre) __syncthreads();
    }

    Raw ring[2][kSets][kRows][kStride];
    const auto fetch = [&](auto slot_tag, int group) {
        constexpr int slot = decltype(slot_tag)::value;
#pragma unroll
        for (int s = 0; s < kSets; ++s)
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                const int l = local0 + r;
                if (l < a.rows) {
#pragma unroll
                    for (int q = 0; q < kStride; ++q) {
                        ring[slot][s][r][q] = C::load(
                            a, a.row_begin + l + (s == 1 ? a.up_offset : 0), group * kGroup + q);
                    }
                }
            }
    };
    using Slot0 = std::integral_constant<int, 0>;
    using Slot1 = std::integral_constant<int, 1>;
    if (lane < groups) fetch(Slot0{}, lane);

    // GemmArgs::pdl_prefetch: the first group's weights are in flight; x and the output are the
    // previous kernel's to finish. Then a joined next call may start (it reads x without waiting).
    if (a.pdl_prefetch != 0) {
        pdl::wait_for_dependencies();
        if constexpr (E == Epilogue::Store && !kPre) {
            if (a.release_next != 0 && threadIdx.x == 0) pdl::trigger_dependents();
        }
    }

    // The activation: token slots past T hold zeros.
    const auto* x = static_cast<const __nv_bfloat16*>(a.x);
    for (int task = tid; !kPre && task < NT * blocks; task += S::kThreads) {
        const int token   = task / blocks;
        const int b       = task - token * blocks;
        unsigned words[8] = {};
        float scale       = 0.0F;
        int qsum          = 0;
        if (token < a.tokens) {
            float v[32];
            w4a8_load_block(x + static_cast<std::int64_t>(token) * a.ldx + b * 32, v);
            w4a8_quantize_block(v, words, scale, qsum);
        }
        x_lo[task]    = make_uint4(words[0], words[1], words[2], words[3]);
        x_hi[task]    = make_uint4(words[4], words[5], words[6], words[7]);
        x_scale[task] = scale;
        if constexpr (C::kMin) x_sum[task] = scale * static_cast<float>(qsum);
    }
    if constexpr (!kPre) __syncthreads();

#pragma unroll 1
    for (;;) {
        float acc[kSets][kRows][NT];
#pragma unroll
        for (int s = 0; s < kSets; ++s)
#pragma unroll
            for (int r = 0; r < kRows; ++r)
#pragma unroll
                for (int n = 0; n < NT; ++n) acc[s][r][n] = 0.0F;

        // Block kb of every token against block sp of load q of each row (kb = group*kGroup + q +
        // kStride*sp): a group's blocks in ascending order, whatever the codec's grouping.
        const auto compute_block = [&](const auto& raws, int q, int sp, int kb) {
            uint4 xl[NT], xh[NT];
            float xs[NT], xm[NT];
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                if constexpr (kPre) {
                    if (n < a.tokens) {
                        const auto* c = reinterpret_cast<const uint4*>(
                            act.codes + static_cast<std::int64_t>(n) * a.k + kb * 32);
                        xl[n] = c[0];
                        xh[n] = c[1];
                        xs[n] = act.scales[n * blocks + kb];
                        xm[n] = C::kMin ? act.sums[n * blocks + kb] : 0.0F;
                    } else {
                        xl[n] = xh[n] = make_uint4(0, 0, 0, 0);
                        xs[n] = xm[n] = 0.0F;
                    }
                } else {
                    xl[n] = x_lo[n * blocks + kb];
                    xh[n] = x_hi[n * blocks + kb];
                    xs[n] = x_scale[n * blocks + kb];
                    xm[n] = C::kMin ? x_sum[n * blocks + kb] : 0.0F;
                }
            }
            // kHalfMin: each token's code sums of values 0..15 and 16..31 (dp4a against ones).
            [[maybe_unused]] float hl[NT], hh[NT];
            if constexpr (kHalfMin) {
                const uint4 ones = make_uint4(0x01010101U, 0x01010101U, 0x01010101U, 0x01010101U);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    hl[n] =
                        int_accumulator_to_float(dp4a_words(ones, xl[n], int_accumulator_zero()));
                    hh[n] =
                        int_accumulator_to_float(dp4a_words(ones, xh[n], int_accumulator_zero()));
                }
            }
#pragma unroll
            for (int s = 0; s < kSets; ++s)
#pragma unroll
                for (int r = 0; r < kRows; ++r) {
                    if (local0 + r >= a.rows) continue;
                    uint4 lo, hi;
                    float s0 = 0.0F, s1 = 0.0F;
                    [[maybe_unused]] float m0 = 0.0F, m1 = 0.0F;
                    if constexpr (kHalfMin) {
                        C::decode(raws[s][r][q], sp, lo, hi, s0, s1, m0, m1);
                    } else if constexpr (kGridWords > 0) {
                        C::decode(raws[s][r][q], lo, hi, s0, s1, SharedGrid{grid_words});
                    } else if constexpr (kSpan == 1) {
                        C::decode(raws[s][r][q], lo, hi, s0, s1);
                    } else {
                        C::decode(raws[s][r][q], sp, lo, hi, s0, s1);
                    }
#pragma unroll
                    for (int n = 0; n < NT; ++n) {
                        if constexpr (kHalves) {
                            const int d0 = dp4a_words(lo, xl[n], int_accumulator_zero());
                            const int d1 = dp4a_words(hi, xh[n], int_accumulator_zero());
                            float w      = fmaf(int_accumulator_to_float(d1), s1,
                                                int_accumulator_to_float(d0) * s0);
                            if constexpr (kHalfMin) w = fmaf(hh[n], m1, fmaf(hl[n], m0, w));
                            acc[s][r][n] = fmaf(w, xs[n], acc[s][r][n]);
                        } else {
                            const int d = dp4a_words(hi, xh[n],
                                                     dp4a_words(lo, xl[n], int_accumulator_zero()));
                            acc[s][r][n] =
                                fmaf(int_accumulator_to_float(d) * s0, xs[n], acc[s][r][n]);
                            if constexpr (C::kMin) acc[s][r][n] = fmaf(s1, xm[n], acc[s][r][n]);
                        }
                    }
                }
        };
        const auto compute = [&](auto slot_tag, int group) {
            constexpr int slot = decltype(slot_tag)::value;
#pragma unroll
            for (int j = 0; j < kGroup; ++j) {
                compute_block(ring[slot], j % kStride, j / kStride, group * kGroup + j);
            }
        };

#pragma unroll 1
        for (int g = lane; g < groups; g += 64) {
            if (g + 32 < groups) fetch(Slot1{}, g + 32);
            compute(Slot0{}, g);
            if (g + 32 >= groups) break;
            if (g + 64 < groups) fetch(Slot0{}, g + 64);
            compute(Slot1{}, g + 32);
        }

        // The next row block's first weights go in flight before this one's reduction and stores.
        const int row0 = local0;
        row_block += static_cast<int>(gridDim.x);
        const bool more = row_block < row_blocks;
        if (more) {
            local0 = row_block * S::kRowsPerCta + warp * kRows;
            if (lane < groups) fetch(Slot0{}, lane);
        }

        // Butterfly: every lane ends with the same sum (each step adds a pair in either order).
#pragma unroll
        for (int s = 0; s < kSets; ++s)
#pragma unroll
            for (int r = 0; r < kRows; ++r)
#pragma unroll
                for (int n = 0; n < NT; ++n)
#pragma unroll
                    for (int m = 16; m > 0; m >>= 1)
                        acc[s][r][n] += __shfl_xor_sync(0xffffffffU, acc[s][r][n], m);

#pragma unroll
        for (int r = 0; r < kRows; ++r) {
            const int l = row0 + r;
            if (l >= a.rows) continue;
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                if (n >= a.tokens || lane != ((r * NT + n) & 31)) continue;
                float value = acc[0][r][n];
                if constexpr (kSwiGlu) value = generic_gate(a, value) * acc[kSets - 1][r][n];
                store_output<E>(a, l, n, value);
            }
        }
        if (!more) break;
    }
    // A joined launch (Call::join_previous) ran alongside the kernel before it; the grid completes
    // only after that one has, so the stream's next kernel follows both.
    if constexpr (E == Epilogue::Store) {
        if (a.join_previous != 0) pdl::wait_for_dependencies();
    }
}

// ---------------------------------------------------------------------------------------------
// Launch. Keep the schedule list in the order of kR8Schedules in generic_gemm.cpp.
// ---------------------------------------------------------------------------------------------

#define NINFERNO_GENERIC_R8_SCHEDULES(X)                                                           \
    X(0, 4, 1)                                                                                     \
    X(1, 4, 2)                                                                                     \
    X(2, 8, 2)                                                                                     \
    X(3, 8, 1)

template <class C, class S, int NT, Epilogue E, bool kPre = false>
void r8_launch_tokens(const GemmArgs& args, cudaStream_t stream, W4a8Activation act = {}) {
    const auto smem =
        (kPre ? std::size_t{0} : static_cast<std::size_t>(r8_smem_bytes(NT, args.k, C::kMin))) +
        (r8_stage_grid() && r8_grid_words<C>() > 0
             ? sizeof(std::uint32_t) * r8_grid_words<C>() + r8_grid_pad
             : 0);
    const int row_blocks = (args.rows + S::kRowsPerCta - 1) / S::kRowsPerCta;
    int ctas             = row_blocks;
    if (!kPre && r8_resident_grid()) {
        // As many CTAs as the SMs hold at once (the device's SM count times this kernel's
        // occupancy at its block size and shared bytes): each quantizes the activation once.
        static thread_local int device_ = -1, resident_ = 0;
        static thread_local std::size_t smem_ = 0;
        int device                            = 0;
        CUDA_CHECK(cudaGetDevice(&device));
        if (device != device_ || smem != smem_) {
            int per_sm = 0, sms = 0;
            CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(
                &per_sm, r8_kernel<C, S, NT, E, kPre>, S::kThreads, smem));
            CUDA_CHECK(cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, device));
            device_   = device;
            smem_     = smem;
            resident_ = per_sm * sms;
        }
        if (resident_ > 0) ctas = std::min(row_blocks, resident_);
    }
    const dim3 grid(static_cast<unsigned>(ctas));
    // kPre launches in stream order: the quantizer it reads is the kernel right before it.
    if (kPre || (args.join_previous == 0 && args.pdl_prefetch == 0) || !pdl::supported()) {
        r8_kernel<C, S, NT, E, kPre><<<grid, S::kThreads, smem, stream>>>(args, act);
    } else {
        cudaLaunchAttribute attribute{};
        attribute.id = cudaLaunchAttributeProgrammaticStreamSerialization;
        attribute.val.programmaticStreamSerializationAllowed = 1;
        cudaLaunchConfig_t config{};
        config.gridDim          = grid;
        config.blockDim         = dim3(S::kThreads);
        config.dynamicSmemBytes = smem;
        config.stream           = stream;
        config.attrs            = &attribute;
        config.numAttrs         = 1;
        CUDA_CHECK(cudaLaunchKernelEx(&config, r8_kernel<C, S, NT, E, kPre>, args, act));
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class C, class S, Epilogue E, bool kVerify>
void r8_launch_schedule(const GemmArgs& args, cudaStream_t stream, W4a8Activation pre) {
    if (pre.codes != nullptr) {
        if constexpr (kVerify) {
            if (args.tokens <= 2) return r8_launch_tokens<C, S, 2, E, true>(args, stream, pre);
            if (args.tokens <= 4) return r8_launch_tokens<C, S, 4, E, true>(args, stream, pre);
            if (args.tokens <= 8) return r8_launch_tokens<C, S, 8, E, true>(args, stream, pre);
        }
        throw std::invalid_argument("generic R8: no verify-width form for this call");
    }
    if (args.tokens <= 1) return r8_launch_tokens<C, S, 1, E>(args, stream);
    if (args.tokens <= 2) return r8_launch_tokens<C, S, 2, E>(args, stream);
    if (args.tokens <= 4) return r8_launch_tokens<C, S, 4, E>(args, stream);
    throw std::invalid_argument("generic R8: more than kR8MaxTokens tokens");
}

template <class C, Epilogue E, bool kVerify>
void r8_launch_epilogue(int schedule, const GemmArgs& args, cudaStream_t stream,
                        W4a8Activation pre) {
    switch (schedule) {
#define NINFERNO_GENERIC_CASE(I, W, R)                                                             \
    case I:                                                                                        \
        r8_launch_schedule<C, R8Schedule<W, R>, E, kVerify>(args, stream, pre);                    \
        return;
        NINFERNO_GENERIC_R8_SCHEDULES(NINFERNO_GENERIC_CASE)
#undef NINFERNO_GENERIC_CASE
    default:
        throw std::invalid_argument("generic R8: unknown schedule");
    }
}

// kVerify: the codec also has the verify-width form (a pre-quantized activation, T
// = 2..kR8VerifyTokens); r8_verify_compiled lists the same formats.
template <class C, bool kVerify = false>
void r8_launch_codec(int schedule, Epilogue epilogue, const GemmArgs& args, cudaStream_t stream,
                     W4a8Activation pre = {}) {
    // A lane loads kSpan * kStride blocks at once (r8_k_values on the host side).
    if (args.k % (32 * w4a8_span<C>() * w4a8_stride<C>()) != 0) {
        throw std::invalid_argument("generic R8: K is not a whole number of the codec's groups");
    }
    switch (epilogue) {
    case Epilogue::Store:
        r8_launch_epilogue<C, Epilogue::Store, kVerify>(schedule, args, stream, pre);
        return;
    case Epilogue::AddResidual:
        r8_launch_epilogue<C, Epilogue::AddResidual, kVerify>(schedule, args, stream, pre);
        return;
    case Epilogue::SwiGlu:
        r8_launch_epilogue<C, Epilogue::SwiGlu, kVerify>(schedule, args, stream, pre);
        return;
    }
    throw std::invalid_argument("generic R8: unknown epilogue");
}

} // namespace ninferno::ops::detail::generic
