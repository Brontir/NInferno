#pragma once

// The grouped R8 (W8A8 decode) kernel of the generic MoE Op: codebook experts (IQ2_XXS, IQ2_XS,
// IQ2_S, IQ3_XXS, IQ3_S), the K-quants (Q3_K, Q4_K, Q5_K, Q6_K) and the 32-value ggml blocks
// (Q4_0, Q4_1, Q5_0, Q5_1, IQ4_NL, IQ4_XS) at decode widths under an A8 expert policy
// (MoeWeights::expert_policy).
//
// It is the generic path's R8 (ops/generic/generic_r8.cuh, llama.cpp's MMVQ) with the MoE
// addressing of moe_gemm.cuh: blockIdx.y is a job (one expert and its slots, at most NT of them at
// these widths: a token selects an expert once), blockIdx.x a run of that expert's output rows.
// Every CTA quantizes its job's activation columns to int8 per 32 values into shared memory with
// the W4A8 quantizer's arithmetic (w4a8_quantize_block, as R8 does), while its first weight blocks
// are in flight. A warp owns kRows output rows (GateUp: each with its gate and up row), its lanes
// take whole 32-value blocks through the format's W4A8 codec (the grid read through L1 with
// __ldg: no per-CTA staging), eight dp4a per block and slot, and one butterfly ends each row.
// A slot's result depends on its own activation only: its bytes do not change with T or the job.
//
// A short K (a down bank's expert width: K 512 of Q5_K is eight loads of two blocks) would leave
// most of a warp's lanes idle, so the warp splits into 32 / a.r8_lanes groups of a.r8_lanes lanes
// (8 or 16, the fewest that cover the row's loads; moe.cpp r8_lanes), each on its own kRows rows,
// and a CTA covers that many times more rows. A lane then holds the same one load per row as at 32
// lanes and the butterfly adds the same values (the lanes past the loads hold +0): the bytes do
// not depend on the split.

#include "ops/generic/generic_r8.cuh"
#include "ops/moe/moe_gemm.cuh"

#include <cuda_bf16.h>

#include <cassert>
#include <cstdint>
#include <stdexcept>
#include <type_traits>

namespace ninferno::ops::detail::moe {

// The R8 codecs of a ggml_frag_v1 bank (core/ggml_frag.h): the published codec's decode over the
// same bytes, gathered from their fragment-order positions.
//   IQ3_XXS: sub-block ib's index bytes t = 0..7 sit at 32h + 8(t % 4) + 2(ib % 4) + t / 4
//            (h = ib / 4): four 2-byte loads, each holding bytes t and t + 4.
//   IQ2_S:   qs byte l of sub-block ib at 32h + 16(l % 2) + 2(ib % 4) + l / 2 and its sign byte
//            8 bytes further: two 2-byte loads each, holding bytes l and l + 2.
struct W4a8Iq3XxsFrag : generic::W4a8Iq3Xxs {
    __device__ static __forceinline__ Raw load(const generic::GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int ib           = kb & 7;
        const auto* half       = reinterpret_cast<const unsigned short*>(
            a.codes + blk * 64 + 32 * (ib >> 2) + 2 * (ib & 3));
        const unsigned u0 = half[0], u1 = half[4], u2 = half[8], u3 = half[12];
        const unsigned lo = __byte_perm(u0, u1, 0x5140), hi = __byte_perm(u2, u3, 0x5140);
        return {make_uint2(__byte_perm(lo, hi, 0x5410), __byte_perm(lo, hi, 0x7632)),
                reinterpret_cast<const unsigned*>(a.high + blk * 32)[ib],
                reinterpret_cast<const unsigned short*>(a.scales)[blk]};
    }
};

struct W4a8Iq2SFrag : generic::W4a8Iq2S {
    __device__ static __forceinline__ Raw load(const generic::GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const int ib           = kb & 7;
        const std::uint8_t* cb = a.codes + blk * 72;
        const auto* half =
            reinterpret_cast<const unsigned short*>(cb + 32 * (ib >> 2) + 2 * (ib & 3));
        // Halfwords 0 / 8: qs bytes (0, 2) / (1, 3); 4 / 12: the same sign bytes.
        return {__byte_perm(half[0], half[8], 0x5140), __byte_perm(half[4], half[12], 0x5140),
                cb[64 + ib], a.high[blk * 8 + ib],
                reinterpret_cast<const unsigned short*>(a.scales)[blk]};
    }
};

//   Q4_K:    each 64-byte half of a superblock's 128 nibble bytes is 4 x 4-transposed by words
//            (new[16q + 4i + t] = old[16i + 4q + t]): sub-blocks 2p and 2p + 1 are the nibbles of
//            16-byte chunks i = 2(p % 2) and i + 1 of half p / 2, whose words q sit side by side
//            at 16q + 4i: four 8-byte loads.
//   Q4_0:    each whole 64-byte span of four blocks of a row is transposed the same way (block
//            kb % 4 = i, word q at 16q + 4i of its span); a row's last K/32 % 4 blocks stay
//            natural.
struct W4a8Q4KFrag : generic::W4a8Q45K<4> {
    __device__ static __forceinline__ Raw load(const generic::GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + (kb >> 3);
        const std::uint8_t* b  = a.codes + blk * kBytes;
        const int p            = (kb & 7) >> 1;
        const std::uint8_t* q  = b + 16 + 64 * (p >> 1) + 8 * (p & 1);
        uint2 w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) w[i] = *reinterpret_cast<const uint2*>(q + 16 * i);
        Raw r;
        r.head = *reinterpret_cast<const uint4*>(b);
        r.qa   = make_uint4(w[0].x, w[1].x, w[2].x, w[3].x);
        r.qb   = make_uint4(w[0].y, w[1].y, w[2].y, w[3].y);
        r.ha = r.hb = make_uint4(0, 0, 0, 0);
        r.j         = static_cast<unsigned>(kb & 7);
        return r;
    }
};

struct W4a8Q40Frag : generic::W4a8Ggml32<generic::Ggml32::Q4_0> {
    __device__ static __forceinline__ Raw load(const generic::GemmArgs& a, int row, int kb) {
        const std::int64_t blk = static_cast<std::int64_t>(row) * a.groups_per_row + kb;
        Raw r;
        if (kb < (a.groups_per_row & ~3)) {
            const auto* span =
                reinterpret_cast<const unsigned*>(a.codes + (blk - (kb & 3)) * 16) + (kb & 3);
            r.qs = make_uint4(span[0], span[4], span[8], span[12]);
        } else {
            r.qs = *reinterpret_cast<const uint4*>(a.codes + blk * 16);
        }
        r.qh = 0U;
        r.dm = reinterpret_cast<const unsigned short*>(a.scales)[blk];
        return r;
    }
};

// The fused combine (GemmArgs::combine_tickets): every CTA of a run of rows [r0, r1) counts
// itself in once its y stores are made visible; the run's last one adds, for each token and row,
// the token's experts in rank order with combine_kernel's arithmetic (fmaf by the routing weight
// from zero, then the residual), so the bytes are the separate combine's. It sets the ticket back
// to zero for the next call.
__device__ __forceinline__ void fused_combine_tail(const GemmArgs& a, int r0, int r1) {
    // Every thread's y stores before the CTA counts itself in (no static shared flag: the barrier
    // reduces it; a function-local __shared__ of a template cost each sm_89 kernel 48 KiB at -rdc
    // device link).
    __threadfence();
    __syncthreads();
    int mine = 0;
    if (threadIdx.x == 0) {
        mine = atomicAdd(a.combine_tickets + blockIdx.x, 1) + 1 == *a.job_total;
        if (mine) {
            a.combine_tickets[blockIdx.x] = 0;
            __threadfence();
        }
    }
    if (__syncthreads_or(mine) == 0) return;
    const CombineArgs& c = a.combine;
    const auto* y        = static_cast<const __nv_bfloat16*>(c.y);
    const int width      = r1 - r0;
    for (int task = static_cast<int>(threadIdx.x); task < c.tokens * width;
         task += static_cast<int>(blockDim.x)) {
        const int t          = task / width;
        const int h          = r0 + (task - t * width);
        const int* slots     = c.slot_of + static_cast<std::int64_t>(t) * c.top_k;
        const float* weights = c.weights + static_cast<std::int64_t>(t) * c.top_k;
        auto* out =
            static_cast<__nv_bfloat16*>(c.out) + static_cast<std::int64_t>(t) * c.hidden + h;
        float acc = 0.0F;
        for (int k = 0; k < c.top_k; ++k) {
            const float v =
                __bfloat162float(__ldcg(y + static_cast<std::int64_t>(slots[k]) * c.hidden + h));
            acc = fmaf(weights[k], v, acc);
        }
        if (c.add_residual) acc += __bfloat162float(*out);
        *out = __float2bfloat16_rn(acc);
    }
}

template <class C, class S, int NT, Stage St, bool kPre = false>
__global__ __launch_bounds__(S::kThreads) void moe_r8_kernel(GemmArgs a) {
    static_assert(St != Stage::Router, "the router runs on the register MMA");
    constexpr int kSets    = St == Stage::GateUp ? 2 : 1;
    constexpr bool kFirst  = St == Stage::GateUp || St == Stage::Up; // slots gather tokens
    constexpr int kRows    = S::kRows;
    constexpr bool kHalves = generic::w4a8_halves<C>();
    constexpr int kSpan    = generic::w4a8_span<C>();
    constexpr int kStride  = generic::w4a8_stride<C>();
    constexpr int kGroup   = kSpan * kStride;
    static_assert(4 % kGroup == 0, "a 128-value K tile holds whole groups");
    using Raw = typename C::Raw;
    constexpr bool kOneSlot = std::is_same_v<C, generic::W4a8Q6K> || std::is_same_v<C, generic::W4a8Q3K>;
    extern __shared__ __align__(16) unsigned char moe_r8_shared[];

    // GemmArgs::pdl: the plan and the activation come from the producer.
    pdl::wait_for_dependencies();
    if (threadIdx.x == 0) pdl::trigger_dependents();
    const int job = static_cast<int>(blockIdx.y);
    if (job >= *a.job_total) return;
    const int expert = a.job_expert[job];
    const int col0   = a.job_begin[job];
    // At most NT slots: the launch takes NT >= T, and a job has at most T.
    assert(a.job_count[job] <= NT);
    const int count = min(NT, a.job_count[job]);

    const int tid    = static_cast<int>(threadIdx.x);
    const int lane   = tid & 31;
    const int warp   = tid >> 5;
    const int blocks = a.w.k / 32;
    const int groups = blocks / kGroup;
    const int lanes  = a.r8_lanes;
    const int sl     = lane & (lanes - 1); // lane within its row group
    const int local0 = static_cast<int>(blockIdx.x) * S::kRowsPerCta * (32 / lanes) +
                       (warp * (32 / lanes) + lane / lanes) * kRows;
    const std::int64_t row_base = static_cast<std::int64_t>(expert) * a.expert_rows;
    auto* x_lo    = reinterpret_cast<uint4*>(moe_r8_shared);
    auto* x_hi    = x_lo + NT * blocks;
    auto* x_scale = reinterpret_cast<float*>(x_hi + NT * blocks);
    auto* x_sum   = x_scale + NT * blocks; // codecs with a min: scale times the codes' sum

    const auto weight_row = [&](int l, int s) -> int {
        return static_cast<int>(row_base + static_cast<std::int64_t>(l) * a.row_step +
                                (s == 1 ? a.up_offset : 0));
    };

    Raw ring[kOneSlot ? 1 : 2][kSets][kRows][kStride];
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
                        ring[slot][s][r][q] = C::load(a.w, weight_row(l, s), group * kGroup + q);
                    }
                }
            }
    };
    using Slot0 = std::integral_constant<int, 0>;
    using Slot1 = std::integral_constant<int, 1>;
    if (sl < groups) fetch(Slot0{}, sl);

    // The job's activation columns (GateUp: the slot's token; Down: the slot's intermediate row);
    // slots past the job's count hold zeros.
    // kPre (GemmArgs::x_codes): the router's CTAs quantized x already; a slot reads its token's
    // row of codes.
    const auto* x = static_cast<const __nv_bfloat16*>(a.x);
    int src_token[NT];
#pragma unroll
    for (int n = 0; n < NT; ++n) src_token[n] = kPre && n < count ? a.packed_token[col0 + n] : 0;
    for (int task = tid; !kPre && task < NT * blocks; task += S::kThreads) {
        const int n = task / blocks;
        const int b = task - n * blocks;
        unsigned words[8] = {};
        float scale       = 0.0F;
        int qsum          = 0;
        if (n < count) {
            int src = col0 + n;
            if constexpr (kFirst) src = a.packed_token[src];
            float v[32];
            generic::w4a8_load_block(x + static_cast<std::int64_t>(src) * a.ldx + b * 32, v);
            generic::w4a8_quantize_block(v, words, scale, qsum);
        }
        x_lo[task]    = make_uint4(words[0], words[1], words[2], words[3]);
        x_hi[task]    = make_uint4(words[4], words[5], words[6], words[7]);
        x_scale[task] = scale;
        if constexpr (C::kMin) x_sum[task] = scale * static_cast<float>(qsum);
    }
    if constexpr (!kPre) __syncthreads();

    float acc[kSets][kRows][NT];
#pragma unroll
    for (int s = 0; s < kSets; ++s)
#pragma unroll
        for (int r = 0; r < kRows; ++r)
#pragma unroll
            for (int n = 0; n < NT; ++n) acc[s][r][n] = 0.0F;

    // Block kb of every slot against block sp of load q of each row (kb = group*kGroup + q +
    // kStride*sp): a group's blocks in ascending order, as generic R8 sums them.
    const auto compute_block = [&](const auto& raws, int q, int sp, int kb) {
        uint4 xl[NT], xh[NT];
        float xs[NT], xm[NT];
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            if constexpr (kPre) {
                const std::int64_t at =
                    static_cast<std::int64_t>(src_token[n]) * (a.x_ld / 32) + kb;
                const auto* c = reinterpret_cast<const uint4*>(a.x_codes + at * 32);
                xl[n]         = c[0];
                xh[n]         = c[1];
                xs[n]         = n < count ? a.x_scales[at] : 0.0F;
                xm[n]         = C::kMin && n < count ? a.x_sums[at] : 0.0F;
            } else {
                xl[n] = x_lo[n * blocks + kb];
                xh[n] = x_hi[n * blocks + kb];
                xs[n] = x_scale[n * blocks + kb];
                xm[n] = C::kMin ? x_sum[n * blocks + kb] : 0.0F;
            }
        }
#pragma unroll
        for (int s = 0; s < kSets; ++s)
#pragma unroll
            for (int r = 0; r < kRows; ++r) {
                if (local0 + r >= a.rows) continue;
                uint4 lo, hi;
                float s0 = 0.0F, s1 = 0.0F;
                if constexpr (kSpan == 1) {
                    C::decode(raws[s][r][q], lo, hi, s0, s1);
                } else {
                    C::decode(raws[s][r][q], sp, lo, hi, s0, s1);
                }
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    if (n >= count) break; // CTA-uniform: an empty slot adds nothing
                    if constexpr (kHalves) {
                        const int d0  = generic::dp4a_words(lo, xl[n], int_accumulator_zero());
                        const int d1  = generic::dp4a_words(hi, xh[n], int_accumulator_zero());
                        const float w = fmaf(int_accumulator_to_float(d1), s1,
                                             int_accumulator_to_float(d0) * s0);
                        acc[s][r][n] = fmaf(w, xs[n], acc[s][r][n]);
                    } else {
                        const int d = generic::dp4a_words(
                            hi, xh[n], generic::dp4a_words(lo, xl[n], int_accumulator_zero()));
                        acc[s][r][n] = fmaf(int_accumulator_to_float(d) * s0, xs[n], acc[s][r][n]);
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

    if constexpr (kOneSlot) {
        // One slot: the codec's loads of a row set hold too many registers to keep two in flight
        // (EXPERIMENT); the CTA's warps hide the latency instead.
        if (sl < groups) compute(Slot0{}, sl);
#pragma unroll 1
        for (int g = sl + lanes; g < groups; g += lanes) {
            fetch(Slot0{}, g);
            compute(Slot0{}, g);
        }
    } else {
#pragma unroll 1
        for (int g = sl; g < groups; g += 2 * lanes) {
            if (g + lanes < groups) fetch(Slot1{}, g + lanes);
            compute(Slot0{}, g);
            if (g + lanes >= groups) break;
            if (g + 2 * lanes < groups) fetch(Slot0{}, g + 2 * lanes);
            compute(Slot1{}, g + lanes);
        }
    }

    // Butterfly within each row group: every lane ends with its group's sum (each step adds a pair
    // in either order); the job's empty slots take none.
#pragma unroll
    for (int s = 0; s < kSets; ++s)
#pragma unroll
        for (int r = 0; r < kRows; ++r)
#pragma unroll
            for (int n = 0; n < NT; ++n) {
                if (n >= count) break; // CTA-uniform
                for (int m = lanes >> 1; m > 0; m >>= 1)
                    acc[s][r][n] += __shfl_xor_sync(0xffffffffU, acc[s][r][n], m);
            }

    // The epilogue of moe_gemm_kernel (the R8 codecs carry every scale: no row scale).
#pragma unroll
    for (int r = 0; r < kRows; ++r) {
        const int l = local0 + r;
        if (l >= a.rows) continue;
#pragma unroll
        for (int n = 0; n < NT; ++n) {
            if (n >= count || sl != ((r * NT + n) & (lanes - 1))) continue;
            const int row0 = weight_row(l, 0);
            float value    = acc[0][r][n] + moe_bias(a, row0);
            if constexpr (St == Stage::GateUp) {
                const int row1 = weight_row(l, 1);
                value          = moe_activation(a, value, acc[kSets - 1][r][n] + moe_bias(a, row1));
            } else if constexpr (St == Stage::Up) {
                value = moe_activation(a, value, value);
            }
            static_cast<__nv_bfloat16*>(a.out_bf16)[static_cast<std::int64_t>(col0 + n) * a.ldo +
                                                    l] = __float2bfloat16_rn(value);
        }
    }
    if constexpr (St == Stage::Down) {
        if (a.combine_tickets != nullptr) {
            const int r0 = static_cast<int>(blockIdx.x) * S::kRowsPerCta * (32 / lanes);
            fused_combine_tail(a, r0, min(a.rows, r0 + S::kRowsPerCta * (32 / lanes)));
        }
    }
}

template <class C, class S, int NT, Stage St, bool kPre>
void launch_moe_r8_kernel(const GemmArgs& args, dim3 grid, std::size_t smem, cudaStream_t stream) {
    CUDA_CHECK(pdl::launch({grid, dim3(S::kThreads), smem, stream}, args.pdl != 0, false,
                           moe_r8_kernel<C, S, NT, St, kPre>, args));
}

template <class C, class S, int NT, Stage St>
void launch_moe_r8_tokens(const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    if constexpr (St != Stage::Down) {
        if (args.x_codes != nullptr)
            return launch_moe_r8_kernel<C, S, NT, St, true>(args, grid, 0, stream);
    }
    const auto smem = static_cast<std::size_t>(generic::r8_smem_bytes(NT, args.w.k, C::kMin));
    launch_moe_r8_kernel<C, S, NT, St, false>(args, grid, smem, stream);
}

template <class C, class S, Stage St>
void launch_moe_r8_schedule(int tokens, const GemmArgs& args, dim3 grid, cudaStream_t stream) {
    if (tokens <= 1) return launch_moe_r8_tokens<C, S, 1, St>(args, grid, stream);
    if (tokens <= 2) return launch_moe_r8_tokens<C, S, 2, St>(args, grid, stream);
    if (tokens <= 4) return launch_moe_r8_tokens<C, S, 4, St>(args, grid, stream);
    throw std::invalid_argument("moe R8: more than kR8Tokens tokens");
}

template <class C, Stage St>
void launch_moe_r8_stage(int schedule, int tokens, const GemmArgs& args, dim3 grid,
                         cudaStream_t stream) {
    switch (schedule) {
#define NINFERNO_MOE_R8_CASE(I, W, R)                                                              \
    case I:                                                                                        \
        launch_moe_r8_schedule<C, generic::R8Schedule<W, R>, St>(tokens, args, grid, stream);      \
        return;
        NINFERNO_MOE_R8_SCHEDULES(NINFERNO_MOE_R8_CASE)
#undef NINFERNO_MOE_R8_CASE
    default:
        throw std::invalid_argument("moe R8: unknown schedule");
    }
}

template <class C>
void launch_moe_r8(Stage stage, int schedule, int tokens, const GemmArgs& args, dim3 grid,
                   cudaStream_t stream) {
    switch (stage) {
    case Stage::GateUp:
        launch_moe_r8_stage<C, Stage::GateUp>(schedule, tokens, args, grid, stream);
        return;
    case Stage::Down:
        launch_moe_r8_stage<C, Stage::Down>(schedule, tokens, args, grid, stream);
        return;
    case Stage::Up:
        launch_moe_r8_stage<C, Stage::Up>(schedule, tokens, args, grid, stream);
        return;
    case Stage::Router:
        break;
    }
    throw std::invalid_argument("moe R8: the router runs on the register MMA");
}

} // namespace ninferno::ops::detail::moe
