// Volta's decode and speculative-verify widths (T = 1..16) on its tensor cores. sm_70 has no mma.sync of the
// shapes the register-MMA tiles issue (m16n8k16 BF16/F16, m16n8k32 s8): there those tiles run on
// FP32 FMAs over warp shuffles (ops/common/mma.cuh) and ggml's vector kernel re-reads the
// activation for every column on dp4a, so both grow with T and neither streams the weight at the
// DRAM rate (GV100, Qwen3.5-9B Q4_K_M, dflash:7 round: Q6_K LM head 163 GB/s, target verify
// 176-312 GB/s, of 870). Volta's one tensor-core shape is mma.m8n8k4 with f16 operands
// (HMMA.884): this kernel turns each weight code into f16 once, straight from the lane's own loads
// into an A fragment, and multiplies it against eight (or sixteen) token columns.
//
// The m8n8k4 fragments (PTX ISA): a warp runs four independent products, one per quad-pair q
// (lanes 4q..4q+3 and 16+4q..16+4q+3). Lane f of a quad-pair (f = lane % 4, + 4 for the high
// lanes) holds row f of A (k 0..3) and column f of B (k 0..3); accumulator i of a lane is row
// (lane & 1) + (i & 2) (+ 4 for the high lanes), column (i & 4) + (lane & 2) + (i & 1). Which four
// K positions the four k slots stand for is free, as long as A and B agree: so each lane takes the
// codes of its own bytes in the order they come out of them, and the CTA stages the activation in
// that order once (bf16 to f16, as the T >= 17 tiled route does).
//
// Layout: a warp holds eight weight rows (fragment row f = weight row f) and splits each 256-value
// superblock over its quad-pairs, 64 values each (the format's own runs of 32: Q4_K / Q5_K
// sub-blocks 2q and 2q+1, Q6_K quarters q % 2 and q % 2 + 2 of half q / 2, Q8 groups 2q and
// 2q+1). One code word (four bytes) gives two half2 per run: (byte 0, byte 2) and (byte 1, byte 3),
// so the staged activation swaps the middle two of every four values.
//
// Arithmetic: the A operand is the code itself (less its zero point; Q4_K / Q5_K: plus 1024), an
// exact f16 integer, so the tensor cores sum exact products of codes and f16 activations in FP32
// over one scale group (16 or 32 values); the group's scale (and a K-quant's min against the
// group's activation sum) is applied in FP32 after it. Q4_K / Q5_K leave the 1024 the f16 bit
// trick puts under each code in the operand and take 1024 d times the group's activation sum off
// with the min, which already reads that sum (HK45::kBias): one FFMA a group instead of an HSUB2 a
// code pair. The accumulator then holds a sum ~2^7 larger; against FP64 the outputs' error over
// the reduced |w x| stayed as it was without the fold (GV100, K = 4096 / 17408, activations of
// one sign over all of K: max 3.5e-3 / 2.9e-3 both ways; check_volta_hmma_accuracy). No weight
// value is ever rounded to f16, whatever its scale. The groups add in K order, the quad-pairs'
// sums meet at the end in a fixed order ((q0 + q1) + (q2 + q3)): a token's bytes depend on its
// own column only, never on the call's T or its other tokens, and repeat run to run.
#include "core/device.h"
#include "core/device_class.h"
#include "ops/common/mma.cuh"
#include "ops/generic/generic_gemm.cuh"
#include "ops/generic/generic_gemm_internal.h"
#include "ops/generic/generic_volta_hmma.cuh"

#include <cuda_fp16.h>

#include <stdexcept>

namespace ninferno::ops::detail::generic {
namespace {

using namespace volta;

// Stage superblock `sb` of the activation as f16 in fragment order: run src of token t goes to
// quad-pair D / 2, half D % 2 of it (D = dest_run(src)), the middle two of every four swapped.
// For a format with a min term, also each staged run's sum over its 32 values (the f16 values the
// MMA multiplies, summed in FP32 in a fixed order) for token t: sums[t * 8 + D].
template <class F, int NTok>
__device__ __forceinline__ void stage_load(const GemmArgs& a, int sb, uint2 (&v)[NTok / 2]) {
    const auto* x = static_cast<const __nv_bfloat16*>(a.x);
#pragma unroll
    for (int i = 0; i < NTok / 2; ++i) {
        const int task = static_cast<int>(threadIdx.x) + i * kWarps * 32; // NTok * 64 tasks
        const int t    = task >> 6;
        const int g    = task & 63; // four-value group of the superblock
        // Zero past K (a Q8 row ends inside its last superblock; HQ8::load).
        v[i] = t < a.tokens && 256 * sb + 4 * g < a.k
                   ? *reinterpret_cast<const uint2*>(x + static_cast<std::int64_t>(t) * a.ldx +
                                                     256 * sb + 4 * g)
                   : make_uint2(0, 0);
    }
}

template <class F, Epilogue E, int NG, int Ahead>
__global__ __launch_bounds__(kWarps * 32) void volta_hmma_kernel(GemmArgs a) {
#if NINFERNO_HMMA884
    constexpr int kSets   = E == Epilogue::SwiGlu ? 2 : 1;
    constexpr int kNTok   = 8 * NG;
    constexpr int kGroups = F::kGroups;
    constexpr int kSteps  = 16 / kGroups; // MMAs a scale group
    __shared__ __align__(16) __half act[2][4 * kQuadStride<kNTok>];
    __shared__ float sums[2][F::kMin ? kNTok * 8 : 1];

    const int lane  = static_cast<int>(threadIdx.x) & 31;
    const int warp  = static_cast<int>(threadIdx.x) >> 5;
    const int q     = (lane >> 2) & 3;
    const int f     = (lane & 3) + ((lane >> 4) << 2);
    const int l     = static_cast<int>(blockIdx.x) * kCtaRows + warp * kWarpRows + f; // A row
    const bool live = l < a.rows;
    const int sbs   = (a.k + 255) >> 8;
    // The lanes holding the A rows of this lane's accumulators: rows (lane & 1) + 4 (lane / 16)
    // and that + 2, in this quad-pair.
    const int src_a = 4 * q + (lane & 1) + ((lane >> 4) << 4);

    float acc[kSets][NG][8];
#    pragma unroll
    for (int s = 0; s < kSets; ++s)
#    pragma unroll
        for (int g = 0; g < NG; ++g)
#    pragma unroll
            for (int i = 0; i < 8; ++i) acc[s][g][i] = 0.0F;

    // The weight stream Ahead superblocks ahead: cur, then nx1 (and nx2) in flight. Little's law:
    // an SM needs ~5.7 KB in flight for the measured 660 GB/s over ~1000 cycles of latency at the
    // 1.44 GHz it runs under load (GV100), and a CTA keeps 32 rows of a superblock in flight (Q4_K
    // 4.6 KB, SwiGLU 9.2; the other formats' superblocks are larger). With two CTAs or more an SM
    // one superblock ahead covers it; the second one held 23 more registers a lane (gate/up
    // 160 -> 137) and added issue work to a loop that is issue-bound. A grid of fewer CTAs than
    // two an SM keeps the second (launch_format).
    F cur[kSets], nx1[kSets], nx2[kSets];
#    pragma unroll
    for (int s = 0; s < kSets; ++s) {
        const int row = a.row_begin + l + (s == 1 ? a.up_offset : 0);
        cur[s].clear();
        nx1[s].clear();
        nx2[s].clear();
        if (live) {
            cur[s].load(a, row, 0, q);
            if (Ahead == 2 && sbs > 1) nx1[s].load(a, row, 1, q);
        }
    }
    uint2 xv[kNTok / 2];
    stage_load<F, kNTok>(a, 0, xv);
    stage_store<F, kNTok>(act[0], sums[0], xv);
    __syncthreads();

#    pragma unroll 1
    for (int sb = 0; sb < sbs; ++sb) {
        const bool more = sb + 1 < sbs;
#    pragma unroll
        for (int s = 0; s < kSets; ++s) {
            const int row = a.row_begin + l + (s == 1 ? a.up_offset : 0);
            if (Ahead == 1 && more && live) nx1[s].load(a, row, sb + 1, q);
            if (Ahead == 2 && sb + 2 < sbs && live) nx2[s].load(a, row, sb + 2, q);
        }
        if (more) stage_load<F, kNTok>(a, sb + 1, xv);

        const __half* at = act[sb & 1] + q * kQuadStride<kNTok>;
#    pragma unroll
        for (int gr = 0; gr < kGroups; ++gr) {
            float part[kSets][NG][8];
#    pragma unroll
            for (int s = 0; s < kSets; ++s)
#    pragma unroll
                for (int g = 0; g < NG; ++g)
#    pragma unroll
                    for (int i = 0; i < 8; ++i) part[s][g][i] = 0.0F;
#    pragma unroll
            for (int st = 0; st < kSteps; ++st) {
                const int m = gr * kSteps + st;
                uint2 bf[NG];
#    pragma unroll
                for (int g = 0; g < NG; ++g) {
                    bf[g] = *reinterpret_cast<const uint2*>(
                        at + (m * kNTok + token_slot<kNTok>(m, 8 * g + f)) * 4);
                }
#    pragma unroll
                for (int s = 0; s < kSets; ++s) {
                    unsigned a0, a1;
                    cur[s].fragment(m, q, a0, a1);
#    pragma unroll
                    for (int g = 0; g < NG; ++g)
                        mma_m8n8k4_f16(part[s][g], a0, a1, bf[g].x, bf[g].y);
                }
            }
            // The group's scale (and min) of the accumulators' rows, from the lanes holding them.
#    pragma unroll
            for (int s = 0; s < kSets; ++s) {
                float ds, dm;
                cur[s].scale(q, gr, ds, dm);
                if constexpr (requires { F::kBias; }) dm = fmaf(F::kBias, ds, dm);
                const float ds0 = __shfl_sync(0xffffffffU, ds, src_a);
                const float ds2 = __shfl_sync(0xffffffffU, ds, src_a + 2);
                float dm0 = 0.0F, dm2 = 0.0F;
                if constexpr (F::kMin) {
                    dm0 = __shfl_sync(0xffffffffU, dm, src_a);
                    dm2 = __shfl_sync(0xffffffffU, dm, src_a + 2);
                }
#    pragma unroll
                for (int g = 0; g < NG; ++g)
#    pragma unroll
                    for (int i = 0; i < 8; ++i) {
                        const float d = (i & 2) ? ds2 : ds0;
                        acc[s][g][i]  = fmaf(d, part[s][g][i], acc[s][g][i]);
                        if constexpr (F::kMin) {
                            const int token = 8 * g + (i & 4) + (lane & 2) + (i & 1);
                            const float xs  = sums[sb & 1][token * 8 + 2 * q + gr];
                            acc[s][g][i]    = fmaf(-((i & 2) ? dm2 : dm0), xs, acc[s][g][i]);
                        }
                    }
            }
        }
        if (more) stage_store<F, kNTok>(act[(sb + 1) & 1], sums[(sb + 1) & 1], xv);
        __syncthreads();
#    pragma unroll
        for (int s = 0; s < kSets; ++s) {
            cur[s] = nx1[s];
            if (Ahead == 2) nx1[s] = nx2[s];
        }
    }

    // The quad-pairs' partial sums: ((q0 + q1) + (q2 + q3)) in every lane of the four.
#    pragma unroll
    for (int s = 0; s < kSets; ++s)
#    pragma unroll
        for (int g = 0; g < NG; ++g)
#    pragma unroll
            for (int i = 0; i < 8; ++i) {
                acc[s][g][i] += __shfl_xor_sync(0xffffffffU, acc[s][g][i], 4);
                acc[s][g][i] += __shfl_xor_sync(0xffffffffU, acc[s][g][i], 8);
            }
    // Quad-pair q stores accumulators 2q and 2q + 1.
    const int row0 = static_cast<int>(blockIdx.x) * kCtaRows + warp * kWarpRows;
#    pragma unroll
    for (int i = 0; i < 8; ++i) {
        if ((i >> 1) != q) continue;
        const int row = row0 + (lane & 1) + (i & 2) + ((lane >> 4) << 2);
        if (row >= a.rows) continue;
#    pragma unroll
        for (int g = 0; g < NG; ++g) {
            const int token = 8 * g + (i & 4) + (lane & 2) + (i & 1);
            if (token >= a.tokens) continue;
            float value = acc[0][g][i];
            if constexpr (E == Epilogue::SwiGlu)
                value = generic_gate(a, value) * acc[kSets - 1][g][i];
            store_output<E>(a, row, token, value);
        }
    }
#else
    __trap();
#endif
}

template <class F, Epilogue E>
void launch_format(const GemmArgs& args, cudaStream_t stream) {
    const unsigned blocks = static_cast<unsigned>((args.rows + kCtaRows - 1) / kCtaRows);
    // Two CTAs an SM or more: one superblock ahead (volta_hmma_kernel). GV100 dflash:7 round:
    // gate/up (384 CTAs) 156.5 -> 152.6 us; a 4096-row add (128 CTAs) 71.1 -> 73.2 one ahead.
    const bool one_ahead = blocks >= 2U * static_cast<unsigned>(device_class().sms);
    if (args.tokens <= 8) {
        if (one_ahead) {
            volta_hmma_kernel<F, E, 1, 1><<<blocks, kWarps * 32, 0, stream>>>(args);
        } else {
            volta_hmma_kernel<F, E, 1, 2><<<blocks, kWarps * 32, 0, stream>>>(args);
        }
    } else if (one_ahead) {
        volta_hmma_kernel<F, E, 2, 1><<<blocks, kWarps * 32, 0, stream>>>(args);
    } else {
        volta_hmma_kernel<F, E, 2, 2><<<blocks, kWarps * 32, 0, stream>>>(args);
    }
    CUDA_CHECK(cudaGetLastError());
}

template <class F>
void launch_epilogue(Epilogue epilogue, const GemmArgs& args, cudaStream_t stream) {
    switch (epilogue) {
    case Epilogue::Store:
        launch_format<F, Epilogue::Store>(args, stream);
        return;
    case Epilogue::AddResidual:
        launch_format<F, Epilogue::AddResidual>(args, stream);
        return;
    case Epilogue::SwiGlu:
        launch_format<F, Epilogue::SwiGlu>(args, stream);
        return;
    }
}

} // namespace

bool volta_hmma_codec(WeightCodec codec) {
    return codec == WeightCodec::GgmlQ4K || codec == WeightCodec::GgmlQ5K ||
           codec == WeightCodec::GgmlQ6K || codec == WeightCodec::Q8;
}

int volta_hmma_k_multiple(WeightCodec codec) { return codec == WeightCodec::Q8 ? 64 : 256; }

bool volta_hmma_takes(WeightCodec codec, const GemmArgs& args) {
    // A join (Call::join_previous) needs programmatic dependent launch, which sm_70 does not have;
    // the segmented chain asks for one only after a call that released, so a later part of a
    // segmented projection takes this kernel (and the lane-local GEMV, which refuses a join) as a
    // separate call does.
    if (args.lane_local == 0 || args.tokens < kVoltaHmmaMinTokens ||
        args.tokens > kVoltaHmmaMaxTokens) {
        return false;
    }
    // Whole superblocks, or for Q8 G32 whole pairs of groups (gpt-oss's K = 2880: its LM head ran
    // the register-MMA tile on FP32 FMAs, 1.9 ms a token on the GV100); the staging reads four
    // bf16 values at a time.
    return volta_hmma_codec(codec) && (args.k % volta_hmma_k_multiple(codec)) == 0 &&
           (args.ldx % 4) == 0 &&
           (reinterpret_cast<std::uintptr_t>(args.x) % 8) == 0;
}

void launch_volta_hmma(WeightCodec codec, Epilogue epilogue, const GemmArgs& args,
                       cudaStream_t stream) {
    switch (codec) {
    case WeightCodec::GgmlQ4K:
        launch_epilogue<HK45<4>>(epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ5K:
        launch_epilogue<HK45<5>>(epilogue, args, stream);
        return;
    case WeightCodec::GgmlQ6K:
        launch_epilogue<HK6>(epilogue, args, stream);
        return;
    case WeightCodec::Q8:
        launch_epilogue<HQ8>(epilogue, args, stream);
        return;
    default:
        throw std::invalid_argument("volta_hmma: unsupported codec");
    }
}

} // namespace ninferno::ops::detail::generic
