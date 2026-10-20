#pragma once

#include "ops/common/mma.cuh"
#include "ops/linear/q4/q4_schedule.cuh"

namespace ninferno::ops::detail {

template <int BlockRows>
struct Q4IdentityRows {
    static constexpr int kOutputRowsPerCta = BlockRows;

    __host__ __device__ int output_rows(int rows) const { return rows; }

    __device__ __forceinline__ int weight_row(int row_begin, int local_row) const {
        return row_begin + local_row;
    }
};

__device__ __forceinline__ int q4_sliced_k_swizzle(int row, int col) {
    return (((col >> 3) ^ (row & 7)) << 3) | (col & 7);
}

template <class Schedule, bool Full, bool FullTokens, class Output, class Epilogue, class RowPolicy>
__global__
__launch_bounds__(Schedule::kThreads, Schedule::kMinBlocksPerSm) void q4_a16_sliced_k_mma_kernel(
    const __nv_bfloat16* __restrict__ x, const std::uint8_t* __restrict__ codes,
    const std::uint8_t* __restrict__ scales, Output output, Epilogue epilogue, int rows, int k,
    int tokens, int padded_k, int token_begin, RowPolicy row_policy) {
    constexpr int R  = Schedule::kBlockRows;
    constexpr int T  = Schedule::kBlockTokens;
    constexpr int W  = Schedule::kKWarps;
    constexpr int S  = Schedule::kStages;
    constexpr int MT = Schedule::kMmaRows;
    constexpr int NT = Schedule::kMmaTokens;

    // A code row holds the W warps' 32-byte groups and a 32-byte pad: a half-warp's 64-bit code
    // loads span four rows, and the pad staggers them by eight banks (row stride 32 or 96 mod 128)
    // where W * 32 bytes alone would put all four on the same banks.
    constexpr int kCodeRow = W * 32 + Schedule::kCodeRowPad;
    union SharedStorage {
        struct {
            alignas(16) std::uint8_t codes[S][R][kCodeRow];
            alignas(16) std::uint16_t scales[S][R][W];
            alignas(16) __nv_bfloat16 x[S][W][T][64];
        } staging;

        float partial[W][MT][NT][32][4];
    };

    static_assert(sizeof(SharedStorage) == Schedule::kSharedBytes);
    __shared__ SharedStorage shared;
    const int tid        = static_cast<int>(threadIdx.x);
    const int warp       = tid >> 5;
    const int lane       = tid & 31;
    const int gid        = lane >> 2;
    const int lid        = lane & 3;
    const int row_begin  = static_cast<int>(blockIdx.x) * RowPolicy::kOutputRowsPerCta;
    const int token0     = static_cast<int>(blockIdx.y) * Schedule::kTokenCapacity;
    const int logical_k  = Schedule::kStaticK ? Schedule::kStaticK : k;
    const int groups     = logical_k / 64;
    const int row_groups = (Schedule::kStaticK ? Schedule::kStaticK : padded_k) / 64;
    const int tiles      = (groups + W - 1) / W;

    const auto stage_inputs = [&](int slot, int tile) {
        const int first_group = tile * W;
        for (int item = tid; item < R * W * 2; item += Schedule::kThreads) {
            const int half   = item & 1;
            const int group  = (item >> 1) % W;
            const int row    = item / (W * 2);
            const bool valid = Full || (row_policy.weight_row(row_begin, row) < rows &&
                                        first_group + group < groups);
            const std::int64_t index =
                valid ? std::int64_t(row_policy.weight_row(row_begin, row)) * row_groups +
                            first_group + group
                      : 0;
            cp_async_zfill<16, Schedule::kWeightCache>(
                &shared.staging.codes[slot][row][group * 32 + half * 16],
                codes + index * 32 + half * 16, valid ? 16 : 0);
        }
        if constexpr (Full) {
            for (int row = tid; row < R; row += Schedule::kThreads) {
                const std::int64_t index =
                    std::int64_t(row_policy.weight_row(row_begin, row)) * row_groups + first_group;
                cp_async<W * 2>(&shared.staging.scales[slot][row][0], scales + index * 2);
            }
        } else {
            // Partial K tiles only guarantee scale-pair alignment of each stored row.
            for (int item = tid; item < R * (W / 2); item += Schedule::kThreads) {
                const int row  = item / (W / 2);
                const int pair = (item % (W / 2)) * 2;
                const bool valid =
                    row_policy.weight_row(row_begin, row) < rows && first_group + pair < groups;
                const std::int64_t index =
                    valid ? std::int64_t(row_policy.weight_row(row_begin, row)) * row_groups +
                                first_group + pair
                          : 0;
                cp_async_zfill<4>(&shared.staging.scales[slot][row][pair], scales + index * 2,
                                  valid ? 4 : 0);
            }
        }
        for (int item = lane; item < Schedule::kTokenCapacity * 8; item += 32) {
            const int token = item / 8;
            const int k8    = (item % 8) * 8;
            const bool valid =
                (FullTokens || token0 + token < tokens) && (Full || first_group + warp < groups);
            const std::int64_t index =
                valid ? std::int64_t(token0 + token) * logical_k + (first_group + warp) * 64 + k8
                      : 0;
            cp_async_zfill<16, Schedule::kActivationCache>(
                &shared.staging.x[slot][warp][token][q4_sliced_k_swizzle(token, k8)], x + index,
                valid ? 16 : 0);
        }
        cp_commit();
    };

    float acc[MT][NT][4] = {};
#pragma unroll
    for (int slot = 0; slot < S; ++slot) {
        if (slot < tiles)
            stage_inputs(slot, slot);
        else
            cp_commit();
    }
    constexpr int kUnroll = Schedule::kStaticK ? (Schedule::kStaticK / 64 + W - 1) / W : 1;
#pragma unroll kUnroll
    for (int tile = 0; tile < tiles; ++tile) {
        const int slot = tile % S;
        cp_wait<S - 1>();
        __syncthreads();
        // K order. The MMA's k slots may carry any permutation of the group's 64 k as long as A and
        // B agree: lane lid takes k 16 * lid .. 16 * lid + 15, so for step ks its slots (2 lid,
        // 2 lid + 1) carry k 16 lid + 4 ks + {0, 1} and slots (2 lid + 8, 2 lid + 9) carry
        // 16 lid + 4 ks + {2, 3}. Its A operand over the group is then eight contiguous code bytes
        // a row (one 64-bit load, decoded without an int-to-float conversion) and its B operand
        // sixteen contiguous activations of one token (two 128-bit loads), where the natural order
        // costs a byte load and two conversions per A pair and an ldmatrix per B fragment. The
        // swizzle moves whole 8-element chunks, so each 128-bit load stays contiguous.
        float group_acc[MT][NT][4] = {};
        uint2 top[MT], bottom[MT];
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const int r0 = mi * 16 + gid;
            top[mi]      = load_vec<uint2>(&shared.staging.codes[slot][r0][warp * 32 + 8 * lid]);
            bottom[mi] = load_vec<uint2>(&shared.staging.codes[slot][r0 + 8][warp * 32 + 8 * lid]);
        }
#pragma unroll
        for (int half = 0; half < 2; ++half) {
            unsigned a_top[MT][4], a_bottom[MT][4];
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
                Q4MmaDecodeAtom::decode_eight(half == 0 ? top[mi].x : top[mi].y, a_top[mi]);
                Q4MmaDecodeAtom::decode_eight(half == 0 ? bottom[mi].x : bottom[mi].y,
                                              a_bottom[mi]);
            }
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                const int token = ni * 8 + gid;
                const uint4 b   = load_vec<uint4>(
                    &shared.staging
                         .x[slot][warp][token][q4_sliced_k_swizzle(token, 16 * lid + 8 * half)]);
#pragma unroll
                for (int mi = 0; mi < MT; ++mi) {
                    float (&g)[4] = group_acc[mi][ni];
                    mma_bf16(g[0], g[1], g[2], g[3], a_top[mi][0], a_bottom[mi][0], a_top[mi][1],
                             a_bottom[mi][1], b.x, b.y);
                    mma_bf16(g[0], g[1], g[2], g[3], a_top[mi][2], a_bottom[mi][2], a_top[mi][3],
                             a_bottom[mi][3], b.z, b.w);
                }
            }
        }
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
            const float top =
                __half2float(__ushort_as_half(shared.staging.scales[slot][mi * 16 + gid][warp]));
            const float bottom = __half2float(
                __ushort_as_half(shared.staging.scales[slot][mi * 16 + gid + 8][warp]));
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                acc[mi][ni][0] = fmaf(group_acc[mi][ni][0], top, acc[mi][ni][0]);
                acc[mi][ni][1] = fmaf(group_acc[mi][ni][1], top, acc[mi][ni][1]);
                acc[mi][ni][2] = fmaf(group_acc[mi][ni][2], bottom, acc[mi][ni][2]);
                acc[mi][ni][3] = fmaf(group_acc[mi][ni][3], bottom, acc[mi][ni][3]);
            }
        }
        if (tile + S < tiles) {
            // Only a reused stage needs a barrier here. The final reduction
            // barrier covers the drained stages without an extra CTA stall.
            __syncthreads();
            stage_inputs(slot, tile + S);
        } else if constexpr (S > 1) {
            cp_commit();
        }
    }
    cp_wait<0>();
    __syncthreads();
    constexpr bool kPairwise = Schedule::kReduction == Q4SlicedKReduction::Pairwise;
    if (!kPairwise || (warp & 1) != 0) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                store_vec(
                    shared.partial[warp][mi][ni][lane],
                    make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]));
            }
        }
    }
    __syncthreads();
    if constexpr (kPairwise) {
        // Pair neighboring K warps before the final owner loads shared partials.
        // This trades one CTA barrier for half as many loads by the owner warp.
        if ((warp & 1) == 0) {
#pragma unroll
            for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
                for (int ni = 0; ni < NT; ++ni) {
                    const float4 partner = load_vec<float4>(shared.partial[warp + 1][mi][ni][lane]);
                    acc[mi][ni][0] += partner.x;
                    acc[mi][ni][1] += partner.y;
                    acc[mi][ni][2] += partner.z;
                    acc[mi][ni][3] += partner.w;
                    if (warp != 0)
                        store_vec(shared.partial[warp][mi][ni][lane],
                                  make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2],
                                              acc[mi][ni][3]));
                }
            }
        }
        __syncthreads();
    }
    if (warp == 0) {
#pragma unroll
        for (int mi = 0; mi < MT; ++mi) {
#pragma unroll
            for (int ni = 0; ni < NT; ++ni) {
                float4 sum;
                if constexpr (kPairwise)
                    sum =
                        make_float4(acc[mi][ni][0], acc[mi][ni][1], acc[mi][ni][2], acc[mi][ni][3]);
                else
                    sum = load_vec<float4>(shared.partial[0][mi][ni][lane]);
#pragma unroll
                for (int part = kPairwise ? 2 : 1; part < W; part += kPairwise ? 2 : 1) {
                    const float4 value = load_vec<float4>(shared.partial[part][mi][ni][lane]);
                    sum.x += value.x;
                    sum.y += value.y;
                    sum.z += value.z;
                    sum.w += value.w;
                }
                // The fragment's first row: a row policy that folds several weight rows into one
                // output row names it (fragment_row), otherwise the weight row itself.
                const int r0 = [&] {
                    if constexpr (requires { row_policy.fragment_row(row_begin, mi, gid); })
                        return row_policy.fragment_row(row_begin, mi, gid);
                    else
                        return row_begin + mi * 16 + gid;
                }();
                const int t0        = token0 + ni * 8 + 2 * lid;
                const int token_end = token_begin + min(tokens, token0 + Schedule::kTokenCapacity);
                if constexpr (requires {
                                  epilogue.store_fragment(output, r0, token_begin + t0, sum, rows,
                                                          token_end);
                              }) {
                    // Fully reduced m16n8 fragment: x/y belong to r0, z/w to
                    // r0+8. The owning Op receives absolute bounds for its store.
                    epilogue.store_fragment(output, r0, token_begin + t0, sum, rows, token_end);
                } else {
                    const auto store = [&](int row, int token, float value) {
                        if (token - token0 < Schedule::kTokenCapacity && (Full || row < rows) &&
                            (FullTokens || token < tokens)) {
                            const int global_token = token_begin + token;
                            output.store(row, global_token,
                                         epilogue.apply(row, global_token, value));
                        }
                    };
                    store(r0, t0, sum.x);
                    store(r0, t0 + 1, sum.y);
                    store(r0 + 8, t0, sum.z);
                    store(r0 + 8, t0 + 1, sum.w);
                }
            }
        }
    }
}

} // namespace ninferno::ops::detail
