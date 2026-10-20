#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q5/q5_operands.h"
#include "ops/linear/q5/q5_a16_sliced_k_mma.cuh"

#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75
#    include "ops/linear/q5/q5_simt_launch.cuh"

#    include <type_traits>
#endif

namespace ninferno::ops::detail {
#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75
// The direct tile at exactly `tokens` columns for 2..8 when K fills its 1024-wide steps; false
// when it does not apply.
template <class Output, class Epilogue>
bool launch_q5_volta_direct(const Q5LinearOperands& operands, Output output, Epilogue epilogue,
                            cudaStream_t stream) {
    if (operands.k != operands.padded_k || operands.k % 1024 != 0) return false;
    const auto run = [&]<int T>() {
        launch_q5_a16_direct_simt<Q5A16DirectSimtSchedule<1, T, 2, 8, 16, 0, true>>(
            operands, output, epilogue, stream);
        return true;
    };
    switch (operands.tokens) {
    case 2: return run.template operator()<2>();
    case 3: return run.template operator()<3>();
    case 4: return run.template operator()<4>();
    case 5: return run.template operator()<5>();
    case 6: return run.template operator()<6>();
    case 7: return run.template operator()<7>();
    case 8: return run.template operator()<8>();
    default: return false;
    }
}
#endif

template <class Schedule, class Output, class Epilogue,
          class RowPolicy = Q5IdentityRows<Schedule::kBlockRows>>
void launch_q5_a16_sliced_k_mma(const Q5LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream, RowPolicy row_policy = {}) {
    validate_q5_operands(operands);
    // NINFERNO-SPECIAL(arch:sm70): without tensor mma.sync this tile's MMA runs on FP32 FMAs over
    // warp shuffles (ops/common/mma.cuh) - V100 35B C4: the Q8 input projections' sliced tile 453
    // us at T=4 against 34 us for the T=1 GEMV. The SIMT tile of the same format (up to eight
    // columns a CTA, codes decoded once per row for every column) takes the call where the rows are
    // the identity.
// V100 27B MTP4 verify (T = 5): the eight-column tile spent 234 us on the 17408-K down projection,
// FP32 issue-bound on columns it does not have; the exact-column direct tile (one row a CTA, two
// warps along K) reads the row once for exactly T columns.
#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75
    if constexpr (std::is_same_v<RowPolicy, Q5IdentityRows<Schedule::kBlockRows>>) {
        if (launch_q5_volta_direct(operands, output, epilogue, stream)) return;
        // T <= 4 on the four-column tile: the eight-column one carries idle accumulator columns
        // (V100 27B C4: gate_up T=4 348 us; ninfer-v100-sm70 runs its r8 c4 SIMT tile there).
        if (operands.tokens <= 4) {
            launch_q5_a16_simt<Q5A16SimtSchedule<8, 4, 1, 16, 2, Cache::ca, 1>>(operands, output,
                                                                                epilogue, stream);
        } else {
            launch_q5_a16_simt<Q5A16SimtSchedule<8, 8, 1, 16, 2, Cache::ca, 1>>(operands, output,
                                                                                epilogue, stream);
        }
        return;
    }
#endif
    if (operands.k % 8 != 0) throw std::invalid_argument("Q5 MMA requires K aligned to 8");
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
            throw std::invalid_argument("Q5 sliced-K static geometry requires K == padded K");
    }
    for_each_token_slice(operands.tokens, Schedule::kTokenCapacity, [&](int offset, int count) {
        const int output_rows = row_policy.output_rows(operands.rows);
        const dim3 grid(div_up(output_rows, RowPolicy::kOutputRowsPerCta),
                        div_up(count, Schedule::kTokenCapacity));
        const auto* x   = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        const bool full = operands.rows % Schedule::kBlockRows == 0 &&
                          output_rows % RowPolicy::kOutputRowsPerCta == 0 &&
                          operands.k == operands.padded_k && operands.k % Schedule::kBlockK == 0 &&
                          reinterpret_cast<std::uintptr_t>(operands.scales) %
                                  (Schedule::kKWarps * sizeof(__half)) ==
                              0;
        const bool full_tokens = count % Schedule::kTokenCapacity == 0;
        const auto launch      = [&]<bool Full, bool FullTokens>() {
            q5_a16_sliced_k_mma_kernel<Schedule, Full, FullTokens>
                <<<grid, Schedule::kThreads, 0, stream>>>(
                    x, operands.codes, operands.high, operands.scales, output, epilogue,
                    operands.rows, operands.k, count, operands.padded_k, offset, row_policy);
        };
        if (full && full_tokens)
            launch.template operator()<true, true>();
        else if (full)
            launch.template operator()<true, false>();
        else if (full_tokens)
            launch.template operator()<false, true>();
        else
            launch.template operator()<false, false>();
        CUDA_CHECK(cudaGetLastError());
    });
}
} // namespace ninferno::ops::detail
