#pragma once
#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q4/q4_operands.h"
#include "ops/linear/q4/q4_a16_sliced_k_mma.cuh"

#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75
#    include "ops/linear/q4/q4_simt_launch.cuh"

#    include <type_traits>
#endif

namespace ninferno::ops::detail {
template <class Schedule, class Output, class Epilogue,
          class RowPolicy = Q4IdentityRows<Schedule::kBlockRows>>
void launch_q4_a16_sliced_k_mma(const Q4LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream, RowPolicy row_policy = {}) {
    validate_q4_operands(operands);
    // NINFERNO-SPECIAL(arch:sm70): without tensor mma.sync this tile's MMA runs on FP32 FMAs over
    // warp shuffles (ops/common/mma.cuh) - V100 35B C4: the Q8 input projections' sliced tile 453
    // us at T=4 against 34 us for the T=1 GEMV. The SIMT tile of the same format (up to eight
    // columns a CTA, codes decoded once per row for every column) takes the call where the rows are
    // the identity.
#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75
    if constexpr (std::is_same_v<RowPolicy, Q4IdentityRows<Schedule::kBlockRows>>) {
        // T <= 4 on the four-column tile: the eight-column one carries idle accumulator columns
        // (V100 27B C4: gate_up T=4 348 us; ninfer-v100-sm70 runs its r8 c4 SIMT tile there).
        // T = 5..7 on a tile of exactly T columns: the eight-column tile with idle columns cost
        // more than a full one (V100 Q4 34816x5120: T=5 468 us, T=8 450; ninfer-v100-sm70 avoids
        // it).
        const auto run = [&]<int Cols>() {
            launch_q4_a16_simt<Q4A16SimtSchedule<8, Cols, 1, 16, 2, Cache::ca, 1>>(
                operands, output, epilogue, stream);
        };
        switch (operands.tokens) {
        case 5: run.template operator()<5>(); break;
        case 6: run.template operator()<6>(); break;
        case 7: run.template operator()<7>(); break;
        default:
            if (operands.tokens <= 4)
                run.template operator()<4>();
            else
                run.template operator()<8>();
        }
        return;
    }
#endif
    if constexpr (Schedule::kStaticK > 0) {
        if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
            throw std::invalid_argument("Q4 sliced-K static geometry requires K == padded K");
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
            q4_a16_sliced_k_mma_kernel<Schedule, Full, FullTokens>
                <<<grid, Schedule::kThreads, 0, stream>>>(
                    x, operands.codes, operands.scales, output, epilogue, operands.rows, operands.k,
                    count, operands.padded_k, offset, row_policy);
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
