#pragma once

#include "core/device.h"
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/common/epilogue.cuh"
#include "ops/linear/q6/q6_operands.h"
#include "ops/linear/q6/q6_a16_sliced_k_mma.cuh"

#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75
#    include "ops/linear/q6/q6_simt_launch.cuh"

#    include <type_traits>
#endif

namespace ninferno::ops::detail {

template <class Schedule, class Output, class Epilogue = LinearIdentityEpilogue>
void launch_q6_a16_sliced_k_mma(const Q6LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream) {
    validate_q6_operands(operands);
    // NINFERNO-SPECIAL(arch:sm70): without tensor mma.sync this tile's MMA runs on FP32 FMAs over
    // warp shuffles (ops/common/mma.cuh) - V100 35B C4: the Q8 input projections' sliced tile 453
    // us at T=4 against 34 us for the T=1 GEMV. The SIMT tile of the same format (up to eight
    // columns a CTA, codes decoded once per row for every column) takes the call where the rows are
    // the identity.
#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75
    if constexpr (true) {
        // T <= 4 on the four-column tile: the eight-column one carries idle accumulator columns
        // (V100 27B C4: gate_up T=4 348 us; ninfer-v100-sm70 runs its r8 c4 SIMT tile there).
        // T = 5..7 on a tile of exactly T columns: the eight-column tile with idle columns cost
        // more than a full one (V100 Q4 34816x5120: T=5 468 us, T=8 450; ninfer-v100-sm70 avoids
        // it).
        const auto run = [&]<int Cols>() {
            launch_q6_a16_simt<Q6A16SimtSchedule<8, Cols, 1, 16, 2, Cache::ca, 1>>(
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
    for_each_token_slice(operands.tokens, Schedule::kBlockTokens, [&](int offset, int count) {
        const dim3 grid(static_cast<unsigned>(div_up(operands.rows, Schedule::kBlockRows)),
                        static_cast<unsigned>(div_up(count, Schedule::kBlockTokens)));
        const auto* x = operands.x + static_cast<std::int64_t>(offset) * operands.k;
        // The full kernel copies all warp scales together; the masked kernel
        // copies four-byte scale pairs and supports the minimum input alignment.
        const bool full = operands.rows % Schedule::kBlockRows == 0 &&
                          count % Schedule::kBlockTokens == 0 && operands.k == operands.padded_k &&
                          operands.k % Schedule::kBlockK == 0 &&
                          reinterpret_cast<std::uintptr_t>(operands.scales) %
                                  (Schedule::kKWarps * sizeof(__half)) ==
                              0;
        if (full) {
            q6_a16_sliced_k_mma_kernel<Schedule, true><<<grid, Schedule::kThreads, 0, stream>>>(
                x, operands.codes, operands.high, operands.scales, output, epilogue, operands.rows,
                operands.k, count, operands.padded_k, offset);
        } else {
            q6_a16_sliced_k_mma_kernel<Schedule, false><<<grid, Schedule::kThreads, 0, stream>>>(
                x, operands.codes, operands.high, operands.scales, output, epilogue, operands.rows,
                operands.k, count, operands.padded_k, offset);
        }
        CUDA_CHECK(cudaGetLastError());
    });
}

} // namespace ninferno::ops::detail
