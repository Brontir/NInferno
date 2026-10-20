#pragma once
#include "ops/common/math.h"
#include "ops/common/token_slices.h"
#include "ops/linear/q8/q8_a16_sliced_k_mma.cuh"

#include <stdexcept>

#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75
#    include "ops/linear/q8/q8_simt_launch.cuh"

#    include <type_traits>
#endif

namespace ninferno::ops::detail {
template <class Schedule, class RowPolicy = Q8SlicedKIdentityRows, class Output, class Epilogue>
void launch_q8_a16_sliced_k_mma(const Q8LinearOperands& operands, Output output, Epilogue epilogue,
                                cudaStream_t stream, RowPolicy row_policy = {}) {
    // NINFERNO-SPECIAL(arch:sm70): without tensor mma.sync this tile's MMA runs on FP32 FMAs over
    // warp shuffles (ops/common/mma.cuh) - V100 35B C4: the Q8 input projections' sliced tile 453
    // us at T=4 against 34 us for the T=1 GEMV. The SIMT tile of the same format (up to eight
    // columns a CTA, codes decoded once per row for every column) takes the call where the rows are
    // the identity.
#if defined(NINFERNO_BUILD_ARCH) && NINFERNO_BUILD_ARCH < 75
    // An epilogue that takes a row's Tokens columns in one call (kWholeRowTokens) keeps the
    // sliced-K tile for calls wider than the SIMT tile's eight columns.
    constexpr int whole_row_tokens = [] {
        if constexpr (requires { Epilogue::kWholeRowTokens; })
            return Epilogue::kWholeRowTokens;
        else
            return 0;
    }();
    if constexpr (std::is_same_v<RowPolicy, Q8SlicedKIdentityRows> && whole_row_tokens <= 8) {
        // T <= 4 on the four-column tile: the eight-column one carries idle accumulator columns
        // (V100 27B C4: gate_up T=4 348 us; ninfer-v100-sm70 runs its r8 c4 SIMT tile there), and
        // T = 5..7 on a tile of exactly T columns (an eight-column tile with idle columns cost more
        // than a full one: V100 Q4 34816x5120 T=5 468 us, T=8 450). A whole-row epilogue needs a tile
        // at least as wide as its row.
        const auto run = [&]<int Cols>() {
            if constexpr (whole_row_tokens <= Cols) {
                launch_q8_a16_simt<Q8A16SimtSchedule<8, Cols, 1, 16, 2, Cache::ca, 1>>(
                    operands, output, epilogue, stream);
                return true;
            }
            return false;
        };
        bool done = false;
        switch (operands.tokens) {
        case 5: done = run.template operator()<5>(); break;
        case 6: done = run.template operator()<6>(); break;
        case 7: done = run.template operator()<7>(); break;
        default:
            if (operands.tokens <= 4) done = run.template operator()<4>();
        }
        if (!done) run.template operator()<8>();
        return;
    }
#endif
    // NINFERNO-SPECIAL(arch:sm75): a token tile whose staging exceeds the build's shared memory
    // (Turing's 64 KiB) runs as twice the CTAs over half the columns each, or, where no narrower
    // tile exists (8 tokens, an exact tile), with one staging buffer instead of two. Every
    // column's K split and its sum order stay those of the tuned tile. A tile with neither is
    // refused by q8_prepare_shared (and listed by the NINFERNO_SHARED_AUDIT build).
    if constexpr (!Schedule::kFitsSharedLimit &&
                  (Schedule::kExactTokens || Schedule::kBlockTokens <= 8) &&
                  Schedule::kStages == 2 && Schedule::template with_stages<1>::kFitsSharedLimit) {
        launch_q8_a16_sliced_k_mma<typename Schedule::template with_stages<1>, RowPolicy>(
            operands, output, epilogue, stream, row_policy);
    } else if constexpr (!Schedule::kFitsSharedLimit && !Schedule::kExactTokens &&
                         Schedule::kBlockTokens > 8) {
        using Half =
            typename Schedule::template with_block_tokens<(Schedule::kBlockTokens / 2 + 7) / 8 * 8>;
        launch_q8_a16_sliced_k_mma<Half, RowPolicy>(operands, output, epilogue, stream, row_policy);
    } else {
        validate_q8_operands(operands);
        if (operands.k % 8 != 0)
            throw std::invalid_argument("Q8 sliced-K MMA requires K aligned to 8");
        if constexpr (Schedule::kStaticK > 0) {
            if (operands.k != Schedule::kStaticK || operands.padded_k != Schedule::kStaticK)
                throw std::invalid_argument(
                    "Q8 sliced-K static K requires matching, padding-free operands");
        }
        if constexpr (Schedule::kExactTokens) {
            if (operands.tokens != Schedule::kTokenCapacity)
                throw std::invalid_argument("Q8 sliced-K exact tokens differ from the schedule");
        }
        constexpr int ratio = Schedule::kBlockRows / RowPolicy::kOutputRowsPerCta;
        static_assert(Schedule::kBlockRows % RowPolicy::kOutputRowsPerCta == 0);
        if (operands.rows % ratio != 0)
            throw std::invalid_argument("Q8 sliced-K row mapping requires complete row pairs");
        for_each_token_slice(operands.tokens, Schedule::kTokenCapacity, [&](int offset, int count) {
            const dim3 grid(div_up(operands.rows / ratio, RowPolicy::kOutputRowsPerCta),
                            div_up(count, Schedule::kTokenCapacity));
            const auto launch = [&]<bool Full>() {
                static constexpr auto kernel =
                    q8_a16_sliced_k_mma_kernel<Schedule, Full, Output, Epilogue, RowPolicy>;
                const int shared = q8_prepare_shared<Schedule::kSharedBytes, kernel>();
                kernel<<<grid, Schedule::kThreads, shared, stream>>>(operands, output, epilogue,
                                                                     row_policy, offset);
                CUDA_CHECK(cudaGetLastError());
            };
            if (operands.rows % Schedule::kBlockRows == 0 && operands.k == operands.padded_k &&
                operands.k % Schedule::kBlockK == 0)
                launch.template operator()<true>();
            else
                launch.template operator()<false>();
        });
    }
}
} // namespace ninferno::ops::detail
