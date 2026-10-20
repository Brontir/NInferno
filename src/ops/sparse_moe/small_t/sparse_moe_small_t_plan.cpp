#include "core/weight.h"
#include "ops/sparse_moe/small_t/sparse_moe_small_t.h"

#include "core/layout.h"

#include <cstdlib>
#include <stdexcept>

namespace ninferno::ops::detail {

bool sparse_moe_uses_small_t(std::int32_t tokens) noexcept {
    return tokens >= kSparseMoeSmallTMin && tokens <= kSparseMoeSmallTMax;
}

std::size_t sparse_moe_small_t_workspace_bytes(std::int32_t tokens) {
    if (!sparse_moe_uses_small_t(tokens)) {
        throw std::invalid_argument("sparse_moe small-T: tokens must be in [2,46]");
    }
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_small_t_workspace(layout, tokens);
    return layout.peak_bytes(1);
}

namespace {


SparseMoeSmallTPlan measured_plan(std::int32_t tokens, QType routed_gate_up, QType routed_down) {
    if (!sparse_moe_uses_small_t(tokens)) {
        throw std::invalid_argument("sparse_moe small-T: unsupported token count");
    }
    const bool main_profile =
        routed_gate_up == QType::Q4_G64_FP16 &&
        (routed_down == QType::Q5_G64_FP16 || routed_down == QType::Q6_G64_FP16);
    const bool mtp_profile =
        routed_gate_up == QType::Q8_G32_FP16 && routed_down == QType::Q8_G32_FP16;
    const bool nvfp4_profile = routed_gate_up == QType::NVFP4 && routed_down == QType::NVFP4;
    if (!main_profile && !mtp_profile && !nvfp4_profile) {
        throw std::invalid_argument("sparse_moe small-T: unsupported routed codec profile");
    }

    SparseMoeSmallTPlan plan{tokens, sparse_moe_small_t_workspace_bytes(tokens)};
    if (nvfp4_profile) {
        // Swept over the whole [2,46] window against all nine pairs, cold, with the prefill
        // frontier held open; the figures below are that sweep's, and the frontier comment in
        // sparse_moe_prefill.h quotes a later one, so the two differ in the third digit. One path
        // per warp and two output rows per block win everywhere NVFP4 keeps the window: 34.8 us at
        // T=2 and 79.9 at T=8, against 38.4 and 90.1 for the three-path, one-row pair. Rows4
        // overtakes Rows2 somewhere between T=16 and T=32, which is beyond the frontier this
        // profile stops at, so it has no interval of its own. Swept again with the current kernels
        // (RTX 5090, cold, graph), Rows4 wins from T = 3: 32.8 / 65.5 / 90.1 us at
        // T = 3 / 8 / 12 against 34.8 / 67.5 / 94.3 for Rows2.
        plan.d3_schedule = SparseMoeSmallTD3Schedule::Paths9;
        plan.d4_schedule = SparseMoeSmallTD4Schedule::Rows4;
        return plan;
    }
    if (mtp_profile) {
        plan.d3_schedule =
            tokens <= 5 ? SparseMoeSmallTD3Schedule::Paths1 : SparseMoeSmallTD3Schedule::Paths9;
        // Rows4 at every width: the Rows1 interval to T = 8 lost 10..16 % on the RTX 5090 at
        // T = 3..8 (63.5 against 73.7 us at T = 8), 6..12 % on the 4090.
        plan.d4_schedule = SparseMoeSmallTD4Schedule::Rows4;
        return plan;
    }

    plan.d3_schedule = SparseMoeSmallTD3Schedule::Paths3;
    // Rows is only a grid split: (kHidden / Rows, tokens). Rows2 runs twice the blocks, so it has
    // twice the parallelism to cover memory latency while the grid is small, and it re-reads the
    // same per-expert activations in every block - +63 % L1 traffic, L2 flat (lts__t_bytes
    // x1.004); dram__* is not exposed on this part. Those counters are from Q6 at T=17; the kernel
    // is codec-parametric, so the mechanism is assumed to carry to Q5, not measured there.
    //
    // The Rows1 / Rows2 intervals below T = 12..17 were crossovers of earlier kernels. Swept
    // again on the current ones (cold, graph, every token sharing one set of eight experts and
    // trace-like routing), Rows4 is the fastest from T = 2 on every chip measured: RTX 5090
    // Q4/Q5 +4..10 % at T = 2..17, RTX 4090 +3..7 %, RTX 3090 +4..10 %,
    // Q4/Q6 alike. One schedule for all chips, no per-chip crossover left to keep.
    plan.d4_schedule = SparseMoeSmallTD4Schedule::Rows4;
    return plan;
}

} // namespace

SparseMoeSmallTPlan resolve_sparse_moe_small_t_plan(std::int32_t tokens, QType routed_gate_up,
                                                    QType routed_down) {
    return measured_plan(tokens, routed_gate_up, routed_down);
}

} // namespace ninferno::ops::detail
