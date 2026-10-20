#include "core/weight.h"
#include "core/device_class.h"
#include "core/measured_boards.h"
#include "ops/common/native_nvfp4.h"
#include "ops/sparse_moe/prefill/sparse_moe_prefill.h"

#include "core/layout.h"
#include "ninferno/ops/prefill_routes.h"

#include <algorithm>
#include <cstdlib>
#include <span>
#include <stdexcept>

namespace ninferno::ops::detail {
namespace {

// The Q4/Q5 and Q4/Q6 groupwise frontier per chip: the per-token small-T kernels hold their lead
// to T = 46 on Blackwell and Ada (kSparseMoePrefillQ4Q5Min), while Ampere loses it early.
// Qwen3.6-35B DFlash7 C4 tok/s by frontier 47 / 25 / 17 / 13 (ABBA, plain C8 unmoved everywhere):
//   RTX 5090  843 / 812 / 815 / 802
//   RTX 4090  636 / 536 / 520 / 528
//   RTX 3090  292 / 389 / 403 / 411
// 17 is the first prefill-sized call, where the prefill profile switches too. Another board takes
// the nearest measured board's frontier within measured::kCarryDistance, else 47.
struct BoardFrontier {
    Board board;
    std::int32_t min_tokens;
};

constexpr BoardFrontier kQ4GroupwiseFrontiers[] = {
    {measured::kRtx5090, kSparseMoePrefillQ4Q5Min},
    {measured::kRtx4090, kSparseMoePrefillQ4Q5Min},
    {measured::kRtx3090, 17},
};

std::int32_t q4_groupwise_frontier() noexcept {
    std::int32_t frontier = kSparseMoePrefillQ4Q5Min;
    try {
        const BoardFrontier* f =
            nearest_measured(std::span(kQ4GroupwiseFrontiers), board_of(device_class()),
                             measured::kCarryDistance);
        if (f != nullptr) frontier = f->min_tokens;
    } catch (...) {}
    return frontier;
}

} // namespace

std::int32_t sparse_moe_prefill_min_tokens(QType routed_gate_up, QType routed_down,
                                           bool prefill_profile) noexcept {
    std::int32_t minimum = 0;
    if (routed_gate_up == QType::Q4_G64_FP16) {
        if (routed_down == QType::Q5_G64_FP16 || routed_down == QType::Q6_G64_FP16) {
            minimum = q4_groupwise_frontier();
        }
    }
    if (routed_gate_up == QType::Q8_G32_FP16 && routed_down == QType::Q8_G32_FP16) {
        minimum = kSparseMoePrefillQ8Q8Min;
    }
    if (routed_gate_up == QType::NVFP4 && routed_down == QType::NVFP4) {
        minimum = kSparseMoePrefillNvfp4Min;
    }
    if (minimum != 0 && prefill_profile) {
        minimum = std::min(minimum, kSparseMoePrefillProfileMin);
    }
    // The invariant route profile (--activations exact): every width takes the prefill kernels,
    // which compute a token alike at every call size; the decode and small-T kernels sum the
    // experts and K their own way, so a verify row's bytes followed its batch.
    if (minimum != 0 && invariant_routes_active()) { minimum = 1; }
    return minimum;
}

namespace {

std::int32_t prefill_min_tokens(QType routed_gate_up, QType routed_down) noexcept {
    return sparse_moe_prefill_min_tokens(routed_gate_up, routed_down, prefill_routes_active());
}

} // namespace

bool sparse_moe_uses_prefill(std::int32_t tokens, QType routed_gate_up,
                             QType routed_down) noexcept {
    const std::int32_t minimum = prefill_min_tokens(routed_gate_up, routed_down);
    return minimum != 0 && tokens >= minimum;
}

std::size_t sparse_moe_prefill_workspace_bytes(std::int32_t max_tokens, bool nvfp4) {
    if (max_tokens < kSparseMoePrefillWorkspaceMin) {
        throw std::invalid_argument("sparse_moe prefill: max_tokens must be at least 20");
    }
    const std::int32_t capacity_tokens = std::min(max_tokens, kSparseMoePrefillSliceMax);
    WorkspaceLayoutBuilder layout;
    (void)allocate_sparse_moe_prefill_workspace(layout, capacity_tokens, nvfp4);
    return layout.peak_bytes(1);
}

SparseMoePrefillPlan resolve_sparse_moe_prefill_plan(std::int32_t tokens, QType routed_gate_up,
                                                     QType routed_down) {
#ifdef NINFERNO_WITHOUT_NVFP4
    if (routed_gate_up == QType::NVFP4 || routed_down == QType::NVFP4) {
        no_native_nvfp4("sparse_moe prefill");
    }
#endif
    const std::int32_t minimum = prefill_min_tokens(routed_gate_up, routed_down);
    if (minimum == 0) {
        throw std::invalid_argument("sparse_moe prefill: unsupported routed codec profile");
    }
    if (tokens < minimum) {
        throw std::invalid_argument("sparse_moe prefill: unsupported token count");
    }

    // The workspace query has a floor of twenty tokens, and NVFP4 enters prefill at thirteen, so a
    // call in [13,19] has to allocate to that floor or the execution high-water mark falls short of
    // what the query promised. Slicing is unaffected: the loop still steps by the real token
    // count.
    const std::int32_t slice_tokens =
        std::max(std::min(tokens, kSparseMoePrefillSliceMax), kSparseMoePrefillWorkspaceMin);
    const bool nvfp4 = routed_gate_up == QType::NVFP4;
    return {tokens, slice_tokens, sparse_moe_prefill_workspace_bytes(slice_tokens, nvfp4), nvfp4};
}

} // namespace ninferno::ops::detail
