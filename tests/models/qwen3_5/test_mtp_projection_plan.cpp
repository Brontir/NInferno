// The MTP layer's packed Q/K/gate/V parent (host only): the fused attn_input_proj Op is taken
// only for the parent forms it is registered for; any other geometry (the 35B-A3B MoE's MTP layer
// stored in BF16, as in the NVFP4 releases of nvidia and RedHatAI) is sized for, and run by, the
// generic linear plus the split, instead of the Op's "unsupported BF16 profile" refusal.

#include "models/qwen3_5/execution/mtp.h"
#include "ninferno/ops/attn_input_proj.h"

#include <cstdint>
#include <iostream>
#include <stdexcept>

namespace {

using namespace ninferno;
using namespace ninferno::models::qwen3_5;

int failures = 0;

void check(bool condition, const char* message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

execution::MtpProjectionParameters packed(QType qtype, std::int32_t rows, std::int32_t hidden) {
    execution::MtpProjectionParameters parameters;
    ops::SingleProjectionWeight parent;
    parent.weight.qtype = qtype;
    parent.weight.n     = rows;
    parent.weight.k     = hidden;
    parent.policy       = ops::LinearPolicy::A16Only;
    parameters.packed   = parent;
    return parameters;
}

bool capacity_refuses(QType qtype, std::int32_t rows, std::int32_t hidden) {
    try {
        (void)ops::attn_input_proj_workspace_capacity_bytes(qtype, rows, hidden,
                                                            ops::LinearPolicy::A16Only, 1, 8);
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

} // namespace

int main() {
    // 35B-A3B: hidden 2048, 16 query and 2 KV heads of 256. 27B: hidden 5120, 24 and 4 heads.
    const AttentionConfig moe{16, 2, 256};
    const AttentionConfig dense{24, 4, 256};

    // The Op's registered forms.
    check(ops::attn_input_proj_fused_supported(QType::Q8_G32_FP16, 2048, 4096, 512),
          "Q8 [9216,2048] is a fused form");
    check(ops::attn_input_proj_fused_supported(QType::BF16, 5120, 6144, 1024),
          "BF16 [14336,5120] is a fused form");
    check(ops::attn_input_proj_fused_supported(QType::FP8_E4M3FN_ROW_BF16, 5120, 6144, 1024),
          "FP8 [14336,5120] is a fused form");
    // The head split is part of the form: right total rows, other split.
    check(!ops::attn_input_proj_fused_supported(QType::BF16, 5120, 7168, 0),
          "a different head split is not a fused form");
    // The 35B-A3B geometry in every format the Op lists is not registered except Q8.
    check(!ops::attn_input_proj_fused_supported(QType::BF16, 2048, 4096, 512),
          "BF16 [9216,2048] is not a fused form");
    check(!ops::attn_input_proj_fused_supported(QType::NVFP4, 2048, 4096, 512),
          "NVFP4 [9216,2048] is not a fused form");
    check(!ops::attn_input_proj_fused_supported(QType::FP8_E4M3FN_ROW_BF16, 2048, 4096, 512),
          "FP8 [9216,2048] is not a fused form");
    check(!ops::attn_input_proj_fused_supported(QType::Q8_G32_FP16, 5120, 6144, 1024),
          "Q8 [14336,5120] is not a fused form");
    check(!ops::attn_input_proj_fused_supported(QType::Q6_G64_FP16, 2048, 4096, 512),
          "other formats are not fused forms");

    // The capacity query still refuses what the Op cannot run; the MTP planner no longer asks it.
    check(capacity_refuses(QType::BF16, 9216, 2048), "the Op's capacity query refuses BF16 9216");

    // BF16 MoE parent: sized (packed BF16 output plus the linear's workspace), not refused.
    try {
        const auto bytes =
            execution::mtp_projection_workspace_bytes(packed(QType::BF16, 9216, 2048), moe, 1, 8);
        check(bytes >= std::size_t(9216) * 8 * 2, "BF16 MoE parent reserves the packed output");
    } catch (const std::exception& e) {
        std::cerr << "FAIL: BF16 MoE parent refused: " << e.what() << '\n';
        ++failures;
    }
    // The 27B BF16 parent stays on the fused Op (zero workspace).
    try {
        check(execution::mtp_projection_workspace_bytes(packed(QType::BF16, 14336, 5120), dense, 1,
                                                        8) == 0,
              "BF16 27B parent stays on the fused Op");
    } catch (const std::exception& e) {
        std::cerr << "FAIL: BF16 27B parent: " << e.what() << '\n';
        ++failures;
    }
    // The same parent with a head split the Op does not know takes the generic route.
    try {
        const auto bytes = execution::mtp_projection_workspace_bytes(
            packed(QType::BF16, 14336, 5120), AttentionConfig{28, 0, 256}, 1, 8);
        check(bytes >= std::size_t(14336) * 8 * 2, "an unregistered split reserves the packed output");
    } catch (const std::exception& e) {
        std::cerr << "FAIL: unregistered split refused: " << e.what() << '\n';
        ++failures;
    }
    return failures == 0 ? 0 : 1;
}
