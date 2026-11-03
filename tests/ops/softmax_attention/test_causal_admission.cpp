// The causal cache attention's one admitted domain (#312, causal_cache/plan_rules.h
// causal_domain_refusal), no GPU: each refusal names its reason and its kind (not a call, the
// execution contract, not compiled), and the plans' admission and the workspace query agree on
// every row count, width and envelope at the boundaries for every KV storage.
#include "ninferno/ops/softmax_attention.h"
#include "ops/softmax_attention/common/causal_geometry.h"
#include "ops/softmax_attention/dense/causal_cache/plan_rules.h"

#include <cstdio>
#include <stdexcept>
#include <string>

namespace {
namespace detail = ninferno::ops::detail;
using detail::CausalRestriction;
using ninferno::ops::CausalAttentionExecutionEnvelope;
using ninferno::ops::kCausalAttentionMaximumVisibleKeys;

int failures = 0;

void expect(bool ok, const std::string& what) {
    if (!ok && ++failures <= 12) std::fprintf(stderr, "FAIL: %s\n", what.c_str());
}

bool refused_as(int heads, int kv, int batch, int width, CausalAttentionExecutionEnvelope e,
                CausalRestriction kind) {
    const auto refusal = detail::causal_domain_refusal(heads, kv, batch, width, width, e);
    return refusal && refusal->kind == kind;
}

} // namespace

int main() {
    const CausalAttentionExecutionEnvelope ok{1, 4096};
    // Admitted at the boundaries: 8 rows of 16 columns, one row of any width, every pair.
    for (const detail::CausalHeadPair p : detail::kCausalHeadPairs) {
        expect(!detail::causal_domain_refusal(p.query_heads, p.kv_heads, 8, 16, 16, ok) &&
                   !detail::causal_domain_refusal(p.query_heads, p.kv_heads, 1, 1, 4096, ok),
               "an admitted call refused");
    }
    expect(!detail::causal_domain_refusal(24, 4, 1, 1, 1,
                                          {1, kCausalAttentionMaximumVisibleKeys}),
           "the partition's whole capacity refused");
    // Each refusal and its kind.
    expect(refused_as(24, 4, 0, 1, ok, CausalRestriction::Semantic), "no row: not a call");
    expect(refused_as(24, 4, 1, 0, ok, CausalRestriction::Semantic), "no column: not a call");
    expect(refused_as(24, 4, 1, 1, {0, 16}, CausalRestriction::Semantic), "an empty envelope");
    expect(refused_as(24, 4, 1, 1, {32, 16}, CausalRestriction::Semantic), "an inverted envelope");
    expect(refused_as(24, 4, 9, 1, ok, CausalRestriction::Contract), "nine rows: the contract");
    expect(refused_as(24, 4, 2, 17, ok, CausalRestriction::Contract),
           "17 columns in a call of two rows: the contract");
    expect(refused_as(24, 4, 1, 1, {1, kCausalAttentionMaximumVisibleKeys + 1},
                      CausalRestriction::Contract),
           "past the partition's capacity: the contract");
    expect(refused_as(32, 4, 1, 1, ok, CausalRestriction::Coverage), "32/4: not compiled");
    // The message names the reason and the kind.
    try {
        detail::require_causal_plan_inputs("BF16", 24, 4, 17, 2, ok);
        expect(false, "a plan admitted 2 rows of 17 columns");
    } catch (const std::invalid_argument& error) {
        const std::string what = error.what();
        expect(what.find("BF16 attention: more than 16 columns") == 0 &&
                   what.find("(the execution contract)") != std::string::npos,
               "the plan's refusal: " + what);
    }
    // The plans and the workspace query agree on every boundary, for every storage.
    for (const ninferno::KvCacheStorage storage :
         {ninferno::KvCacheStorage::BFloat16, ninferno::KvCacheStorage::Int8Group64,
          ninferno::KvCacheStorage::Fp8E4M3Row256, ninferno::KvCacheStorage::Nvfp4Group16,
          ninferno::KvCacheStorage::Fp8KeyNvfp4Value}) {
        for (const int batch : {0, 1, 2, 8, 9}) {
            for (const int width : {0, 1, 16, 17}) {
                for (const CausalAttentionExecutionEnvelope e :
                     {ok, CausalAttentionExecutionEnvelope{0, 16},
                      CausalAttentionExecutionEnvelope{
                          1, kCausalAttentionMaximumVisibleKeys + 1}}) {
                    const bool admitted = !detail::causal_domain_refusal(24, 4, batch, width, width, e);
                    bool sized          = true;
                    try {
                        (void)ninferno::ops::causal_softmax_attention_workspace_capacity_bytes(
                            {detail::kCausalHeadDim, 24, 4}, storage, e, batch, width, width,
                            {nullptr, 80});
                    } catch (const std::invalid_argument&) { sized = false; }
                    expect(admitted == sized, "plan and workspace disagree at B=" +
                                                  std::to_string(batch) + " W=" +
                                                  std::to_string(width) + " keys " +
                                                  std::to_string(e.max_visible_keys));
                }
            }
        }
    }
    if (failures != 0) {
        std::fprintf(stderr, "causal admission: %d failure(s)\n", failures);
        return 1;
    }
    std::printf("causal admission: one domain, each refusal named with its kind\n");
    return 0;
}
