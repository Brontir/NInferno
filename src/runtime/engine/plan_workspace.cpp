#include "runtime/engine/plan_workspace.h"

#include "artifact/formats.h"
#include "core/weight_view.h"
#include "models/gemma4/config.h"
#include "models/gemma4/workspace.h"
#include "models/qwen3_5/program/planning/sequence_plan.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/prefill_routes.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/speculative_round.h"
#include "runtime/engine/plan_dry_run.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

namespace ninferno::runtime {
namespace {

namespace gemma4 = models::gemma4;
namespace detail = models::qwen3_5::detail;

// The drafter's own pass beside the target's (gemma4::DrafterPass::workspace_bytes: its fixed
// 16 MiB reserve and the round's 1 KiB of ids): its rows are a few small layers the facts do not
// describe, under 1 MiB at the widths it runs.
constexpr std::uint64_t kGemmaDrafterPass = (std::uint64_t(17) << 20) + 1024;

// A weight described by its format and shape alone, in the layout the converter stores it in (an
// aligned address stands in, its bytes never read; ops/generic/route_query.cpp's description). Held
// whole: its view and weight point into its parent.
struct DescribedWeight {
    WeightParent parent;
    WeightView view;
    Weight weight;

    DescribedWeight(QType qtype, std::uint64_t n, std::uint64_t k) {
        parent.data = reinterpret_cast<const std::byte*>(std::uintptr_t{1} << 32);
        const std::uint64_t shape[2] = {n, k};
        parent.geometry = weight_geometry(qtype, default_weight_layout(qtype), shape);
        view.shape      = {n, k};
        view.parts      = {WeightRegion{&parent, 0, n * k}};
        weight          = native_weight(view, 0.0F);
    }
    DescribedWeight(const DescribedWeight&)            = delete;
    DescribedWeight& operator=(const DescribedWeight&) = delete;
};

// A generic MoE layer's capacity over a call of up to `columns` tokens, as the start sizes it
// (models/gemma4/target.cu: ops::moe_workspace_bytes over the layer's banks and the load's config,
// gemma4/load_target.cpp): the router [E, H] in BF16, the stacked gate/up [E 2F, H] and down
// [E H, F] in the facts' expert formats. nullopt where a bank's format is unknown. The start takes
// this Op for experts the fused NVFP4 kernel does not read; where it does, gemma4_moe's capacity
// (target_op_bytes) is the larger, so counting both bounds either.
std::optional<std::size_t> generic_moe_bytes(const PlanModelFacts& m, const models::gemma4::Config& c,
                                             std::int32_t columns) {
    // A bank whose layers differ in format is one linear a format (PlanLinear): every pair's.
    std::vector<const PlanLinear*> gate_ups, downs;
    for (const PlanLinear& l : m.linears) {
        if (l.expert) { (l.gated ? gate_ups : downs).push_back(&l); }
    }
    if (gate_ups.empty() || downs.empty() || m.experts == 0 || m.experts_used == 0) {
        return std::nullopt;
    }
    const auto experts = static_cast<std::uint64_t>(m.experts);
    const auto hidden  = static_cast<std::uint64_t>(c.hidden_size);
    const auto router  = std::make_unique<DescribedWeight>(QType::BF16, experts, hidden);
    ops::MoeConfig config;
    config.top_k          = static_cast<std::int32_t>(m.experts_used);
    config.scoring        = ops::MoeScoring::Softmax;
    config.activation     = ops::MoeActivation::GeGluTanh;
    config.gate_up_layout = ops::MoeGateUpLayout::Concatenated;
    std::size_t widest = 0;
    for (const PlanLinear* g : gate_ups) {
        for (const PlanLinear* d : downs) {
            if (g->format.empty() || d->format.empty()) { return std::nullopt; }
            const auto inter = static_cast<std::uint64_t>(d->k);
            const auto up    = std::make_unique<DescribedWeight>(artifact::parse_format(g->format),
                                                                 experts * 2 * inter, hidden);
            const auto dn    = std::make_unique<DescribedWeight>(artifact::parse_format(d->format),
                                                                 experts * hidden, inter);
            ops::MoeWeights weights;
            weights.router  = router->weight;
            weights.gate_up = up->weight;
            weights.down    = dn->weight;
            widest = std::max(widest, ops::moe_workspace_bytes(weights, config, 1, columns));
        }
    }
    return widest;
}

// The policy a start's projection carries for a stored format (the converter's: BF16 and FP32
// keep A16, NVFP4 allows four-bit activations, every other quantized format eight).
ops::LinearPolicy policy_of(QType qtype) {
    if (qtype == QType::BF16 || qtype == QType::FP32) { return ops::LinearPolicy::A16Only; }
    if (qtype == QType::NVFP4) { return ops::LinearPolicy::AllowA4; }
    return ops::LinearPolicy::AllowA8;
}

// Gemma 4's arena (models/gemma4/program_family.cpp workspace_plan) from the facts: the text
// prefill, the ordinary round and, under MTP, its prefill and round; nullopt where the facts lack
// the geometry or a projection's format.
std::optional<std::uint64_t> gemma4_workspace(const PlanModelFacts& m, const PlanRung& rung,
                                              const EngineOptions& options,
                                              std::uint32_t concurrency,
                                              const detail::SequencePlanImpl* plan) {
    if (m.model_type != "gemma4_text") { return std::nullopt; }
    if (m.hidden == 0 || m.vocab == 0 || m.layers == 0 || m.query_heads == 0 || !m.ple_known ||
        m.full_layers == 0 || m.full_key_dim == 0 || m.full_kv_heads_total % m.full_layers != 0 ||
        (m.sliding_layers != 0 && (m.sliding_key_dim == 0 ||
                                   m.sliding_kv_heads_total % m.sliding_layers != 0)) ||
        m.linears.empty()) {
        return std::nullopt;
    }
    gemma4::Config c;
    c.hidden_size       = static_cast<std::int32_t>(m.hidden);
    c.vocab_size        = static_cast<std::int32_t>(m.vocab);
    c.layers            = static_cast<std::int32_t>(m.layers);
    c.query_heads       = static_cast<std::int32_t>(m.query_heads);
    c.sliding_window    = static_cast<std::int32_t>(m.window);
    c.intermediate_size = static_cast<std::int32_t>(m.dense_ffn);
    c.experts           = static_cast<std::int32_t>(m.experts);
    c.ple_width         = static_cast<std::int32_t>(m.ple_width);
    c.full.head_dim     = static_cast<std::int32_t>(m.full_key_dim);
    c.full.kv_heads     = static_cast<std::int32_t>(m.full_kv_heads_total / m.full_layers);
    c.sliding           = c.full;
    if (m.sliding_layers != 0) {
        c.sliding.head_dim = static_cast<std::int32_t>(m.sliding_key_dim);
        c.sliding.kv_heads = static_cast<std::int32_t>(m.sliding_kv_heads_total / m.sliding_layers);
    }
    // The trunk's projections (an expert bank runs in the MoE kernel, sized apart).
    std::vector<gemma4::WorkspaceLinear> linears;
    for (const PlanLinear& l : m.linears) {
        if (l.expert) { continue; }
        if (l.format.empty()) { return std::nullopt; }
        QType qtype{};
        try {
            qtype = artifact::parse_format(l.format);
        } catch (const std::exception&) { return std::nullopt; }
        // A gated feed-forward's fused GeGLU is queried too: a start does wherever its layout
        // takes it, so its capacity is the larger side.
        linears.push_back({qtype, static_cast<std::int32_t>(l.n), static_cast<std::int32_t>(l.k),
                           policy_of(qtype), l.role == LinearRole::FfnUp && l.gated});
    }
    // The head (the token table it is tied to): its format is not in the facts; at the few
    // columns a round runs, BF16's capacity stands for it.
    const gemma4::WorkspaceLinear head{QType::BF16, c.vocab_size, c.hidden_size,
                                       ops::LinearPolicy::A16Only};
    const auto chunk   = static_cast<std::int32_t>(std::min(rung.prefill_chunk, rung.max_context));
    const auto batch   = static_cast<std::int32_t>(concurrency);
    const auto context = static_cast<std::int32_t>(rung.max_context);
    // A MoE's generic layer, at the start's widest call (its chunk) for every pass, as the start
    // sizes it once; unknown expert formats leave the formula to stand.
    std::size_t routed = 0;
    if (c.experts > 0) {
        const auto bytes = generic_moe_bytes(m, c, std::max<std::int32_t>(
                                                       static_cast<std::int32_t>(std::min(rung.prefill_chunk, rung.max_context)), 1));
        if (!bytes) { return std::nullopt; }
        routed = *bytes;
    }
    const auto target  = [&](std::int32_t columns, std::int32_t head_columns) -> std::uint64_t {
        return gemma4::target_activation_bytes(c, columns, head_columns) +
               gemma4::target_op_bytes(c, linears, head, context, columns, head_columns, routed) +
               gemma4::kTargetWorkspaceReserve;
    };
    const ops::ScopedInvariantRoutes routes(options.activations == ActivationMode::Exact);
    std::uint64_t text_prefill = 0;
    {
        const ops::ScopedPrefillRoutes prefill(true);
        text_prefill = target(chunk, 1) + ops::sampling_workspace_capacity_bytes(c.vocab_size, 1, 1);
    }
    const std::uint64_t ordinary =
        target(batch, batch) + ops::sampling_workspace_capacity_bytes(c.vocab_size, 1, batch);
    std::uint64_t general = std::max(text_prefill, ordinary);
    if (plan != nullptr && plan->speculative_backend == SpeculativeBackend::Mtp &&
        plan->draft_window != 0) {
        const auto width   = static_cast<std::int32_t>(plan->draft_window) + 1;
        const auto rows    = static_cast<std::int32_t>(detail::speculative_round_batches(*plan));
        const auto columns = static_cast<std::int32_t>(detail::speculative_round_columns(*plan));
        general            = std::max(general, text_prefill + kGemmaDrafterPass);
        if (columns > 0) {
            general = std::max(general,
                               target(columns, columns) +
                                   ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                       c.vocab_size, 1, width - 1, 1, rows) +
                                   kGemmaDrafterPass);
        }
    }
    return general;
}

} // namespace

void register_plan_workspace_queries() {
    set_plan_workspace_query([](const PlanModelFacts& model, const PlanRung& rung,
                                const EngineOptions& options, std::uint32_t concurrency,
                                const detail::SequencePlanImpl* plan) -> std::optional<std::uint64_t> {
        try {
            return gemma4_workspace(model, rung, options, concurrency, plan);
        } catch (const std::exception&) {
            // A geometry an Op's query refuses: the formula stands, as for a family without one.
            return std::nullopt;
        }
    });
}

} // namespace ninferno::runtime
