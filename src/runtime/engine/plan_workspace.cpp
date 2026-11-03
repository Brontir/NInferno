#include "runtime/engine/plan_workspace.h"

#include "artifact/formats.h"
#include "core/weight_view.h"
#include "models/gemma4/config.h"
#include "models/gemma4/workspace.h"
#include "models/qwen3_5/program/planning/sequence_bytes.h"
#include "models/qwen3_5/program/planning/sequence_plan.h"
#include "ninferno/ops/gemma4_attention.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/linear_swiglu.h"
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
#include <string>
#include <vector>

namespace ninferno::runtime {
namespace {

namespace gemma4 = models::gemma4;
namespace detail = models::qwen3_5::detail;

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

// The policy a start's projection carries for a stored format (the converter's: BF16 and FP32
// keep A16, NVFP4 allows four-bit activations, every other quantized format eight).
ops::LinearPolicy policy_of(QType qtype) {
    if (qtype == QType::BF16 || qtype == QType::FP32) { return ops::LinearPolicy::A16Only; }
    if (qtype == QType::NVFP4) { return ops::LinearPolicy::AllowA4; }
    return ops::LinearPolicy::AllowA8;
}

// The assistant's own pass over `rows` rows beside the target's, as DrafterPass::workspace_bytes
// sizes it (gemma4::drafter_pass_bytes) from its config's width, feed-forward and head rows,
// every projection and the head in each format it is or may be stored in (the widest of them; a
// start prepares them A16: ActivationProfile::A16, gemma4/load_mtp.cpp); its backbone width,
// query heads and head geometry are the target's `c` (the load refuses another). Plus the round's
// 1 KiB of ids.
// nullopt with why where the facts lack the assistant.
std::optional<std::uint64_t> gemma_drafter_pass(const gemma4::Config& c,
                                                const std::optional<PlanDrafterGeometry>& d,
                                                std::int32_t context, std::int32_t rows,
                                                std::string& why) {
    if (!d || d->hidden == 0 || d->intermediate == 0 || d->head_rows == 0) {
        why = "the assistant's dimensions are not in the facts";
        return std::nullopt;
    }
    if (d->projection_formats.empty() || d->head_formats.empty()) {
        why = "the assistant's formats are not in the facts";
        return std::nullopt;
    }
    const auto format = [&](const std::string& name) -> std::optional<QType> {
        try {
            return artifact::parse_format(name);
        } catch (const std::exception&) {
            why = "the assistant's format " + name + " is not one a start reads";
            return std::nullopt;
        }
    };
    gemma4::DrafterConfig dc;
    dc.hidden_size       = static_cast<std::int32_t>(d->hidden);
    dc.backbone_size     = c.hidden_size;
    dc.query_heads       = c.query_heads;
    dc.intermediate_size = static_cast<std::int32_t>(d->intermediate);
    dc.sliding           = c.sliding;
    dc.full              = c.full;
    const auto h = static_cast<std::int32_t>(d->hidden), b = c.hidden_size,
               ir = static_cast<std::int32_t>(d->intermediate);
    std::vector<gemma4::WorkspaceLinear> projections;
    for (const std::string& name : d->projection_formats) {
        const auto qtype = format(name);
        if (!qtype) { return std::nullopt; }
        const auto a16 = ops::LinearPolicy::A16Only;
        projections.push_back({*qtype, h, 2 * b, a16});
        projections.push_back({*qtype, b, h, a16});
        for (const auto* g : {&c.sliding, &c.full}) {
            const std::int32_t q_rows = c.query_heads * g->head_dim;
            projections.push_back({*qtype, q_rows, h, a16});
            projections.push_back({*qtype, h, q_rows, a16});
        }
        const DescribedWeight gate_up(*qtype, 2 * static_cast<std::uint64_t>(ir), d->hidden);
        projections.push_back({*qtype, 2 * ir, h, a16, ops::linear_geglu_supports(gate_up.weight)});
        projections.push_back({*qtype, h, ir, a16});
    }
    // The pass's logits rows are the head's in any format: each head format as a projection, the
    // first as the head.
    std::optional<gemma4::WorkspaceLinear> head;
    for (const std::string& name : d->head_formats) {
        const auto qtype = format(name);
        if (!qtype) { return std::nullopt; }
        const gemma4::WorkspaceLinear linear{*qtype, static_cast<std::int32_t>(d->head_rows), h,
                                             ops::LinearPolicy::A16Only};
        if (!head) { head = linear; }
        projections.push_back(linear);
    }
    return gemma4::drafter_pass_bytes(dc, projections, *head, context, std::max(rows, 1)) + 1024;
}

// A generic MoE layer's capacity over a call of up to `columns` tokens, as the start sizes it
// (models/gemma4/target.cu: ops::moe_workspace_bytes over the layer's banks and the load's config,
// gemma4/load_target.cpp): the router [E, H] in BF16, the stacked gate/up [E 2F, H] and down
// [E H, F] in the tensors' expert formats (every pair). nullopt where they are unknown. The start takes
// this Op for experts the fused NVFP4 kernel does not read; where it does, gemma4_moe's capacity
// (target_op_bytes) is the larger, so counting both bounds either.
std::optional<std::size_t> generic_moe_bytes(const PlanModelFacts& m, const models::gemma4::Config& c,
                                             std::int32_t columns, std::string& why) {
    // The banks' shapes from the linears' expert entries, their formats from the tensors
    // (expert_up_formats, expert_down_formats): every pair of them.
    const PlanLinear* gate_up = nullptr;
    const PlanLinear* down    = nullptr;
    for (const PlanLinear& l : m.linears) {
        if (l.expert) { (l.gated ? gate_up : down) = &l; }
    }
    if (gate_up == nullptr || down == nullptr || m.experts == 0 || m.experts_used == 0) {
        why = "the facts list no expert banks or counts";
        return std::nullopt;
    }
    if (m.expert_up_formats.empty() || m.expert_down_formats.empty()) {
        why = "the expert banks' formats are not in the facts";
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
    const auto inter   = static_cast<std::uint64_t>(down->k);
    std::size_t widest = 0;
    for (const std::string& up_format : m.expert_up_formats) {
        for (const std::string& down_format : m.expert_down_formats) {
            const auto up = std::make_unique<DescribedWeight>(artifact::parse_format(up_format),
                                                              experts * 2 * inter, hidden);
            const auto dn = std::make_unique<DescribedWeight>(artifact::parse_format(down_format),
                                                              experts * hidden, inter);
            ops::MoeWeights weights;
            weights.router  = router->weight;
            weights.gate_up = up->weight;
            weights.down    = dn->weight;
            // The banks' policy as a load sets it (models::loading::expert_policy): under A8 a
            // prefill quantizes its activations into the workspace (the grouped W4A8).
            weights.expert_policy =
                ops::moe_policy(policy_of(up->weight.qtype), policy_of(dn->weight.qtype));
            widest = std::max(widest, ops::moe_workspace_bytes(weights, config, 1, columns));
        }
    }
    return widest;
}

// Gemma 4's arena (models/gemma4/program_family.cpp workspace_plan) from the facts: the text
// prefill, the ordinary round and, under MTP, its prefill and round; nullopt where the facts lack
// the geometry or a projection's format.
PlanWorkspaceAnswer gemma4_workspace(const PlanModelFacts& m, const PlanRung& rung,
                                     const EngineOptions& options, std::uint32_t concurrency,
                                     const detail::SequencePlanImpl* plan) {
    if (m.model_type != "gemma4_text") { return {}; }
    // A vision start sizes the arena for the encoder and its handoff too (workspace_plan): the
    // facts do not describe the tower, so the formula stands.
    if (options.enable_vision) { return {std::nullopt, "vision: the encoder's arena is not in the facts"}; }
    if (m.hidden == 0 || m.vocab == 0 || m.layers == 0 || m.query_heads == 0 || !m.ple_known ||
        m.full_layers == 0 || m.full_key_dim == 0 || m.full_kv_heads_total % m.full_layers != 0 ||
        (m.sliding_layers != 0 && (m.sliding_key_dim == 0 ||
                                   m.sliding_kv_heads_total % m.sliding_layers != 0)) ||
        m.linears.empty()) {
        return {std::nullopt, "the facts lack Gemma 4's geometry"};
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
        if (l.format.empty()) { return {std::nullopt, "a projection's format is not in the facts"}; }
        QType qtype{};
        try {
            qtype = artifact::parse_format(l.format);
        } catch (const std::exception&) {
            return {std::nullopt, "a projection's format " + l.format + " is not one a start reads"};
        }
        // A gated feed-forward's fused GeGLU is queried too: a start does wherever its layout
        // takes it, so its capacity is the larger side.
        linears.push_back({qtype, static_cast<std::int32_t>(l.n), static_cast<std::int32_t>(l.k),
                           policy_of(qtype), l.role == LinearRole::FfnUp && l.gated});
    }
    // The head (or the token table it is tied to) in its own format: a quantized head's wide calls
    // take a plane of its rows (target_op_bytes queries its policy and A16, as the start does).
    if (m.head_format.empty()) { return {std::nullopt, "the output head's format is not in the facts"}; }
    QType head_qtype{};
    try {
        head_qtype = artifact::parse_format(m.head_format);
    } catch (const std::exception&) {
        return {std::nullopt, "the output head's format " + m.head_format + " is not one a start reads"};
    }
    const gemma4::WorkspaceLinear head{
        head_qtype,
        static_cast<std::int32_t>(m.head_rows != 0 ? m.head_rows : m.vocab), c.hidden_size,
        policy_of(head_qtype)};
    const auto chunk   = static_cast<std::int32_t>(std::min(rung.prefill_chunk, rung.max_context));
    const auto batch   = static_cast<std::int32_t>(concurrency);
    const auto context = static_cast<std::int32_t>(rung.max_context);
    // A MoE's generic layer at each pass's own columns, as the start sizes it
    // (gemma4::Target::workspace_bytes); unknown expert formats leave the formula to stand.
    if (c.experts > 0) {
        std::string why;
        if (!generic_moe_bytes(m, c, std::max<std::int32_t>(chunk, 1), why)) {
            return {std::nullopt, why};
        }
    }
    const auto target = [&](std::int32_t columns, std::int32_t head_columns) -> std::uint64_t {
        std::string why;
        const std::size_t routed = c.experts > 0 ? generic_moe_bytes(m, c, columns, why).value_or(0) : 0;
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
    // The rung's launch windows as the start's planner lays them out (Gemma 4 plans without the
    // family's sequence plan: its windows from the same inputs, the copy window resolved by the
    // family's rule, as a start resolves it: an MTP start captures depth 15 for copy rounds).
    detail::SequencePlanImpl own;
    if (plan == nullptr && options.speculative.backend == SpeculativeBackend::Mtp) {
        const bool copy_rounds  = detail::family_copy_rounds(models::StateFamily::Gemma4);
        EngineOptions planned   = options;
        planned.max_concurrency = concurrency;
        planned.max_context     = rung.max_context;
        planned.prefill_chunk   = rung.prefill_chunk;
        planned.speculative     = detail::resolve_copy_window(planned.speculative, copy_rounds);
        const detail::SequencePlanningInputs in =
            detail::sequence_planning_inputs(planned, copy_rounds, 0, false);
        own.max_concurrency         = in.max_concurrency;
        own.draft_window            = in.draft_window;
        own.launch_windows          = in.launch_windows;
        own.launch_window_batches   = in.launch_window_batches;
        own.speculative_batch_limit = in.speculative_batch_limit;
        own.speculative_backend     = in.speculative_backend;
        plan                        = &own;
    }
    std::string sized = "no speculative round";
    if (plan != nullptr && plan->speculative_backend == SpeculativeBackend::Mtp &&
        plan->draft_window != 0) {
        sized = "the MTP round at window " + std::to_string(plan->draft_window) + ", " +
                std::to_string(detail::speculative_round_batches(*plan)) + " row(s)" +
                (plan == &own ? " (the start's launch windows)" : "");
        const auto width   = static_cast<std::int32_t>(plan->draft_window) + 1;
        const auto rows    = static_cast<std::int32_t>(detail::speculative_round_batches(*plan));
        const auto columns = static_cast<std::int32_t>(detail::speculative_round_columns(*plan));
        // The prefill drafts one row; a round drafts its rows (program_family.cpp mtp_prefill,
        // mtp_round).
        std::string why;
        const auto one   = gemma_drafter_pass(c, m.drafter_geometry, context, 1, why);
        const auto round = gemma_drafter_pass(c, m.drafter_geometry, context, rows, why);
        if (!one || !round) { return {std::nullopt, why}; }
        general = std::max(general, text_prefill + *one);
        if (columns > 0) {
            general = std::max(general,
                               target(columns, columns) +
                                   ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                                       c.vocab_size, 1, width - 1, 1, rows) +
                                   *round);
        }
    }
    return {general, sized};
}

} // namespace

void register_plan_workspace_queries() {
    set_plan_workspace_query([](const PlanModelFacts& model, const PlanRung& rung,
                                const EngineOptions& options, std::uint32_t concurrency,
                                const detail::SequencePlanImpl* plan) -> PlanWorkspaceAnswer {
        try {
            return gemma4_workspace(model, rung, options, concurrency, plan);
        } catch (const std::exception& error) {
            // A geometry an Op's query refuses: the formula stands, as for a family without one,
            // and the plan says why.
            return {std::nullopt, std::string("an Op's capacity query refused the facts: ") + error.what()};
        }
    });
}

} // namespace ninferno::runtime
