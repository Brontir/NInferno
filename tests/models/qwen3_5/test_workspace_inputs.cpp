// The Qwen 3.5 workspace planner over a view of prepared forms (#615 W1,
// models/qwen3_5/program/planning/workspace_plan.h): no Model, no device, no files.
//
// A start passes its Parameters' own forms through workspace_inputs (qwen3_5_workspace_plan). Here
// those forms are prepared by the loader's own functions (ops::prepare_*, as
// execution/parameters.cpp calls them) over weights described by format and shape alone, in a
// converter's packing (each bank one parent) or a GGUF's (one parent a tensor), for the geometries
// of Qwen3.5-9B and Qwen3.6-35B-A3B. Checked:
// - the start's view (workspace_inputs over owned forms) and a view built field by field over a
//   second set of the same forms, described at other addresses, give the same plan to the byte:
//   the bytes are a function of (format, rows, inputs, policy) and the forms, which a plan
//   without the weights can describe;
// - the arenas the plan's features ask for are sized, the others are 0, and the widest is the
//   capacity (no Vision region: qwen3_5_workspace_plan adds it);
// - a plan whose features read a part the view does not carry is refused.
// The capacity queries answer for a substituted board (the RTX 5090, the RTX 4060 Ti).

#include "core/device_class.h"
#include "core/weight_view.h"
#include "models/qwen3_5/config.h"
#include "models/qwen3_5/execution/parameter_types.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/planning/sequence_bytes.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/planning/workspace_plan.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/weight_input.h"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <iostream>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <variant>
#include <vector>

namespace {

using namespace ninferno;
namespace qwen   = ninferno::models::qwen3_5;
namespace ex     = ninferno::models::qwen3_5::execution;
namespace detail = ninferno::models::qwen3_5::detail;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) { return; }
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

// The policy a converter records for a stored format (BF16 and FP32 keep A16, every other
// format here allows eight-bit activations).
ops::LinearPolicy policy_of(QType qtype) {
    if (qtype == QType::BF16 || qtype == QType::FP32) { return ops::LinearPolicy::A16Only; }
    if (qtype == QType::NVFP4) { return ops::LinearPolicy::AllowA4; }
    return ops::LinearPolicy::AllowA8;
}

// Weights described by format and shape alone: one parent per bank, its inputs consecutive row
// ranges of it, at an address that is never read (each parent its own, from `base`).
class Described {
public:
    explicit Described(std::uintptr_t base) : next_(base) {}

    Described(const Described&)            = delete;
    Described& operator=(const Described&) = delete;

    std::vector<ops::WeightInput> bank(QType qtype, std::uint64_t k,
                                       std::span<const std::uint64_t> rows) {
        std::uint64_t total = 0;
        for (const std::uint64_t r : rows) { total += r; }
        WeightParent& parent        = parents_.emplace_back();
        const std::uint64_t shape[] = {total, k};
        parent.geometry             = weight_geometry(qtype, default_weight_layout(qtype), shape);
        parent.data                 = reinterpret_cast<const std::byte*>(next_);
        parent.weight_scale_divisor = 1.0F;
        next_ += std::uintptr_t{1} << 32;
        std::vector<ops::WeightInput> out;
        std::uint64_t row = 0;
        for (const std::uint64_t r : rows) {
            WeightView& view = views_.emplace_back();
            view.shape       = {r, k};
            view.parts       = {WeightRegion{&parent, row * k, (row + r) * k}};
            out.push_back(ops::WeightInput{view, policy_of(qtype)});
            row += r;
        }
        return out;
    }

    std::vector<ops::WeightInput> bank(QType qtype, std::uint64_t k,
                                       std::initializer_list<std::uint64_t> rows) {
        return bank(qtype, k, std::span<const std::uint64_t>(rows.begin(), rows.size()));
    }

    ops::WeightInput one(QType qtype, std::uint64_t n, std::uint64_t k) {
        return bank(qtype, k, {n}).front();
    }

    // A bank's inputs as one parent (`packed`) or one parent each.
    std::vector<ops::WeightInput> inputs(bool packed, QType qtype, std::uint64_t k,
                                         std::initializer_list<std::uint64_t> rows) {
        if (packed) { return bank(qtype, k, rows); }
        std::vector<ops::WeightInput> out;
        for (const std::uint64_t r : rows) { out.push_back(one(qtype, r, k)); }
        return out;
    }

private:
    std::deque<WeightParent> parents_;
    std::deque<WeightView> views_;
    std::uintptr_t next_;
};

// How a load stores the model: a converter packs each bank into one parent, a GGUF keeps one
// tensor a parent; the formats of the trunk's projections, the GDN control, the routed experts
// and the head.
struct Layout {
    bool packed       = true;
    QType linear      = QType::Q8_G32_FP16;
    QType control     = QType::BF16;
    QType expert_up   = QType::Q8_G32_FP16;
    QType expert_down = QType::Q8_G32_FP16;
    QType head        = QType::Q8_G32_FP16;
};

// The forms a load prepares (execution/parameters.cpp, Prepare), over described weights.
class Forms {
public:
    Forms(const qwen::Config& config, const Layout& layout, std::uintptr_t base)
        : c_(config), f_(layout), d_(base) {}

    ex::FfnParameters ffn() {
        const std::uint64_t h = c_.text.hidden_size;
        if (const auto* dense = std::get_if<qwen::DenseConfig>(&c_.text.ffn)) {
            const std::uint64_t i = dense->intermediate_size;
            const auto gu         = d_.inputs(f_.packed, f_.linear, h, {i, i});
            auto gate_up          = ops::prepare_dense_gate_up_weights(gu[0], gu[1]);
            ex::DenseParameters p;
            p.down = ops::prepare_linear_weight(d_.one(f_.linear, h, i));
            if (auto* split = std::get_if<ops::SegmentedProjectionWeights>(&gate_up)) {
                p.split_gate_up = std::move(*split);
            } else {
                p.gate_up = std::get<ops::SingleProjectionWeight>(gate_up);
            }
            return p;
        }
        const auto& m         = std::get<qwen::MoeConfig>(c_.text.ffn);
        const std::uint64_t e = m.num_experts, ff = m.moe_intermediate_size,
                            s = m.shared_expert_intermediate_size;
        // The router and the shared expert's score: one BF16 parent in a converter's packing (the
        // SparseMoe router bank).
        const auto routers = d_.inputs(f_.packed, QType::BF16, h, {e, 1});
        const auto& router = routers[0];
        const auto& score  = routers[1];
        const std::vector<std::uint64_t> gate_up_rows(2 * e, ff), down_rows(e, h);
        const auto gate_up     = d_.bank(f_.expert_up, h, gate_up_rows);
        const auto down        = d_.bank(f_.expert_down, ff, down_rows);
        const auto shared      = d_.bank(f_.linear, h, {s, s});
        const auto shared_down = d_.one(f_.linear, h, s);
        auto generic = ops::prepare_moe_weights(router, score, gate_up, down, shared[0], shared[1],
                                                shared_down);
        // --moe-path auto: SparseMoe where it runs the layer (Prepare::ffn).
        if (m.num_experts_per_tok == 8 && ops::sparse_moe_admits(generic)) {
            return ops::prepare_sparse_moe_weights(router, score, gate_up, down, shared[0],
                                                   shared[1], shared_down);
        }
        ops::MoeConfig config;
        config.top_k          = static_cast<std::int32_t>(m.num_experts_per_tok);
        config.scoring        = ops::MoeScoring::Softmax;
        config.renormalize    = true;
        config.activation     = ops::MoeActivation::SwiGlu;
        config.gate_up_layout = ops::MoeGateUpLayout::Concatenated;
        return ex::GenericMoeParameters{std::move(generic), config};
    }

    ex::BlockParameters block(qwen::MixerKind kind) {
        const std::uint64_t h = c_.text.hidden_size;
        ex::BlockParameters out;
        out.ffn = ffn();
        if (kind == qwen::MixerKind::FullAttention) {
            const std::uint64_t q  = c_.text.attention->query_width(),
                                kv = c_.text.attention->key_width();
            const auto in          = d_.inputs(f_.packed, f_.linear, h, {q, kv, q, kv});
            out.mixer              = ex::AttentionParameters{
                ops::prepare_attn_input_proj_weights(in[0], in[1], in[2], in[3]),
                             {},
                             {},
                ops::prepare_linear_weight(d_.one(f_.linear, h, q))};
        } else {
            const auto& g          = *c_.text.gdn;
            const std::uint64_t kw = g.key_width(), vw = g.value_width(),
                                heads = g.linear_num_value_heads;
            const auto in             = d_.inputs(f_.packed, f_.linear, h, {kw, kw, vw, vw});
            const auto ab             = d_.inputs(f_.packed, f_.control, h, {heads, heads});
            out.mixer =
                ex::GdnParameters{ops::prepare_gdn_input_proj_weights(in[0], in[1], in[2], in[3]),
                                  ops::prepare_gdn_gating_proj_weights(ab[0], ab[1]),
                                  {},
                                  {},
                                  {},
                                  {},
                                  ops::prepare_linear_weight(d_.one(f_.linear, h, vw))};
        }
        return out;
    }

    ex::TextParameters text() {
        ex::TextParameters out;
        out.output_head =
            ops::prepare_linear_weight(d_.one(f_.head, c_.text.vocab_size, c_.text.hidden_size));
        for (const qwen::MixerKind kind : c_.text.layer_types) {
            out.layers.push_back(block(kind));
        }
        return out;
    }

    ex::MtpParameters mtp() {
        const std::uint64_t h = c_.text.hidden_size, q = c_.text.attention->query_width(),
                            kv = c_.text.attention->key_width();
        ex::MtpParameters out;
        out.input_projection = ops::prepare_linear_weight(d_.one(f_.linear, h, 2 * h));
        const auto in        = d_.inputs(f_.packed, f_.linear, h, {q, kv, q, kv});
        const bool dense     = c_.text.architecture == models::Architecture::Qwen3_5;
        if (!dense || ops::consecutive_rows_of_one_parent(in)) {
            out.projection.packed =
                ops::prepare_linear_weight(std::span<const ops::WeightInput>(in));
        }
        if (dense) {
            out.projection.rows =
                std::array{ops::prepare_linear_weight(in[0]), ops::prepare_linear_weight(in[1]),
                           ops::prepare_linear_weight(in[2]), ops::prepare_linear_weight(in[3])};
        }
        out.output      = ops::prepare_linear_weight(d_.one(f_.linear, h, q));
        out.ffn         = ffn();
        out.output_head = ops::prepare_linear_weight(d_.one(f_.head, c_.text.vocab_size, h));
        return out;
    }

    ex::ProposalParameters proposal(std::uint32_t rows) {
        return {ops::prepare_linear_weight(d_.one(f_.head, rows, c_.text.hidden_size)),
                std::nullopt, rows};
    }

private:
    const qwen::Config& c_;
    Layout f_;
    Described d_;
};

std::vector<qwen::MixerKind> hybrid_layers(std::uint32_t layers) {
    std::vector<qwen::MixerKind> out;
    for (std::uint32_t i = 0; i < layers; ++i) {
        out.push_back((i + 1) % 4 == 0 ? qwen::MixerKind::FullAttention
                                       : qwen::MixerKind::LinearAttention);
    }
    return out;
}

qwen::Config text_config(models::Architecture architecture, std::uint32_t hidden,
                         std::uint32_t layers, qwen::AttentionConfig attention,
                         qwen::GdnConfig gdn) {
    qwen::Config c;
    c.mtp                          = true;
    c.text.architecture            = architecture;
    c.text.hidden_size             = hidden;
    c.text.vocab_size              = 248320;
    c.text.num_hidden_layers       = layers;
    c.text.max_position_embeddings = 262144;
    c.text.layer_types             = hybrid_layers(layers);
    c.text.full_attention_layers   = layers / 4;
    c.text.linear_attention_layers = layers - layers / 4;
    c.text.attention               = attention;
    c.text.gdn                     = gdn;
    return c;
}

// Qwen3.5-9B: 32 layers of 4096, 16 query and 4 KV heads of 256, GDN 16 key and 32 value heads of
// 128, a dense FFN of 12288.
qwen::Config qwen35_9b() {
    auto c =
        text_config(models::Architecture::Qwen3_5, 4096, 32, {16, 4, 256}, {16, 128, 32, 128, 4});
    c.text.ffn = qwen::DenseConfig{12288};
    return c;
}

// Qwen3.6-35B-A3B: 40 layers of 2048, 16 query and 2 KV heads of 256, GDN 16 key and 32 value
// heads of 128, 256 experts of 512 (8 routed) beside a shared expert of 512.
qwen::Config qwen36_35b_a3b() {
    auto c     = text_config(models::Architecture::Qwen3_5Moe, 2048, 40, {16, 2, 256},
                             {16, 128, 32, 128, 4});
    c.text.ffn = qwen::MoeConfig{256, 8, 512, 512};
    return c;
}

DeviceClass board(std::uint32_t pci, int major, int minor, int sms, std::size_t l2_mib) {
    DeviceClass d = synthetic_device_class(pci, major, minor, sms, l2_mib << 20, 0);
    d.facts       = machine_facts_of(d);
    return d;
}

// The plan a start's planner lays out for `options` (make_sequence_planner_impl: the options the
// family validated, its copy window, the smallest candidate), its workspace not yet planned.
detail::SequencePlanImpl plan_of(const qwen::Config& config, const EngineOptions& options,
                                 std::int32_t sms) {
    detail::validate_program_options(options, config.text.max_position_embeddings,
                                     qwen::kMaximumMtpDraftTokens, true);
    const detail::SequencePlanningInputs inputs = detail::sequence_planning_inputs(
        options, detail::family_copy_rounds(models::StateFamily::Qwen35), sms, false);
    const auto range = detail::sequence_page_range(inputs.capacity, inputs.max_concurrency);
    return std::move(*detail::sequence_plan_shell(inputs, range.minimum));
}

EngineOptions options_of(std::uint32_t context, std::uint32_t chunk, std::uint32_t concurrency,
                         SpeculativeBackend backend, ProposalHead head) {
    EngineOptions o;
    o.max_context               = context;
    o.prefill_chunk             = chunk;
    o.max_concurrency           = concurrency;
    o.kv_capacity               = KvCapacityPolicy::explicit_capacity(context * concurrency);
    o.kv_cache                  = KvCacheStorage::Fp8E4M3Row256;
    o.speculative.backend       = backend;
    o.speculative.draft_tokens  = backend == SpeculativeBackend::Mtp ? 3 : 0;
    o.speculative.proposal_head = head;
    return o;
}

bool same(const detail::WorkspacePlan& a, const detail::WorkspacePlan& b) {
    return a.text_prefill == b.text_prefill && a.ordinary_round == b.ordinary_round &&
           a.mtp_prefill == b.mtp_prefill && a.mtp_round == b.mtp_round &&
           a.dflash_context == b.dflash_context && a.dflash_round == b.dflash_round &&
           a.dflash_round_append == b.dflash_round_append && a.causal_score == b.causal_score &&
           a.general_capacity == b.general_capacity &&
           a.vision.has_value() == b.vision.has_value() && a.capacity == b.capacity;
}

std::string bytes(const detail::WorkspacePlan& p) {
    return "prefill " + std::to_string(p.text_prefill) + ", ordinary " +
           std::to_string(p.ordinary_round) + ", mtp prefill " + std::to_string(p.mtp_prefill) +
           ", mtp round " + std::to_string(p.mtp_round) + ", capacity " +
           std::to_string(p.capacity);
}

const char* form(const ops::ProjectionWeights& p) {
    return std::holds_alternative<ops::SingleProjectionWeight>(p)    ? "single"
           : std::holds_alternative<ops::PairedProjectionWeights>(p) ? "paired"
                                                                     : "segmented";
}

// The forms the load gave the first attention and GDN layers' projections and the feed-forward:
// "attention/gdn input/gdn control/ffn".
std::string forms(const ex::TextParameters& text) {
    std::string attention, gdn;
    for (const auto& block : text.layers) {
        if (const auto* a = std::get_if<ex::AttentionParameters>(&block.mixer)) {
            if (attention.empty()) { attention = form(a->projection); }
        } else if (gdn.empty()) {
            const auto& g = std::get<ex::GdnParameters>(block.mixer);
            gdn           = std::string(form(g.projection)) + "/" + form(g.control);
        }
    }
    const auto& ffn = text.layers.front().ffn;
    const char* f   = std::holds_alternative<ops::SparseMoeWeights>(ffn)      ? "sparse moe"
                      : std::holds_alternative<ex::GenericMoeParameters>(ffn) ? "generic moe"
                      : std::get<ex::DenseParameters>(ffn).split_gate_up      ? "dense split"
                                                                              : "dense";
    return attention + "/" + gdn + "/" + f;
}

struct Case {
    std::string name;
    std::function<qwen::Config()> config;
    Layout layout;
    DeviceClass device;
    EngineOptions options;
    std::string forms; // the layout branches the case takes (forms(), above)
};

void run(const Case& k) {
    const ScopedDeviceClass substitute(k.device);
    const qwen::Config config             = k.config();
    const std::int32_t public_tokens      = static_cast<std::int32_t>(config.text.vocab_size);
    const auto plan                       = plan_of(config, k.options, k.device.sms);
    const bool mtp                        = plan.features.mtp();
    const bool optimal                    = plan.proposal_head == ProposalHead::Optimized;
    constexpr std::uint32_t kProposalRows = 32768;

    // The start's forms, as its Parameters owns them, and its view of them.
    Forms start_forms(config, k.layout, std::uintptr_t{1} << 40);
    const ex::TextParameters text = start_forms.text();
    std::optional<ex::MtpParameters> start_mtp;
    std::optional<ex::ProposalParameters> start_proposal;
    if (mtp) { start_mtp = start_forms.mtp(); }
    if (mtp && optimal) { start_proposal = start_forms.proposal(kProposalRows); }
    const auto start = detail::build_workspace_plan(
        detail::workspace_inputs(config, public_tokens, text, start_mtp, start_proposal,
                                 std::nullopt),
        plan);

    // The same forms described again at other addresses, the view built field by field.
    Forms described_forms(config, k.layout, std::uintptr_t{1} << 44);
    const ex::TextParameters described_text = described_forms.text();
    std::optional<ex::MtpParameters> described_mtp;
    std::optional<ex::ProposalParameters> described_proposal;
    if (mtp) { described_mtp = described_forms.mtp(); }
    if (mtp && optimal) { described_proposal = described_forms.proposal(kProposalRows); }
    const detail::WorkspaceInputs view{.config        = config.text,
                                       .public_tokens = public_tokens,
                                       .text          = described_text,
                                       .mtp           = described_mtp ? &*described_mtp : nullptr,
                                       .proposal =
                                           described_proposal ? &*described_proposal : nullptr};
    const auto described = detail::build_workspace_plan(view, plan);

    std::cout << k.name << " (" << forms(text) << "): " << bytes(start) << '\n';
    check(forms(text) == k.forms, k.name + ": the forms are " + forms(text) + ", not " + k.forms);
    check(same(start, described), k.name + ": the described forms' plan (" + bytes(described) +
                                      ") is the start's (" + bytes(start) + ")");
    check(start.text_prefill > 0 && start.ordinary_round > 0, k.name + ": text arenas sized");
    check(mtp == (start.mtp_prefill > 0 && start.mtp_round > 0),
          k.name + ": MTP arenas sized exactly under MTP");
    check(start.dflash_context == 0 && start.dflash_round == 0 && start.causal_score == 0,
          k.name + ": no drafter or scoring arena");
    check(start.general_capacity == std::max({start.text_prefill, start.ordinary_round,
                                              start.mtp_prefill, start.mtp_round}) &&
              start.capacity == start.general_capacity && !start.vision,
          k.name + ": the capacity is the widest arena, no Vision region");

    if (mtp) {
        // A plan that reads the MTP layer (or the optimized head) the view leaves out is refused.
        detail::WorkspaceInputs missing = view;
        missing.mtp                     = nullptr;
        bool refused                    = false;
        try {
            (void)detail::build_workspace_plan(missing, plan);
        } catch (const std::invalid_argument&) { refused = true; }
        check(refused, k.name + ": an MTP plan without the MTP layer is refused");
        if (optimal) {
            detail::WorkspaceInputs headless = view;
            headless.proposal                = nullptr;
            refused                          = false;
            try {
                (void)detail::build_workspace_plan(headless, plan);
            } catch (const std::invalid_argument&) { refused = true; }
            check(refused, k.name + ": an optimized-head plan without the head is refused");
        }
    }
}

} // namespace

int main() {
    try {
        const DeviceClass rtx5090   = board(0x2B85, 12, 0, 170, 96);
        const DeviceClass rtx4060ti = board(0x2805, 8, 9, 34, 32);
        const Layout converter_q8{};
        // A GGUF: one parent a tensor (the generic segmented projections), K-quant experts (the
        // generic MoE), a Q8_0 control.
        const Layout gguf{.packed      = false,
                          .linear      = QType::Q8_G32_FP16,
                          .control     = QType::Q8_G32_FP16,
                          .expert_up   = QType::GGML_Q4_K,
                          .expert_down = QType::GGML_Q6_K,
                          .head        = QType::GGML_Q6_K};
        // The 9B's geometry has no fused input or control projection (every load segments them);
        // the 35B-A3B's Q8 parents take the fused forms and SparseMoe, a GGUF's the generic ones.
        const std::string nine_b      = "segmented/segmented/segmented/dense";
        const std::string a3b_q8      = "single/single/single/sparse moe";
        const std::vector<Case> cases = {
            {"Qwen3.5-9B Q8_0, RTX 5090, MTP 3, 1 stream", qwen35_9b, converter_q8, rtx5090,
             options_of(32768, 2048, 1, SpeculativeBackend::Mtp, ProposalHead::Full), nine_b},
            {"Qwen3.5-9B Q8_0, RTX 5090, MTP 3 optimized head, 4 streams", qwen35_9b, converter_q8,
             rtx5090, options_of(32768, 2048, 4, SpeculativeBackend::Mtp, ProposalHead::Optimized),
             nine_b},
            {"Qwen3.5-9B GGUF, RTX 4060 Ti, no speculation, 2 streams", qwen35_9b, gguf, rtx4060ti,
             options_of(16384, 1024, 2, SpeculativeBackend::None, ProposalHead::Full),
             "segmented/segmented/segmented/dense split"},
            {"Qwen3.6-35B-A3B Q8, RTX 4060 Ti, MTP 3 optimized head, 1 stream", qwen36_35b_a3b,
             converter_q8, rtx4060ti,
             options_of(32768, 2048, 1, SpeculativeBackend::Mtp, ProposalHead::Optimized), a3b_q8},
            {"Qwen3.6-35B-A3B Q8, RTX 5090, MTP 3, 4 streams", qwen36_35b_a3b, converter_q8,
             rtx5090, options_of(65536, 4096, 4, SpeculativeBackend::Mtp, ProposalHead::Full),
             a3b_q8},
            {"Qwen3.6-35B-A3B GGUF, RTX 4060 Ti, no speculation, 1 stream", qwen36_35b_a3b, gguf,
             rtx4060ti, options_of(32768, 2048, 1, SpeculativeBackend::None, ProposalHead::Full),
             "segmented/segmented/segmented/generic moe"},
        };
        for (const Case& k : cases) { run(k); }
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "workspace inputs: pass\n";
    return 0;
}
