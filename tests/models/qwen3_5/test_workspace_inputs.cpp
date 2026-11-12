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
// - a plan whose features read a part the view does not carry is refused;
// - the plan without the weights (#615 W2): the same model as a converted artifact's directory, in
//   the same packing, read by --plan's facts (product::plan_facts_of_directory), sizes the dry
//   run's rung (runtime::plan_rung_bytes, the workspace query registered as ninferno-serve
//   registers it) at the start's capacity to the byte, with "start" as its basis; facts without
//   the directory (a config.json, a GGUF header) keep the formula and say so.
// The capacity queries answer for a substituted board (the RTX 5090, the RTX 4060 Ti).

#include "artifact/formats.h"
#include "artifact/schema.h"
#include "core/device_class.h"
#include "product/plan/plan_facts.h"
#include "runtime/engine/plan_dry_run.h"
#include "runtime/engine/plan_workspace.h"
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
namespace art    = ninferno::artifact;
namespace rt     = ninferno::runtime;

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

// The same model as a converted artifact's directory, without its payload: Forms' parents as its
// objects (each bank one object where the layout packs it, else one object a tensor), every
// parameter a load binds (load/text.cpp, load/mtp.cpp, bind_proposal) as a row range of its
// object with the Uses it reads, each recording the converter's policy for its format, the MTP
// component and an indexed proposal head of `proposal_rows`, and the text config as the converter
// writes it (convert/qwen3_5.cpp text_config).
class DirectoryOf {
public:
    DirectoryOf(const qwen::Config& config, const Layout& layout) : c_(config), f_(layout) {}

    art::Directory build(std::uint32_t proposal_rows) {
        const std::uint64_t h = c_.text.hidden_size, v = c_.text.vocab_size;
        const bool moe               = std::holds_alternative<qwen::MoeConfig>(c_.text.ffn);
        d_.components["text"].config = text_config(moe);
        d_.components["mtp"].config =
            art::Json{{"architectures", art::Json::array({moe ? "Qwen3_5MoeMTP" : "Qwen3_5MTP"})}};
        d_.components["mtp"].target = "text";
        object(f_.head, h, {{"text/token_embedding", v, {}}});
        object(f_.head, h, {{"text/output_head", v, {"text/final_hidden", "mtp/final_hidden"}}});
        direct("text/final_norm", {h});
        for (std::size_t i = 0; i < c_.text.layer_types.size(); ++i) {
            block("text/layers/" + std::to_string(i) + "/", c_.text.layer_types[i]);
        }
        object(f_.linear, 2 * h, {{"mtp/input_projection", h, {"mtp/stem_input"}}});
        for (const char* norm : {"mtp/embedding_norm", "mtp/hidden_norm", "mtp/final_norm"}) {
            direct(norm, {h});
        }
        block("mtp/layers/0/", qwen::MixerKind::FullAttention);
        art::Proposal proposal;
        proposal.indexed               = true;
        proposal.rows                  = proposal_rows;
        d_.components["text"].proposal = proposal;
        object(f_.head, h, {{"proposal/head", proposal_rows, {"mtp/final_hidden"}}});
        direct("proposal/token_ids", {proposal_rows}, QType::INT32);
        return std::move(d_);
    }

private:
    struct Member {
        std::string name;
        std::uint64_t rows = 0;
        std::vector<std::string> inputs;
    };

    // One object of the members' rows stacked, `k` wide, in `qtype`: each member a row range.
    void object(QType qtype, std::uint64_t k, const std::vector<Member>& members) {
        std::uint64_t total = 0;
        for (const Member& m : members) { total += m.rows; }
        const std::vector<std::uint64_t> shape{total, k};
        add(qtype, shape, members, k);
    }

    // A bank as one object where the layout packs it, else one object a member.
    void inputs(bool packed, QType qtype, std::uint64_t k, const std::vector<Member>& members) {
        if (packed) {
            object(qtype, k, members);
            return;
        }
        for (const Member& m : members) { object(qtype, k, {m}); }
    }

    // A parameter a load reads as a Tensor (a norm, a convolution, the ids): its own object.
    void direct(const std::string& name, std::vector<std::uint64_t> shape,
                QType qtype = QType::BF16) {
        std::uint64_t elements = 1;
        for (const std::uint64_t d : shape) { elements *= d; }
        add(qtype, shape, {{name, elements, {}}}, 1);
    }

    void add(QType qtype, const std::vector<std::uint64_t>& shape,
             const std::vector<Member>& members, std::uint64_t row_elements) {
        const QuantLayout layout = default_weight_layout(qtype);
        const WeightGeometry g   = weight_geometry(qtype, layout, shape);
        offset_                  = (offset_ + g.alignment - 1) / g.alignment * g.alignment;
        art::TensorObject t;
        t.id     = members.front().name + "#object";
        t.shape  = shape;
        t.format = std::string(art::format_name(qtype));
        t.layout = std::string(art::layout_name(layout));
        t.offset = offset_;
        t.bytes  = g.bytes;
        offset_ += g.bytes;
        const art::ObjectHandle handle{d_.objects.size()};
        d_.objects.emplace_back(t);
        d_.object_index[t.id] = handle;
        std::uint64_t row     = 0;
        for (const Member& m : members) {
            art::Binding b;
            b.whole_object = members.size() == 1;
            b.elements     = m.rows * row_elements;
            b.parts        = {art::Part{handle, row * row_elements, (row + m.rows) * row_elements}};
            row += m.rows;
            d_.bindings[m.name] = b;
            for (const std::string& input : m.inputs) {
                art::Use use;
                use.parameter             = m.name;
                use.input                 = input;
                const ops::LinearPolicy p = policy_of(qtype);
                use.activation_policy =
                    p == ops::LinearPolicy::A16Only   ? art::ActivationPolicy::A16Only
                    : p == ops::LinearPolicy::AllowA4 ? art::ActivationPolicy::AllowA4
                                                      : art::ActivationPolicy::AllowA8;
                d_.uses[{m.name, input}] = use;
            }
        }
    }

    void block(const std::string& p, qwen::MixerKind kind) {
        const std::uint64_t h = c_.text.hidden_size;
        direct(p + "input_norm", {h});
        direct(p + "post_attention_norm", {h});
        const std::vector<std::string> mixer{p + "mixer_input"};
        if (kind == qwen::MixerKind::FullAttention) {
            const auto& a         = *c_.text.attention;
            const std::uint64_t q = a.query_width(), kv = a.key_width();
            inputs(f_.packed, f_.linear, h,
                   {{p + "attention/query", q, mixer},
                    {p + "attention/key", kv, mixer},
                    {p + "attention/gate", q, mixer},
                    {p + "attention/value", kv, mixer}});
            direct(p + "attention/query_norm", {a.head_dim});
            direct(p + "attention/key_norm", {a.head_dim});
            object(f_.linear, q, {{p + "attention/output", h, {p + "attention/gated_output"}}});
        } else {
            const auto& g          = *c_.text.gdn;
            const std::uint64_t kw = g.key_width(), vw = g.value_width(),
                                heads = g.linear_num_value_heads;
            inputs(f_.packed, f_.linear, h,
                   {{p + "gdn/query", kw, mixer},
                    {p + "gdn/key", kw, mixer},
                    {p + "gdn/value", vw, mixer},
                    {p + "gdn/z", vw, mixer}});
            inputs(
                f_.packed, f_.control, h,
                {{p + "gdn/a_projection", heads, mixer}, {p + "gdn/b_projection", heads, mixer}});
            direct(p + "gdn/a_log", {heads}, QType::FP32);
            direct(p + "gdn/dt_bias", {heads}, QType::FP32);
            direct(p + "gdn/convolution", {g.linear_conv_kernel_dim, g.conv_channels()});
            direct(p + "gdn/norm", {g.linear_value_head_dim});
            object(f_.linear, vw, {{p + "gdn/output", h, {p + "gdn/gated_output"}}});
        }
        const std::vector<std::string> in{p + "ffn_input"};
        if (const auto* dense = std::get_if<qwen::DenseConfig>(&c_.text.ffn)) {
            const std::uint64_t i = dense->intermediate_size;
            inputs(f_.packed, f_.linear, h, {{p + "mlp/gate", i, in}, {p + "mlp/up", i, in}});
            object(f_.linear, i, {{p + "mlp/down", h, {p + "mlp/product"}}});
            return;
        }
        const auto& m         = std::get<qwen::MoeConfig>(c_.text.ffn);
        const std::uint64_t e = m.num_experts, ff = m.moe_intermediate_size,
                            s = m.shared_expert_intermediate_size;
        inputs(f_.packed, QType::BF16, h,
               {{p + "moe/router", e, in}, {p + "moe/shared_score", 1, in}});
        std::vector<Member> gate_up, down;
        for (std::uint64_t x = 0; x < e; ++x) {
            const std::string ep = p + "moe/experts/" + std::to_string(x) + "/";
            gate_up.push_back({ep + "gate", ff, in});
            gate_up.push_back({ep + "up", ff, in});
            down.push_back({ep + "down", h, {ep + "product"}});
        }
        object(f_.expert_up, h, gate_up);
        object(f_.expert_down, ff, down);
        object(f_.linear, h, {{p + "moe/shared/gate", s, in}, {p + "moe/shared/up", s, in}});
        object(f_.linear, s, {{p + "moe/shared/down", h, {p + "moe/shared/product"}}});
    }

    art::Json text_config(bool moe) const {
        const auto& t    = c_.text;
        art::Json layers = art::Json::array();
        for (const qwen::MixerKind kind : t.layer_types) {
            layers.push_back(kind == qwen::MixerKind::FullAttention ? "full_attention"
                                                                    : "linear_attention");
        }
        art::Json out = {{"architectures",
                          art::Json::array({moe ? "Qwen3_5MoeForCausalLM" : "Qwen3_5ForCausalLM"})},
                         {"model_type", moe ? "qwen3_5_moe_text" : "qwen3_5_text"},
                         {"hidden_size", t.hidden_size},
                         {"vocab_size", t.vocab_size},
                         {"num_hidden_layers", t.num_hidden_layers},
                         {"max_position_embeddings", t.max_position_embeddings},
                         {"tie_word_embeddings", false},
                         {"rms_norm_eps", 1e-6},
                         {"layer_types", layers},
                         {"num_attention_heads", t.attention->num_attention_heads},
                         {"num_key_value_heads", t.attention->num_key_value_heads},
                         {"head_dim", t.attention->head_dim},
                         {"rope_parameters",
                          {{"rope_theta", 10000000.0},
                           {"partial_rotary_factor", 0.25},
                           {"mrope_section", art::Json::array({11, 11, 10})}}},
                         {"linear_num_key_heads", t.gdn->linear_num_key_heads},
                         {"linear_key_head_dim", t.gdn->linear_key_head_dim},
                         {"linear_num_value_heads", t.gdn->linear_num_value_heads},
                         {"linear_value_head_dim", t.gdn->linear_value_head_dim},
                         {"linear_conv_kernel_dim", t.gdn->linear_conv_kernel_dim}};
        if (const auto* m = std::get_if<qwen::MoeConfig>(&t.ffn)) {
            out["num_experts"]                     = m->num_experts;
            out["num_experts_per_tok"]             = m->num_experts_per_tok;
            out["moe_intermediate_size"]           = m->moe_intermediate_size;
            out["shared_expert_intermediate_size"] = m->shared_expert_intermediate_size;
        } else {
            out["intermediate_size"] = std::get<qwen::DenseConfig>(t.ffn).intermediate_size;
        }
        return out;
    }

    const qwen::Config& c_;
    Layout f_;
    art::Directory d_;
    std::uint64_t offset_ = 0;
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

    // The plan without the weights (#615 W2): the same model as an artifact's directory, its
    // facts as --plan reads them, the dry run's rung at the case's options and context.
    const art::Directory directory = DirectoryOf(config, k.layout).build(kProposalRows);
    const rt::PlanModelFacts facts = product::plan_facts_of_directory(directory, k.name).model;
    check(facts.offline_bytes && facts.artifact_directory != nullptr,
          k.name + ": the artifact's facts lay out its state and keep its directory (" +
              facts.offline_reason + ")");
    rt::PlanRung rung;
    rung.max_context   = k.options.max_context;
    rung.prefill_chunk = k.options.prefill_chunk;
    const auto dry = rt::plan_rung_bytes(facts, rung, k.options, k.options.max_concurrency, false);
    // The same facts without the directory: a config.json's or a GGUF header's.
    rt::PlanModelFacts bare = facts;
    bare.artifact_directory.reset();
    const auto formula =
        rt::plan_rung_bytes(bare, rung, k.options, k.options.max_concurrency, false);
    std::cout << "  --plan workspace " << dry.workspace << " (" << dry.workspace_basis
              << "), the formula's " << formula.workspace << " (" << formula.workspace_basis
              << ")\n";
    check(dry.workspace_basis.rfind("start", 0) == 0,
          k.name + ": the dry run sizes the workspace by the start's planner (" +
              dry.workspace_basis + ")");
    check(dry.workspace == start.capacity,
          k.name + ": the dry run's workspace " + std::to_string(dry.workspace) +
              " is the start's capacity " + std::to_string(start.capacity));
    check(formula.workspace_basis.rfind("formula: ", 0) == 0,
          k.name + ": facts without the artifact's bindings keep the formula and say why (" +
              formula.workspace_basis + ")");
    // The rest of the rung is the start's whatever sizes the workspace.
    check(dry.total() - dry.workspace == formula.total() - formula.workspace,
          k.name + ": only the workspace differs");
}

} // namespace

int main() {
    try {
        // The plan's workspace query, as ninferno-serve registers it.
        rt::register_plan_workspace_queries();
        const DeviceClass rtx5090   = board(0x2B85, 12, 0, 170, 96);
        const DeviceClass rtx4060ti = board(0x2805, 8, 9, 34, 32);
        const DeviceClass rtx4070ti = board(0x2782, 8, 9, 60, 48);
        const DeviceClass rtx2080ti = board(0x1E07, 7, 5, 68, 5);
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
            // The boards of the lab's --plan against the start's log (#615 W2).
            {"Qwen3.5-9B Q8_0, RTX 4070 Ti, MTP 3 optimized head, 2 streams", qwen35_9b,
             converter_q8, rtx4070ti,
             options_of(32768, 2048, 2, SpeculativeBackend::Mtp, ProposalHead::Optimized), nine_b},
            {"Qwen3.5-9B Q8_0, RTX 2080 Ti, no speculation, 1 stream", qwen35_9b, converter_q8,
             rtx2080ti, options_of(16384, 1024, 1, SpeculativeBackend::None, ProposalHead::Full),
             nine_b},
            {"Qwen3.6-35B-A3B Q8, RTX 2080 Ti, MTP 3, 1 stream", qwen36_35b_a3b, converter_q8,
             rtx2080ti, options_of(16384, 1024, 1, SpeculativeBackend::Mtp, ProposalHead::Full),
             a3b_q8},
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
