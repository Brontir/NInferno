// The plan without the weights for the dense decoder (dense_decoder, moe_decoder) and NemotronH
// (#615 W3): --plan sizes the workspace with the start's own planner (target_workspace_plan, which
// the start's execution sizes its arena with) over the model the artifact's directory describes
// (Model::describe: the parameters load binds and prepares, without the payload or a device).
//
// Each case is a lab model's geometry as a converted artifact's directory (the converter's packing
// or a GGUF's one-tensor-a-parent, its formats, the Uses with their policies) read by --plan's
// facts (product::plan_facts_of_directory). Checked:
// - the dry run's rung (runtime::plan_rung_bytes, the workspace query registered as
//   ninferno-serve registers it) carries the start planner's capacity to the byte for the start's
//   sequence plan and the tokenizer's public ids, with "start" as its basis, under the fast and
//   the exact profile (A16 activations, the invariant routes);
// - the facts without the directory, without the tokenizer's ids, under a --moe-path pin (a model
//   that routes experts; a dense one answers as unpinned, #892), under
//   speculation (the drafter is an artifact of its own), under --activations int8,
//   --rope-yarn-factor or --state-dtype bf16 (the start refuses them), with more
//   tokenizer ids than embedding rows, and with a Use that reads the payload (an activation
//   divisor) keep the formula and say why.
// The capacity queries answer for a substituted board. No GPU, no files.
// With every CUDA device hidden (as ctest runs it) the bytes are compared as with one visible: the
// attention decode plan prices its residency from the kernels' committed facts (#830, closed), so
// nothing on the start's planner reads the host GPU (the GPU test
// ninferno_decoder_plan_workspace_real_test compares them against a real start).
//
// With NINFERNO_PLAN_WORKSPACE_ARTIFACTS (paths joined by ':'), each artifact file of these
// families is read as --plan reads it (product::plan_facts_of_artifact: its directory and its
// tokenizer's ids) and checked the same way; unset, that part is skipped.

#include "artifact/formats.h"
#include "artifact/schema.h"
#include "core/cuda_error.h"
#include "core/device_class.h"
#include "core/weight_view.h"
#include "models/dense/model.h"
#include "models/dense/program_family.h"
#include "models/nemotron_h/model.h"
#include "models/nemotron_h/program_family.h"
#include "models/qwen3_5/program/internal.h"
#include "models/qwen3_5/program/planning/sequence_bytes.h"
#include "models/qwen3_5/program/planning/sequence_plan.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "ninferno/ops/prefill_routes.h"
#include "ops/generic/kernel_facts.h"
#include "ops/route_pins.h"
#include "product/plan/plan_facts.h"
#include "runtime/engine/plan_dry_run.h"
#include "runtime/engine/plan_workspace.h"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <vector>

namespace {

using namespace ninferno;
namespace detail = ninferno::models::qwen3_5::detail;
namespace art    = ninferno::artifact;
namespace rt     = ninferno::runtime;
namespace dense  = ninferno::models::dense;
namespace nemo   = ninferno::models::nemotron_h;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) { return; }
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

// The policy a converter records for a stored format (BF16 and FP32 keep A16, NVFP4 allows four-bit
// activations, every other format eight).
art::ActivationPolicy policy_of(QType qtype) {
    if (qtype == QType::BF16 || qtype == QType::FP32) { return art::ActivationPolicy::A16Only; }
    if (qtype == QType::NVFP4) { return art::ActivationPolicy::AllowA4; }
    return art::ActivationPolicy::AllowA8;
}

// A converted artifact's directory without its payload: each object a stack of its members' rows,
// every member a binding of its row range with the Uses it is read through, each recording the
// converter's policy for its format.
class DirectoryOf {
public:
    struct Member {
        std::string name;
        std::uint64_t rows = 0;
        std::vector<std::string> inputs;
    };

    explicit DirectoryOf(art::Json config) { d_.components["text"].config = std::move(config); }

    // One object of the members' rows stacked, `k` wide, in `qtype`.
    void object(QType qtype, std::uint64_t k, const std::vector<Member>& members) {
        std::uint64_t total = 0;
        for (const Member& m : members) { total += m.rows; }
        add(qtype, {total, k}, members, k);
    }

    // The members as one object (`packed`, the converter's) or one object each (a GGUF's).
    void objects(bool packed, QType qtype, std::uint64_t k, const std::vector<Member>& members) {
        if (packed) {
            object(qtype, k, members);
            return;
        }
        for (const Member& m : members) { object(qtype, k, {m}); }
    }

    // A parameter a load reads as a Tensor (a norm, a bias, a convolution): its own object.
    void direct(const std::string& name, std::vector<std::uint64_t> shape,
                QType qtype = QType::BF16) {
        std::uint64_t elements = 1;
        for (const std::uint64_t d : shape) { elements *= d; }
        add(qtype, shape, {{name, elements, {}}}, 1);
    }

    [[nodiscard]] art::Directory build() { return std::move(d_); }

private:
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
                use.parameter            = m.name;
                use.input                = input;
                use.activation_policy    = policy_of(qtype);
                d_.uses[{m.name, input}] = use;
            }
        }
    }

    art::Directory d_;
    std::uint64_t offset_ = 0;
};

// How a dense decoder is stored: the converter packs Q/K/V and gate/up into one parent each, a
// GGUF keeps one tensor a parent; the formats of each projection.
struct DenseLayout {
    bool packed       = true;
    QType query_key   = QType::Q8_G32_FP16; // query, key, output, gate, up
    QType value_down  = QType::Q8_G32_FP16; // value, down (a K-quant mix's wider rows)
    QType head        = QType::Q8_G32_FP16;
    QType expert_up   = QType::MXFP4;
    QType expert_down = QType::MXFP4;
};

// A dense decoder's (or a moe_decoder's) directory, as models/dense/load.cpp binds it.
art::Directory dense_directory(const art::Json& c, const DenseLayout& f) {
    DirectoryOf d(c);
    const std::uint64_t h = c.at("hidden_size"), v = c.at("vocab_size");
    const std::uint64_t hd = c.at("head_dim");
    const std::uint64_t q  = hd * c.at("num_attention_heads").get<std::uint64_t>();
    const std::uint64_t kv = hd * c.at("num_key_value_heads").get<std::uint64_t>();
    const bool moe         = c.contains("moe");
    const bool bias        = c.value("attention_bias", false);
    d.object(f.query_key, h, {{"text/token_embedding", v, {}}});
    d.object(f.head, h, {{"text/output_head", v, {"text/final_hidden"}}});
    d.direct("text/final_norm", {h});
    for (std::uint64_t i = 0; i < c.at("num_hidden_layers").get<std::uint64_t>(); ++i) {
        const std::string p = "text/layers/" + std::to_string(i) + "/";
        d.direct(p + "attention_norm", {h});
        d.direct(p + "ffn_norm", {h});
        const std::vector<std::string> in{p + "attention_input"};
        if (f.packed) {
            d.object(f.query_key, h,
                     {{p + "attention/query", q, in},
                      {p + "attention/key", kv, in},
                      {p + "attention/value", kv, in}});
        } else {
            d.object(f.query_key, h, {{p + "attention/query", q, in}});
            d.object(f.query_key, h, {{p + "attention/key", kv, in}});
            d.object(f.value_down, h, {{p + "attention/value", kv, in}});
        }
        d.object(f.query_key, q, {{p + "attention/output", h, {p + "attention/context"}}});
        if (c.value("qk_norm", std::string("none")) == "head") {
            d.direct(p + "attention/query_norm", {hd});
            d.direct(p + "attention/key_norm", {hd});
        }
        if (bias) {
            d.direct(p + "attention/query_bias", {q});
            d.direct(p + "attention/key_bias", {kv});
            d.direct(p + "attention/value_bias", {kv});
        }
        if (c.value("attention_output_bias", false)) { d.direct(p + "attention/output_bias", {h}); }
        if (c.value("attention_sinks", false)) {
            d.direct(p + "attention/sinks", {c.at("num_attention_heads").get<std::uint64_t>()},
                     QType::FP32);
        }
        const std::vector<std::string> ffn{p + "ffn_input"};
        if (moe) {
            const auto& m         = c.at("moe");
            const std::uint64_t e = m.at("num_experts"), ff = m.at("moe_intermediate_size");
            d.object(QType::BF16, h, {{p + "mlp/router", e, ffn}});
            if (m.value("router_bias", false)) { d.direct(p + "mlp/router_bias", {e}); }
            std::vector<DirectoryOf::Member> gate_up, down;
            for (std::uint64_t x = 0; x < e; ++x) {
                const std::string ep = p + "mlp/experts/" + std::to_string(x) + "/";
                gate_up.push_back({ep + "gate_up", 2 * ff, ffn});
                down.push_back({ep + "down", h, {p + "mlp/expert_product"}});
            }
            d.object(f.expert_up, h, gate_up);
            d.object(f.expert_down, ff, down);
            if (m.value("expert_bias", false)) {
                d.direct(p + "mlp/experts/gate_up_bias", {e * 2 * ff});
                d.direct(p + "mlp/experts/down_bias", {e * h});
            }
            continue;
        }
        const std::uint64_t ir = c.at("intermediate_size");
        d.objects(f.packed, f.query_key, h, {{p + "mlp/gate", ir, ffn}, {p + "mlp/up", ir, ffn}});
        d.object(f.value_down, ir, {{p + "mlp/down", h, {p + "mlp/product"}}});
    }
    return d.build();
}

art::Json layer_list(std::uint64_t layers, const std::function<std::string(std::uint64_t)>& kind) {
    art::Json out = art::Json::array();
    for (std::uint64_t i = 0; i < layers; ++i) { out.push_back(kind(i)); }
    return out;
}

// The converter's dense_decoder config (convert's dense text config) for a Llama-lineage model.
art::Json dense_config(std::uint64_t hidden, std::uint64_t layers, std::uint64_t heads,
                       std::uint64_t kv_heads, std::uint64_t ffn, std::uint64_t vocab,
                       std::uint64_t positions, bool qk_norm) {
    return {{"architectures", art::Json::array({"DenseDecoderForCausalLM"})},
            {"model_type", "dense_decoder"},
            {"hidden_size", hidden},
            {"vocab_size", vocab},
            {"num_hidden_layers", layers},
            {"max_position_embeddings", positions},
            {"num_attention_heads", heads},
            {"num_key_value_heads", kv_heads},
            {"head_dim", 128},
            {"intermediate_size", ffn},
            {"rms_norm_eps", 1e-6},
            {"hidden_act", "silu"},
            {"norm_placement", "pre"},
            {"qk_norm", qk_norm ? "head" : "none"},
            {"attention_scale", 0.08838834764831845},
            {"layer_types", layer_list(layers, [](std::uint64_t) { return "full_attention"; })},
            {"layer_rope", layer_list(layers, [](std::uint64_t) { return "default"; })},
            {"rope_groups",
             {{"default",
               {{"rotary_dim", 128}, {"inv_freq", art::Json(std::vector<double>(64, 0.5))}}}}}};
}

// Qwen3-8B: 36 layers of 4096, 32 query and 8 KV heads of 128 with per-head QK norms, a
// feed-forward of 12288, 151936 embedding rows (its tokenizer's 151669 public ids).
art::Json qwen3_8b() { return dense_config(4096, 36, 32, 8, 12288, 151936, 40960, true); }

// Llama-3.1-8B: 32 layers of 4096, 32 query and 8 KV heads of 128, a feed-forward of 14336.
art::Json llama31_8b() { return dense_config(4096, 32, 32, 8, 14336, 128256, 131072, false); }

// gpt-oss-20b as the converter writes it (moe_decoder): 24 layers of 2880 alternating a
// 128-token window and full attention, 64 query and 8 KV heads of 64 with sinks and biases, 32
// experts of 2880 (4 routed), interleaved gate/up, clamped SwiGLU.
art::Json gpt_oss_20b() {
    const std::uint64_t layers = 24;
    const auto kind            = [](std::uint64_t i) {
        return i % 2 == 0 ? "sliding_attention" : "full_attention";
    };
    return {{"architectures", art::Json::array({"MoeDecoderForCausalLM"})},
            {"model_type", "moe_decoder"},
            {"hidden_size", 2880},
            {"vocab_size", 201088},
            {"num_hidden_layers", layers},
            {"max_position_embeddings", 131072},
            {"num_attention_heads", 64},
            {"num_key_value_heads", 8},
            {"head_dim", 64},
            {"intermediate_size", 2880},
            {"rms_norm_eps", 1e-5},
            {"hidden_act", "silu"},
            {"norm_placement", "pre"},
            {"qk_norm", "none"},
            {"moe",
             {{"num_experts", 32},
              {"num_experts_per_tok", 4},
              {"moe_intermediate_size", 2880},
              {"scoring", "softmax"},
              {"renormalize", true},
              {"activation", "clamped_swiglu"},
              {"alpha", 1.702},
              {"limit", 7.0},
              {"gate_up_layout", "interleaved"},
              {"router_bias", true},
              {"expert_bias", true},
              {"shared_intermediate_size", 0}}},
            {"attention_sinks", true},
            {"attention_bias", true},
            {"attention_output_bias", true},
            {"attention_scale", 0.125},
            {"sliding_window", 128},
            {"layer_types", layer_list(layers, kind)},
            {"layer_rope", layer_list(layers, kind)},
            {"rope_groups",
             {{"sliding_attention",
               {{"rotary_dim", 64}, {"inv_freq", art::Json(std::vector<double>(32, 0.5))}}},
              {"full_attention",
               {{"rotary_dim", 64}, {"inv_freq", art::Json(std::vector<double>(32, 0.5))}}}}}};
}

// Nemotron 3 Nano 30B-A3B's 52 blocks: Mamba2 (M), MoE (E) and attention (*).
std::string nemotron_pattern() { return "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEMEME"; }

// Its converter config: 2688 wide, 32 query and 2 KV heads of 128, Mamba2 64 heads of 64 over 8
// groups of state 128, 128 experts of 1856 (6 routed) beside a shared expert of 3712.
art::Json nemotron_nano() {
    art::Json layers = art::Json::array();
    for (const char ch : nemotron_pattern()) {
        layers.push_back(ch == 'M' ? "mamba" : ch == '*' ? "attention" : "moe");
    }
    return {{"architectures", art::Json::array({"NemotronHForCausalLM"})},
            {"model_type", "nemotron_h"},
            {"hidden_size", 2688},
            {"vocab_size", 131072},
            {"num_hidden_layers", nemotron_pattern().size()},
            {"max_position_embeddings", 262144},
            {"rms_norm_eps", 1e-5},
            {"num_attention_heads", 32},
            {"num_key_value_heads", 2},
            {"head_dim", 128},
            {"attention_scale", 0.08838834764831845},
            {"layer_types", layers},
            {"mamba",
             {{"num_heads", 64},
              {"head_dim", 64},
              {"n_groups", 8},
              {"state_size", 128},
              {"conv_kernel", 4}}},
            {"moe",
             {{"num_experts", 128},
              {"num_experts_per_tok", 6},
              {"moe_intermediate_size", 1856},
              {"shared_intermediate_size", 3712}}}};
}

// NemotronH's directory, as models/nemotron_h/load.cpp binds it: Q8_0 projections and expert
// banks (one parent each), a BF16 router.
art::Directory nemotron_directory(const art::Json& c, QType linear, QType experts) {
    DirectoryOf d(c);
    const std::uint64_t h = c.at("hidden_size"), v = c.at("vocab_size");
    const std::uint64_t hd    = c.at("head_dim");
    const std::uint64_t q     = hd * c.at("num_attention_heads").get<std::uint64_t>();
    const std::uint64_t kv    = hd * c.at("num_key_value_heads").get<std::uint64_t>();
    const auto& mb            = c.at("mamba");
    const std::uint64_t mh    = mb.at("num_heads"),
                        inner = mh * mb.at("head_dim").get<std::uint64_t>();
    const std::uint64_t gn =
        mb.at("n_groups").get<std::uint64_t>() * mb.at("state_size").get<std::uint64_t>();
    const std::uint64_t in_rows = 2 * inner + 2 * gn + mh, conv = inner + 2 * gn;
    const auto& moe       = c.at("moe");
    const std::uint64_t e = moe.at("num_experts"), ff = moe.at("moe_intermediate_size"),
                        fs = moe.at("shared_intermediate_size");
    d.object(linear, h, {{"text/token_embedding", v, {}}});
    d.object(linear, h, {{"text/output_head", v, {"text/final_hidden"}}});
    d.direct("text/final_norm", {h});
    const auto& types = c.at("layer_types");
    for (std::size_t i = 0; i < types.size(); ++i) {
        const std::string p = "text/layers/" + std::to_string(i) + "/";
        const std::vector<std::string> in{p + "mixer_input"};
        d.direct(p + "norm", {h});
        const std::string kind = types[i];
        if (kind == "mamba") {
            d.object(linear, h, {{p + "mamba/in_proj", in_rows, in}});
            d.direct(p + "mamba/conv_weight", {conv, 4});
            d.direct(p + "mamba/conv_bias", {conv});
            d.direct(p + "mamba/dt_bias", {mh}, QType::FP32);
            d.direct(p + "mamba/a_log", {mh}, QType::FP32);
            d.direct(p + "mamba/d", {mh}, QType::FP32);
            d.direct(p + "mamba/norm", {inner});
            d.object(linear, inner, {{p + "mamba/out_proj", h, {p + "mamba/gated"}}});
        } else if (kind == "attention") {
            d.object(linear, h,
                     {{p + "attention/query", q, in},
                      {p + "attention/key", kv, in},
                      {p + "attention/value", kv, in}});
            d.object(linear, q, {{p + "attention/output", h, {p + "attention/context"}}});
        } else {
            d.object(QType::BF16, h, {{p + "mlp/router", e, in}});
            d.direct(p + "mlp/selection_bias", {e}, QType::FP32);
            std::vector<DirectoryOf::Member> up, down;
            for (std::uint64_t x = 0; x < e; ++x) {
                const std::string ep = p + "mlp/experts/" + std::to_string(x) + "/";
                up.push_back({ep + "up", ff, in});
                down.push_back({ep + "down", h, {p + "mlp/expert_product"}});
            }
            d.object(experts, h, up);
            d.object(experts, ff, down);
            d.object(linear, h, {{p + "mlp/shared_up", fs, in}});
            d.object(linear, fs, {{p + "mlp/shared_down", h, {p + "mlp/shared_product"}}});
        }
    }
    return d.build();
}

DeviceClass board(std::uint32_t pci, int major, int minor, int sms, std::size_t l2_mib) {
    DeviceClass d = synthetic_device_class(pci, major, minor, sms, l2_mib << 20, 0);
    d.facts       = machine_facts_of(d);
    return d;
}

EngineOptions options_of(std::uint32_t context, std::uint32_t chunk, std::uint32_t concurrency,
                         ActivationMode activations = ActivationMode::Fast) {
    EngineOptions o;
    o.max_context     = context;
    o.prefill_chunk   = chunk;
    o.max_concurrency = concurrency;
    o.kv_capacity     = KvCapacityPolicy::explicit_capacity(context * concurrency);
    o.kv_cache        = KvCacheStorage::BFloat16;
    o.activations     = activations;
    return o;
}

// The plan a start's planner lays out for `options` (make_sequence_planner_impl: the options the
// family validated, its copy window, the smallest candidate), its workspace not yet planned.
detail::SequencePlanImpl plan_of(const EngineOptions& options, models::StateFamily family,
                                 std::uint64_t positions, std::int32_t sms) {
    detail::validate_program_options(options, positions, 0);
    const detail::SequencePlanningInputs inputs =
        detail::sequence_planning_inputs(options, detail::family_copy_rounds(family), sms, false);
    const auto range = detail::sequence_page_range(inputs.capacity, inputs.max_concurrency);
    return std::move(*detail::sequence_plan_shell(inputs, range.minimum));
}

// The start planner's arenas over the described model, as the start's execution sizes them
// (DenseExecution / NemotronExecution::workspace_plan without a drafter or the MTP head).
detail::WorkspacePlan start_plan(const art::Directory& directory, const EngineOptions& options,
                                 std::int32_t tokens, const detail::SequencePlanImpl& plan) {
    const bool exact = options.activations == ActivationMode::Exact;
    const ops::ScopedInvariantRoutes routes(exact);
    const std::string type =
        directory.components.at("text").config.value("model_type", std::string());
    if (type == "nemotron_h") {
        const auto model = nemo::Model::describe(
            directory, exact ? nemo::ActivationProfile::A16 : nemo::ActivationProfile::Artifact);
        return nemo::target_workspace_plan(*model, tokens, plan);
    }
    const auto model = dense::Model::describe(
        directory, exact ? dense::ActivationProfile::A16 : dense::ActivationProfile::Artifact);
    return dense::target_workspace_plan(*model, tokens, plan);
}

std::string bytes(const detail::WorkspacePlan& p) {
    return "prefill " + std::to_string(p.text_prefill) + ", ordinary " +
           std::to_string(p.ordinary_round) + ", capacity " + std::to_string(p.capacity);
}

std::string with_commas(std::uint64_t n) {
    std::string s = std::to_string(n);
    for (int i = static_cast<int>(s.size()) - 3; i > 0; i -= 3) { s.insert(i, ","); }
    return s;
}

// No CUDA device is visible (ctest runs this with CUDA_VISIBLE_DEVICES=-1).
bool device_hidden = false;

// No known escape: #830 (the attention decode plan, which DensePass and NemotronH's Pass size their
// attention with, read the host GPU's occupancy) is closed; the plan reads the kernels' facts.
// Those price a board only where its kernels are this build's: the decode tile is a constexpr of
// the build (#912), so a board of another arch refuses by name and keeps the formula.
bool own_tile(const DeviceClass& device) {
    namespace kf = ninferno::ops::detail::generic::kernel_facts;
    return kf::release_arch(device.major, device.minor) == kf::build_arch();
}

// A board of another arch: the query refuses by name (#912) and the formula stands.
void check_refused_other_arch(const std::string& name, const std::string& basis) {
    check(basis.rfind("formula: ", 0) == 0 && basis.find("no kernel facts") != std::string::npos &&
              basis.find("#912") != std::string::npos,
          name + ": a board of another arch than the build's keeps the formula, by name (" + basis +
              ")");
}

// The rung's dry run against the start planner's capacity; prints the formula's beside it.
void compare(const std::string& name, const rt::PlanModelFacts& facts,
             const art::Directory& directory, const EngineOptions& options,
             const DeviceClass& device) {
    const ScopedDeviceClass substitute(device);
    const auto family = facts.model_type == "nemotron_h" ? models::StateFamily::NemotronH
                                                         : models::StateFamily::DenseDecoder;
    const auto plan   = plan_of(options, family, facts.native_context, device.sms);
    rt::PlanRung rung;
    rung.max_context   = options.max_context;
    rung.prefill_chunk = options.prefill_chunk;
    const auto dry     = rt::plan_rung_bytes(facts, rung, options, options.max_concurrency, false);
    rt::PlanModelFacts bare = facts;
    bare.artifact_directory.reset();
    const auto formula = rt::plan_rung_bytes(bare, rung, options, options.max_concurrency, false);
    if (!own_tile(device)) {
        std::cout << name << ": another arch than the build's, --plan workspace " << dry.workspace
                  << " (" << dry.workspace_basis << ")" << std::endl;
        check_refused_other_arch(name, dry.workspace_basis);
        return;
    }
    const auto start =
        start_plan(directory, options, static_cast<std::int32_t>(facts.tokenizer_tokens), plan);
    std::cout << name << ": start " << bytes(start) << "\n  --plan workspace " << dry.workspace
              << " (" << dry.workspace_basis << "), the formula's " << formula.workspace << "\n";
    std::cout << "  | " << name << " | " << with_commas(formula.workspace) << " | "
              << with_commas(dry.workspace) << " |\n";
    check(start.text_prefill > 0 && start.ordinary_round > 0 && start.causal_score == 0 &&
              start.mtp_prefill == 0 && start.dflash_context == 0 &&
              start.capacity == std::max(start.text_prefill, start.ordinary_round),
          name + ": the target's arenas sized, the capacity their widest");
    check(dry.workspace_basis.rfind("start", 0) == 0,
          name + ": the dry run sizes the workspace by the start's planner (" +
              dry.workspace_basis + ")");
    check(dry.workspace == start.capacity,
          name + ": the dry run's workspace " + std::to_string(dry.workspace) +
              " is the start's capacity " + std::to_string(start.capacity));
    check(formula.workspace_basis.rfind("formula: ", 0) == 0,
          name + ": facts without the artifact's bindings keep the formula and say why (" +
              formula.workspace_basis + ")");
    check(dry.total() - dry.workspace == formula.total() - formula.workspace,
          name + ": only the workspace differs");
}

// The query's answer for `facts` at `options` (the rung's sequence plan laid out as above).
std::string basis_of(const rt::PlanModelFacts& facts, const EngineOptions& options,
                     const DeviceClass& device) {
    const ScopedDeviceClass substitute(device);
    const auto family     = facts.model_type == "nemotron_h" ? models::StateFamily::NemotronH
                                                             : models::StateFamily::DenseDecoder;
    EngineOptions planned = options;
    planned.speculative   = {};
    const auto plan       = plan_of(planned, family, facts.native_context, device.sms);
    rt::PlanRung rung;
    rung.max_context   = options.max_context;
    rung.prefill_chunk = options.prefill_chunk;
    std::string basis;
    (void)rt::plan_workspace_estimate(facts, rung, options, options.max_concurrency, &plan, &basis);
    return basis;
}

void declines(const std::string& name, const rt::PlanModelFacts& facts,
              const art::Directory& directory, const DeviceClass& device,
              SpeculativeBackend backend) {
    const EngineOptions options = options_of(32768, 2048, 1);
    const auto refuses          = [&](const std::string& basis, const std::string& why,
                             const std::string& what) {
        check(basis.rfind("formula: ", 0) == 0 && basis.find(why) != std::string::npos,
                       name + ": " + what + " keeps the formula (" + basis + ")");
    };
    const std::string answered = basis_of(facts, options, device);
    if (!own_tile(device)) {
        check_refused_other_arch(name, answered);
        return;
    }
    check(answered.rfind("start", 0) == 0,
          name + ": the query answers at the decline cases' options (" + answered + ")");
    rt::PlanModelFacts unread = facts;
    unread.tokenizer_tokens   = 0;
    refuses(basis_of(unread, options, device), "tokenizer", "without the tokenizer's ids, it");
    rt::PlanModelFacts wide = facts;
    wide.tokenizer_tokens   = static_cast<std::uint32_t>(facts.vocab) + 1;
    refuses(basis_of(wide, options, device), "embedding has rows",
            "a tokenizer wider than the embedding");
    {
        // The overrides of the options planned decline; a pin left in the process by an earlier
        // engine is not these options' (#842 review).
        EngineOptions pinned       = options;
        pinned.maintainer.moe_path = MoePath::Generic;
        // The facts themselves, not only the rule: a dense decoder routes no experts.
        check(facts.model_type != "dense_decoder" || facts.experts <= 1,
              name + ": a dense_decoder's facts name no routed experts");
        if (facts.model_type == "dense_decoder" && facts.experts <= 1) {
            // No MoE layer for the pin to move: the answer is the unpinned one (#892).
            check(basis_of(facts, pinned, device) == answered,
                  name + ": a --moe-path pin leaves a model without experts as it is");
        } else {
            refuses(basis_of(facts, pinned, device), "--moe-path", "a pinned MoE path");
        }
        EngineOptions routed           = options;
        routed.maintainer.decode_route = DecodeRoute::A16;
        refuses(basis_of(facts, routed, device), "route is pinned", "a pinned decode route");
        EngineOptions evidence           = options;
        evidence.maintainer.route_policy = RoutePolicy::Evidence;
        refuses(basis_of(facts, evidence, device), "--route-policy", "the evidence policy");
        ops::RoutePins stale;
        stale.moe = MoePath::Generic;
        ops::set_route_pins(stale);
        const std::string unpinned = basis_of(facts, options, device);
        ops::set_route_pins({});
        check(unpinned.find("--moe-path") == std::string::npos,
              name + ": a stale process pin does not decline the plan (" + unpinned + ")");
    }
    EngineOptions speculative            = options;
    speculative.speculative.backend      = backend;
    speculative.speculative.draft_tokens = 3;
    refuses(basis_of(facts, speculative, device), "speculation", "speculation");
    refuses(basis_of(facts, options_of(32768, 2048, 1, ActivationMode::Int8), device), "int8",
            "--activations int8");
    // The start's other refusals of the Qwen-only options, in its words (#892).
    EngineOptions yarn    = options;
    yarn.rope_yarn_factor = 4.0F;
    refuses(basis_of(facts, yarn, device), "--rope-yarn-factor", "--rope-yarn-factor");
    EngineOptions narrow   = options;
    narrow.recurrent_state = RecurrentStateStorage::BF16;
    refuses(basis_of(facts, narrow, device), "--state-dtype", "--state-dtype bf16");
    // A Use that reads the payload: its activation divisor (an NVFP4 A4 route's calibration).
    art::Directory divided = directory;
    {
        const std::string head = "text/output_head";
        const auto use         = divided.uses.find({head, "text/final_hidden"});
        art::TensorObject t;
        t.id     = "divisor#object";
        t.shape  = {1};
        t.format = std::string(art::format_name(QType::FP32));
        t.layout = std::string(art::layout_name(default_weight_layout(QType::FP32)));
        t.offset = 0;
        t.bytes  = 4;
        const art::ObjectHandle handle{divided.objects.size()};
        divided.objects.emplace_back(t);
        divided.object_index[t.id] = handle;
        art::Binding b;
        b.whole_object                                      = true;
        b.elements                                          = 1;
        b.parts                                             = {art::Part{handle, 0, 1}};
        use->second.auxiliaries["activation_input_divisor"] = b;
    }
    rt::PlanModelFacts payload = facts;
    payload.artifact_directory = std::make_shared<const art::Directory>(divided);
    refuses(basis_of(payload, options, device), "does not hold",
            "a binding that reads the payload");
    rt::PlanModelFacts bare = facts;
    bare.artifact_directory.reset();
    refuses(basis_of(bare, options, device), "artifact", "facts without the directory");
}

struct Case {
    std::string name;
    std::function<art::Directory()> directory;
    std::uint32_t tokens = 0; // the tokenizer's public ids
    DeviceClass device;
    EngineOptions options;
};

// A lab board of this build's arch (the decode tile is the build's, #912), or none: each case also
// runs on it where its own board is of another arch, so every stand compares a start.
std::optional<DeviceClass> own_board() {
    namespace kf             = ninferno::ops::detail::generic::kernel_facts;
    const std::string_view a = kf::build_arch();
    if (a == "70") return board(0x1DBA, 7, 0, 80, 6);      // Quadro GV100
    if (a == "75") return board(0x1E07, 7, 5, 68, 5);      // RTX 2080 Ti
    if (a == "86") return board(0x2204, 8, 6, 82, 6);      // RTX 3090
    if (a == "89") return board(0x2782, 8, 9, 60, 48);     // RTX 4070 Ti
    if (a == "120a") return board(0x2B85, 12, 0, 170, 96); // RTX 5090
    return std::nullopt;
}

void run(const Case& k) {
    const art::Directory directory = k.directory();
    rt::PlanModelFacts facts       = product::plan_facts_of_directory(directory, k.name).model;
    check(facts.offline_bytes && facts.artifact_directory != nullptr,
          k.name + ": the artifact's facts lay out its state and keep its directory (" +
              facts.offline_reason + ")");
    // As plan_facts_of_artifact reads them from the file's resources.
    facts.tokenizer_tokens = k.tokens;
    compare(k.name, facts, directory, k.options, k.device);
    if (!own_tile(k.device)) {
        if (const auto own = own_board()) {
            compare(k.name + " (on this build's arch, sm_" +
                        std::string(ninferno::ops::detail::generic::kernel_facts::build_arch()) +
                        ")",
                    facts, directory, k.options, *own);
        }
    }
}

// The artifact files NINFERNO_PLAN_WORKSPACE_ARTIFACTS names, read as --plan reads them.
void real_artifacts(const std::vector<std::pair<std::string, DeviceClass>>& boards) {
    const char* env = std::getenv("NINFERNO_PLAN_WORKSPACE_ARTIFACTS");
    if (env == nullptr || *env == '\0') {
        std::cout << "real artifacts: skipped (NINFERNO_PLAN_WORKSPACE_ARTIFACTS unset)\n";
        return;
    }
    std::stringstream paths(env);
    std::string path;
    while (std::getline(paths, path, ':')) {
        if (path.empty()) { continue; }
        // What the query adds to a start's (and a --plan's) reads of the facts: the tokenizer
        // parsed once a path (the first read), the model described once a profile (the first
        // query); the later reads and queries are the caches'.
        using clock   = std::chrono::steady_clock;
        const auto ms = [](clock::duration d) {
            return std::chrono::duration<double, std::milli>(d).count();
        };
        const auto t0     = clock::now();
        const auto source = product::plan_facts_of_artifact(path);
        const auto t1     = clock::now();
        (void)product::plan_facts_of_artifact(path);
        const auto t2                  = clock::now();
        const rt::PlanModelFacts facts = source.model;
        check(facts.artifact_directory != nullptr && facts.tokenizer_tokens != 0,
              path + ": the facts keep the directory and the tokenizer's ids");
        if (facts.artifact_directory == nullptr || facts.tokenizer_tokens == 0) { continue; }
        std::cout << path << ": " << facts.model_type << ", tokenizer " << facts.tokenizer_tokens
                  << " of " << facts.vocab << " rows\n";
        {
            const ScopedDeviceClass substitute(boards.front().second);
            rt::PlanRung rung;
            rung.max_context          = 32768;
            rung.prefill_chunk        = 2048;
            const EngineOptions first = options_of(32768, 2048, 1);
            const auto t3             = clock::now();
            (void)rt::plan_rung_bytes(facts, rung, first, 1, false);
            const auto t4 = clock::now();
            (void)rt::plan_rung_bytes(facts, rung, first, 1, false);
            const auto t5 = clock::now();
            std::cout << "  facts " << ms(t1 - t0) << " ms first (tokenizer parsed), "
                      << ms(t2 - t1) << " ms again; rung " << ms(t4 - t3)
                      << " ms first (model described), " << ms(t5 - t4) << " ms again\n";
        }
        const std::string stem = std::filesystem::path(path).stem().string();
        // A long context within the model's positions.
        const std::uint32_t wide = std::min<std::uint32_t>(65536, facts.native_context);
        for (const auto& [label, device] : boards) {
            compare(stem + ", " + label + ", ctx 32k, chunk 2048, 1 stream", facts,
                    *facts.artifact_directory, options_of(32768, 2048, 1), device);
            compare(stem + ", " + label + ", ctx " + std::to_string(wide / 1024) +
                        "k, chunk 4096, 4 streams",
                    facts, *facts.artifact_directory, options_of(wide, 4096, 4), device);
        }
    }
}

} // namespace

int main() {
    try {
        // The plan's workspace query (and the tokenizer's), as ninferno-serve registers them.
        rt::register_plan_workspace_queries();
        int devices   = 0;
        device_hidden = cudaGetDeviceCount(&devices) != cudaSuccess || devices == 0;
        if (device_hidden) {
            std::cout << "every CUDA device hidden: the bytes are compared all the same (#830)\n";
        }
        const DeviceClass rtx5090   = board(0x2B85, 12, 0, 170, 96);
        const DeviceClass rtx4070ti = board(0x2782, 8, 9, 60, 48);
        const DeviceClass rtx4060ti = board(0x2805, 8, 9, 34, 32);
        const DenseLayout converter_q8{};
        // A Q4_K_M GGUF: one parent a tensor, Q4_K projections with Q6_K value and down rows and
        // head.
        const DenseLayout gguf_q4km{.packed     = false,
                                    .query_key  = QType::GGML_Q4_K,
                                    .value_down = QType::GGML_Q6_K,
                                    .head       = QType::GGML_Q6_K};
        // gpt-oss-20b: Q8_0 attention and head, MXFP4 experts, the converter's packing.
        const DenseLayout gpt_oss{};
        const auto qwen3    = [&] { return dense_directory(qwen3_8b(), gguf_q4km); };
        const auto llama    = [&] { return dense_directory(llama31_8b(), converter_q8); };
        const auto oss      = [&] { return dense_directory(gpt_oss_20b(), gpt_oss); };
        const auto nemotron = [] {
            return nemotron_directory(nemotron_nano(), QType::Q8_G32_FP16, QType::Q8_G32_FP16);
        };
        const std::vector<Case> cases = {
            {"Qwen3-8B Q4_K_M, RTX 4060 Ti, ctx 32k, chunk 2048, 1 stream", qwen3, 151669,
             rtx4060ti, options_of(32768, 2048, 1)},
            {"Qwen3-8B Q4_K_M, RTX 5090, ctx 40k, chunk 4096, 4 streams", qwen3, 151669, rtx5090,
             options_of(40960, 4096, 4)},
            {"Llama-3.1-8B Q8_0, RTX 4070 Ti, ctx 32k, chunk 2048, 2 streams", llama, 128256,
             rtx4070ti, options_of(32768, 2048, 2)},
            {"Llama-3.1-8B Q8_0, RTX 4070 Ti, exact, ctx 32k, chunk 2048, 1 stream", llama, 128256,
             rtx4070ti, options_of(32768, 2048, 1, ActivationMode::Exact)},
            {"gpt-oss-20b MXFP4, RTX 5090, ctx 64k, chunk 4096, 4 streams", oss, 200019, rtx5090,
             options_of(65536, 4096, 4)},
            {"gpt-oss-20b MXFP4, RTX 4070 Ti, ctx 32k, chunk 2048, 1 stream", oss, 200019,
             rtx4070ti, options_of(32768, 2048, 1)},
            {"gpt-oss-20b MXFP4, RTX 4060 Ti, exact, ctx 16k, chunk 1024, 1 stream", oss, 200019,
             rtx4060ti, options_of(16384, 1024, 1, ActivationMode::Exact)},
            {"Nemotron-3-Nano-30B-A3B Q8_0, RTX 5090, ctx 64k, chunk 4096, 4 streams", nemotron,
             131072, rtx5090, options_of(65536, 4096, 4)},
            {"Nemotron-3-Nano-30B-A3B Q8_0, RTX 4070 Ti, ctx 32k, chunk 2048, 1 stream", nemotron,
             131072, rtx4070ti, options_of(32768, 2048, 1)},
            {"Nemotron-3-Nano-30B-A3B Q8_0, RTX 4060 Ti, exact, ctx 16k, chunk 1024, 1 stream",
             nemotron, 131072, rtx4060ti, options_of(16384, 1024, 1, ActivationMode::Exact)},
        };
        for (const Case& k : cases) { run(k); }

        // The declines, once a family.
        for (const auto& [name, make, tokens, backend] :
             std::vector<std::tuple<std::string, std::function<art::Directory()>, std::uint32_t,
                                    SpeculativeBackend>>{
                 {"Llama-3.1-8B", llama, 128256, SpeculativeBackend::DFlash},
                 {"gpt-oss-20b", oss, 200019, SpeculativeBackend::DFlash},
                 {"Nemotron-3-Nano", nemotron, 131072, SpeculativeBackend::Mtp}}) {
            const art::Directory directory = make();
            rt::PlanModelFacts facts = product::plan_facts_of_directory(directory, name).model;
            facts.tokenizer_tokens   = tokens;
            declines(name, facts, directory, rtx4070ti, backend);
        }

        real_artifacts({{"RTX 5090", rtx5090}, {"RTX 4070 Ti", rtx4070ti}});
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return 1;
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "decoder plan workspace: pass\n";
    return 0;
}
