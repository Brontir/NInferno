// The plan without the weights takes a start's bytes: for every family, the state shape the dry
// run reads from a config.json or a GGUF header (product/plan/plan_facts.cpp) is the one the
// family's own planner reads from its bound config (each family's state_shape), and a dry-run
// rung's bytes (runtime/engine/plan_dry_run.h, plan_rung_bytes) are the start's planner's to the
// byte, but for the workspace: its Op scratch the bound weights' routes size, an estimate.
//
// The start's side is the planner a start builds (make_sequence_planner_impl, startup.h, over its
// device-free seam: the options the family validated and the multiprocessor count): its copy
// window resolved from the model's traits, its inputs, its candidates and its affine page curve,
// then the plan finalize_sequence_plan_impl lays out at the rung's pages. The model under it is a
// ModelExecution that lays out the family's synthetic shape (the families' persistent_layout is
// plan_persistent_layout over their state_shape; test_shapes checks the shapes) with the family's
// traits (family_copy_rounds, which the families' traits read) and no workspace: the real
// families need bound weights and a device to construct. Synthetic shapes of the lab's models
// over concurrencies, speculation (none, MTP, DFlash), codecs, contexts, chunks, cached states and
// the copy window (automatic as a server's default options leave it, and off). CPU, no GPU, no
// files.

#include "models/dense/program_family.h"
#include "models/gemma4/program_family.h"
#include "models/nemotron_h/program_family.h"
#include "models/qwen3_5/program/model_family.h"
#include "models/qwen3_5/program/planning/sequence_bytes.h"
#include "models/qwen3_5/program/planning/startup.h"
#include "models/qwen3_5/program/qwen3_5_execution.h"
#include "product/plan/plan_facts.h"
#include "product/plan/plan_quants.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/plan_dry_run.h"
#include "runtime/engine/plan_ladder.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <functional>
#include <iostream>
#include <memory>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace ninferno::models::qwen3_5::detail {
namespace {

// A model the start's planner plans over without weights: a family's state shape laid out by the
// family's rule, the family's traits, no workspace (the dry run estimates it). Nothing runs on it.
class ShapeExecution final : public ModelExecution {
public:
    explicit ShapeExecution(const StateShape& shape)
        : shape_(shape), traits_{.hidden        = shape.hidden,
                                 .output_rows   = shape.output_rows,
                                 .public_tokens = shape.public_tokens,
                                 .copy_rounds   = family_copy_rounds(shape.family)} {}

    [[nodiscard]] const ModelTraits& traits() const noexcept override { return traits_; }

    [[nodiscard]] models::LoadOptions load_options() const override { return {}; }

    void validate_options(DeviceContext&, const EngineOptions&) const override { unused(); }

    [[nodiscard]] PersistentLayout persistent_layout(const SequencePlanImpl& plan) const override {
        return plan_persistent_layout(shape_, plan);
    }

    [[nodiscard]] WorkspacePlan workspace_plan(const SequencePlanImpl&) const override {
        return {};
    }

    [[nodiscard]] std::uint32_t prefill_call_tokens(std::uint32_t, std::uint32_t,
                                                    std::uint32_t) const override {
        unused();
    }

    [[nodiscard]] execution::PrefillChunkResult
    prefill_text_chunk(execution::PrefillContext&, std::span<const TokenId>, std::uint32_t,
                       std::optional<std::uint32_t>, bool) const override {
        unused();
    }

    void output_logits(const Tensor&, Tensor&, WorkspaceArena&, cudaStream_t) const override {
        unused();
    }

    void capture_ordinary_decode_batch(execution::OrdinaryBatchContext&, std::int32_t,
                                       ops::CausalAttentionExecutionEnvelope,
                                       DecodeGraphDefinition&) const override {
        unused();
    }

    void ordinary_decode_batch(execution::OrdinaryBatchContext&, std::int32_t,
                               ops::CausalAttentionExecutionEnvelope,
                               DecodeGraphExecutable*) const override {
        unused();
    }

private:
    [[noreturn]] static void unused() {
        throw std::logic_error("the planning-only model runs nothing");
    }

    StateShape shape_;
    ModelTraits traits_;
};

} // namespace
} // namespace ninferno::models::qwen3_5::detail

namespace {

namespace rt     = ninferno::runtime;
namespace pr     = ninferno::product;
namespace m      = ninferno::models;
namespace detail = ninferno::models::qwen3_5::detail;
using ninferno::EngineOptions;
using ninferno::KvCacheStorage;
using ninferno::RecurrentStateStorage;
using ninferno::ProposalHead;
using ninferno::SpeculativeBackend;
using nlohmann::json;

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;

int failures         = 0;
std::size_t compared = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

std::string shape_text(const m::StateShape& s) {
    std::string out = "family " + std::to_string(static_cast<int>(s.family)) + " full " +
                      std::to_string(s.full_layers) + "x" + std::to_string(s.kv_heads) + "x" +
                      std::to_string(s.head_dim) + " sliding " + std::to_string(s.sliding_layers) +
                      "x" + std::to_string(s.sliding_kv_heads) + "x" +
                      std::to_string(s.sliding_head_dim) + " w" + std::to_string(s.window) +
                      " recurrent " + std::to_string(s.recurrent_layers) + " v" +
                      std::to_string(s.value_heads) + "x" + std::to_string(s.value_dim) + " k" +
                      std::to_string(s.key_heads) + "x" + std::to_string(s.key_dim) + " conv" +
                      std::to_string(s.conv_kernel) + " h" + std::to_string(s.hidden) + " vocab" +
                      std::to_string(s.output_rows) + "/" + std::to_string(s.public_tokens);
    if (s.drafter) {
        const auto& d = *s.drafter;
        out += " drafter full " + std::to_string(d.full_layers) + " local " +
               std::to_string(d.local_layers) + " w" + std::to_string(d.window) + " " +
               std::to_string(d.kv_heads) + "x" + std::to_string(d.head_dim) + " features " +
               std::to_string(d.feature_rows) + "/" + std::to_string(d.prefill_feature_rows) +
               (d.dflash2 ? " dflash2" : "");
    }
    return out;
}

// ---- The families' configs (as their loaders bind them) and the same models' config.json.

struct Subject {
    std::string name;
    m::StateShape family; // the family planner's shape, from its bound config
    rt::PlanModelFacts facts;
    std::vector<KvCacheStorage> kv;    // the codecs the family lays out
    std::vector<KvCacheStorage> rings; // its sliding rings' (BF16 alone without rings)
    std::vector<SpeculativeBackend> spec;
    bool recurrent = false;
};

json qwen27b_config() {
    return {{"architectures", {"Qwen3_5ForConditionalGeneration"}},
            {"model_type", "qwen3_5"},
            {"text_config",
             {{"model_type", "qwen3_5_text"},
              {"hidden_size", 5120},
              {"num_hidden_layers", 64},
              {"num_attention_heads", 24},
              {"num_key_value_heads", 4},
              {"head_dim", 256},
              {"max_position_embeddings", 262144},
              {"vocab_size", 248320},
              {"intermediate_size", 17408},
              {"full_attention_interval", 4},
              {"linear_num_value_heads", 48},
              {"linear_value_head_dim", 128},
              {"linear_num_key_heads", 16},
              {"linear_key_head_dim", 128},
              {"linear_conv_kernel_dim", 4}}}};
}

// The GGUF header of the same model (llama.cpp's qwen35 keys, the MTP block after the trunk).
json qwen27b_gguf() {
    return {{"general.architecture", "qwen35"},     {"qwen35.block_count", 65},
            {"qwen35.nextn_predict_layers", 1},     {"qwen35.context_length", 262144},
            {"qwen35.embedding_length", 5120},      {"qwen35.vocab_size", 248320},
            {"qwen35.feed_forward_length", 17408},  {"qwen35.attention.head_count", 24},
            {"qwen35.attention.head_count_kv", 4},  {"qwen35.attention.key_length", 256},
            {"qwen35.attention.value_length", 256}, {"qwen35.full_attention_interval", 4},
            {"qwen35.ssm.inner_size", 6144},        {"qwen35.ssm.state_size", 128},
            {"qwen35.ssm.group_count", 16},         {"qwen35.ssm.time_step_rank", 48},
            {"qwen35.ssm.conv_kernel", 4}};
}

// A DFlash drafter's config (the artifact's dflash component).
json qwen_drafter_config(bool dflash2) {
    json config = {
        {"architectures", {dflash2 ? "DFlash2DraftModel" : "DFlashDraftModel"}},
        {"num_hidden_layers", 5},
        {"num_attention_heads", 32},
        {"num_key_value_heads", 8},
        {"head_dim", 128},
        {"sliding_window", 2048},
        {"layer_types",
         {"sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention",
          "full_attention"}},
        {"dflash_config", {{"target_layer_ids", {1, 16, 31, 46, 61}}, {"mask_token_id", 0}}}};
    if (dflash2) { config["dflash_config"]["selector_rank"] = 64; }
    return config;
}

m::qwen3_5::Config qwen27b_family(bool drafter, bool dflash2) {
    m::qwen3_5::Config c;
    c.text.hidden_size             = 5120;
    c.text.vocab_size              = 248320;
    c.text.num_hidden_layers       = 64;
    c.text.max_position_embeddings = 262144;
    c.text.full_attention_layers   = 16;
    c.text.linear_attention_layers = 48;
    c.text.attention               = m::qwen3_5::AttentionConfig{24, 4, 256};
    c.text.gdn                     = m::qwen3_5::GdnConfig{.linear_num_key_heads   = 16,
                                                           .linear_key_head_dim    = 128,
                                                           .linear_num_value_heads = 48,
                                                           .linear_value_head_dim  = 128,
                                                           .linear_conv_kernel_dim = 4};
    if (drafter) {
        m::qwen3_5::DraftConfig d;
        d.attention         = {32, 8, 128};
        d.num_hidden_layers = 5;
        d.layer_types.assign(4, m::qwen3_5::DraftAttentionKind::SlidingAttention);
        d.layer_types.push_back(m::qwen3_5::DraftAttentionKind::FullAttention);
        d.sliding_window   = 2048;
        d.target_layer_ids = {1, 16, 31, 46, 61};
        if (dflash2) { d.dflash2 = m::qwen3_5::DFlash2Config{4, 16, 64, 8}; }
        c.draft = d;
    }
    return c;
}

json qwen35b_config() {
    return {{"architectures", {"Qwen3_5MoeForConditionalGeneration"}},
            {"text_config",
             {{"model_type", "qwen3_5_moe_text"},
              {"hidden_size", 2048},
              {"num_hidden_layers", 40},
              {"num_attention_heads", 16},
              {"num_key_value_heads", 2},
              {"head_dim", 256},
              {"max_position_embeddings", 262144},
              {"vocab_size", 248320},
              {"full_attention_interval", 4},
              {"num_experts", 256},
              {"num_experts_per_tok", 8},
              {"moe_intermediate_size", 512},
              {"shared_expert_intermediate_size", 512},
              {"linear_num_value_heads", 32},
              {"linear_value_head_dim", 128},
              {"linear_num_key_heads", 16},
              {"linear_key_head_dim", 128},
              {"linear_conv_kernel_dim", 4}}}};
}

m::qwen3_5::Config qwen35b_family() {
    m::qwen3_5::Config c;
    c.text.hidden_size             = 2048;
    c.text.vocab_size              = 248320;
    c.text.num_hidden_layers       = 40;
    c.text.max_position_embeddings = 262144;
    c.text.full_attention_layers   = 10;
    c.text.linear_attention_layers = 30;
    c.text.attention               = m::qwen3_5::AttentionConfig{16, 2, 256};
    c.text.gdn                     = m::qwen3_5::GdnConfig{.linear_num_key_heads   = 16,
                                                           .linear_key_head_dim    = 128,
                                                           .linear_num_value_heads = 32,
                                                           .linear_value_head_dim  = 128,
                                                           .linear_conv_kernel_dim = 4};
    return c;
}

json llama8b_config() {
    return {{"architectures", {"LlamaForCausalLM"}},
            {"model_type", "llama"},
            {"hidden_size", 4096},
            {"num_hidden_layers", 32},
            {"num_attention_heads", 32},
            {"num_key_value_heads", 8},
            {"head_dim", 128},
            {"max_position_embeddings", 131072},
            {"vocab_size", 128256},
            {"intermediate_size", 14336}};
}

m::dense::Config llama8b_family() {
    m::dense::Config c;
    c.hidden_size = 4096;
    c.vocab_size  = 128256;
    c.layers      = 32;
    c.kv_heads    = 8;
    c.head_dim    = 128;
    c.layer_specs.assign(32, m::dense::LayerSpec{.window = 0});
    return c;
}

json gptoss20b_config() {
    json types = json::array();
    for (int i = 0; i < 24; ++i) {
        types.push_back(i % 2 == 0 ? "sliding_attention" : "full_attention");
    }
    return {{"architectures", {"GptOssForCausalLM"}},
            {"model_type", "gpt_oss"},
            {"hidden_size", 2880},
            {"num_hidden_layers", 24},
            {"num_attention_heads", 64},
            {"num_key_value_heads", 8},
            {"head_dim", 64},
            {"sliding_window", 128},
            {"layer_types", types},
            {"max_position_embeddings", 131072},
            {"vocab_size", 201088},
            {"intermediate_size", 2880},
            {"num_local_experts", 32},
            {"experts_per_token", 4}};
}

// gpt-oss's DFlash drafter (a separate artifact: the caller reads its config).
json gptoss_drafter_config() {
    return {
        {"architectures", {"DFlashDraftModel"}},
        {"model_type", "dflash"},
        {"num_hidden_layers", 4},
        {"num_attention_heads", 32},
        {"num_key_value_heads", 8},
        {"head_dim", 64},
        {"sliding_window", 2048},
        {"layer_types",
         {"sliding_attention", "sliding_attention", "sliding_attention", "full_attention"}},
        {"dflash_config", {{"target_layer_ids", {1, 6, 11, 16, 21}}, {"mask_token_id", 200000}}}};
}

m::dense::Config gptoss20b_family() {
    m::dense::Config c;
    c.hidden_size = 2880;
    c.vocab_size  = 201088;
    c.layers      = 24;
    c.kv_heads    = 8;
    c.head_dim    = 64;
    for (int i = 0; i < 24; ++i) { c.layer_specs.push_back({.window = i % 2 == 0 ? 128 : 0}); }
    return c;
}

m::speculation::DFlashConfig gptoss_drafter_family() {
    m::speculation::DFlashConfig d;
    d.hidden_size    = 2880;
    d.layers         = 4;
    d.kv_heads       = 8;
    d.head_dim       = 64;
    d.sliding_window = 2048;
    d.windows        = {2048, 2048, 2048, 0};
    d.taps           = {1, 6, 11, 16, 21};
    return d;
}

json gemma_config(int layers, int shared) {
    json types = json::array();
    for (int i = 0; i < layers; ++i) {
        types.push_back((i + 1) % 6 == 0 ? "full_attention" : "sliding_attention");
    }
    json c = {{"architectures", {"Gemma4ForConditionalGeneration"}},
              {"text_config",
               {{"model_type", "gemma4_text"},
                {"hidden_size", 5376},
                {"num_hidden_layers", layers},
                {"num_attention_heads", 32},
                {"num_key_value_heads", 16},
                {"head_dim", 256},
                {"global_head_dim", 512},
                {"num_global_key_value_heads", 4},
                {"sliding_window", 1024},
                {"layer_types", types},
                {"max_position_embeddings", 262144},
                {"vocab_size", 262144},
                {"intermediate_size", 21504}}}};
    if (shared != 0) { c["text_config"]["num_kv_shared_layers"] = shared; }
    return c;
}

m::gemma4::Config gemma_family(int layers, int shared) {
    m::gemma4::Config c;
    c.hidden_size    = 5376;
    c.vocab_size     = 262144;
    c.layers         = layers;
    c.sliding_window = 1024;
    c.full           = {.head_dim = 512, .kv_heads = 4};
    c.sliding        = {.head_dim = 256, .kv_heads = 16};
    for (int i = 0; i < layers; ++i) {
        c.kinds.push_back((i + 1) % 6 == 0 ? m::gemma4::LayerKind::Full
                                           : m::gemma4::LayerKind::Sliding);
    }
    // KV sharing: a sharing layer attends over the last layer of its kind before the first one.
    const int first_shared = layers - shared;
    for (int i = 0; i < layers; ++i) {
        int source = i;
        if (i >= first_shared) {
            for (int j = first_shared - 1; j >= 0; --j) {
                if (c.kinds[static_cast<std::size_t>(j)] == c.kinds[static_cast<std::size_t>(i)]) {
                    source = j;
                    break;
                }
            }
        }
        c.kv_source.push_back(source);
    }
    return c;
}

std::string nemotron_pattern() {
    // 52 blocks: Mamba2 (M), MoE (E) and attention (*), Nemotron 3 Nano's mix.
    return "MEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEM*EMEMEMEMEME";
}

json nemotron_config() {
    return {{"architectures", {"NemotronHForCausalLM"}},
            {"model_type", "nemotron_h"},
            {"hidden_size", 2688},
            {"num_hidden_layers", static_cast<int>(nemotron_pattern().size())},
            {"hybrid_override_pattern", nemotron_pattern()},
            {"num_attention_heads", 32},
            {"num_key_value_heads", 2},
            {"head_dim", 128},
            {"mamba_num_heads", 64},
            {"mamba_head_dim", 64},
            {"ssm_state_size", 128},
            {"n_groups", 8},
            {"conv_kernel", 4},
            {"max_position_embeddings", 262144},
            {"vocab_size", 131072},
            {"n_routed_experts", 128},
            {"num_experts_per_tok", 6},
            {"moe_intermediate_size", 1856}};
}

m::nemotron_h::Config nemotron_family() {
    m::nemotron_h::Config c;
    c.hidden_size    = 2688;
    c.vocab_size     = 131072;
    c.kv_heads       = 2;
    c.head_dim       = 128;
    c.mamba.geometry = {.heads = 64, .head_dim = 64, .groups = 8, .state = 128};
    for (const char ch : nemotron_pattern()) {
        c.blocks.push_back(ch == 'M'   ? m::nemotron_h::Block::Mamba
                           : ch == '*' ? m::nemotron_h::Block::Attention
                                       : m::nemotron_h::Block::Moe);
    }
    c.layers = static_cast<std::int32_t>(c.blocks.size());
    return c;
}

rt::PlanModelFacts facts_of(const json& config) {
    return pr::plan_facts_of_config(config, 10 * kGiB, rt::WeightBasis::FileSize).model;
}

std::vector<Subject> subjects() {
    const std::vector<KvCacheStorage> all = {
        KvCacheStorage::BFloat16, KvCacheStorage::Int8Group64, KvCacheStorage::Fp8E4M3Row256,
        KvCacheStorage::Nvfp4Group16, KvCacheStorage::Fp8KeyNvfp4Value};
    const std::vector<KvCacheStorage> ring = {KvCacheStorage::BFloat16,
                                              KvCacheStorage::Fp8E4M3Row256};
    const std::vector<KvCacheStorage> bf16 = {KvCacheStorage::BFloat16};
    std::vector<Subject> out;
    {
        // The artifact's own DFlash drafter: the dry run reads its component's config.
        Subject s{.name = "qwen27b+dflash"};
        s.family              = detail::qwen3_5_state_shape(qwen27b_family(true, false), 248320);
        s.facts               = facts_of(qwen27b_config());
        s.facts.state.drafter = pr::plan_drafter_state_of_config(qwen_drafter_config(false), 5120);
        s.kv = all, s.rings = bf16, s.recurrent = true;
        s.spec = {SpeculativeBackend::None, SpeculativeBackend::Mtp, SpeculativeBackend::DFlash};
        out.push_back(s);
    }
    {
        Subject s{.name = "qwen27b+dflash2"};
        s.family              = detail::qwen3_5_state_shape(qwen27b_family(true, true), 248320);
        s.facts               = facts_of(qwen27b_config());
        s.facts.state.drafter = pr::plan_drafter_state_of_config(qwen_drafter_config(true), 5120);
        s.kv = {KvCacheStorage::Int8Group64}, s.rings = bf16, s.recurrent = true;
        s.spec = {SpeculativeBackend::DFlash2};
        out.push_back(s);
    }
    {
        Subject s{.name = "qwen27b gguf"};
        s.family = detail::qwen3_5_state_shape(qwen27b_family(false, false), 248320);
        s.facts  = pr::plan_facts_of_gguf_metadata(qwen27b_gguf(), 16 * kGiB).model;
        s.kv = all, s.rings = bf16, s.recurrent = true;
        s.spec = {SpeculativeBackend::None, SpeculativeBackend::Mtp};
        out.push_back(s);
    }
    {
        Subject s{.name = "qwen35b-a3b"};
        s.family = detail::qwen3_5_state_shape(qwen35b_family(), 248320);
        s.facts  = facts_of(qwen35b_config());
        s.kv = all, s.rings = bf16, s.recurrent = true;
        s.spec = {SpeculativeBackend::None, SpeculativeBackend::Mtp};
        out.push_back(s);
    }
    {
        Subject s{.name = "llama8b"};
        s.family = m::dense::state_shape(llama8b_family(), nullptr, 0, 0, 128256);
        s.facts  = facts_of(llama8b_config());
        s.kv = ring, s.rings = bf16;
        s.spec = {SpeculativeBackend::None};
        out.push_back(s);
    }
    {
        const auto drafter = gptoss_drafter_family();
        Subject s{.name = "gpt-oss-20b+dflash"};
        s.family = m::dense::state_shape(gptoss20b_family(), &drafter, 2880 * 5, 2880 * 5, 201088);
        s.facts  = facts_of(gptoss20b_config());
        s.facts.state.drafter = pr::plan_drafter_state_of_config(gptoss_drafter_config(), 2880);
        s.kv = ring, s.rings = ring;
        s.spec = {SpeculativeBackend::None, SpeculativeBackend::DFlash};
        out.push_back(s);
    }
    {
        Subject s{.name = "gemma31b"};
        s.family = m::gemma4::state_shape(gemma_family(60, 0), 262144);
        s.facts  = facts_of(gemma_config(60, 0));
        s.kv = ring, s.rings = ring;
        s.spec = {SpeculativeBackend::None, SpeculativeBackend::Mtp};
        out.push_back(s);
    }
    {
        Subject s{.name = "gemma kv-shared"};
        s.family = m::gemma4::state_shape(gemma_family(42, 18), 262144);
        s.facts  = facts_of(gemma_config(42, 18));
        s.kv = ring, s.rings = ring;
        s.spec = {SpeculativeBackend::None};
        out.push_back(s);
    }
    {
        Subject s{.name = "nemotron30b"};
        s.family = m::nemotron_h::state_shape(nemotron_family(), 131072);
        s.facts  = facts_of(nemotron_config());
        s.kv = ring, s.rings = bf16;
        s.spec = {SpeculativeBackend::None, SpeculativeBackend::Mtp};
        out.push_back(s);
    }
    return out;
}

// ---- The start's bytes: the planner a start builds (make_sequence_planner_impl), its curve at
// the rung's pages and the plan it finalizes there.

struct StartPlan {
    std::uint64_t bytes        = 0; // the finalized plan's reservation (no workspace here)
    std::uint32_t draft_window = 0;
    std::uint32_t copy_window  = 0; // the planner's resolved ngram_draft_tokens
};

StartPlan start_plan(const m::StateShape& shape, const EngineOptions& planned,
                     std::uint32_t pages) {
    const detail::ShapeExecution model(shape);
    auto planner = detail::make_sequence_planner_impl(model, planned, 0, false);
    StartPlan out{.draft_window = planner->inputs.draft_window,
                  .copy_window  = planner->inputs.ngram_draft_tokens};
    const std::uint64_t curve = planner->curve.reservation_bytes(pages);
    const auto plan           = detail::finalize_sequence_plan_impl(std::move(planner), pages);
    check(plan->device_reservation_bytes == curve, "the start's plan is on its curve");
    check(plan->workspace.capacity == 0, "the planning-only model plans no workspace");
    out.bytes = plan->device_reservation_bytes;
    return out;
}

std::uint64_t start_bytes(const m::StateShape& shape, const EngineOptions& planned,
                          std::uint32_t pages) {
    return start_plan(shape, planned, pages).bytes;
}

EngineOptions options_of(std::uint32_t concurrency, SpeculativeBackend spec, KvCacheStorage kv,
                         KvCacheStorage ring, RecurrentStateStorage recurrent) {
    EngineOptions o;
    o.max_concurrency           = concurrency;
    o.kv_cache                  = kv;
    o.kv_cache_automatic        = false;
    o.sliding_kv_cache          = ring;
    o.recurrent_state           = recurrent;
    o.recurrent_state_automatic = false;
    o.use_cuda_graph            = true;
    o.speculative.backend       = spec;
    o.speculative.draft_tokens  = spec == SpeculativeBackend::None  ? 0U
                                  : spec == SpeculativeBackend::Mtp ? 3U
                                                                    : 7U;
    if (spec == SpeculativeBackend::DFlash || spec == SpeculativeBackend::DFlash2) {
        o.speculative.request_draft_tokens = {3U};
    }
    // The copy window stays the Engine's default (automatic), as a server leaves it.
    return o;
}

void test_shapes(const std::vector<Subject>& all) {
    for (const Subject& s : all) {
        check(s.facts.offline_bytes,
              s.name + ": the facts name the state (" + s.facts.offline_reason + ")");
        check(s.facts.state == s.family,
              s.name + ": the dry run's shape is the family's\n  dry run " +
                  shape_text(s.facts.state) + "\n  family  " + shape_text(s.family));
    }
}

void test_bytes(const std::vector<Subject>& all) {
    for (const Subject& s : all) {
        if (!s.facts.offline_bytes) { continue; }
        for (const std::uint32_t concurrency : {1U, 4U}) {
            for (const SpeculativeBackend spec : s.spec) {
                for (const KvCacheStorage kv : s.kv) {
                    for (const KvCacheStorage ring : s.rings) {
                        for (const RecurrentStateStorage recurrent :
                             s.recurrent
                                 ? std::vector<RecurrentStateStorage>{RecurrentStateStorage::FP32,
                                                                      RecurrentStateStorage::FP16}
                                 : std::vector<RecurrentStateStorage>{
                                       RecurrentStateStorage::FP32}) {
                            for (const std::uint32_t context : {4096U, 32768U, 131072U}) {
                                for (const std::uint32_t chunk : {1024U, 2048U}) {
                                    for (const std::uint32_t cached : {0U, 3U}) {
                                        EngineOptions o =
                                            options_of(concurrency, spec, kv, ring, recurrent);
                                        const std::uint32_t logical =
                                            rt::plan_context_pages(context);
                                        const std::uint32_t low  = std::max(logical, concurrency);
                                        const std::uint32_t high = logical * concurrency;
                                        for (const std::uint32_t pages :
                                             {low, (low + high) / 2, high}) {
                                            const rt::PlanRung rung{.max_context   = context,
                                                                    .kv_pages      = pages,
                                                                    .cached_states = cached,
                                                                    .prefill_chunk = chunk};
                                            const rt::PlanRungBytes dry = rt::plan_rung_bytes(
                                                s.facts, rung, o, concurrency, false);
                                            // A start plans the rung with these options
                                            // (planned_at).
                                            EngineOptions planned                    = o;
                                            planned.max_context                      = context;
                                            planned.prefill_chunk                    = chunk;
                                            planned.context_cache.device_state_slots = cached;
                                            const std::uint64_t start =
                                                start_bytes(s.family, planned, pages);
                                            ++compared;
                                            const std::uint64_t ours = dry.total() - dry.workspace;
                                            check(ours == start,
                                                  s.name + " C" + std::to_string(concurrency) +
                                                      " spec " +
                                                      std::to_string(static_cast<int>(spec)) +
                                                      " kv " +
                                                      std::to_string(static_cast<int>(kv)) +
                                                      " ctx " + std::to_string(context) +
                                                      " chunk " + std::to_string(chunk) +
                                                      " cached " + std::to_string(cached) +
                                                      " pages " + std::to_string(pages) +
                                                      ": dry run " + std::to_string(ours) +
                                                      " != start " + std::to_string(start));
                                            check(dry.kv != 0 && dry.fixed != 0 &&
                                                      dry.host_state_image != 0,
                                                  s.name + ": the bytes are laid out");
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

// The copy window a server's default options leave automatic (ngram_draft_tokens
// kCopyWindowAuto, copy rounds on): a start plans MTP on a family whose round runs copy rounds
// with the 15-draft window (its 16-column round logits, replay width, MTP pages and graphs), and
// the dry run now plans the same (the round-2 review's M1: it planned the depth's window alone).
// Under none and DFlash, and with the window off, no copy window.
void test_default_copy_window(const std::vector<Subject>& all) {
    std::size_t windows = 0;
    for (const Subject& s : all) {
        if (!s.facts.offline_bytes) { continue; }
        for (const SpeculativeBackend spec : s.spec) {
            const char* name = spec == SpeculativeBackend::None     ? "none"
                               : spec == SpeculativeBackend::Mtp    ? "mtp"
                               : spec == SpeculativeBackend::DFlash ? "dflash"
                                                                    : "dflash2";
            for (const std::uint32_t concurrency : {1U, 4U}) {
                // ninferno-serve's options for --spec <name> (product/plan, launch_options), the
                // codecs and the depth named as the dry run normalizes them.
                EngineOptions o =
                    pr::launch_options(
                        {.max_context = 32768, .max_concurrency = concurrency, .spec = name})
                        .options;
                check(o.speculative.ngram_draft_tokens ==
                              ninferno::SpeculativeOptions::kCopyWindowAuto &&
                          o.speculative.ngram_min_match != 0,
                      "a server's default options leave the copy window automatic");
                const EngineOptions shaped = options_of(
                    concurrency, spec, s.kv.front(), s.rings.front(),
                    s.recurrent ? RecurrentStateStorage::FP16 : RecurrentStateStorage::FP32);
                o.kv_cache                         = shaped.kv_cache;
                o.kv_cache_automatic               = false;
                o.sliding_kv_cache                 = shaped.sliding_kv_cache;
                o.recurrent_state                  = shaped.recurrent_state;
                o.recurrent_state_automatic        = false;
                o.speculative.draft_tokens         = shaped.speculative.draft_tokens;
                o.speculative.request_draft_tokens = shaped.speculative.request_draft_tokens;
                EngineOptions off                  = o;
                off.speculative.ngram_draft_tokens = 0;
                const bool copies =
                    spec == SpeculativeBackend::Mtp && detail::family_copy_rounds(s.family.family);
                for (const std::uint32_t cached : {0U, 2U}) {
                    const std::uint32_t logical = rt::plan_context_pages(32768);
                    const std::uint32_t low     = std::max(logical, concurrency);
                    const rt::PlanRung rung{.max_context   = 32768,
                                            .kv_pages      = low,
                                            .cached_states = cached,
                                            .prefill_chunk = 1024};
                    EngineOptions planned                      = o;
                    planned.max_context                        = 32768;
                    planned.prefill_chunk                      = 1024;
                    planned.context_cache.device_state_slots   = cached;
                    EngineOptions planned_off                  = planned;
                    planned_off.speculative.ngram_draft_tokens = 0;
                    const StartPlan start                      = start_plan(s.family, planned, low);
                    const StartPlan start_off = start_plan(s.family, planned_off, low);
                    const rt::PlanRungBytes dry =
                        rt::plan_rung_bytes(s.facts, rung, o, concurrency, false);
                    const rt::PlanRungBytes dry_off =
                        rt::plan_rung_bytes(s.facts, rung, off, concurrency, false);
                    const std::string where = s.name + " --spec " + name + " C" +
                                              std::to_string(concurrency) + " cached " +
                                              std::to_string(cached);
                    ++compared;
                    check(dry.total() - dry.workspace == start.bytes,
                          where + ": default options: dry run " +
                              std::to_string(dry.total() - dry.workspace) + " != start " +
                              std::to_string(start.bytes));
                    check(dry_off.total() - dry_off.workspace == start_off.bytes,
                          where + ": copy window off: dry run " +
                              std::to_string(dry_off.total() - dry_off.workspace) + " != start " +
                              std::to_string(start_off.bytes));
                    check(start.copy_window == (copies ? 15U : 0U) && start_off.copy_window == 0,
                          where + ": the start resolves the copy window by the family's rule");
                    if (copies) {
                        ++windows;
                        check(start.draft_window == 15 && start_off.draft_window == 3,
                              where + ": the start's widest window is the copy window");
                        check(start.bytes > start_off.bytes && dry.workspace > dry_off.workspace,
                              where + ": the copy window's bytes and round logits are planned");
                    } else {
                        check(start.bytes == start_off.bytes && dry.workspace == dry_off.workspace,
                              where + ": no copy window outside MTP on a copy-round family");
                    }
                    // An explicit copy window deeper than the draft (validate_program_options
                    // requires one): the start and the dry run plan the same bytes, and the axes'
                    // round stays the draft window + 1 on both sides (plan_round_tokens), so a
                    // start's consistency check with the dry run's axes holds.
                    EngineOptions explicit_window                   = o;
                    explicit_window.speculative.ngram_draft_tokens  = 8;
                    EngineOptions planned_explicit                  = planned;
                    planned_explicit.speculative.ngram_draft_tokens = 8;
                    const StartPlan start_explicit = start_plan(s.family, planned_explicit, low);
                    const rt::PlanRungBytes dry_explicit =
                        rt::plan_rung_bytes(s.facts, rung, explicit_window, concurrency, false);
                    check(dry_explicit.total() - dry_explicit.workspace == start_explicit.bytes,
                          where + ": explicit copy window 8: dry run " +
                              std::to_string(dry_explicit.total() - dry_explicit.workspace) +
                              " != start " + std::to_string(start_explicit.bytes));
                    check(!copies || start_explicit.copy_window == 8U,
                          where + ": an explicit copy window is the start's");
                    const rt::PlanAxes axes = rt::plan_axes(planned_explicit, 262144, false, 0);
                    check(axes.round_tokens ==
                                  rt::plan_round_tokens(planned_explicit.speculative) &&
                              axes.round_tokens ==
                                  std::max(planned_explicit.speculative.draft_tokens,
                                           planned_explicit.speculative.request_draft_tokens.empty()
                                               ? 0U
                                               : planned_explicit.speculative.request_draft_tokens
                                                     .front()) +
                                      1U,
                          where + ": the axes' round is the draft window + 1 with a copy window");
                }
            }
        }
    }
    check(windows != 0, "copy windows compared");
    // The review's row: Qwen3.8-27B UD-IQ3_XXS on the RTX 5070 Ti, --spec mtp at depth 3 with a
    // server's default options, the copy window planned (now) and not (the dry run before M1).
    rt::PlanDeviceFacts device;
    device.name               = "RTX 5070 Ti";
    device.total_bytes        = rt::plan_board_total_bytes(16);
    device.compute_capability = 120;
    rt::PlanModelFacts facts  = pr::plan_facts_of_gguf_metadata(qwen27b_gguf(), 0).model;
    facts.weight_bytes        = static_cast<std::uint64_t>(10.20 * kGiB);
    facts.mtp_bytes           = static_cast<std::uint64_t>(0.33 * kGiB);
    for (const auto& [concurrency, free_gib] :
         std::vector<std::tuple<std::uint32_t, double>>{{1, 5.21}, {4, 5.17}}) {
        device.free_bytes = facts.weight_bytes + static_cast<std::uint64_t>(free_gib * kGiB);
        EngineOptions now =
            pr::launch_options({.max_context = 0, .max_concurrency = concurrency, .spec = "mtp"})
                .options;
        now.speculative.draft_tokens          = 3;
        EngineOptions before                  = now;
        before.speculative.ngram_draft_tokens = 0;
        const rt::DryRunPlan plan_now         = rt::dry_run_plan(facts, device, now);
        const rt::DryRunPlan plan_before      = rt::dry_run_plan(facts, device, before);
        check(plan_now.planned && plan_before.planned, "the 27B MTP rows plan");
        if (!plan_now.planned || !plan_before.planned || plan_now.ladder.empty()) { continue; }
        // The ladder's first rung under both, and the rung each chooses.
        const rt::DryRunRung& top_now    = plan_now.ladder.front();
        const rt::DryRunRung& top_before = plan_before.ladder.front();
        std::cout << "5070 Ti 27B IQ3 --spec mtp C" << concurrency << " (" << free_gib
                  << " GiB free), rung 1 (" << top_before.rung.max_context << "/"
                  << top_before.rung.kv_pages << " -> " << top_now.rung.max_context << "/"
                  << top_now.rung.kv_pages << " tokens/pages): total " << top_before.bytes.total()
                  << " -> " << top_now.bytes.total() << " ("
                  << static_cast<long long>(top_now.bytes.total()) -
                         static_cast<long long>(top_before.bytes.total())
                  << ": kv " << top_before.bytes.kv << " -> " << top_now.bytes.kv << ", fixed "
                  << top_before.bytes.fixed << " -> " << top_now.bytes.fixed << ", graphs "
                  << top_before.bytes.graphs << " -> " << top_now.bytes.graphs << ", workspace "
                  << top_before.bytes.workspace << " -> " << top_now.bytes.workspace << "); chosen "
                  << (plan_before.fits ? std::to_string(plan_before.chosen()->rung.max_context)
                                       : std::string("none"))
                  << " -> "
                  << (plan_now.fits ? std::to_string(plan_now.chosen()->rung.max_context)
                                    : std::string("none"))
                  << " tokens\n";
        check(top_now.rung.max_context != top_before.rung.max_context ||
                  top_now.bytes.total() > top_before.bytes.total(),
              "the copy window costs the 27B MTP rung bytes");
    }
}

// An artifact's proposal head (proposal/*) is on the Device only while it drafts: under DFlash
// (which drafts with it) or when asked for; plain, MTP and DFlash2 plans leave it off (Codex on
// #170: the default head counted for every plan, a false 484 MB on the 9B).
void test_proposal_bytes_only_when_drafting() {
    rt::PlanDeviceFacts device;
    device.name               = "RTX 5070 Ti";
    device.total_bytes        = rt::plan_board_total_bytes(16);
    device.compute_capability = 120;
    device.free_bytes         = device.total_bytes;
    rt::PlanModelFacts facts  = pr::plan_facts_of_gguf_metadata(qwen27b_gguf(), 0).model;
    facts.weight_bytes        = static_cast<std::uint64_t>(10.20 * kGiB);
    facts.proposal_bytes      = static_cast<std::uint64_t>(0.45 * kGiB);
    const auto weights = [&](SpeculativeBackend backend, ProposalHead head) {
        EngineOptions options;
        options.speculative.backend       = backend;
        options.speculative.draft_tokens  = backend == SpeculativeBackend::None ? 0 : 3;
        options.speculative.proposal_head = head;
        return rt::dry_run_plan(facts, device, options).weight_bytes;
    };
    const std::uint64_t without = facts.weight_bytes - facts.proposal_bytes;
    check(weights(SpeculativeBackend::None, ProposalHead::Full) == without, "plain: no head");
    check(weights(SpeculativeBackend::Mtp, ProposalHead::Full) == without, "MTP: no head");
    check(weights(SpeculativeBackend::DFlash2, ProposalHead::Full) == without, "DFlash2: no head");
    check(weights(SpeculativeBackend::DFlash, ProposalHead::Full) == facts.weight_bytes,
          "DFlash: the head");
    check(weights(SpeculativeBackend::Mtp, ProposalHead::Optimized) == facts.weight_bytes,
          "an asked-for head");
}

// The default proposal head a Qwen conversion with a DFlash companion adds, per quantization: each
// row's own weights (Codex on #170: every row of the --plan table took the header's size).
void test_proposal_bytes_per_quant() {
    pr::PlanSourceFacts q4 = pr::plan_facts_of_gguf_metadata(qwen27b_gguf(), 0);
    q4.model.weight_bytes  = 5117970432;
    q4.model.head_bytes    = 834355200;
    q4.model.vocab         = 248320;
    q4.model.hidden        = 4096;
    pr::DryRunInputs inputs;
    inputs.proposal_drafter_bytes       = 1372703744.0;
    inputs.proposal_header_weight_bytes = q4.model.weight_bytes;
    pr::PlanSourceFacts q8 = q4;
    q8.model.weight_bytes  = 2 * q4.model.weight_bytes; // a quantization twice the size
    const std::uint64_t at_q4 = pr::proposal_bytes_at(q4, inputs);
    const std::uint64_t at_q8 = pr::proposal_bytes_at(q8, inputs);
    check(at_q4 == 147456ULL * 3360 + 147456ULL * 4, "the header's row: 147,456 rows and ids");
    check(at_q8 > at_q4, "a larger quantization's head is larger");
    pr::DryRunInputs none = inputs;
    none.proposal_drafter_bytes = 0;
    check(pr::proposal_bytes_at(q4, none) == 0, "no companion: no head");
}

// What the planner adds over the approximate estimate's KV pages: the MTP layer's pool, the
// replay records, a DFlash drafter's pool and ring (the review's M2: all of them were missing).
void test_what_the_estimate_missed() {
    rt::PlanModelFacts facts = facts_of(qwen27b_config());
    facts.state.drafter      = pr::plan_drafter_state_of_config(qwen_drafter_config(false), 5120);
    const rt::PlanRung rung{
        .max_context = 32768, .kv_pages = 512, .cached_states = 1, .prefill_chunk = 1024};
    const auto bytes = [&](SpeculativeBackend spec) {
        return rt::plan_rung_bytes(facts, rung,
                                   options_of(1, spec, KvCacheStorage::Int8Group64,
                                              KvCacheStorage::BFloat16,
                                              RecurrentStateStorage::FP16),
                                   1, false);
    };
    const rt::PlanRungBytes plain = bytes(SpeculativeBackend::None);
    const rt::PlanRungBytes mtp   = bytes(SpeculativeBackend::Mtp);
    const rt::PlanRungBytes flash = bytes(SpeculativeBackend::DFlash);
    check(mtp.kv > plain.kv, "the MTP layer's KV pool is counted");
    check(mtp.fixed > plain.fixed, "MTP's GDN replay records are counted");
    check(flash.kv > plain.kv && flash.rings > plain.rings,
          "the DFlash drafter's paged KV and its ring per StateImage slot are counted");
    // The plain rung: the estimate and the planner, side by side (the start's is the planner's).
    rt::PlanModelFacts approximate = facts;
    approximate.offline_bytes      = false;
    const EngineOptions o = options_of(1, SpeculativeBackend::None, KvCacheStorage::Int8Group64,
                                       KvCacheStorage::BFloat16, RecurrentStateStorage::FP16);
    const rt::PlanRungBytes estimate = rt::plan_rung_estimate(approximate, rung, o, 1);
    std::cout << "qwen27b 32K x1 plain: estimate " << estimate.total() << " (kv " << estimate.kv
              << " states " << estimate.states << " graphs " << estimate.graphs << " fixed "
              << estimate.fixed << "), planner " << plain.total() << " (kv " << plain.kv
              << " states " << plain.states << " graphs " << plain.graphs << " fixed "
              << plain.fixed << ")\n";
}

// The dry run's ladder over the planner's bytes: a fit when there is room, approximate and why
// without the state, refused with the family's words for a codec it does not lay out.
void test_dry_run_basis() {
    rt::PlanDeviceFacts device;
    device.name               = "RTX 5070 Ti";
    device.total_bytes        = rt::plan_board_total_bytes(16);
    device.compute_capability = 120;
    // The review's rows (docs/reference/serving.md, the 21.10 stand logs): Qwen3.8-27B UD-IQ3_XXS,
    // 10.20 GiB of weights, 5.21 GiB free after them at C=1 (5.17 at C=4); the start's ladder
    // served 142,912 tokens at C=1 and 131,072 at C=4.
    rt::PlanModelFacts facts = pr::plan_facts_of_gguf_metadata(qwen27b_gguf(), 0).model;
    facts.weight_bytes       = static_cast<std::uint64_t>(10.20 * kGiB);
    for (const auto& [concurrency, free_gib, start] :
         std::vector<std::tuple<std::uint32_t, double, std::uint32_t>>{{1, 5.21, 142912},
                                                                       {4, 5.17, 131072}}) {
        device.free_bytes = facts.weight_bytes + static_cast<std::uint64_t>(free_gib * kGiB);
        const EngineOptions o =
            pr::launch_options({.max_context = 0, .max_concurrency = concurrency, .spec = "none"})
                .options;
        const rt::DryRunPlan plan = rt::dry_run_plan(facts, device, o);
        check(plan.planned && plan.fits && plan.estimate_basis == "family_planner",
              "a GGUF's facts plan with the family planner");
        rt::PlanModelFacts approximate = facts;
        approximate.offline_bytes      = false;
        approximate.offline_reason     = "test";
        const rt::DryRunPlan old       = rt::dry_run_plan(approximate, device, o);
        check(old.estimate_basis == "approximate" && old.estimate_reason == "test",
              "facts without the state are approximate, with the reason");
        if (!plan.fits || !old.fits) { continue; }
        const rt::DryRunRung& r = *plan.chosen();
        std::cout << "5070 Ti 27B IQ3 C" << concurrency << " (" << free_gib
                  << " GiB free): " << r.rung.max_context << " tokens, rung " << plan.rung + 1
                  << " (kv " << r.bytes.kv << " states " << r.bytes.states << " fixed "
                  << r.bytes.fixed << " graphs " << r.bytes.graphs << " workspace "
                  << r.bytes.workspace << "); approximate " << old.chosen()->rung.max_context
                  << ", rung " << old.rung + 1 << "; the start " << start << '\n';
    }
    // The MTP layers are uploaded under MTP only (the 21.10 5070 Ti 32K x 4 row: a plain start
    // read 9.84 GiB of weights against the GGUF's 10.17, the NextN block left in the file).
    facts.mtp_bytes   = static_cast<std::uint64_t>(0.33 * kGiB);
    device.free_bytes = 15 * kGiB;
    EngineOptions plain_o =
        pr::launch_options({.max_context = 32768, .max_concurrency = 4, .spec = "none"}).options;
    EngineOptions mtp_o             = plain_o;
    mtp_o.speculative.backend       = SpeculativeBackend::Mtp;
    mtp_o.speculative.draft_tokens  = 3;
    const rt::DryRunPlan plain_plan = rt::dry_run_plan(facts, device, plain_o);
    const rt::DryRunPlan mtp_plan   = rt::dry_run_plan(facts, device, mtp_o);
    check(plain_plan.weight_bytes + facts.mtp_bytes == mtp_plan.weight_bytes &&
              mtp_plan.weight_bytes == facts.weight_bytes,
          "the MTP layers' weights only under MTP");
    // The C=4 rows of the 21.10 stand runs (the start's free memory, MTP at depth 3 as an
    // example of what --spec auto serves): what the planner now counts beside the KV.
    for (const auto& [name, total_gib, weights_gib, free_gib] :
         std::vector<std::tuple<std::string, double, double, double>>{
             {"5070 Ti 27B IQ3 auto c4", 16, 10.17, 5.27},
             {"A100 27B Q4_K_XL auto c4", 40, 16.34, 22.95}}) {
        device.total_bytes = rt::plan_board_total_bytes(total_gib);
        facts.weight_bytes = static_cast<std::uint64_t>(weights_gib * kGiB);
        device.free_bytes  = facts.weight_bytes + static_cast<std::uint64_t>(free_gib * kGiB);
        EngineOptions o =
            pr::launch_options({.max_context = 0, .max_concurrency = 4, .spec = "none"}).options;
        o.speculative.backend          = SpeculativeBackend::Mtp;
        o.speculative.draft_tokens     = 3;
        const rt::DryRunPlan now       = rt::dry_run_plan(facts, device, o);
        rt::PlanModelFacts approximate = facts;
        approximate.offline_bytes      = false;
        const rt::DryRunPlan old       = rt::dry_run_plan(approximate, device, o);
        if (now.fits && old.fits) {
            std::cout << name << " MTP3: planner rung " << now.rung + 1 << " kv "
                      << now.chosen()->rung.kv_pages * 64ULL << " tokens; approximate rung "
                      << old.rung + 1 << " kv " << old.chosen()->rung.kv_pages * 64ULL << '\n';
        }
    }
    device.total_bytes = rt::plan_board_total_bytes(16);
    device.free_bytes.reset();
    rt::PlanModelFacts gemma = facts_of(gemma_config(60, 0));
    gemma.weight_bytes       = 10 * kGiB;
    EngineOptions o;
    o.kv_cache                   = KvCacheStorage::Nvfp4Group16;
    o.kv_cache_automatic         = false;
    device.total_bytes           = rt::plan_board_total_bytes(32);
    const rt::DryRunPlan refused = rt::dry_run_plan(gemma, device, o);
    check(!refused.planned && refused.reason.find("BF16 or FP8") != std::string::npos,
          "a codec the family does not lay out is refused with its words");
    // DFlash without the drafter's shape: approximate.
    rt::PlanModelFacts qwen      = facts_of(qwen27b_config());
    qwen.weight_bytes            = 15 * kGiB;
    o                            = EngineOptions{};
    o.speculative.backend        = SpeculativeBackend::DFlash;
    o.speculative.draft_tokens   = 7;
    const rt::DryRunPlan unknown = rt::dry_run_plan(qwen, device, o);
    check(unknown.estimate_basis == "approximate", "a DFlash plan without its drafter's shape");
}

// The advice is the start's: the asked concurrency only when an axis was pinned, then each lower
// one that fits longer, until one fits the ceiling.
void test_fits_instead() {
    // Contexts each concurrency fits: 1 -> 64K, 2 -> 32K, 3 -> 32K, 4 -> 8K.
    const rt::PlanLongestContextFn longest = [](std::uint32_t c,
                                                bool) -> std::optional<std::uint32_t> {
        switch (c) {
        case 1:
            return 65536U;
        case 2:
        case 3:
            return 32768U;
        case 4:
            return 8192U;
        }
        return std::nullopt;
    };
    const rt::PlanFitsInstead automatic = rt::plan_fits_instead(4, 131072, false, false, longest);
    check(automatic.fits.size() == 2 && automatic.fits[0].max_concurrency == 3 &&
              automatic.fits[1].max_concurrency == 1 && !automatic.narrowed,
          "automatic: the lower concurrencies that fit longer");
    const rt::PlanFitsInstead pinned = rt::plan_fits_instead(4, 131072, true, false, longest);
    check(pinned.fits.size() == 3 && pinned.fits[0].max_concurrency == 4 &&
              pinned.fits[1].max_concurrency == 3 && pinned.fits[2].max_concurrency == 1,
          "pinned: the asked concurrency first");
    const rt::PlanFitsInstead ceiling = rt::plan_fits_instead(4, 32768, false, false, longest);
    check(ceiling.fits.size() == 1 && ceiling.fits[0].max_concurrency == 3,
          "stops at the first that fits the ceiling");
    const rt::PlanLongestContextFn chunked = [](std::uint32_t c,
                                                bool kept) -> std::optional<std::uint32_t> {
        if (kept) { return std::nullopt; }
        return c == 1 ? std::optional<std::uint32_t>(4096U) : std::nullopt;
    };
    const rt::PlanFitsInstead narrowed = rt::plan_fits_instead(2, 131072, false, true, chunked);
    check(narrowed.narrowed && narrowed.fits.size() == 1, "an explicit chunk narrowed");
}

} // namespace

int main() {
    const std::vector<Subject> all = subjects();
    test_shapes(all);
    test_bytes(all);
    test_default_copy_window(all);
    test_what_the_estimate_missed();
    test_dry_run_basis();
    test_fits_instead();
    test_proposal_bytes_only_when_drafting();
    test_proposal_bytes_per_quant();
    std::cout << "rung bytes compared: " << compared << '\n';
    check(compared > 1000, "many rungs compared");
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "plan bytes OK\n";
    return 0;
}
