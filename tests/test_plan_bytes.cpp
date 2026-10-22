// The plan without the weights takes a start's bytes: for every family, the state shape the dry
// run reads from a config.json or a GGUF header (product/plan/plan_facts.cpp) is the one the
// family's own planner reads from its bound config (each family's state_shape), and a dry-run
// rung's bytes (runtime/engine/plan_dry_run.h, plan_rung_bytes) are the start's planner's
// (models/qwen3_5/program/planning/sequence_bytes.h, over the start's affine page curve) to the
// byte, but for the workspace: its Op scratch the bound weights' routes size, an estimate on both
// sides of the comparison. Synthetic shapes of the lab's models over concurrencies, speculation
// (none, MTP, DFlash), codecs, contexts, chunks and cached states. CPU, no GPU, no files.

#include "models/dense/program_family.h"
#include "models/gemma4/program_family.h"
#include "models/nemotron_h/program_family.h"
#include "models/qwen3_5/program/planning/sequence_bytes.h"
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
#include <optional>
#include <string>
#include <tuple>
#include <vector>

namespace {

namespace rt     = ninferno::runtime;
namespace pr     = ninferno::product;
namespace m      = ninferno::models;
namespace detail = ninferno::models::qwen3_5::detail;
using ninferno::EngineOptions;
using ninferno::KvCacheStorage;
using ninferno::RecurrentStateStorage;
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

// ---- The start's bytes: its planner's curve (make_sequence_planner_impl) at the rung's pages.

std::uint64_t start_bytes(const m::StateShape& shape, const EngineOptions& planned,
                          std::uint32_t pages) {
    const detail::SequencePlanningInputs inputs =
        detail::sequence_planning_inputs(planned, 0, false);
    const detail::SequencePageRange range =
        detail::sequence_page_range(inputs.capacity, inputs.max_concurrency);
    const auto at = [&](std::uint32_t p) {
        auto plan        = detail::sequence_plan_shell(inputs, p);
        plan->persistent = detail::plan_persistent_layout(shape, *plan);
        detail::plan_graph_allowance(*plan, false);
        return std::uint64_t{plan->persistent.bytes} + plan->graph_allowance_bytes;
    };
    const std::uint64_t minimum = at(range.minimum);
    const std::uint64_t stride =
        range.minimum < range.maximum ? at(range.minimum + 1) - minimum : 0;
    return minimum + static_cast<std::uint64_t>(pages - range.minimum) * stride;
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
    test_what_the_estimate_missed();
    test_dry_run_basis();
    test_fits_instead();
    std::cout << "rung bytes compared: " << compared << '\n';
    check(compared > 1000, "many rungs compared");
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "plan bytes OK\n";
    return 0;
}
