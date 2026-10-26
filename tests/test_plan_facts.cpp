// The plan facts (src/product/plan/plan_facts.h): the shape a GGUF header, a config.json and an
// artifact's text config name, read into the PlanModelFacts the plan without the weights plans
// from, for each family (Qwen3.5 dense and MoE, Gemma 4, gpt-oss, NemotronH, a Llama-lineage
// dense decoder); the weight basis (a whole tensor table, a split GGUF's file sizes); and the
// named boards of gpu_facts.json. Synthetic headers and configs shaped like the published models.
// CPU.

#include "product/plan/plan_facts.h"

#include "convert/ggml_quants.h"
#include "convert/gguf.h"
#include "runtime/engine/plan_dry_run.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

namespace pr = ninferno::product;
namespace rt = ninferno::runtime;
using nlohmann::json;
using ninferno::convert::GgufFile;
using ninferno::convert::GgufTensor;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

template <class A, class B>
int same(const A& actual, const B& expected, const std::string& what) {
    if (actual == static_cast<A>(expected)) { return 0; }
    std::cerr << "FAIL: " << what << ": " << actual << " (expected " << expected << ")\n";
    return 1;
}

bool has(const std::vector<std::string>& v, const std::string& s) {
    return std::find(v.begin(), v.end(), s) != v.end();
}

json qwen35_metadata() {
    return {{"general.architecture", "qwen35"},
            {"qwen35.block_count", 65}, // 64 trunk layers and the MTP block
            {"qwen35.nextn_predict_layers", 1},
            {"qwen35.context_length", 262144},
            {"qwen35.embedding_length", 5120},
            {"qwen35.feed_forward_length", 17408},
            {"qwen35.attention.head_count", 24},
            {"qwen35.attention.head_count_kv", 4},
            {"qwen35.attention.key_length", 256},
            {"qwen35.attention.value_length", 256},
            {"qwen35.full_attention_interval", 4},
            {"qwen35.ssm.conv_kernel", 4},
            {"qwen35.ssm.state_size", 128},
            {"qwen35.ssm.group_count", 16},
            {"qwen35.ssm.time_step_rank", 48},
            {"qwen35.ssm.inner_size", 6144},
            {"general.base_model.0.repo_url", "https://huggingface.co/Qwen/Qwen3.6-27B"}};
}

GgufTensor tensor(std::string name, std::vector<std::uint64_t> ne, std::uint32_t type,
                  std::uint64_t bytes) {
    GgufTensor t;
    t.name  = std::move(name);
    t.ne    = std::move(ne);
    t.type  = type;
    t.bytes = bytes;
    return t;
}

// The Qwen3.5 27B facts every spelling of it must give.
int qwen35_dense_facts(const pr::PlanSourceFacts& f, const std::string& where) {
    const auto& m = f.model;
    int failures  = 0;
    failures += same(m.model_type, std::string("qwen3_5_text"), where + " model_type");
    failures += same(m.native_context, 262144, where + " context");
    failures += same(m.full_layers, 16, where + " full layers");
    failures += same(m.full_kv_heads_total, 64, where + " full kv heads");
    failures += same(m.full_key_dim, 256, where + " key dim");
    failures += same(m.full_value_dim, 256, where + " value dim");
    failures += same(m.sliding_layers, 0, where + " sliding layers");
    failures += same(m.recurrent_layers, 48, where + " recurrent layers");
    // 48 layers x 48 value heads x 128 x 128.
    failures += same(m.recurrent_state_elements, 48ULL * 6144 * 128, where + " state elements");
    // 48 layers x 3 rows x (6144 + 2 x 16 x 128) channels x 2 bytes.
    failures += same(m.recurrent_conv_bytes, 48ULL * 3 * 10240 * 2, where + " conv bytes");
    failures += same(m.hidden, 5120, where + " hidden");
    failures += same(m.ffn_active, 17408, where + " ffn_active");
    failures += check(f.shape.builtin_mtp, where + ": MTP layers not seen");
    failures += same(f.shape.num_hidden_layers, 64, where + " trunk layers");
    failures +=
        check(m.params > 25e9 && m.params < 29e9, where + ": params " + std::to_string(m.params));
    // int8 KV at D256: 264 key + 264 value bytes a head, 16 layers x 4 heads.
    failures += same(rt::plan_kv_bytes_per_token(m, ninferno::KvCacheStorage::Int8Group64), 33792,
                     where + " int8 KV bytes per token");
    return failures;
}

int gguf_qwen35() {
    int failures = 0;
    GgufFile file;
    file.metadata = qwen35_metadata();
    file.tensors.push_back(
        tensor("token_embd.weight", {5120, 248320}, ninferno::convert::ggml::Q6_K, 1000));
    file.tensors.push_back(
        tensor("output.weight", {5120, 248320}, ninferno::convert::ggml::Q6_K, 1000));
    file.tensors.push_back(
        tensor("blk.0.attn_qkv.weight", {5120, 10240}, ninferno::convert::ggml::Q4_K, 500));
    file.tensors.push_back(
        tensor("blk.0.attn_norm.weight", {5120}, ninferno::convert::ggml::F32, 20));
    const auto f = pr::plan_facts_of_gguf(file, 999999);
    failures += qwen35_dense_facts(f, "qwen35 gguf");
    failures += same(f.model.weight_bytes, 2520, "qwen35 gguf tensor bytes");
    failures += check(f.model.weight_basis == rt::WeightBasis::GgufTensors, "qwen35 gguf basis");
    failures += same(f.model.vocab, 248320, "qwen35 gguf vocab from token_embd");
    failures += check(has(f.shape.weight_formats, "ggml_q4_k") &&
                          has(f.shape.weight_formats, "ggml_q6_k") &&
                          has(f.shape.weight_formats, "fp32") && f.shape.weight_formats.size() == 3,
                      "qwen35 gguf weight formats");
    failures += same(f.shape.base_models.empty() ? std::string() : f.shape.base_models.front(),
                     std::string("Qwen/Qwen3.6-27B"), "qwen35 base model");
    failures += same(f.model.architecture, std::string("qwen35"), "qwen35 architecture");
    // Bits per weight over the tensor table's elements (exact), not the geometry's estimate.
    const double elements = 2.0 * 5120 * 248320 + 5120.0 * 10240 + 5120;
    failures += check(f.model.weight_elements == elements,
                      "qwen35 gguf elements " + std::to_string(f.model.weight_elements));
    failures += check(std::abs(pr::bits_per_weight(f.model, 2520) - 2520 * 8 / elements) < 1e-12,
                      "bits per weight over the elements");

    // A split GGUF's head lists its own part's tensors: the listed file sizes stand.
    GgufFile split                = file;
    split.metadata["split.count"] = 3;
    const auto s                  = pr::plan_facts_of_gguf(split, 17000000000ULL);
    failures += check(s.model.weight_basis == rt::WeightBasis::FileSize, "split gguf basis");
    failures += same(s.model.weight_bytes, 17000000000ULL, "split gguf bytes");
    failures += same(s.model.full_layers, 16, "split gguf shape kept");
    failures += check(s.model.weight_elements == 0 &&
                          pr::bits_per_weight(s.model, 8) == (s.model.params > 0 ? 64 / s.model.params : 0),
                      "a split part's elements unknown: the parameter count");

    // The metadata alone (ninferno-ui): FileSize.
    const auto meta = pr::plan_facts_of_gguf_metadata(file.metadata, 16000000000ULL);
    failures += check(meta.model.weight_basis == rt::WeightBasis::FileSize, "metadata basis");
    failures += same(meta.model.weight_bytes, 16000000000ULL, "metadata bytes");
    failures += same(meta.model.full_kv_heads_total, 64, "metadata kv heads");

    // An explicit recurrent-layer pattern wins over the interval.
    json flagged = file.metadata;
    json flags   = json::array();
    for (int i = 0; i < 64; ++i) { flags.push_back(i % 2 == 0); } // every other layer recurrent
    flagged["qwen35.attention.recurrent_layers"] = flags;
    const auto fl                                = pr::plan_facts_of_gguf_metadata(flagged, 1);
    failures += same(fl.model.full_layers, 32, "recurrent_layers flags: full layers");
    failures += same(fl.model.recurrent_layers, 32, "recurrent_layers flags: recurrent layers");

    // with_weights: another quantization, the shape kept.
    const auto w = pr::with_weights(f, 12345, rt::WeightBasis::FileSize);
    failures += same(w.model.weight_bytes, 12345, "with_weights bytes");
    failures += check(w.model.weight_basis == rt::WeightBasis::FileSize, "with_weights basis");
    failures += same(w.model.full_layers, 16, "with_weights shape");
    return failures;
}

int config_qwen35() {
    json types = json::array();
    for (int i = 0; i < 64; ++i) {
        types.push_back((i + 1) % 4 == 0 ? "full_attention" : "linear_attention");
    }
    const json config = {{"architectures", {"Qwen3_5ForConditionalGeneration"}},
                         {"model_type", "qwen3_5"},
                         {"text_config",
                          {{"model_type", "qwen3_5_text"},
                           {"hidden_size", 5120},
                           {"num_hidden_layers", 64},
                           {"layer_types", types},
                           {"num_attention_heads", 24},
                           {"num_key_value_heads", 4},
                           {"head_dim", 256},
                           {"intermediate_size", 17408},
                           {"max_position_embeddings", 262144},
                           {"vocab_size", 248320},
                           {"linear_num_value_heads", 48},
                           {"linear_value_head_dim", 128},
                           {"linear_key_head_dim", 128},
                           {"linear_num_key_heads", 16},
                           {"linear_conv_kernel_dim", 4},
                           {"mtp_num_hidden_layers", 1}}}};
    int failures      = 0;
    const auto f      = pr::plan_facts_of_config(config, 54000000000ULL, rt::WeightBasis::FileSize);
    failures += qwen35_dense_facts(f, "qwen35 config");
    failures += same(f.model.weight_bytes, 54000000000ULL, "qwen35 config bytes");
    failures += same(f.model.vocab, 248320, "qwen35 config vocab");
    // The artifact's text config spells the same keys with its own model_type.
    json text             = config["text_config"];
    text["architectures"] = json::array({"Qwen3_5ForCausalLM"});
    failures += qwen35_dense_facts(pr::plan_facts_of_config(text, 1, rt::WeightBasis::Artifact),
                                   "qwen35 artifact config");
    return failures;
}

int qwen35_moe() {
    int failures      = 0;
    const json md     = {{"general.architecture", "qwen35moe"},
                         {"qwen35moe.block_count", 41},
                         {"qwen35moe.nextn_predict_layers", 1},
                         {"qwen35moe.context_length", 262144},
                         {"qwen35moe.embedding_length", 2048},
                         {"qwen35moe.attention.head_count", 16},
                         {"qwen35moe.attention.head_count_kv", 2},
                         {"qwen35moe.attention.key_length", 256},
                         {"qwen35moe.attention.value_length", 256},
                         {"qwen35moe.full_attention_interval", 4},
                         {"qwen35moe.ssm.conv_kernel", 4},
                         {"qwen35moe.ssm.state_size", 128},
                         {"qwen35moe.ssm.group_count", 16},
                         {"qwen35moe.ssm.time_step_rank", 32},
                         {"qwen35moe.ssm.inner_size", 4096},
                         {"qwen35moe.expert_count", 256},
                         {"qwen35moe.expert_used_count", 8},
                         {"qwen35moe.expert_feed_forward_length", 512},
                         {"qwen35moe.expert_shared_feed_forward_length", 512}};
    const json config = {{"architectures", {"Qwen3_5MoeForConditionalGeneration"}},
                         {"text_config",
                          {{"model_type", "qwen3_5_moe_text"},
                           {"hidden_size", 2048},
                           {"num_hidden_layers", 40},
                           {"full_attention_interval", 4},
                           {"num_attention_heads", 16},
                           {"num_key_value_heads", 2},
                           {"head_dim", 256},
                           {"max_position_embeddings", 262144},
                           {"vocab_size", 248320},
                           {"linear_num_value_heads", 32},
                           {"linear_value_head_dim", 128},
                           {"linear_key_head_dim", 128},
                           {"linear_num_key_heads", 16},
                           {"linear_conv_kernel_dim", 4},
                           {"num_experts", 256},
                           {"num_experts_per_tok", 8},
                           {"moe_intermediate_size", 512},
                           {"shared_expert_intermediate_size", 512}}}};
    for (const auto& [f, where] :
         {std::pair{pr::plan_facts_of_gguf_metadata(md, 1), std::string("qwen35moe gguf")},
          std::pair{pr::plan_facts_of_config(config, 1, rt::WeightBasis::FileSize),
                    std::string("qwen35moe config")}}) {
        failures += same(f.model.model_type, std::string("qwen3_5_moe_text"), where + " type");
        failures += same(f.model.full_layers, 10, where + " full layers");
        failures += same(f.model.full_kv_heads_total, 20, where + " kv heads");
        failures += same(f.model.recurrent_layers, 30, where + " recurrent layers");
        failures += same(f.model.recurrent_state_elements, 30ULL * 4096 * 128, where + " state");
        failures += same(f.model.recurrent_conv_bytes, 30ULL * 3 * (4096 + 2 * 16 * 128) * 2,
                         where + " conv bytes");
        failures += same(f.model.ffn_active, 8 * 512 + 512, where + " ffn_active");
        failures += same(f.shape.experts, 256, where + " experts");
        failures += same(f.shape.experts_per_token, 8, where + " experts per token");
        failures += check(f.model.params > 30e9 && f.model.params < 40e9,
                          where + ": params " + std::to_string(f.model.params));
        failures += check(f.shape.active_params > 1.5e9 && f.shape.active_params < 4.5e9,
                          where + ": active " + std::to_string(f.shape.active_params));
        failures +=
            check(f.shape.routed_expert_params > 0.8 * f.model.params, where + ": routed share");
    }
    return failures;
}

int gemma4_facts(const pr::PlanSourceFacts& f, const std::string& where) {
    int failures  = 0;
    const auto& m = f.model;
    failures += same(m.model_type, std::string("gemma4_text"), where + " type");
    failures += same(m.sliding_layers, 50, where + " sliding layers");
    failures += same(m.sliding_kv_heads_total, 800, where + " sliding kv heads");
    failures += same(m.sliding_key_dim, 256, where + " sliding key");
    failures += same(m.sliding_value_dim, 256, where + " sliding value");
    failures += same(m.window, 1024, where + " window");
    failures += same(m.full_layers, 10, where + " full layers");
    failures += same(m.full_kv_heads_total, 40, where + " full kv heads");
    failures += same(m.full_key_dim, 512, where + " full key");
    failures += same(m.full_value_dim, 512, where + " full value");
    failures += same(m.recurrent_layers, 0, where + " recurrent");
    failures += same(m.ffn_active, 21504, where + " ffn_active");
    failures += same(m.native_context, 262144, where + " context");
    return failures;
}

int gemma4() {
    int failures = 0;
    json pattern = json::array(), kv = json::array(), types = json::array();
    for (int i = 0; i < 60; ++i) {
        const bool sliding = i % 6 != 5;
        pattern.push_back(sliding);
        kv.push_back(sliding ? 16 : 4);
        types.push_back(sliding ? "sliding_attention" : "full_attention");
    }
    const json md = {{"general.architecture", "gemma4"},
                     {"gemma4.block_count", 60},
                     {"gemma4.context_length", 262144},
                     {"gemma4.embedding_length", 5376},
                     {"gemma4.feed_forward_length", 21504},
                     {"gemma4.attention.head_count", 32},
                     {"gemma4.attention.head_count_kv", kv},
                     {"gemma4.attention.key_length", 512},
                     {"gemma4.attention.value_length", 512},
                     {"gemma4.attention.key_length_swa", 256},
                     {"gemma4.attention.value_length_swa", 256},
                     {"gemma4.attention.sliding_window", 1024},
                     {"gemma4.attention.sliding_window_pattern", pattern}};
    failures += gemma4_facts(pr::plan_facts_of_gguf_metadata(md, 1), "gemma4 gguf");
    const json config = {{"architectures", {"Gemma4ForConditionalGeneration"}},
                         {"text_config",
                          {{"model_type", "gemma4_text"},
                           {"hidden_size", 5376},
                           {"num_hidden_layers", 60},
                           {"layer_types", types},
                           {"sliding_window", 1024},
                           {"num_attention_heads", 32},
                           {"head_dim", 256},
                           {"num_key_value_heads", 16},
                           {"global_head_dim", 512},
                           {"num_global_key_value_heads", 4},
                           {"attention_k_eq_v", true},
                           {"intermediate_size", 21504},
                           {"max_position_embeddings", 262144},
                           {"vocab_size", 262144}}}};
    failures += gemma4_facts(pr::plan_facts_of_config(config, 1, rt::WeightBasis::FileSize),
                             "gemma4 config");
    // Without attention_k_eq_v the full layers keep num_key_value_heads (gemma4.cpp).
    json no_share                               = config;
    no_share["text_config"]["attention_k_eq_v"] = false;
    failures += same(
        pr::plan_facts_of_config(no_share, 1, rt::WeightBasis::FileSize).model.full_kv_heads_total,
        160, "gemma4 config without k_eq_v: full kv heads");
    // The artifact's spelling: attention_geometry, and KV sharing as kv_source_layers (the last
    // two layers read layers 52 and 53's keys and values: no KV of their own).
    json artifact = {{"architectures", {"Gemma4ForCausalLM"}},
                     {"model_type", "gemma4_text"},
                     {"hidden_size", 5376},
                     {"num_hidden_layers", 60},
                     {"layer_types", types},
                     {"sliding_window", 1024},
                     {"num_attention_heads", 32},
                     {"attention_geometry",
                      {{"sliding_attention", {{"head_dim", 256}, {"num_key_value_heads", 16}}},
                       {"full_attention", {{"head_dim", 512}, {"num_key_value_heads", 4}}}}},
                     {"intermediate_size", 21504},
                     {"max_position_embeddings", 262144},
                     {"vocab_size", 262144}};
    failures += gemma4_facts(pr::plan_facts_of_config(artifact, 1, rt::WeightBasis::Artifact),
                             "gemma4 artifact");
    json sources = json::array();
    for (int i = 0; i < 60; ++i) { sources.push_back(i < 58 ? i : i - 6); }
    artifact["kv_source_layers"] = sources; // 58 sliding -> 52, 59 full -> 53
    const auto shared            = pr::plan_facts_of_config(artifact, 1, rt::WeightBasis::Artifact);
    failures += same(shared.model.sliding_layers, 49, "gemma4 shared kv: sliding layers");
    failures += same(shared.model.full_layers, 9, "gemma4 shared kv: full layers");
    return failures;
}

int gpt_oss() {
    int failures  = 0;
    const json md = {{"general.architecture", "gpt-oss"},
                     {"gpt-oss.block_count", 24},
                     {"gpt-oss.context_length", 131072},
                     {"gpt-oss.embedding_length", 2880},
                     {"gpt-oss.attention.head_count", 64},
                     {"gpt-oss.attention.head_count_kv", 8},
                     {"gpt-oss.attention.key_length", 64},
                     {"gpt-oss.attention.value_length", 64},
                     {"gpt-oss.attention.sliding_window", 128},
                     {"gpt-oss.expert_count", 32},
                     {"gpt-oss.expert_used_count", 4},
                     {"gpt-oss.expert_feed_forward_length", 2880},
                     {"gpt-oss.vocab_size", 201088}};
    json types    = json::array();
    for (int i = 0; i < 24; ++i) {
        types.push_back(i % 2 == 0 ? "sliding_attention" : "full_attention");
    }
    const json config = {{"architectures", {"GptOssForCausalLM"}},
                         {"model_type", "gpt_oss"},
                         {"hidden_size", 2880},
                         {"num_hidden_layers", 24},
                         {"layer_types", types},
                         {"num_attention_heads", 64},
                         {"num_key_value_heads", 8},
                         {"head_dim", 64},
                         {"sliding_window", 128},
                         {"intermediate_size", 2880},
                         {"num_local_experts", 32},
                         {"num_experts_per_tok", 4},
                         {"max_position_embeddings", 131072},
                         {"vocab_size", 201088}};
    for (const auto& [f, where] :
         {std::pair{pr::plan_facts_of_gguf_metadata(md, 1), std::string("gpt-oss gguf")},
          std::pair{pr::plan_facts_of_config(config, 1, rt::WeightBasis::FileSize),
                    std::string("gpt-oss config")}}) {
        failures += same(f.model.model_type, std::string("moe_decoder"), where + " type");
        failures += same(f.model.sliding_layers, 12, where + " sliding layers");
        failures += same(f.model.full_layers, 12, where + " full layers");
        failures += same(f.model.full_kv_heads_total, 96, where + " full kv heads");
        failures += same(f.model.sliding_kv_heads_total, 96, where + " sliding kv heads");
        failures += same(f.model.full_key_dim, 64, where + " key dim");
        failures += same(f.model.window, 128, where + " window");
        failures += same(f.model.ffn_active, 4 * 2880, where + " ffn_active");
        failures += same(f.shape.experts, 32, where + " experts");
        failures += same(f.shape.experts_per_token, 4, where + " top-k");
        failures += check(f.model.params > 18e9 && f.model.params < 23e9,
                          where + ": params " + std::to_string(f.model.params));
    }
    return failures;
}

int nemotron_facts(const pr::PlanSourceFacts& f, const std::string& where) {
    int failures = 0;
    failures += same(f.model.model_type, std::string("nemotron_h"), where + " type");
    failures += same(f.model.full_layers, 2, where + " full layers");
    failures += same(f.model.full_kv_heads_total, 4, where + " kv heads");
    failures += same(f.model.full_key_dim, 128, where + " key dim");
    failures += same(f.model.recurrent_layers, 2, where + " mamba layers");
    failures += same(f.model.recurrent_state_elements, 2ULL * 4096 * 128, where + " state");
    failures += same(f.model.recurrent_conv_bytes, 2ULL * 3 * (4096 + 2 * 8 * 128) * 2,
                     where + " conv bytes");
    failures += same(f.model.ffn_active, 6 * 1856 + 3712, where + " ffn_active");
    failures += same(f.shape.experts, 128, where + " experts");
    return failures;
}

int nemotron_h() {
    int failures = 0;
    // mamba, attention, moe, mamba, attention, moe
    const json md = {{"general.architecture", "nemotron_h_moe"},
                     {"nemotron_h_moe.block_count", 6},
                     {"nemotron_h_moe.context_length", 262144},
                     {"nemotron_h_moe.embedding_length", 2688},
                     {"nemotron_h_moe.attention.head_count", 32},
                     {"nemotron_h_moe.attention.head_count_kv", {0, 2, 0, 0, 2, 0}},
                     {"nemotron_h_moe.feed_forward_length", {0, 0, 1856, 0, 0, 1856}},
                     {"nemotron_h_moe.attention.key_length", 128},
                     {"nemotron_h_moe.attention.value_length", 128},
                     {"nemotron_h_moe.ssm.conv_kernel", 4},
                     {"nemotron_h_moe.ssm.state_size", 128},
                     {"nemotron_h_moe.ssm.group_count", 8},
                     {"nemotron_h_moe.ssm.time_step_rank", 64},
                     {"nemotron_h_moe.ssm.inner_size", 4096},
                     {"nemotron_h_moe.expert_count", 128},
                     {"nemotron_h_moe.expert_used_count", 6},
                     {"nemotron_h_moe.expert_feed_forward_length", 1856},
                     {"nemotron_h_moe.expert_shared_feed_forward_length", 3712}};
    failures += nemotron_facts(pr::plan_facts_of_gguf_metadata(md, 1), "nemotron gguf");
    const json config = {{"architectures", {"NemotronHForCausalLM"}},
                         {"model_type", "nemotron_h"},
                         {"hidden_size", 2688},
                         {"hybrid_override_pattern", "M*EM*E"},
                         {"num_attention_heads", 32},
                         {"num_key_value_heads", 2},
                         {"head_dim", 128},
                         {"mamba_num_heads", 64},
                         {"mamba_head_dim", 64},
                         {"n_groups", 8},
                         {"ssm_state_size", 128},
                         {"conv_kernel", 4},
                         {"n_routed_experts", 128},
                         {"num_experts_per_tok", 6},
                         {"moe_intermediate_size", 1856},
                         {"moe_shared_expert_intermediate_size", 3712},
                         {"n_shared_experts", 1},
                         {"max_position_embeddings", 262144},
                         {"vocab_size", 131072}};
    const auto hf     = pr::plan_facts_of_config(config, 1, rt::WeightBasis::FileSize);
    failures += nemotron_facts(hf, "nemotron config");
    failures += same(hf.shape.num_hidden_layers, 6, "nemotron config layers from the pattern");
    // The converter's artifact config: layer_types, the mamba and moe objects.
    const json artifact = {
        {"architectures", {"NemotronHForCausalLM"}},
        {"model_type", "nemotron_h"},
        {"hidden_size", 2688},
        {"num_hidden_layers", 6},
        {"layer_types", {"mamba", "attention", "moe", "mamba", "attention", "moe"}},
        {"num_attention_heads", 32},
        {"num_key_value_heads", 2},
        {"head_dim", 128},
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
          {"shared_intermediate_size", 3712}}},
        {"max_position_embeddings", 262144},
        {"vocab_size", 131072}};
    failures += nemotron_facts(pr::plan_facts_of_config(artifact, 1, rt::WeightBasis::Artifact),
                               "nemotron artifact");
    return failures;
}

int llama_facts(const pr::PlanSourceFacts& f, const std::string& where) {
    int failures = 0;
    failures += same(f.model.model_type, std::string("dense_decoder"), where + " type");
    failures += same(f.model.full_layers, 32, where + " full layers");
    failures += same(f.model.full_kv_heads_total, 256, where + " kv heads");
    failures += same(f.model.full_key_dim, 128, where + " key dim");
    failures += same(f.model.sliding_layers, 0, where + " sliding");
    failures += same(f.model.recurrent_layers, 0, where + " recurrent");
    failures += same(f.model.ffn_active, 14336, where + " ffn_active");
    failures += same(f.model.native_context, 131072, where + " context");
    failures += check(f.model.params > 7.9e9 && f.model.params < 8.1e9,
                      where + ": params " + std::to_string(f.model.params));
    failures += check(!f.shape.builtin_mtp, where + ": no MTP");
    return failures;
}

int llama() {
    int failures  = 0;
    const json md = {{"general.architecture", "llama"},    {"llama.block_count", 32},
                     {"llama.context_length", 131072},     {"llama.embedding_length", 4096},
                     {"llama.feed_forward_length", 14336}, {"llama.attention.head_count", 32},
                     {"llama.attention.head_count_kv", 8}, {"llama.vocab_size", 128256}};
    failures += llama_facts(pr::plan_facts_of_gguf_metadata(md, 1), "llama gguf");
    const json config = {{"architectures", {"LlamaForCausalLM"}},
                         {"model_type", "llama"},
                         {"hidden_size", 4096},
                         {"num_hidden_layers", 32},
                         {"num_attention_heads", 32},
                         {"num_key_value_heads", 8},
                         {"intermediate_size", 14336},
                         {"max_position_embeddings", 131072},
                         {"vocab_size", 128256}};
    failures +=
        llama_facts(pr::plan_facts_of_config(config, 1, rt::WeightBasis::FileSize), "llama config");
    json artifact             = config;
    artifact["architectures"] = json::array({"DenseDecoderForCausalLM"});
    artifact["model_type"]    = "dense_decoder";
    artifact["layer_types"]   = json::array();
    for (int i = 0; i < 32; ++i) { artifact["layer_types"].push_back("full_attention"); }
    failures += llama_facts(pr::plan_facts_of_config(artifact, 1, rt::WeightBasis::Artifact),
                            "llama artifact");
    return failures;
}

int model_types() {
    int failures    = 0;
    const auto gguf = [](const std::string& arch, json extra = json::object()) {
        json md                    = std::move(extra);
        md["general.architecture"] = arch;
        return pr::plan_model_type_of_gguf(md);
    };
    for (const char* dense : {"llama", "qwen2", "qwen3", "granite", "olmo2", "smollm3", "mistral3",
                              "phi3", "muse-glimmer"}) {
        failures += same(gguf(dense), std::string("dense_decoder"), std::string("gguf ") + dense);
    }
    failures += same(gguf("qwen3moe"), std::string("moe_decoder"), "gguf qwen3moe");
    failures += same(gguf("gpt-oss"), std::string("moe_decoder"), "gguf gpt-oss");
    failures += same(gguf("deepseek2"), std::string(), "gguf deepseek2");
    failures += same(gguf("gemma4-assistant"), std::string(), "gguf gemma4-assistant");
    const auto cfg = [](const std::string& arch) {
        return pr::plan_model_type_of_config({{"architectures", {arch}}});
    };
    failures += same(cfg("Qwen3MoeForCausalLM"), std::string("dense_decoder"),
                     "Qwen3Moe without expert keys");
    failures += same(pr::plan_model_type_of_config(
                         {{"architectures", {"Qwen3MoeForCausalLM"}}, {"num_experts", 128}}),
                     std::string("moe_decoder"), "Qwen3Moe with experts");
    failures += same(cfg("Gemma4AssistantForCausalLM"), std::string(), "Gemma4 assistant");
    failures += same(cfg("NemotronHMtpForCausalLM"), std::string(), "NemotronH MTP drafter");
    failures += same(cfg("DeepseekV3ForCausalLM"), std::string(), "unknown architecture");
    failures += same(pr::plan_model_type_of_config({{"model_type", "moe_decoder"}}),
                     std::string("moe_decoder"), "artifact family");
    return failures;
}

int boards() {
    int failures = 0;
    std::string error;
    const auto b = pr::plan_device_of_board("RTX 4090", &error);
    failures += check(b.has_value(), "RTX 4090 not found: " + error);
    if (b) {
        failures += same(b->name, std::string("GeForce RTX 4090"), "RTX 4090 name");
        failures += same(b->total_bytes, rt::plan_board_total_bytes(24), "RTX 4090 bytes");
        failures += same(b->compute_capability, 89, "RTX 4090 cc");
        failures += same(b->pci_device_id, 0x2684U, "RTX 4090 pci id");
        failures += same(b->sms, 128, "RTX 4090 sms");
        failures += check(b->paravirtual == pr::host_paravirtual(), "RTX 4090 driver model");
        failures += check(b->drives_display == b->paravirtual, "RTX 4090 display default");
        failures += check(!b->free_bytes.has_value(), "a named board measures nothing");
    }
    const auto h = pr::plan_device_of_board("4090,headless,linux", &error);
    failures += check(h && h->name == "GeForce RTX 4090" && !h->paravirtual && !h->drives_display,
                      "4090,headless,linux");
    const auto w = pr::plan_device_of_board("nvidia geforce rtx 4090, wddm", &error);
    failures += check(w && w->paravirtual && w->drives_display, "4090 wddm: a display assumed");
    const auto laptop = pr::plan_device_of_board("4090 laptop", &error);
    failures += check(laptop && laptop->name == "GeForce RTX 4090 Laptop GPU", "4090 laptop");
    const auto pci = pr::plan_device_of_board("0x2B85", &error);
    failures += check(pci && pci->name == "GeForce RTX 5090" && pci->compute_capability == 120,
                      "PCI id 0x2B85");
    error.clear();
    failures +=
        check(!pr::plan_device_of_board("RTX 9999", &error) && !error.empty(), "an unknown board");
    error.clear();
    failures += check(!pr::plan_device_of_board("Voodoo 9999", &error) &&
                          error.find("close names") == std::string::npos,
                      "no close names for a query no row shares a word with: " + error);
    error.clear();
    failures += check(!pr::plan_device_of_board("A100 Ultra", &error) &&
                          error.find("A100 SXM4 40GB") != std::string::npos,
                      "close names list rows with a word of the query: " + error);
    error.clear();
    failures += check(!pr::plan_device_of_board("4090,turbo", &error) && !error.empty(),
                      "an unknown board fact");
    const auto s =
        pr::plan_device_of_stats("NVIDIA GeForce RTX 5090", 32ULL << 30, 12, 0, false, 0x2B85, 170);
    failures += check(s.compute_capability == 120 && !s.drives_display &&
                          s.pci_device_id == 0x2B85 && s.sms == 170 && s.source == "GPU (NVML)",
                      "plan_device_of_stats");
    const auto u = pr::plan_device_of_stats("x", 1, 8, 9, std::nullopt);
    failures += check(u.drives_display == u.paravirtual, "stats: unknown display");
    return failures;
}

int artifact_errors() {
    const auto path = std::filesystem::temp_directory_path() / "ninferno_plan_facts_no_artifact";
    try {
        (void)pr::plan_facts_of_artifact(path);
    } catch (const std::runtime_error&) { return 0; }
    return check(false, "a missing artifact did not throw");
}

} // namespace

int main() {
    const int failures = gguf_qwen35() + config_qwen35() + qwen35_moe() + gemma4() + gpt_oss() +
                         nemotron_h() + llama() + model_types() + boards() + artifact_errors();
    if (failures == 0) { std::cout << "plan facts OK\n"; }
    return failures == 0 ? 0 : 1;
}
