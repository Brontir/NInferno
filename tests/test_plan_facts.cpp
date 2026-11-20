// The plan facts (src/product/plan/plan_facts.h): the shape a GGUF header, a config.json and an
// artifact's text config name, read into the PlanModelFacts the plan without the weights plans
// from, for each family (Qwen3.5 dense and MoE, Gemma 4, gpt-oss, NemotronH, a Llama-lineage
// dense decoder); the weight basis (a whole tensor table, a split GGUF's file sizes); and the
// named boards of gpu_facts.json. Synthetic headers and configs shaped like the published models.
// CPU.

#include "product/plan/plan_cached.h"
#include "product/plan/plan_facts.h"
#include "product/plan/plan_choice.h"
#include "product/plan/plan_quants.h"
#include "product/plan/plan_speculation.h"
#include "spec/round_cost.h"

#include "convert/ggml_quants.h"
#include "convert/gguf.h"
#include "runtime/engine/plan_dry_run.h"
#include "artifact/formats.h"
#include "artifact/schema.h"
#include "core/ggml_frag.h"
#include "core/weight_view.h"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <initializer_list>
#include <iostream>
#include <optional>
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
    failures += check(std::abs(f.model.bits_per_weight(2520) - 2520 * 8 / elements) < 1e-12,
                      "bits per weight over the elements");

    // A split GGUF's head lists its own part's tensors: the listed file sizes stand.
    GgufFile split                = file;
    split.metadata["split.count"] = 3;
    const auto s                  = pr::plan_facts_of_gguf(split, 17000000000ULL);
    failures += check(s.model.weight_basis == rt::WeightBasis::FileSize, "split gguf basis");
    failures += same(s.model.weight_bytes, 17000000000ULL, "split gguf bytes");
    failures += same(s.model.full_layers, 16, "split gguf shape kept");
    failures += check(s.model.weight_elements == 0 &&
                          s.model.bits_per_weight(8) == (s.model.params > 0 ? 64 / s.model.params : 0),
                      "a split part's elements unknown: the parameter count");
    // Its part 0's token table kept whether or not it is the head (complete_split_gguf_head
    // settles the tie from the later parts; Codex on #199).
    GgufFile split0 = split;
    split0.tensors.erase(split0.tensors.begin() + 1); // output.weight in a later part
    const auto s0   = pr::plan_facts_of_gguf(split0, 17000000000ULL);
    failures += same(s0.model.token_table_bytes, 1000, "split part 0's token table");
    failures += same(s0.model.head_bytes, 0, "split part 0: the head not known");
    failures += same(s0.model.embedding_bytes, 0, "split part 0: the embedding not known");
    // Its parameter count, left out while the tie is unknown, from the geometry once settled
    // (complete_split_gguf_head; Opus on #202): the untied count of the part naming its head,
    // the token table once less when tied.
    failures += check(s0.model.params == 0 && s.model.params > 0, "split part 0: params unknown");
    const double table = 5120.0 * 248320;
    failures += check(std::abs(pr::params_of_geometry(s0, false) - s.model.params) <
                              1e-9 * s.model.params &&
                          std::abs(pr::params_of_geometry(s0, true) - (s.model.params - table)) <
                              1e-9 * s.model.params,
                      "split params from the geometry " +
                          std::to_string(pr::params_of_geometry(s0, false)) + " vs " +
                          std::to_string(s.model.params));

    // Routed expert banks as stored (*_exps), the trunk's and the MTP layers' apart (Codex on
    // #199: a mixed-quant model's experts are not their parameter share of the bytes).
    GgufFile moe = file;
    moe.tensors.push_back(
        tensor("blk.0.ffn_up_exps.weight", {5120, 512, 64}, ninferno::convert::ggml::Q4_K, 300));
    moe.tensors.push_back(
        tensor("blk.0.ffn_up_shexp.weight", {5120, 512}, ninferno::convert::ggml::Q8_0, 40));
    moe.tensors.push_back(
        tensor("blk.64.ffn_down_exps.weight", {512, 5120, 64}, ninferno::convert::ggml::Q6_K, 70));
    const auto e = pr::plan_facts_of_gguf(moe, 999999);
    failures += same(e.model.routed_expert_bytes, 300, "routed expert banks as stored");
    failures += same(e.model.mtp_routed_expert_bytes, 70, "the MTP layer's routed bank apart");
    // Their prefill GEMM (ops/moe w4a8_bank_refusal, bank by bank): Q4_K banks (fragment order) on
    // the grouped W4A8; IQ3_XXS and IQ3_S banks in fragment order the W4A8 does not read, so the
    // 16-bit register MMA; MXFP4 banks their own route.
    failures += check(e.model.expert_prefill == rt::ExpertPrefill::Int8, "Q4_K banks: int8");
    failures +=
        check(!e.model.expert_prefill_assumed, "a whole table's banks: the file's own route");
    // A split part's banks are the first part's only; a later part may hold another format, so
    // the route read off them is an assumed one (#813), as gguf_tensor_table already says.
    GgufFile moe_split                = moe;
    moe_split.metadata["split.count"] = 3;
    const auto es                     = pr::plan_facts_of_gguf(moe_split, 17000000000ULL);
    failures +=
        check(es.model.expert_prefill == rt::ExpertPrefill::Int8 && es.model.expert_prefill_assumed,
              "a split part's banks: the route kept, assumed");
    GgufFile iq3 = file;
    iq3.tensors.push_back(
        tensor("blk.0.ffn_gate_exps.weight", {5120, 512, 64}, ninferno::convert::ggml::IQ3_XXS, 200));
    iq3.tensors.push_back(
        tensor("blk.0.ffn_down_exps.weight", {512, 5120, 64}, ninferno::convert::ggml::IQ3_S, 100));
    failures += check(pr::plan_facts_of_gguf(iq3, 999999).model.expert_prefill == rt::ExpertPrefill::A16,
                      "IQ3_XXS banks: the register MMA");
    GgufFile mx = file;
    mx.tensors.push_back(
        tensor("blk.0.ffn_gate_exps.weight", {5120, 512, 64}, ninferno::convert::ggml::MXFP4, 200));
    mx.tensors.push_back(
        tensor("blk.0.ffn_down_exps.weight", {512, 5120, 64}, ninferno::convert::ggml::MXFP4, 100));
    failures += check(pr::plan_facts_of_gguf(mx, 999999).model.expert_prefill == rt::ExpertPrefill::Mxfp4,
                      "MXFP4 banks");
    // The op's block rule (core/moe_w4a8_blocks.h): a gate/up bank zero-padded to K 1024 over a
    // hidden size of 1000 (a multiple of 8, not of 32) quantizes x of 1000 values, not whole
    // blocks, so the op refuses its W4A8 and the bank runs the 16-bit register MMA; the down bank
    // (K 512) keeps the int8 route. The gate/up's bytes are the most: the layer is A16. At hidden
    // 1024 (whole blocks) both banks are int8.
    for (const int hidden : {1000, 1024}) {
        GgufFile padded                            = file;
        padded.metadata["qwen35.embedding_length"] = hidden;
        padded.tensors.push_back(tensor("blk.0.ffn_gate_exps.weight", {1024, 512, 64},
                                        ninferno::convert::ggml::Q4_K, 300));
        padded.tensors.push_back(tensor("blk.0.ffn_down_exps.weight",
                                        {512, static_cast<std::uint64_t>(hidden), 64},
                                        ninferno::convert::ggml::Q4_K, 100));
        const auto route = pr::plan_facts_of_gguf(padded, 999999).model.expert_prefill;
        failures +=
            check(route == (hidden % 32 == 0 ? rt::ExpertPrefill::Int8 : rt::ExpertPrefill::A16),
                  "a gate/up bank over a padded x of " + std::to_string(hidden) +
                      " values: the op's W4A8 block rule");
        // The same layer of MXFP4 banks: the op refuses the gate/up bank there too (KGroup) and
        // runs it A16, so the gate/up's bytes, the most, make the layer A16, not MXFP4; in whole
        // blocks both banks take the MXFP4 route.
        GgufFile padded_mx                            = file;
        padded_mx.metadata["qwen35.embedding_length"] = hidden;
        padded_mx.tensors.push_back(tensor("blk.0.ffn_gate_exps.weight", {1024, 512, 64},
                                           ninferno::convert::ggml::MXFP4, 300));
        padded_mx.tensors.push_back(tensor("blk.0.ffn_down_exps.weight",
                                           {512, static_cast<std::uint64_t>(hidden), 64},
                                           ninferno::convert::ggml::MXFP4, 100));
        const auto mx_route = pr::plan_facts_of_gguf(padded_mx, 999999).model.expert_prefill;
        failures += check(
            mx_route == (hidden % 32 == 0 ? rt::ExpertPrefill::Mxfp4 : rt::ExpertPrefill::A16),
            "MXFP4 banks over a padded x of " + std::to_string(hidden) +
                " values: the op's block rule");
    }
    failures += check(pr::plan_facts_of_gguf(file, 999999).model.expert_prefill == rt::ExpertPrefill::Unknown,
                      "a dense model: no route");

    // The routed layers by kind, as the MoE op's route query is asked for them (#616 step 7): two
    // layers of Q4_K gate and up over a Q6_K down are one kind; a layer whose gate and up are two
    // formats is a kind with no formats (not read whole); the MTP layer's banks are not the
    // trunk's.
    GgufFile kinds = file;
    for (const int block : {0, 1, 2}) {
        const std::string b = "blk." + std::to_string(block) + ".";
        kinds.tensors.push_back(tensor(b + "ffn_gate_exps.weight", {5120, 512, 64},
                                       ninferno::convert::ggml::Q4_K, 100));
        kinds.tensors.push_back(tensor(
            b + "ffn_up_exps.weight", {5120, 512, 64},
            block == 2 ? ninferno::convert::ggml::IQ3_XXS : ninferno::convert::ggml::Q4_K, 100));
        kinds.tensors.push_back(tensor(b + "ffn_down_exps.weight", {512, 5120, 64},
                                       ninferno::convert::ggml::Q6_K, 120));
    }
    kinds.tensors.push_back(
        tensor("blk.64.ffn_down_exps.weight", {512, 5120, 64}, ninferno::convert::ggml::Q6_K, 70));
    const auto layers = pr::plan_facts_of_gguf(kinds, 999999).model.expert_layers;
    const auto read = std::find_if(layers.begin(), layers.end(), [](const rt::PlanExpertLayers& l) {
        return !l.gate_up_format.empty();
    });
    const auto unread =
        std::find_if(layers.begin(), layers.end(),
                     [](const rt::PlanExpertLayers& l) { return l.gate_up_format.empty(); });
    failures += check(layers.size() == 2 && read != layers.end() && unread != layers.end(),
                      "routed layers: two kinds");
    if (read != layers.end() && unread != layers.end()) {
        failures += check(read->gate_up_format == "ggml_q4_k" && read->down_format == "ggml_q6_k" &&
                              !read->gate_up_layout.empty() && !read->down_layout.empty() &&
                              read->gate_up_k == 5120 && read->down_k == 512 && read->gated &&
                              read->policy == rt::PlanExpertPolicy::AllowA8 && read->layers == 2 &&
                              read->bytes == 640,
                          "routed layers: Q4_K gate/up over Q6_K down, two layers, int8 kept");
        failures +=
            check(unread->layers == 1 && unread->bytes == 320 && unread->down_format.empty() &&
                      unread->policy == rt::PlanExpertPolicy::Unknown,
                  "routed layers: gate and up of two formats are not read whole");
    }
    // The GGUF banks' policy (gguf_expert_policy): MXFP4 kept as stored (AllowA8, qwen3_5's and
    // moe_recipe's); NVFP4 without an activation divisor at BF16 inputs; NVFP4 with one
    // (<stem>.input_scale) on every bank as its recipe decides: unknown (one bank without:
    // A16Only). `divisors`: the banks given an activation divisor (<stem>.input_scale).
    const auto policy_of = [&](std::uint32_t type, std::initializer_list<const char*> divisors) {
        GgufFile banks = file;
        banks.tensors.push_back(tensor("blk.0.ffn_gate_exps.weight", {5120, 512, 64}, type, 100));
        banks.tensors.push_back(tensor("blk.0.ffn_up_exps.weight", {5120, 512, 64}, type, 100));
        banks.tensors.push_back(tensor("blk.0.ffn_down_exps.weight", {512, 5120, 64}, type, 100));
        for (const char* stem : divisors) {
            banks.tensors.push_back(tensor(std::string("blk.0.") + stem + ".input_scale", {64},
                                           ninferno::convert::ggml::F32, 256));
        }
        const auto l = pr::plan_facts_of_gguf(banks, 999999).model.expert_layers;
        // No kind at all is a failure of its own, not an Unknown policy.
        return l.size() == 1 ? std::optional<rt::PlanExpertPolicy>(l[0].policy) : std::nullopt;
    };
    failures +=
        check(policy_of(ninferno::convert::ggml::MXFP4, {}) == rt::PlanExpertPolicy::AllowA8,
              "gguf MXFP4 banks: AllowA8");
    failures +=
        check(policy_of(ninferno::convert::ggml::NVFP4, {}) == rt::PlanExpertPolicy::A16Only,
              "gguf NVFP4 banks without a divisor: A16Only");
    failures += check(policy_of(ninferno::convert::ggml::NVFP4,
                                {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"}) ==
                          rt::PlanExpertPolicy::Unknown,
                      "gguf NVFP4 banks with a divisor: unknown");
    // Only the gate has a divisor: the up bank (A16Only, no divisor) makes the packed gate/up bank
    // A16 (ops::common_policy, the strictest of its objects), so the layer is A16Only (#796).
    failures += check(policy_of(ninferno::convert::ggml::NVFP4, {"ffn_gate_exps"}) ==
                          rt::PlanExpertPolicy::A16Only,
                      "gguf NVFP4, a divisor on the gate only: A16Only (the up bank's)");
    // #796 item 3: the GGUF path's recorded policy reaches the prefill route: NVFP4 gate/up
    // without a divisor (A16Only) beside an int8-coded Q4_K down (kept AllowA8) makes the layer
    // A16. The recorded policy here equals NVFP4's kept one (A16Only); no GGUF case has the two
    // differ in a way that moves the route, so this checks the layer policy, not that difference.
    {
        GgufFile banks = file;
        for (const char* name : {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"}) {
            banks.tensors.push_back(
                tensor(name, {5120, 512, 64}, ninferno::convert::ggml::NVFP4, 100));
        }
        // The Q4_K down holds most of the bytes, so the route is its: Int8 if the gate/up's
        // A16Only did not reach the layer.
        banks.tensors.push_back(tensor("blk.0.ffn_down_exps.weight", {512, 5120, 64},
                                       ninferno::convert::ggml::Q4_K, 500));
        const auto route = pr::plan_facts_of_gguf(banks, 999999).model.expert_prefill;
        failures += check(route == rt::ExpertPrefill::A16,
                          "gguf NVFP4 gate/up without a divisor beside a Q4_K down: A16 (" +
                              std::to_string(static_cast<int>(route)) + ")");
    }
    // #1090: a GGUF NVFP4 MoE without divisors (A16Only) runs the grouped W4A8 under
    // --activations int8; with divisors its policy is the loader's probe and the recipe's
    // (unknown here): the build's under fast activations, the grouped W4A8 under int8.
    for (const bool divisor : {false, true}) {
        GgufFile banks = file;
        for (const char* bank : {"ffn_gate_exps", "ffn_up_exps", "ffn_down_exps"}) {
            const bool down = std::string(bank) == "ffn_down_exps";
            banks.tensors.push_back(tensor(std::string("blk.0.") + bank + ".weight",
                                           down ? std::vector<std::uint64_t>{512, 5120, 64}
                                                : std::vector<std::uint64_t>{5120, 512, 64},
                                           ninferno::convert::ggml::NVFP4, 100));
            if (divisor) {
                banks.tensors.push_back(tensor(std::string("blk.0.") + bank + ".input_scale", {64},
                                               ninferno::convert::ggml::F32, 256));
            }
        }
        const auto facts       = pr::plan_facts_of_gguf(banks, 999999).model;
        const std::string what = divisor ? "with divisors" : "without divisors";
        failures += check(facts.expert_prefill ==
                              (divisor ? rt::ExpertPrefill::Unknown : rt::ExpertPrefill::A16),
                          "gguf NVFP4 banks " + what + ": " + (divisor ? "Unknown" : "A16") + " (" +
                              std::to_string(static_cast<int>(facts.expert_prefill)) + ")");
        failures += check(facts.expert_prefill_int8 == rt::ExpertPrefill::Int8,
                          "gguf NVFP4 banks " + what + " under int8: the grouped W4A8 (" +
                              std::to_string(static_cast<int>(facts.expert_prefill_int8)) + ")");
    }
    // Floating-point weight banks (F16, kept as bf16) are a kind of their own, at BF16 inputs:
    // the op runs them too (Codex on #764); a float auxiliary tensor of a bank (a scale, a bias)
    // is not a bank, and neither moves the prefill route.
    {
        GgufFile banks = file;
        for (const char* name : {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"}) {
            banks.tensors.push_back(
                tensor(name, {5120, 512, 64}, ninferno::convert::ggml::F16, 100));
        }
        banks.tensors.push_back(tensor("blk.0.ffn_down_exps.weight", {512, 5120, 64},
                                       ninferno::convert::ggml::F16, 100));
        banks.tensors.push_back(
            tensor("blk.0.ffn_down_exps.scale", {64}, ninferno::convert::ggml::F32, 256));
        const auto f  = pr::plan_facts_of_gguf(banks, 999999);
        const auto& l = f.model.expert_layers;
        failures +=
            check(l.size() == 1 && l[0].gate_up_format == "bf16" && l[0].down_format == "bf16" &&
                      !l[0].gate_up_layout.empty() && l[0].gate_up_k == 5120 &&
                      l[0].down_k == 512 && l[0].policy == rt::PlanExpertPolicy::A16Only &&
                      l[0].layers == 1 && l[0].bytes == 300,
                  "gguf F16 banks: one bf16 kind at BF16 inputs, the scale apart");
        failures += check(f.model.expert_prefill == rt::ExpertPrefill::Unknown,
                          "gguf F16 banks: the prefill route as before");
        // F32 banks as the published recipes keep them: bf16 (FP32 has no weight codec).
        GgufFile f32 = file;
        for (const char* name : {"blk.0.ffn_gate_exps.weight", "blk.0.ffn_up_exps.weight"}) {
            f32.tensors.push_back(tensor(name, {5120, 512, 64}, ninferno::convert::ggml::F32, 200));
        }
        f32.tensors.push_back(tensor("blk.0.ffn_down_exps.weight", {512, 5120, 64},
                                     ninferno::convert::ggml::F32, 200));
        const auto l32 = pr::plan_facts_of_gguf(f32, 999999).model.expert_layers;
        failures += check(l32.size() == 1 && l32[0].gate_up_format == "bf16" &&
                              l32[0].down_format == "bf16" && !l.empty() &&
                              l32[0].gate_up_layout == l[0].gate_up_layout &&
                              l32[0].policy == rt::PlanExpertPolicy::A16Only,
                          "gguf F32 banks: a bf16 kind");
    }

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
        // The linears by shape (DECISIONS 66), as the census found the calls on the RTX 4090: the
        // gated attention's qkv 9216 x 2048 and output 2048 x 4096 (10 layers), the Gated
        // DeltaNet's in-projection 12288 x 2048 and output 2048 x 4096 (30), the shared expert's
        // 1024 x 2048 and 2048 x 512 (40), the experts' the same at 8 picks a token.
        const auto has = [&](double n, double k, double per, bool expert) {
            return std::any_of(f.model.linears.begin(), f.model.linears.end(), [&](const rt::PlanLinear& l) {
                return l.n == n && l.k == k && std::abs(l.per_token - per) < 1e-9 && l.expert == expert;
            });
        };
        failures += check(has(9216, 2048, 1, false) && has(2048, 4096, 1, false), where + ": attention linears");
        failures += check(has(12288, 2048, 30, false) && has(2048, 4096, 30, false), where + ": recurrent linears");
        failures += check(has(1024, 2048, 40, false) && has(2048, 512, 40, false), where + ": shared expert linears");
        failures += check(has(1024, 2048, 320, true) && has(2048, 512, 320, true), where + ": expert linears");
        failures += same(f.model.linears.size(), 26, where + ": 20 attention + 6 others");
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
    // Gemma 4's MXFP4 experts (#616 step 7): the GGUF decoder maps them onto NVFP4 only where
    // their block exponents fit E4M3 (the weights' bytes), the recipe re-quantizes them otherwise:
    // the policy the plan cannot read off the header is unknown, not MXFP4's AllowA8.
    {
        GgufFile mx;
        mx.metadata = md;
        mx.tensors.push_back(tensor("blk.0.ffn_gate_up_exps.weight", {5376, 1408, 128},
                                    ninferno::convert::ggml::MXFP4, 200));
        mx.tensors.push_back(tensor("blk.0.ffn_down_exps.weight", {704, 5376, 128},
                                    ninferno::convert::ggml::MXFP4, 100));
        const auto facts  = pr::plan_facts_of_gguf(mx, 999999).model;
        const auto layers = facts.expert_layers;
        failures += check(layers.size() == 1 && layers[0].gate_up_format == "mxfp4" &&
                              layers[0].policy == rt::PlanExpertPolicy::Unknown,
                          "gemma4 gguf: MXFP4 experts, the policy unknown");
        // Their prefill (#769): every site the loader keeps them as runs A16 under fast
        // activations (an NVFP4 site with no divisor, or the recipe's BF16 / Q8 at the Recipe
        // default A16Only): A16, on a format the header does not tell (assumed: Transferred).
        // Under --activations int8 every bank runs AllowA8: the file's format decides.
        failures +=
            check(facts.expert_prefill == rt::ExpertPrefill::A16 && facts.expert_prefill_assumed,
                  "gemma4 gguf: MXFP4 experts prefill A16, assumed");
        // expert_prefill_int8 is the header's format (MXFP4), not the site the loader keeps
        // (NVFP4, BF16 or Q8): the assumed flag keeps an estimate on it Transferred.
        failures += check(facts.expert_prefill_int8 == rt::ExpertPrefill::Mxfp4,
                          "gemma4 gguf: MXFP4 experts under int8, the MXFP4 route");
    }
    // End to end (#769): the plan's quantization (product::quant_option_of) and the choice's
    // prefill (choice::speed_of) of a Gemma 4 MoE GGUF with MXFP4 experts and Q8_0 linears beside
    // them, on the RTX 4090: the banks A16 under fast activations, the header's MXFP4 under int8,
    // and an estimate on them Transferred.
    // With an activation divisor (<stem>.input_scale) the NVFP4 site the loader keeps where
    // probe_mx fits is AllowA4 (codex on #792): the policy is the probe's, so the prefill is not
    // forced A16; the kept fallback (MXFP4) stands, assumed.
    {
        GgufFile mx;
        mx.metadata = md;
        for (const char* bank : {"ffn_gate_up_exps", "ffn_down_exps"}) {
            const bool down = std::string(bank) == "ffn_down_exps";
            mx.tensors.push_back(tensor(std::string("blk.0.") + bank + ".weight",
                                        down ? std::vector<std::uint64_t>{704, 5376, 128}
                                             : std::vector<std::uint64_t>{5376, 1408, 128},
                                        ninferno::convert::ggml::MXFP4, down ? 100 : 200));
            mx.tensors.push_back(tensor(std::string("blk.0.") + bank + ".input_scale", {1},
                                        ninferno::convert::ggml::F32, 4));
        }
        const auto facts = pr::plan_facts_of_gguf(mx, 999999).model;
        failures +=
            check(facts.expert_prefill == rt::ExpertPrefill::Mxfp4 && facts.expert_prefill_assumed,
                  "gemma4 gguf: MXFP4 experts with a divisor, the kept route, assumed");
    }
    // `linears`: the ggml type of the linears beside the banks (Q8_0: the tiles price the prefill;
    // another: prefill_linears_seconds gives 0 and the share path stands). `banks`: the experts'
    // ggml type, at the same stored bytes whatever it is.
    const auto gemma4_mx_gguf = [&](std::uint32_t linears,
                                    std::uint32_t banks = ninferno::convert::ggml::MXFP4) {
        json moe_md                                 = md;
        moe_md["gemma4.expert_count"]               = 128;
        moe_md["gemma4.expert_used_count"]          = 8;
        moe_md["gemma4.expert_feed_forward_length"] = 704;
        GgufFile mx;
        mx.metadata = moe_md;
        for (int b = 0; b < 60; ++b) {
            const std::string blk = "blk." + std::to_string(b) + ".";
            mx.tensors.push_back(
                tensor(blk + "ffn_gate_up_exps.weight", {5376, 1408, 128}, banks, 200));
            mx.tensors.push_back(
                tensor(blk + "ffn_down_exps.weight", {704, 5376, 128}, banks, 100));
            mx.tensors.push_back(tensor(blk + "attn_q.weight", {5376, 8192}, linears, 50));
        }
        return pr::plan_facts_of_gguf(mx, 999999);
    };
    rt::PlanDeviceFacts rtx4090;
    rtx4090.name               = "RTX 4090";
    rtx4090.compute_capability = 89;
    rtx4090.pci_device_id      = 0x2684;
    rtx4090.sms                = 128;
    rtx4090.total_bytes        = 24ULL << 30;
    {
        const auto facts = gemma4_mx_gguf(ninferno::convert::ggml::Q8_0);
        const ninferno::product::choice::QuantOption q = pr::quant_option_of(facts, "MXFP4");
        using Banks                                    = ninferno::product::choice::ExpertBanks;
        failures += check(q.expert_banks == Banks::A16 && q.expert_banks_int8 == Banks::Mxfp4 &&
                              q.expert_banks_assumed,
                          "gemma4 gguf e2e: the quantization's banks A16, MXFP4 under int8, "
                          "assumed");
        const ninferno::product::choice::CardFacts card =
            pr::choice_card_of(pr::gpu_key_of(rtx4090));
        const ninferno::product::choice::ModelFacts model = pr::choice_model_of(facts);
        bool transferred                                  = false;
        const double seconds =
            ninferno::product::choice::prefill_linears_seconds(model, card, q, 2048, &transferred);
        failures += check(model.moe() && seconds > 0 && transferred,
                          "gemma4 gguf e2e: the banks' tiles priced, Transferred");
        ninferno::product::choice::Candidate fast;
        fast.kv                                   = ninferno::KvCacheStorage::BFloat16;
        fast.chunk                                = 2048;
        ninferno::product::choice::Candidate int8 = fast;
        int8.activations                          = ninferno::ActivationMode::Int8;
        const ninferno::product::choice::Speed f =
            ninferno::product::choice::speed_of(model, card, q, fast, 1, 4096, 2048);
        const ninferno::product::choice::Speed i =
            ninferno::product::choice::speed_of(model, card, q, int8, 1, 4096, 2048);
        failures += check(f.basis != ninferno::product::choice::SpeedBasis::Own &&
                              i.basis != ninferno::product::choice::SpeedBasis::Own,
                          "gemma4 gguf e2e: fast and int8 estimates Transferred");
        failures += check(f.prefill > 0 && i.prefill > 0 && f.prefill != i.prefill,
                          "gemma4 gguf e2e: an int8 candidate's prefill on another route");
    }
    // The same with Q4_K linears beside the banks (codex on #792): the tiles price nothing
    // (prefill_linears_seconds 0), the share path stands, and the estimate is Transferred still.
    {
        const auto facts = gemma4_mx_gguf(ninferno::convert::ggml::Q4_K);
        const ninferno::product::choice::QuantOption q = pr::quant_option_of(facts, "MXFP4");
        const ninferno::product::choice::CardFacts card =
            pr::choice_card_of(pr::gpu_key_of(rtx4090));
        const ninferno::product::choice::ModelFacts model = pr::choice_model_of(facts);
        ninferno::product::choice::Candidate fast;
        fast.kv    = ninferno::KvCacheStorage::BFloat16;
        fast.chunk = 2048;
        const ninferno::product::choice::Speed f =
            ninferno::product::choice::speed_of(model, card, q, fast, 1, 4096, 2048);
        failures += check(
            q.expert_banks_assumed &&
                ninferno::product::choice::prefill_linears_seconds(model, card, q, 2048) == 0 &&
                f.prefill > 0 && f.basis == ninferno::product::choice::SpeedBasis::Transferred,
            "gemma4 gguf e2e, Q4_K linears: the share path, Transferred");
    }
    // Two quantizations of equal tensor bytes (#775): one of MXFP4 experts (A16, MXFP4 under
    // int8, assumed), one of Q4_0 experts (Int8 both ways, known). A row reads its own header's
    // prefill route, never the parsed file's at equal bytes; a row without its table has none.
    {
        using Banks      = ninferno::product::choice::ExpertBanks;
        const auto mxfp4 = gemma4_mx_gguf(ninferno::convert::ggml::Q8_0);
        const auto q4_0 =
            gemma4_mx_gguf(ninferno::convert::ggml::Q8_0, ninferno::convert::ggml::Q4_0);
        failures += check(mxfp4.model.weight_bytes == q4_0.model.weight_bytes &&
                              q4_0.model.weight_basis == rt::WeightBasis::GgufTensors,
                          "equal-bytes rows: the two files' tensors weigh the same");
        failures += check(q4_0.model.expert_prefill == rt::ExpertPrefill::Int8 &&
                              q4_0.model.expert_prefill_int8 == rt::ExpertPrefill::Int8 &&
                              !q4_0.model.expert_prefill_assumed,
                          "equal-bytes rows: Q4_0 experts prefill Int8, known");
        const auto row_of = [](const pr::PlanSourceFacts& own, const std::string& quant) {
            pr::QuantRow row;
            row.listing.format = "gguf";
            row.listing.quant  = quant;
            row.listing.table  = pr::gguf_tensor_table(own);
            row.weight_bytes   = row.listing.table ? row.listing.table->tensor_bytes : 0;
            row.basis          = rt::WeightBasis::GgufTensors;
            return row;
        };
        const ninferno::product::choice::CardFacts card =
            pr::choice_card_of(pr::gpu_key_of(rtx4090));
        ninferno::product::choice::Candidate fast;
        fast.kv               = ninferno::KvCacheStorage::BFloat16;
        fast.chunk            = 2048;
        const auto prefill_of = [&](const pr::PlanSourceFacts& facts, const std::string& quant) {
            return ninferno::product::choice::speed_of(pr::choice_model_of(facts), card,
                                                       pr::quant_option_of(facts, quant), fast, 1,
                                                       4096, 2048)
                .prefill;
        };
        // The Q4_0 row beside the parsed MXFP4 file, and the reverse.
        const auto q4_row = pr::row_facts(mxfp4, row_of(q4_0, "Q4_0"));
        const auto q4     = pr::quant_option_of(q4_row, "Q4_0");
        failures += check(q4.expert_banks == Banks::Int8 && q4.expert_banks_int8 == Banks::Int8 &&
                              !q4.expert_banks_assumed,
                          "equal-bytes rows: the Q4_0 row's own route, not the parsed MXFP4's");
        const auto mx_row = pr::row_facts(q4_0, row_of(mxfp4, "MXFP4"));
        const auto mx     = pr::quant_option_of(mx_row, "MXFP4");
        failures += check(mx.expert_banks == Banks::A16 && mx.expert_banks_int8 == Banks::Mxfp4 &&
                              mx.expert_banks_assumed,
                          "equal-bytes rows: the MXFP4 row's own route, int8 and assumed");
        failures += check(prefill_of(q4_row, "Q4_0") != prefill_of(mx_row, "MXFP4"),
                          "equal-bytes rows: two prefill estimates");
        pr::QuantRow unread = row_of(q4_0, "Q4_0");
        unread.listing.table.reset();
        const auto bare = pr::row_facts(mxfp4, unread).model;
        failures += check(bare.expert_prefill == rt::ExpertPrefill::Unknown &&
                              bare.expert_prefill_int8 == rt::ExpertPrefill::Unknown &&
                              !bare.expert_prefill_assumed,
                          "equal-bytes rows: a row without its table, no route");
        // An unread Gemma 4 row whose label names MXFP4 banks (#811 P3): the kept site its header
        // would not tell either, so its estimate is Transferred as a read one's is; a label of
        // other banks is not marked.
        pr::QuantRow unread_mx = row_of(mxfp4, "MXFP4_MOE");
        unread_mx.listing.table.reset();
        const auto bare_mx = pr::row_facts(q4_0, unread_mx);
        failures += check(bare_mx.model.expert_prefill == rt::ExpertPrefill::Unknown &&
                              bare_mx.model.expert_prefill_assumed,
                          "unread Gemma 4 MXFP4 row: no route, assumed by its label");
        // An engaged but empty table (the UI caches one for a header past 128 MiB): unread too.
        pr::QuantRow empty_mx     = row_of(mxfp4, "MXFP4_MOE");
        empty_mx.listing.table    = pr::GgufTensorTable{};
        const auto empty_table_mx = pr::row_facts(q4_0, empty_mx);
        failures += check(empty_table_mx.model.expert_prefill_assumed,
                          "Gemma 4 MXFP4 row with an empty table: assumed by its label");
        // Through the quantization the choice prices (linear_prefill: an assumed MoE prefill is
        // Transferred on every path, the share path of Unknown banks too).
        failures += check(
            pr::quant_option_of(bare_mx, "MXFP4_MOE").expert_banks_assumed &&
                !pr::quant_option_of(pr::row_facts(mxfp4, unread), "Q4_0").expert_banks_assumed,
            "unread Gemma 4 rows: the MXFP4 label's assumed, Q4_0's not");
        // Another family's row is not marked by an MXFP4 label (gpt-oss: its banks run as read).
        pr::PlanSourceFacts other_family = q4_0;
        other_family.model.model_type    = "gpt_oss";
        failures += check(!pr::row_facts(other_family, unread_mx).model.expert_prefill_assumed,
                          "an unread MXFP4 row of another family: not assumed");
    }
    // Two quantizations of equal tensor bytes with other linears (#811): Q8_0 beside the banks
    // in one, Q4_K in the other. A row reads its own header's formats (the weight class, whether
    // the tiles price its prefill, its routes), never the parsed file's at equal bytes; a row
    // without its table reads its label's.
    {
        using Class       = ninferno::product::choice::WeightClass;
        const auto q8     = gemma4_mx_gguf(ninferno::convert::ggml::Q8_0);
        const auto k      = gemma4_mx_gguf(ninferno::convert::ggml::Q4_K);
        const auto row_of = [](const pr::PlanSourceFacts& own, const std::string& quant) {
            pr::QuantRow row;
            row.listing.format = "gguf";
            row.listing.quant  = quant;
            row.listing.table  = pr::gguf_tensor_table(own);
            row.weight_bytes   = row.listing.table ? row.listing.table->tensor_bytes : 0;
            row.basis          = rt::WeightBasis::GgufTensors;
            return row;
        };
        failures += check(q8.model.weight_bytes == k.model.weight_bytes &&
                              q8.shape.rest_layer_formats != k.shape.rest_layer_formats,
                          "equal-bytes formats: the same bytes, other linears");
        const ninferno::product::choice::CardFacts card =
            pr::choice_card_of(pr::gpu_key_of(rtx4090));
        const auto tiles = [&](const pr::PlanSourceFacts& facts, const std::string& quant) {
            return ninferno::product::choice::prefill_linears_seconds(
                pr::choice_model_of(facts), card, pr::quant_option_of(facts, quant), 2048);
        };
        // The Q4_K row beside the parsed Q8_0 file: its own K-quant linears (the share path), not
        // the parsed file's Q8_0 (the tiles); the reverse prices the tiles.
        const auto k_row = pr::row_facts(q8, row_of(k, "Q4_K_M"));
        const auto kq    = pr::quant_option_of(k_row, "Q4_K_M");
        failures += check(kq.rest_weights == Class::KQuant && tiles(k_row, "Q4_K_M") == 0,
                          "equal-bytes formats: the Q4_K row's own linears, the share path");
        const auto q8_row = pr::row_facts(k, row_of(q8, "Q8_0"));
        failures += check(pr::quant_option_of(q8_row, "Q8_0").rest_weights == Class::Q8 &&
                              tiles(q8_row, "Q8_0") > 0,
                          "equal-bytes formats: the Q8_0 row's own linears, the tiles");
        // Its routes from its own formats, not its label's: MXFP4 banks take 8-bit activations
        // under a BF16 label; the same row unread takes the label's (BF16: none).
        failures += check(pr::quant_option_of(q8_row, "BF16").int8_route,
                          "a read row's routes: its own formats' (MXFP4: int8)");
        pr::QuantRow unread = row_of(q8, "BF16");
        unread.listing.table.reset();
        const auto bare = pr::row_facts(k, unread);
        failures += check(bare.shape.weight_formats.empty() &&
                              !pr::quant_option_of(bare, "BF16").int8_route,
                          "an unread row's routes: its label's (BF16: no int8)");
    }
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
    // A registry name with a comma (#831): whole, with facts after it, and without the vendor.
    const auto tu116 = pr::plan_device_of_board("GeForce GTX 1650 (TU116, GDDR6)", &error);
    failures += check(tu116 && tu116->name == "GeForce GTX 1650 (TU116, GDDR6)",
                      "a comma in the board's name: " + error);
    const auto tu116h =
        pr::plan_device_of_board("gtx 1650 (tu116, gddr6), headless, linux", &error);
    failures += check(tu116h && tu116h->name == "GeForce GTX 1650 (TU116, GDDR6)" &&
                          !tu116h->paravirtual && !tu116h->drives_display,
                      "a comma name with facts after it: " + error);
    // The speculation key carries the board's mma_slow from the facts table (#858): a GTX 16
    // board has no mma.sync tiles, a Turing RTX board has them.
    if (tu116) {
        const auto key = pr::gpu_key_of(*tu116);
        failures += check(key.mma_slow && !ninferno::spec::card_rates(key).mma_tiles,
                          "a GTX 16 board's key: mma_slow, no mma.sync tiles");
    }
    if (const auto gtx = pr::plan_device_of_board("0x2182", &error)) {
        const auto key = pr::gpu_key_of(*gtx);
        failures += check(key.mma_slow && !ninferno::spec::card_rates(key).mma_tiles,
                          "the GTX 1660 Ti's key: mma_slow, no mma.sync tiles");
    } else {
        failures += check(false, "the GTX 1660 Ti by its PCI id: " + error);
    }
    if (const auto rtx = pr::plan_device_of_board("2080 ti", &error)) {
        const auto key = pr::gpu_key_of(*rtx);
        failures += check(!key.mma_slow && ninferno::spec::card_rates(key).mma_tiles,
                          "the RTX 2080 Ti's key: mma.sync tiles");
    } else {
        failures += check(false, "the RTX 2080 Ti by name: " + error);
    }
    error.clear();
    failures += check(!pr::plan_device_of_board("GTX 1650 (TU116, GDDR6), turbo", &error) &&
                          error.find("turbo") != std::string::npos,
                      "an unknown fact after a comma name: " + error);
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

// A two-layer gpt-oss text config (E32 k4 H2880 F2880): the artifact tests' routed geometry.
json gpt_oss_text_config() {
    return {{"architectures", {"GptOssForCausalLM"}},
            {"model_type", "gpt_oss"},
            {"hidden_size", 2880},
            {"num_hidden_layers", 2},
            {"layer_types", {"sliding_attention", "full_attention"}},
            {"sliding_window", 128},
            {"num_attention_heads", 64},
            {"num_key_value_heads", 8},
            {"head_dim", 64},
            {"intermediate_size", 2880},
            {"num_local_experts", 32},
            {"num_experts_per_tok", 4},
            {"max_position_embeddings", 131072},
            {"vocab_size", 201088}};
}

// An artifact's routed layers by kind (#616 step 7, plan_facts_of_directory): each bank's policy is
// the one its Use records (what a start reads, models/common/weight_use.cpp); a layer of a bank
// whose use records none has an unknown policy; the MTP layer's banks are not the trunk's.
int artifact_expert_layers() {
    namespace art = ninferno::artifact;
    int failures  = 0;
    art::Directory dir;
    dir.components["text"].config = gpt_oss_text_config();
    const std::string mx =
        std::string(art::layout_name(ninferno::default_weight_layout(ninferno::QType::MXFP4)));
    const auto bank = [&](const std::string& name, std::uint64_t rows, std::uint64_t bytes,
                          std::optional<art::ActivationPolicy> policy) {
        art::TensorObject t;
        t.id     = name;
        t.shape  = {32, rows, 2880};
        t.format = "mxfp4";
        t.layout = mx;
        t.bytes  = bytes;
        const art::ObjectHandle handle{dir.objects.size()};
        dir.objects.emplace_back(t);
        dir.object_index[name] = handle;
        art::Binding b;
        b.whole_object     = true;
        b.elements         = 32 * rows * 2880;
        b.parts            = {art::Part{handle, 0, b.elements}};
        dir.bindings[name] = b;
        art::Use use;
        use.parameter         = name;
        use.input             = "x";
        use.activation_policy = policy;
        dir.uses[{name, "x"}] = use;
    };
    const auto a8 = art::ActivationPolicy::AllowA8;
    bank("text/layers/0/experts/gate_up", 5760, 1000, a8);
    bank("text/layers/0/experts/down", 2880, 500, a8);
    bank("text/layers/1/experts/gate_up", 5760, 1000, a8);
    bank("text/layers/1/experts/down", 2880, 500, std::nullopt);
    bank("mtp/layers/0/experts/gate_up", 5760, 1000, a8);
    bank("mtp/layers/0/experts/down", 2880, 500, a8);
    const auto layers   = pr::plan_facts_of_directory(dir, "a test directory").model.expert_layers;
    std::uint32_t total = 0;
    for (const rt::PlanExpertLayers& l : layers) { total += l.layers; }
    const auto kind = [&](rt::PlanExpertPolicy policy) {
        return std::find_if(layers.begin(), layers.end(), [&](const rt::PlanExpertLayers& l) {
            return l.policy == policy && l.gate_up_format == "mxfp4" && l.down_format == "mxfp4";
        });
    };
    const auto recorded = kind(rt::PlanExpertPolicy::AllowA8);
    const auto unknown  = kind(rt::PlanExpertPolicy::Unknown);
    failures += check(layers.size() == 2 && total == 2,
                      "artifact routed layers: the trunk's two, not the MTP layer's");
    failures += check(recorded != layers.end() && recorded->layers == 1 &&
                          recorded->bytes == 1500 && recorded->gate_up_k == 2880 &&
                          recorded->down_k == 2880 && recorded->gate_up_layout == mx,
                      "artifact routed layers: the recorded AllowA8");
    failures += check(unknown != layers.end() && unknown->layers == 1,
                      "artifact routed layers: a use without a policy, unknown");

    // A layer of BF16 banks (Codex on #764): its weight banks (bound by a Use) are a bf16 kind at
    // the recorded A16; its stacked gate/up bias (no Use: an auxiliary) is not a bank.
    art::Directory floats;
    floats.components = dir.components;
    const std::string bf =
        std::string(art::layout_name(ninferno::default_weight_layout(ninferno::QType::BF16)));
    const auto object = [&](const std::string& name, art::Shape shape, std::uint64_t bytes,
                            bool use) {
        art::TensorObject t;
        t.id     = name;
        t.shape  = shape;
        t.format = "bf16";
        t.layout = bf;
        t.bytes  = bytes;
        const art::ObjectHandle handle{floats.objects.size()};
        floats.objects.emplace_back(t);
        floats.object_index[name] = handle;
        art::Binding b;
        b.whole_object = true;
        b.elements     = 1;
        for (const std::uint64_t d : shape) { b.elements *= d; }
        b.parts               = {art::Part{handle, 0, b.elements}};
        floats.bindings[name] = b;
        if (use) {
            art::Use u;
            u.parameter              = name;
            u.input                  = "x";
            u.activation_policy      = art::ActivationPolicy::A16Only;
            floats.uses[{name, "x"}] = u;
        }
    };
    object("text/layers/0/experts/gate_up", {32, 5760, 2880}, 4000, true);
    object("text/layers/0/experts/down", {32, 2880, 2880}, 2000, true);
    object("text/layers/0/experts/gate_up_bias", {32, 5760}, 300, false);
    const auto bf_layers =
        pr::plan_facts_of_directory(floats, "a test directory").model.expert_layers;
    failures += check(bf_layers.size() == 1 && bf_layers[0].gate_up_format == "bf16" &&
                          bf_layers[0].down_format == "bf16" && bf_layers[0].gate_up_layout == bf &&
                          bf_layers[0].policy == rt::PlanExpertPolicy::A16Only &&
                          bf_layers[0].layers == 1 && bf_layers[0].bytes == 6000,
                      "artifact routed layers: BF16 banks a bf16 kind at A16, the bias apart");
    return failures;
}

// An artifact's MoE prefill route reads the policy its banks' Uses record, as the decode kind does
// (#769), not the format's kept policy (convert::kept_block_policy: MXFP4 and Q4_0 AllowA8): banks
// recorded A16Only prefill on the 16-bit register MMA (ops/moe w4a8_bank_refusal Policy), one
// A16-only bank of a layer makes the layer A16 (ops::moe_policy), and the kept policy stands only
// where a Use records none.
int artifact_prefill_policy() {
    namespace art = ninferno::artifact;
    using ninferno::QType;
    using Policy = std::optional<art::ActivationPolicy>;

    struct Bank {
        std::string name;
        std::uint64_t rows;
        Policy policy;
        std::optional<QType> format = std::nullopt; // the directory's format where none
    };

    // A directory of these banks, one object and one use each; `unrecorded_use`: a second use of
    // every object that records no policy (the fold over an object's uses, #1081).
    const auto directory = [](QType format, const std::vector<Bank>& banks,
                              bool unrecorded_use = false) {
        art::Directory dir;
        dir.components["text"].config = gpt_oss_text_config();
        for (const Bank& bank : banks) {
            const QType f                      = bank.format.value_or(format);
            const ninferno::QuantLayout layout = f == QType::MXFP4 || f == QType::BF16 ||
                                                         f == QType::NVFP4 ||
                                                         f == QType::FP8_E4M3FN_ROW_BF16
                                                     ? ninferno::default_weight_layout(f)
                                                     : ninferno::ggml_published_layout(f);
            art::TensorObject t;
            t.id     = bank.name;
            t.shape  = {32, bank.rows, 2880};
            t.format = std::string(art::format_name(f));
            t.layout = std::string(art::layout_name(layout));
            t.bytes  = bank.rows;
            const art::ObjectHandle handle{dir.objects.size()};
            dir.objects.emplace_back(t);
            dir.object_index[bank.name] = handle;
            art::Binding b;
            b.whole_object          = true;
            b.elements              = 32 * bank.rows * 2880;
            b.parts                 = {art::Part{handle, 0, b.elements}};
            dir.bindings[bank.name] = b;
            art::Use use;
            use.parameter              = bank.name;
            use.input                  = "x";
            use.activation_policy      = bank.policy;
            dir.uses[{bank.name, "x"}] = use;
            if (unrecorded_use) {
                use.input                  = "y";
                use.activation_policy      = std::nullopt;
                dir.uses[{bank.name, "y"}] = use;
            }
        }
        return dir;
    };
    // `int8`: the route under --activations int8 (expert_prefill_int8), else fast activations'.
    const auto route = [&](QType format, const std::vector<Bank>& banks, bool int8 = false,
                           bool unrecorded_use = false) {
        const auto facts = pr::plan_facts_of_directory(directory(format, banks, unrecorded_use),
                                                       "a test directory")
                               .model;
        return int8 ? facts.expert_prefill_int8 : facts.expert_prefill;
    };
    // The decode kind's policy of the same directory (PlanExpertLayers::policy, layer_policy).
    const auto kind_policy = [&](QType format, const std::vector<Bank>& banks,
                                 bool unrecorded_use = false) {
        const auto l = pr::plan_facts_of_directory(directory(format, banks, unrecorded_use),
                                                   "a test directory")
                           .model.expert_layers;
        return l.size() == 1 ? std::optional<rt::PlanExpertPolicy>(l[0].policy) : std::nullopt;
    };
    const auto layers = [](Policy gate_up0, Policy down0, Policy gate_up1, Policy down1) {
        return std::vector<Bank>{{"text/layers/0/experts/gate_up", 5760, gate_up0},
                                 {"text/layers/0/experts/down", 2880, down0},
                                 {"text/layers/1/experts/gate_up", 5760, gate_up1},
                                 {"text/layers/1/experts/down", 2880, down1}};
    };
    const Policy a16 = art::ActivationPolicy::A16Only;
    const Policy a8  = art::ActivationPolicy::AllowA8;
    const Policy none;
    int failures = 0;
    failures += check(route(QType::MXFP4, layers(a8, a8, a8, a8)) == rt::ExpertPrefill::Mxfp4,
                      "artifact prefill: MXFP4 banks recorded AllowA8, the MXFP4 route");
    failures += check(route(QType::MXFP4, layers(a16, a16, a16, a16)) == rt::ExpertPrefill::A16,
                      "artifact prefill: MXFP4 banks recorded A16Only, A16 (not the kept AllowA8)");
    failures += check(route(QType::MXFP4, layers(a8, a16, a8, a16)) == rt::ExpertPrefill::A16,
                      "artifact prefill: one A16-only bank a layer makes the layer A16");
    failures +=
        check(route(QType::MXFP4, layers(none, none, none, none)) == rt::ExpertPrefill::Mxfp4,
              "artifact prefill: no recorded policy, the format's kept policy");
    failures += check(route(QType::GGML_Q4_0, layers(a8, a8, a8, a8)) == rt::ExpertPrefill::Int8,
                      "artifact prefill: Q4_0 banks recorded AllowA8, the grouped W4A8");
    failures += check(route(QType::GGML_Q4_0, layers(a16, a16, a16, a16)) == rt::ExpertPrefill::A16,
                      "artifact prefill: Q4_0 banks recorded A16Only, A16 (not the kept AllowA8)");
    // #796 item 3: a Q4_0 layer with one A16-only bank is A16 as MXFP4's is; a BF16 down beside a
    // Q4_K gate/up (a float bank runs at BF16 inputs) makes the layer A16.
    failures += check(route(QType::GGML_Q4_0, layers(a8, a16, a8, a16)) == rt::ExpertPrefill::A16,
                      "artifact prefill: Q4_0 with an A16-only down bank, A16");
    failures +=
        check(route(QType::GGML_Q4_K, {{"text/layers/0/experts/gate_up", 5760, a8},
                                       {"text/layers/0/experts/down", 2880, a16, QType::BF16},
                                       {"text/layers/1/experts/gate_up", 5760, a8},
                                       {"text/layers/1/experts/down", 2880, a16, QType::BF16}}) ==
                  rt::ExpertPrefill::A16,
              "artifact prefill: a BF16 down beside a Q4_K gate/up, A16");
    // #796 item 1: an NVFP4 gate/up at AllowA4 beside a Q4_K down at AllowA8. The op keeps the
    // layer A16 on a build with the W4A4 kernels (sm_120a) and runs the grouped W4A8 on one
    // without; the facts name no build, so the route is Unknown. Under --activations int8 there
    // is no A4 policy, and the route is not Unknown.
    const Policy a4                  = art::ActivationPolicy::AllowA4;
    const std::vector<Bank> nvfp4_a4 = {{"text/layers/0/experts/gate_up", 5760, a4},
                                        {"text/layers/0/experts/down", 2880, a8, QType::GGML_Q4_K},
                                        {"text/layers/1/experts/gate_up", 5760, a4},
                                        {"text/layers/1/experts/down", 2880, a8, QType::GGML_Q4_K}};
    failures += check(route(QType::NVFP4, nvfp4_a4) == rt::ExpertPrefill::Unknown,
                      "artifact prefill: NVFP4 gate/up at AllowA4 beside a Q4_K down, the build's: "
                      "Unknown");
    // codex on #1087: a build-dependent bank too small to move the route leaves it. An FP8 gate/up
    // (no W4A8 tile: A16 on every build) beside a smaller NVFP4 down at AllowA4: A16 with the W4A4
    // kernels (the veto) and without them (the down alone on the grouped W4A8).
    const std::vector<Bank> small_nvfp4 = {
        {"text/layers/0/experts/gate_up", 5760, a8, QType::FP8_E4M3FN_ROW_BF16},
        {"text/layers/0/experts/down", 2880, a4},
        {"text/layers/1/experts/gate_up", 5760, a8, QType::FP8_E4M3FN_ROW_BF16},
        {"text/layers/1/experts/down", 2880, a4}};
    failures +=
        check(route(QType::NVFP4, small_nvfp4) == rt::ExpertPrefill::A16,
              "artifact prefill: a smaller NVFP4 down at AllowA4 beside an FP8 gate/up, A16 "
              "on every build");
    // #1090: NVFP4 runs the grouped W4A8 on every build where its policy is A8 and not A4
    // (nvfp4_prefill_a8_policy), as under --activations int8: the int8 route, not A16.
    failures +=
        check(route(QType::NVFP4, nvfp4_a4, true) == rt::ExpertPrefill::Int8,
              "artifact prefill under int8: NVFP4 at AllowA4 beside Q4_K, the grouped W4A8");
    failures += check(route(QType::NVFP4, layers(a8, a8, a8, a8)) == rt::ExpertPrefill::Int8,
                      "artifact prefill: NVFP4 banks at AllowA8, the grouped W4A8");
    failures += check(route(QType::NVFP4, layers(a16, a16, a16, a16)) == rt::ExpertPrefill::A16,
                      "artifact prefill: NVFP4 banks recorded A16Only, A16");
    // A policy the facts do not hold is the build's or the recipe's, not the kept A16Only (an
    // artifact's start refuses a use with none, weight_use.cpp; a GGUF's divisor bank has none).
    failures +=
        check(route(QType::NVFP4, layers(none, none, none, none)) == rt::ExpertPrefill::Unknown,
              "artifact prefill: NVFP4 banks with no recorded policy, Unknown");
    failures +=
        check(route(QType::NVFP4, layers(none, none, none, none), true) == rt::ExpertPrefill::Int8,
              "artifact prefill under int8: NVFP4 banks with no recorded policy, the grouped W4A8");
    // A bank known A16-only beside one with no recorded policy: the layer is A16 still
    // (ops::moe_policy), not the kept fallback.
    failures += check(route(QType::MXFP4, layers(a16, none, a16, none)) == rt::ExpertPrefill::A16,
                      "artifact prefill: an A16-only bank beside an unknown one, A16");
    // The decode kind says the same (#796): A16Only, not Unknown, where one bank is known A16.
    failures += check(kind_policy(QType::MXFP4, layers(a16, none, a16, none)) ==
                          rt::PlanExpertPolicy::A16Only,
                      "decode kind: an A16-only bank beside an unknown one, A16Only");
    failures += check(kind_policy(QType::MXFP4, layers(none, none, none, none)) ==
                          rt::PlanExpertPolicy::Unknown,
                      "decode kind: no bank records a policy, Unknown");
    // One object read by an A16-only use and by a use that records none (#1081): the object is
    // A16-only still, so the layer is, for the prefill route and the decode kind.
    failures +=
        check(route(QType::MXFP4, layers(a16, a8, a16, a8), false, true) == rt::ExpertPrefill::A16,
              "artifact prefill: an A16-only use beside an unrecorded one, A16");
    failures += check(kind_policy(QType::MXFP4, layers(a16, a8, a16, a8), true) ==
                          rt::PlanExpertPolicy::A16Only,
                      "decode kind: an A16-only use beside an unrecorded one, A16Only");
    failures += check(kind_policy(QType::MXFP4, layers(a8, a8, a8, a8), true) ==
                          rt::PlanExpertPolicy::Unknown,
                      "decode kind: AllowA8 uses beside unrecorded ones, Unknown");
    // Under --activations int8 the start runs every bank at AllowA8 (models::effective_policy),
    // whatever was recorded: the formats decide.
    failures +=
        check(route(QType::MXFP4, layers(a16, a16, a16, a16), true) == rt::ExpertPrefill::Mxfp4,
              "artifact prefill under int8: MXFP4 banks recorded A16Only, the MXFP4 route");
    failures +=
        check(route(QType::GGML_Q4_0, layers(a16, a8, a16, a8), true) == rt::ExpertPrefill::Int8,
              "artifact prefill under int8: Q4_0 banks with A16-only ones, the grouped W4A8");
    return failures;
}

} // namespace

// A GGUF's linears take their tensors' formats by role: a recurrent layer's attn_qkv is its mixer's
// input, a role whose layers differ in format splits by their elements.
int gguf_linear_formats() {
    int failures = 0;
    namespace g  = ninferno::convert::ggml;
    GgufFile file;
    file.metadata = qwen35_metadata();
    file.tensors.push_back(tensor("token_embd.weight", {5120, 248320}, g::Q6_K, 1000));
    file.tensors.push_back(tensor("blk.0.attn_qkv.weight", {5120, 10240}, g::Q3_K, 500));
    file.tensors.push_back(tensor("blk.0.ssm_out.weight", {6144, 5120}, g::Q5_K, 500));
    file.tensors.push_back(tensor("blk.3.attn_q.weight", {5120, 12288}, g::Q4_K, 500));
    file.tensors.push_back(tensor("blk.0.ffn_down.weight", {17408, 5120}, g::Q4_K, 500));
    file.tensors.push_back(tensor("blk.1.ffn_down.weight", {17408, 5120}, g::Q6_K, 500));
    const auto f = pr::plan_facts_of_gguf(file, 999999);
    const auto find = [&](rt::LinearRole role, const std::string& format) -> const rt::PlanLinear* {
        for (const rt::PlanLinear& l : f.model.linears) {
            if (l.role == role && l.format == format) { return &l; }
        }
        return nullptr;
    };
    const rt::PlanLinear* mixer = find(rt::LinearRole::MixerIn, "ggml_q3_k");
    const rt::PlanLinear* out   = find(rt::LinearRole::MixerOut, "ggml_q5_k");
    const rt::PlanLinear* query = find(rt::LinearRole::AttentionIn, "ggml_q4_k");
    const rt::PlanLinear* down4 = find(rt::LinearRole::FfnDown, "ggml_q4_k");
    const rt::PlanLinear* down6 = find(rt::LinearRole::FfnDown, "ggml_q6_k");
    failures += check(mixer != nullptr && mixer->per_token == 48, "the recurrent layers' input in Q3_K");
    failures += check(out != nullptr && out->per_token == 48, "the recurrent layers' output in Q5_K");
    const auto queries = std::count_if(f.model.linears.begin(), f.model.linears.end(), [](const rt::PlanLinear& l) {
        return l.role == rt::LinearRole::AttentionIn && l.format == "ggml_q4_k" && l.per_token == 1;
    });
    failures += check(query != nullptr && queries == 16, "the full layers' inputs in Q4_K, one a layer");
    failures += check(down4 != nullptr && down6 != nullptr && down4->per_token == 32 && down6->per_token == 32,
                      "the FFN down split by its layers' formats");
    failures += check(find(rt::LinearRole::FfnUp, "") != nullptr && find(rt::LinearRole::AttentionOut, "") != nullptr,
                      "a role without tensors keeps the label's format");
    return failures;
}

// A converted artifact's tensor table (#835 item 3, product::tensor_table_of_artifact: what
// local_tensor_table gives a cached artifact): every field its facts read, the banks' formats
// among them, so a plan of the kept artifact prices its own formats; none for facts that are not
// an artifact's tensors or hold no bytes.
int artifact_tensor_table() {
    namespace art = ninferno::artifact;
    int failures  = 0;
    art::Directory dir;
    dir.components["text"].config = gpt_oss_text_config();
    const std::string mx =
        std::string(art::layout_name(ninferno::default_weight_layout(ninferno::QType::MXFP4)));
    for (const auto& [name, rows] :
         {std::pair<std::string, std::uint64_t>{"text/layers/0/moe/experts/gate_up", 5760},
          {"text/layers/0/moe/experts/down", 2880}}) {
        art::TensorObject t;
        t.id     = name;
        t.shape  = {32, rows, 2880};
        t.format = "mxfp4";
        t.layout = mx;
        t.bytes  = rows;
        const art::ObjectHandle handle{dir.objects.size()};
        dir.objects.emplace_back(t);
        dir.object_index[name] = handle;
        art::Binding b;
        b.whole_object     = true;
        b.elements         = 32 * rows * 2880;
        b.parts            = {art::Part{handle, 0, b.elements}};
        dir.bindings[name] = b;
        art::Use use;
        use.parameter         = name;
        use.input             = "x";
        use.activation_policy = art::ActivationPolicy::AllowA8;
        dir.uses[{name, "x"}] = use;
    }
    const pr::PlanSourceFacts facts = pr::plan_facts_of_directory(dir, "a test directory");
    const auto table                = pr::tensor_table_of_artifact(facts);
    failures += check(table.has_value(), "artifact table: an artifact's facts give a table");
    if (table) {
        const rt::PlanModelFacts& m = facts.model;
        failures += check(table->basis == rt::WeightBasis::Artifact &&
                              table->tensor_bytes == m.weight_bytes && table->tensor_bytes > 0 &&
                              table->mtp_bytes == m.mtp_bytes &&
                              table->own_drafter_bytes == m.own_drafter_bytes,
                          "artifact table: the artifact's bytes, its basis Artifact");
        failures += check(table->expert_layers.size() == m.expert_layers.size() &&
                              table->expert_prefill == m.expert_prefill &&
                              table->expert_prefill_int8 == m.expert_prefill_int8 &&
                              table->expert_prefill_assumed == m.expert_prefill_assumed,
                          "artifact table: the routed layers and routes of the facts");
        failures += check(table->weight_formats == facts.shape.weight_formats &&
                              table->layer_formats == facts.shape.layer_formats &&
                              table->rest_layer_formats == facts.shape.rest_layer_formats &&
                              table->head_format == m.head_format,
                          "artifact table: the trunk's, the layers' and the head's formats");
        const auto mxfp4 = [](const std::vector<std::string>& formats) {
            return std::find(formats.begin(), formats.end(), "mxfp4") != formats.end();
        };
        failures += check(table->expert_up_formats == m.expert_up_formats &&
                              table->expert_down_formats == m.expert_down_formats &&
                              mxfp4(table->expert_up_formats) && mxfp4(table->expert_down_formats),
                          "artifact table: the banks' formats (mxfp4), as the facts read them");
    }
    pr::PlanSourceFacts gguf = facts;
    gguf.model.weight_basis  = rt::WeightBasis::GgufTensors;
    failures += check(!pr::tensor_table_of_artifact(gguf).has_value(),
                      "artifact table: none from facts that are not an artifact's");
    pr::PlanSourceFacts empty = facts;
    empty.model.weight_bytes  = 0;
    failures += check(!pr::tensor_table_of_artifact(empty).has_value(),
                      "artifact table: none from an artifact of no bytes");
    return failures;
}

// Which read tables price their rows (#835 item 3, product::table_prices_row: ninferno-ui counts
// those in its tables generation and plans their repository again): a split part's formats alone
// do (#811), as do its banks' route (#775) and a whole table's bytes; an engaged empty table (a
// failed or past-128-MiB read) does not.
int table_counting() {
    int failures = 0;
    pr::GgufTensorTable formats;
    formats.weight_formats = {"q4_k"};
    failures +=
        check(pr::table_prices_row(formats), "table counting: a split part's formats alone");
    pr::GgufTensorTable route;
    route.expert_prefill = rt::ExpertPrefill::A16;
    failures += check(pr::table_prices_row(route), "table counting: a split part's banks' route");
    pr::GgufTensorTable int8_route;
    int8_route.expert_prefill_int8 = rt::ExpertPrefill::A16;
    failures +=
        check(pr::table_prices_row(int8_route), "table counting: a split part's banks' int8 route");
    pr::GgufTensorTable bytes;
    bytes.tensor_bytes = 1;
    failures += check(pr::table_prices_row(bytes), "table counting: a whole table's bytes");
    const pr::GgufTensorTable empty;
    failures +=
        check(!pr::table_prices_row(empty), "table counting: an empty table is not counted");
    return failures;
}

// plan_device_of_cuda's typed failure (#891): what serve's busy-card path reads through a
// std::exception (src/serve/plan_offline.cpp), the cause and the message.
int plan_device_error() {
    int failures    = 0;
    const auto read = [](bool memory) {
        try {
            throw pr::PlanDeviceError("GPU 0: cudaSetDevice: out of memory", memory);
        } catch (const std::exception& e) {
            const auto* cuda = dynamic_cast<const pr::PlanDeviceError*>(&e);
            return std::pair<bool, std::string>(cuda != nullptr && cuda->out_of_memory, e.what());
        }
    };
    failures += check(read(true).first, "plan device error: out of memory is read back");
    failures += check(!read(false).first, "plan device error: another cause is not out of memory");
    failures += check(read(true).second == "GPU 0: cudaSetDevice: out of memory",
                      "plan device error: the message is kept");
    return failures;
}

// A gpt-oss artifact's bank formats (#1059): its converter names the banks
// text/layers/<i>/mlp/experts/<x>/gate_up and .../down, beside the stacked biases
// (mlp/experts/gate_up_bias); the banks' format is read, the biases' is not.
int artifact_bank_formats() {
    namespace art = ninferno::artifact;
    int failures  = 0;
    art::Directory dir;
    dir.components["text"].config = gpt_oss_text_config();
    const auto bind               = [&](const std::string& name, const std::string& format,
                          std::vector<std::uint64_t> shape, std::uint64_t elements) {
        art::TensorObject t;
        t.id     = name;
        t.shape  = std::move(shape);
        t.format = format;
        t.layout = std::string(art::layout_name(ninferno::default_weight_layout(
            format == "mxfp4" ? ninferno::QType::MXFP4 : ninferno::QType::BF16)));
        t.bytes  = elements;
        const art::ObjectHandle handle{dir.objects.size()};
        dir.objects.emplace_back(t);
        dir.object_index[name] = handle;
        art::Binding b;
        b.whole_object     = true;
        b.elements         = elements;
        b.parts            = {art::Part{handle, 0, elements}};
        dir.bindings[name] = b;
    };
    for (int x = 0; x < 2; ++x) {
        const std::string ep = "text/layers/0/mlp/experts/" + std::to_string(x) + "/";
        bind(ep + "gate_up", "mxfp4", {5760, 2880}, 5760ULL * 2880);
        bind(ep + "down", "mxfp4", {2880, 2880}, 2880ULL * 2880);
    }
    bind("text/layers/0/mlp/experts/gate_up_bias", "bf16", {2 * 5760}, 2 * 5760);
    bind("text/layers/0/mlp/experts/down_bias", "bf16", {2 * 2880}, 2 * 2880);
    const auto facts = pr::plan_facts_of_directory(dir, "a test directory").model;
    const std::vector<std::string> mxfp4{"mxfp4"};
    failures += check(facts.expert_up_formats == mxfp4,
                      "gpt-oss banks: gate_up's format read (mlp/experts/<x>), not the bias's");
    failures += check(facts.expert_down_formats == mxfp4,
                      "gpt-oss banks: down's format read (mlp/experts/<x>), not the bias's");
    return failures;
}

int main() {
    const int failures = gguf_qwen35() + config_qwen35() + qwen35_moe() + gemma4() + gpt_oss() +
                         nemotron_h() + llama() + model_types() + boards() + artifact_errors() +
                         artifact_expert_layers() + artifact_prefill_policy() +
                         gguf_linear_formats() + artifact_tensor_table() + table_counting() +
                         plan_device_error() + artifact_bank_formats();
    if (failures == 0) { std::cout << "plan facts OK\n"; }
    return failures == 0 ? 0 : 1;
}
