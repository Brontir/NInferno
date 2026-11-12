// The plan without the weights for a named board reads no GPU (#615 W4). `--plan --no-load
// --plan-device NAME` plans a board this machine may not have, so every capacity function the plan
// reaches must answer for that board's class (core/device_class.h, ScopedDeviceClass) and never
// for the device the process happens to run on.
//
// Every CUDA device is hidden from this process before the runtime starts (support/no_device.h):
// a device read that the substitute does not answer (an unsubstituted device_class(), a PerDevice
// cache keyed on the real ordinal, a direct runtime query) throws, as on a machine with no GPU.
// For each target row of gpu_facts.json, under the row's class as the plan builds a named board's
// (synthetic_device_class over plan_device_of_board's facts, as product::gpu_key_of and the
// registered MoE route query build it):
// - plan_workspace_estimate through the registered workspace query (register_plan_workspace_
//   queries) for Gemma 4 31B (the trunk's linears and the head), Gemma 4 26B-A4B (the generic
//   MoE's capacity) and Gemma 4 31B under MTP (its assistant's pass): each must answer from the
//   start's capacity functions ("start"), but for a known escape it reaches (below);
// - plan_rung_bytes, the family planner's KV, states, rings and graphs beside that workspace, for
//   those and for Qwen3.6-27B and Qwen3.6-35B-A3B (the KV and context capacity);
// - dry_run_plan over the board's own PlanDeviceFacts (the ladder, the margins, the fit);
// - the registered MoE route query (register_plan_moe_route_query) for the 35B-A3B's and the
//   26B-A4B's routed layers, at decode and at a prefill width.
// Every answer is the same with the boards taken in the reverse order: no cache keeps the first
// board's answer for the next. The Qwen 3.5 workspace planner (build_workspace_plan) is checked the
// same way in tests/models/qwen3_5/test_workspace_inputs.cpp. CPU, no files.
//
// Known escapes (each kept by name and checked as it is, so the list only shrinks):
// - Gemma 4's attention decode plan reads the host GPU's occupancy (kGemmaDecodePlanEscape);
// - the product path installs no substitute: product::plan_artifact and runtime::dry_run_plan
//   take the named board's PlanDeviceFacts, but the registered workspace query reads
//   device_class() as it finds it, so `--plan-device` sizes Gemma 4's workspace for this
//   machine's GPU (with none, the query is refused and the formula stands). A probe query shows
//   the dry run of a named board asks it with no substituted class.

#include "artifact/formats.h"
#include "core/device_class.h"
#include "core/machine_facts.h"
#include "core/weight_view.h"
#include "product/plan/plan_facts.h"
#include "product/plan/plan_quants.h"
#include "runtime/engine/plan_dry_run.h"
#include "runtime/engine/plan_moe_routes.h"
#include "runtime/engine/plan_workspace.h"
#include "support/no_device.h"

#include <nlohmann/json.hpp>

#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace {

using namespace ninferno;
namespace rt = ninferno::runtime;
namespace pr = ninferno::product;
using json   = nlohmann::json;

constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;

int failures = 0;

void check(bool condition, const std::string& what) {
    if (condition) { return; }
    std::cerr << "FAIL: " << what << '\n';
    ++failures;
}

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.substr(0, prefix.size()) == prefix;
}

// ---- The models: config.json as published, the formats a GGUF's tensor table names.

json gemma_config(int hidden, int layers, int heads, int kv_heads, int global_kv_heads,
                  int intermediate) {
    json types = json::array();
    for (int i = 0; i < layers; ++i) {
        types.push_back((i + 1) % 6 == 0 ? "full_attention" : "sliding_attention");
    }
    return {{"architectures", {"Gemma4ForConditionalGeneration"}},
            {"text_config",
             {{"model_type", "gemma4_text"},
              {"hidden_size", hidden},
              {"num_hidden_layers", layers},
              {"num_attention_heads", heads},
              {"num_key_value_heads", kv_heads},
              {"head_dim", 256},
              {"global_head_dim", 512},
              {"num_global_key_value_heads", global_kv_heads},
              {"sliding_window", 1024},
              {"layer_types", types},
              {"max_position_embeddings", 262144},
              {"vocab_size", 262144},
              {"intermediate_size", intermediate}}}};
}

json qwen_config(bool moe) {
    json text = {{"hidden_size", moe ? 2048 : 5120},
                 {"num_hidden_layers", moe ? 40 : 64},
                 {"num_attention_heads", moe ? 16 : 24},
                 {"num_key_value_heads", moe ? 2 : 4},
                 {"head_dim", 256},
                 {"max_position_embeddings", 262144},
                 {"vocab_size", 248320},
                 {"full_attention_interval", 4},
                 {"linear_num_value_heads", moe ? 32 : 48},
                 {"linear_value_head_dim", 128},
                 {"linear_num_key_heads", 16},
                 {"linear_key_head_dim", 128},
                 {"linear_conv_kernel_dim", 4}};
    if (moe) {
        text["model_type"]                      = "qwen3_5_moe_text";
        text["num_experts"]                     = 256;
        text["num_experts_per_tok"]             = 8;
        text["moe_intermediate_size"]           = 512;
        text["shared_expert_intermediate_size"] = 512;
    } else {
        text["model_type"]        = "qwen3_5_text";
        text["intermediate_size"] = 17408;
    }
    return {{"architectures",
             {moe ? "Qwen3_5MoeForConditionalGeneration" : "Qwen3_5ForConditionalGeneration"}},
            {"model_type", "qwen3_5"},
            {"text_config", text}};
}

rt::PlanModelFacts facts_of(const json& config, std::uint64_t weights) {
    return pr::plan_facts_of_config(config, weights, rt::WeightBasis::FileSize).model;
}

// The trunk's projections, the head and the routed banks in a GGUF's formats (Q4_K, its down
// projections Q6_K, a Q6_K head; Q8_0 where K is no whole number of K-quant blocks, as the
// quantizer keeps such tensors), as plan_facts_of_gguf reads them from the tensor table.
std::string gguf_format(double k, const char* k_quant) {
    return static_cast<std::uint64_t>(k) % 256 == 0 ? k_quant : "q8_g32_fp16";
}

rt::PlanModelFacts with_gguf_formats(rt::PlanModelFacts m) {
    std::string expert_up, expert_down;
    for (rt::PlanLinear& l : m.linears) {
        const bool down = l.role == rt::LinearRole::FfnDown || (l.expert && !l.gated);
        l.format        = gguf_format(l.k, down ? "ggml_q6_k" : "ggml_q4_k");
        if (l.expert) { (l.gated ? expert_up : expert_down) = l.format; }
    }
    m.head_format = "ggml_q6_k";
    m.head_rows   = m.vocab;
    if (m.experts != 0) {
        m.expert_up_formats   = {expert_up};
        m.expert_down_formats = {expert_down};
    }
    return m;
}

// A routed layer kind as plan_facts reads a GGUF's banks (#616 step 7).
rt::PlanExpertLayers routed(QType gate_up, QType down, std::uint64_t hidden, std::uint64_t inter,
                            std::uint32_t layers) {
    rt::PlanExpertLayers l;
    l.gate_up_format = std::string(artifact::format_name(gate_up));
    l.down_format    = std::string(artifact::format_name(down));
    l.gate_up_layout = std::string(artifact::layout_name(default_weight_layout(gate_up)));
    l.down_layout    = std::string(artifact::layout_name(default_weight_layout(down)));
    l.gate_up_k      = hidden;
    l.down_k         = inter;
    l.policy         = rt::PlanExpertPolicy::AllowA8;
    l.layers         = layers;
    l.bytes          = 1000;
    return l;
}

struct Subject {
    std::string name;
    rt::PlanModelFacts facts;
    EngineOptions options; // as ninferno-serve builds them (launch_options), then the speculation
    std::uint32_t concurrency = 1;
    bool start_workspace      = false; // the family's workspace query answers ("start")
    // A named known escape (below) the subject's workspace query reaches: the query is refused
    // with the hidden device's error, and the formula stands. Empty: none.
    std::string known_escape;
};

// Known escape: Gemma 4's attention decode plan (ops/softmax_attention/gemma4/launch.cu,
// gemma4_attention_decode_plan, which gemma4_attention_workspace_bytes and so models::gemma4::
// target_op_bytes call) counts its splits from the resident CTAs of the decode kernel: it sets the
// kernel's shared-memory opt-in (allow_dynamic_smem, cudaFuncSetAttribute on the current device)
// and asks the runtime's occupancy (kernel_resident_ctas_per_sm,
// cudaOccupancyMaxActiveBlocksPerMultiprocessor, cached by the real ordinal), then multiplies by
// the substitute's SMs; the Volta SIMT decode (gemma4_volta_decode_resident_ctas) does the same.
// Under a substituted board the splits, and so the partials' workspace, follow the host GPU's
// occupancy; with no GPU the Gemma 4 query is refused and the formula stands. The facts' answer
// (core/kernel_occupancy.h resident_ctas_per_sm) needs these kernels in the kernel-facts
// snapshots, which carry none of them. Every Gemma 4 subject reaches it.
constexpr std::string_view kGemmaDecodePlanEscape = "gemma4_attention_decode_plan";

EngineOptions served(std::uint32_t context, std::uint32_t streams, SpeculativeBackend backend) {
    EngineOptions o =
        pr::launch_options({.max_context = context, .max_concurrency = streams}).options;
    o.speculative.backend      = backend;
    o.speculative.draft_tokens = backend == SpeculativeBackend::None ? 0 : 3;
    return o;
}

// The routed models among subjects(), by index.
constexpr std::size_t kGemmaMoe = 1, kQwenMoe = 4;

std::vector<Subject> subjects() {
    std::vector<Subject> out;
    const rt::PlanModelFacts gemma31b =
        with_gguf_formats(facts_of(gemma_config(5376, 60, 32, 16, 4, 21504), 18 * kGiB));
    const std::string escape(kGemmaDecodePlanEscape);
    out.push_back({"Gemma 4 31B Q4_K", gemma31b, served(32768, 2, SpeculativeBackend::None), 2,
                   true, escape});
    {
        json config                                    = gemma_config(2816, 30, 16, 8, 2, 2112);
        config["text_config"]["num_experts"]           = 128;
        config["text_config"]["top_k_experts"]         = 8;
        config["text_config"]["moe_intermediate_size"] = 704;
        out.push_back({"Gemma 4 26B-A4B Q4_K", with_gguf_formats(facts_of(config, 16 * kGiB)),
                       served(16384, 1, SpeculativeBackend::None), 1, true, escape});
    }
    {
        // Its assistant (a separate drafter, BF16 as published): the MTP round's pass.
        rt::PlanModelFacts mtp = gemma31b;
        mtp.drafter_geometry   = rt::PlanDrafterGeometry{.hidden             = 1024,
                                                         .intermediate       = 8192,
                                                         .head_rows          = 262144,
                                                         .head_formats       = {"bf16"},
                                                         .projection_formats = {"bf16"}};
        mtp.drafter_bytes      = kGiB;
        out.push_back({"Gemma 4 31B Q4_K, MTP 3", mtp, served(32768, 1, SpeculativeBackend::Mtp), 1,
                       true, escape});
    }
    out.push_back({"Qwen3.6-27B Q4_K", facts_of(qwen_config(false), 16 * kGiB),
                   served(65536, 4, SpeculativeBackend::None), 4, false});
    out.push_back({"Qwen3.6-35B-A3B Q4_K", facts_of(qwen_config(true), 20 * kGiB),
                   served(32768, 2, SpeculativeBackend::Mtp), 2, false});
    return out;
}

// ---- The boards: every target row of the registry, as --plan-device names it.

struct NamedBoard {
    std::string name;
    rt::PlanDeviceFacts device;
    DeviceClass substitute;
};

// The class the plan reads for a named board (product::gpu_key_of, plan_moe_routes.cpp class_of):
// the chip from the PCI id or the SM count, the architecture's limits, the table's facts.
DeviceClass class_of(const rt::PlanDeviceFacts& device) {
    DeviceClass d =
        synthetic_device_class(device.pci_device_id, device.compute_capability / 10,
                               device.compute_capability % 10, device.sms, 0, device.total_bytes);
    d.facts = machine_facts_of(d);
    return d;
}

std::vector<NamedBoard> boards() {
    std::vector<NamedBoard> out;
    for (const GpuFactsRow& row : gpu_facts_table()) {
        if (!row.target) { continue; }
        // By its PCI id: a name with a comma ("GeForce GTX 1650 (TU116, GDDR6)") reads as a board
        // fact after it.
        char id[16];
        std::snprintf(id, sizeof id, "0x%04X", static_cast<unsigned>(row.pci_device_id));
        std::string error;
        const auto device = pr::plan_device_of_board(std::string(id) + ",linux,headless", &error);
        check(device.has_value() && device->name == row.name,
              row.name + ": --plan-device names it by " + id + " (" + error + ")");
        if (!device) { continue; }
        out.push_back({row.name, *device, class_of(*device)});
    }
    return out;
}

// ---- What the plan reaches, under the board's class.

std::string text_of(const rt::PlanRungBytes& b) {
    return "kv " + std::to_string(b.kv) + ", states " + std::to_string(b.states) + ", rings " +
           std::to_string(b.rings) + ", workspace " + std::to_string(b.workspace) + " (" +
           b.workspace_basis + "), graphs " + std::to_string(b.graphs) + ", fixed " +
           std::to_string(b.fixed);
}

std::string text_of(const rt::DryRunPlan& p) {
    std::string out = std::string(p.planned ? "planned" : "not planned: " + p.reason) +
                      (p.fits ? ", fits rung " + std::to_string(p.rung) : ", no fit") + " of " +
                      std::to_string(p.ladder.size());
    if (const rt::DryRunRung* chosen = p.chosen()) {
        out +=
            ": context " + std::to_string(chosen->rung.max_context) + ", " + text_of(chosen->bytes);
    }
    return out;
}

std::string text_of(const std::optional<rt::PlanExpertRoute>& r) {
    if (!r) { return "no answer"; }
    return std::to_string(static_cast<int>(r->gate_up)) + "/" +
           std::to_string(static_cast<int>(r->down)) + " by " + r->authority;
}

// The workspace a subject's query answers under the board's class: the start's, or for a subject
// that reaches a known escape, the query refused by the hidden device (the formula then stands).
void check_workspace(const NamedBoard& b, const Subject& s, const std::string& basis,
                     const std::string& where) {
    const std::string what = b.name + ", " + s.name + ", " + where;
    if (!s.known_escape.empty()) {
        check(starts_with(basis, "formula: an Op's capacity query refused the facts") &&
                  basis.find("cudaErrorNoDevice") != std::string::npos,
              what + ": the known escape " + s.known_escape +
                  " is reached (if it is gone, remove it): '" + basis + "'");
    } else if (s.start_workspace) {
        check(starts_with(basis, "start"),
              what + ": the workspace is the start's, not '" + basis + "'");
    }
}

// One board: every answer, keyed by what it is, for the order check.
std::map<std::string, std::string> run_board(const NamedBoard& b, const std::vector<Subject>& all) {
    std::map<std::string, std::string> out;
    const auto guarded = [&](const std::string& what, const std::function<std::string()>& f) {
        try {
            out[what] = f();
        } catch (const std::exception& error) {
            check(false, b.name + ", " + what + ": " + error.what());
            out[what] = std::string("threw: ") + error.what();
        }
    };
    {
        const ScopedDeviceClass scoped(b.substitute);
        for (const Subject& s : all) {
            const rt::PlanRung rung{.max_context   = s.options.max_context,
                                    .kv_pages      = 0,
                                    .cached_states = 1,
                                    .prefill_chunk = s.options.prefill_chunk};
            guarded(s.name + " workspace", [&] {
                std::string basis;
                const std::uint64_t bytes = rt::plan_workspace_estimate(
                    s.facts, rung, s.options, s.concurrency, nullptr, &basis);
                check_workspace(b, s, basis, "the estimate");
                return std::to_string(bytes) + " (" + basis + ")";
            });
            guarded(s.name + " rung", [&] {
                const rt::PlanRungBytes bytes =
                    rt::plan_rung_bytes(s.facts, rung, s.options, s.concurrency, false);
                check_workspace(b, s, bytes.workspace_basis, "the rung");
                check(bytes.kv > 0, b.name + ", " + s.name + ": the rung holds KV pages");
                return text_of(bytes);
            });
            guarded(s.name + " dry run", [&] {
                const rt::DryRunPlan plan = rt::dry_run_plan(s.facts, b.device, s.options);
                check(plan.planned && plan.estimate_basis == "family_planner",
                      b.name + ", " + s.name + ": planned by the family planner (" +
                          plan.estimate_basis + ": " + plan.reason + ")");
                // A small board may fit no rung (the weights alone): the ladder is still sized.
                check(!plan.ladder.empty(), b.name + ", " + s.name + ": a ladder is sized");
                if (const rt::DryRunRung* chosen = plan.chosen()) {
                    check_workspace(b, s, chosen->bytes.workspace_basis, "the chosen rung");
                }
                return text_of(plan);
            });
        }
    }
    // The MoE route query substitutes the board's class itself (moe_route_query): asked with no
    // substitute in place, as the configuration choice asks it.
    const auto& moe_routes                         = rt::plan_moe_route_query();
    const std::vector<const Subject*> moe          = {&all[kGemmaMoe], &all[kQwenMoe]};
    const std::vector<rt::PlanExpertLayers> layers = {
        routed(QType::GGML_Q4_K, QType::Q8_G32_FP16, 2816, 704, 30),
        routed(QType::GGML_Q4_K, QType::GGML_Q6_K, 2048, 512, 40)};
    for (std::size_t i = 0; i < moe.size(); ++i) {
        for (const std::uint32_t tokens : {1U, 4U, 512U}) {
            for (const ActivationMode mode : {ActivationMode::Fast, ActivationMode::Exact}) {
                const std::string what = moe[i]->name + " MoE route at " + std::to_string(tokens) +
                                         (mode == ActivationMode::Exact ? " exact" : " fast");
                guarded(what, [&] {
                    const auto route = moe_routes(b.device, moe[i]->facts, layers[i], tokens, mode);
                    check(route.has_value(), b.name + ", " + what + ": answered");
                    return text_of(route);
                });
            }
        }
    }
    return out;
}

} // namespace

int main() {
    if (!hide_cuda_devices()) {
        std::cerr << "FAIL: a CUDA device is visible to the test (run it with "
                     "CUDA_VISIBLE_DEVICES=-1)\n";
        return 1;
    }
    rt::register_plan_workspace_queries();
    rt::register_plan_moe_route_query();
    const std::vector<Subject> all      = subjects();
    const std::vector<NamedBoard> every = boards();
    check(every.size() >= 50, "the registry has its target boards");

    // Known escape: the product path installs no substitute (see the header). A probe in place of
    // the workspace query sees what class the dry run of a named board sizes the workspace under:
    // none. When the plan substitutes the board's class, this fails: drop the check and the
    // ScopedDeviceClass of run_board's dry runs.
    if (!every.empty()) {
        std::optional<std::string> seen; // the substitute's key the query ran under ("" none)
        rt::set_plan_workspace_query(
            [&](const rt::PlanModelFacts&, const rt::PlanRung&, const EngineOptions&, std::uint32_t,
                const ninferno::models::qwen3_5::detail::SequencePlanImpl*) {
                const std::string* key = substituted_device_class_key();
                seen                   = key != nullptr ? *key : std::string();
                return rt::PlanWorkspaceAnswer{};
            });
        (void)rt::dry_run_plan(all[0].facts, every[0].device, all[0].options);
        check(seen.has_value(), "known escape: the dry run asks the workspace query");
        check(seen.has_value() && seen->empty(),
              "known escape: the dry run of a named board now sizes its workspace under a "
              "substituted class (remove the known escape)");
        std::cout << "known escape: the dry run of " << every[0].name
                  << " asks the workspace query with "
                  << (seen && seen->empty() ? "no substituted class" : "a substituted class")
                  << '\n';
        rt::register_plan_workspace_queries();
    }

    std::vector<std::map<std::string, std::string>> forward;
    for (const NamedBoard& b : every) { forward.push_back(run_board(b, all)); }
    for (std::size_t i = every.size(); i-- > 0;) {
        const auto again = run_board(every[i], all);
        for (const auto& [what, answer] : forward[i]) {
            const auto it = again.find(what);
            check(it != again.end() && it->second == answer,
                  every[i].name + ", " + what + ": the same answer after the other boards (" +
                      answer + " then " + (it == again.end() ? "none" : it->second) + ")");
        }
    }

    // What each board's plan reads differs where the boards differ: the substitute is read.
    std::map<std::string, std::size_t> distinct;
    for (const auto& answers : forward) {
        for (const auto& [what, answer] : answers) { (void)distinct[what + "|" + answer]; }
    }
    std::cout << every.size() << " boards, " << all.size() << " models, "
              << (forward.empty() ? 0 : forward.front().size()) << " answers a board, "
              << distinct.size() << " distinct\n";
    if (!forward.empty()) {
        for (const auto& [what, answer] : forward.front()) {
            std::cout << "  " << every.front().name << ", " << what << ": " << answer << '\n';
        }
    }
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "plan device class: pass\n";
    return 0;
}
