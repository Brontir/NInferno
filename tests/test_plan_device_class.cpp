// The plan without the weights for a named board reads no GPU (#615 W4). `--plan --no-load
// --plan-device NAME` plans a board this machine may not have, so every capacity function the plan
// reaches must answer for that board's class (core/device_class.h, ScopedDeviceClass) and never
// for the device the process happens to run on.
//
// Every CUDA device is hidden from this process before the runtime starts (support/no_device.h):
// a device read that the substitute does not answer (an unsubstituted device_class(), a PerDevice
// cache keyed on the real ordinal, a direct runtime call) throws, as on a machine with no GPU. For
// each target row of gpu_facts.json, under the class the plan builds for the named board
// (support/named_board.h, over plan_device_of_board's facts):
// - each capacity function the registered Gemma 4 workspace query calls (runtime/engine/
//   plan_workspace.cpp gemma4_workspace: models::gemma4::target_op_bytes, generic_moe_bytes,
//   gemma_drafter_pass), called directly for Gemma 4 31B, Gemma 4 26B-A4B and Gemma 4 31B under
//   MTP at the columns a start sizes: the trunk's and the head's linear and GeGLU capacities, the
//   generic MoE's (ops::moe_workspace_bytes over described banks) and the fused MoE's, sampling,
//   the greedy acceptance, the assistant's BF16 projections and head. Each must answer; only
//   ops::gemma4_attention_workspace_bytes may throw, and it must, with cudaErrorNoDevice (the
//   known escape below);
// - the registered workspace query itself (plan_workspace_estimate) and plan_rung_bytes, for the
//   three Gemma 4 subjects and for
//   Qwen3.6-27B and Qwen3.6-35B-A3B (the family planner's KV, states, rings, graphs and context
//   capacity; Qwen 3.5's query (#615 W2) declines facts without an artifact's directory, so its
//   workspace is the formula, with that reason); the Gemma 4 query refused by the hidden device
//   (the escape);
// - dry_run_plan over the board's own PlanDeviceFacts with no class substituted by the test: the
//   dry run substitutes the named board's itself (#829), so it plans as above, by the family
//   planner, a ladder sized; a probe query in place of the workspace query sees it run under the
//   board's class (runtime::plan_device_class);
// - the registered MoE route query (register_plan_moe_route_query) for the 26B-A4B's and the
//   35B-A3B's routed layers at 1, 4 and 512 tokens under fast and exact activations: answered.
// Every answer is the same with the boards taken in the reverse order (no cache keeps the first
// board's answer for the next), and each Gemma 4 subject's capacities and each dry run differ
// between boards (the substitute is read). The Qwen 3.5 workspace planner (build_workspace_plan) is
// checked the same way in tests/models/qwen3_5/test_workspace_inputs.cpp. CPU, no files.
//
// Not escapes, by design (#884): w4a8_ws_extra and prefill_a8_format's device read
// (qx_prefill_a8_pays, made on every call; only the Q4/Q5/Q6 G64 answer depends on it) answer
// false where no device class can be read, instead of throwing. Under a substitute they
// read the substitute; outside one, on a host with no GPU, those A8 planes and the producer-warp
// tile are left out of the size, not refused. A new caller outside the substitute's
// scope gets that answer silently: keep the plan's reads inside plan_device_class.
//
// Known escape (kept by name and checked as it is, so the list only shrinks):
// - #830: Gemma 4's attention decode plan reads the host GPU (kDecodePlanEscape, below). While it
//   escapes, the registered Gemma 4 query never answers "start" under a substitute: the pieces
//   above are what is checked of it.
//
// The escaped calls print their CUDA_CHECK line to stderr; the test silences stderr around them
// (QuietStderr) and writes its own failures to stdout.

#include "artifact/formats.h"
#include "core/cuda_error.h"
#include "core/device_class.h"
#include "core/machine_facts.h"
#include "core/weight_view.h"
#include "models/qwen3_5/program/planning/sequence_bytes.h"
#include "models/qwen3_5/program/planning/sequence_plan.h"
#include "ninferno/ops/gemma4_attention.h"
#include "ninferno/ops/gemma4_moe.h"
#include "ninferno/ops/linear.h"
#include "ninferno/ops/linear_swiglu.h"
#include "ninferno/ops/moe.h"
#include "ninferno/ops/sampling.h"
#include "ninferno/ops/speculative_round.h"
#include "product/plan/plan_facts.h"
#include "product/plan/plan_quants.h"
#include "runtime/engine/plan_dry_run.h"
#include "runtime/engine/plan_moe_routes.h"
#include "runtime/engine/plan_workspace.h"
#include "support/named_board.h"
#include "support/no_device.h"

#include <nlohmann/json.hpp>

#include <fcntl.h>
#ifdef _WIN32
#    include <io.h>
#else
#    include <unistd.h>
#endif

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <functional>
#include <iostream>
#include <map>
#include <memory>
#include <optional>
#include <set>
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

// To stdout: stderr is silenced around the escaped calls (QuietStderr).
void check(bool condition, const std::string& what) {
    if (condition) { return; }
    std::cout << "FAIL: " << what << std::endl;
    ++failures;
}

bool starts_with(std::string_view text, std::string_view prefix) {
    return text.substr(0, prefix.size()) == prefix;
}

// While alive, stderr (descriptor 2) writes nowhere: what cuda_check prints for an escaped call.
class QuietStderr {
public:
    QuietStderr() {
        std::fflush(stderr);
#ifdef _WIN32
        saved_          = _dup(2);
        const int empty = _open("NUL", _O_WRONLY);
        if (empty >= 0) {
            _dup2(empty, 2);
            _close(empty);
        }
#else
        saved_          = dup(2);
        const int empty = open("/dev/null", O_WRONLY);
        if (empty >= 0) {
            dup2(empty, 2);
            close(empty);
        }
#endif
    }

    ~QuietStderr() {
        std::fflush(stderr);
        if (saved_ < 0) { return; }
#ifdef _WIN32
        _dup2(saved_, 2);
        _close(saved_);
#else
        dup2(saved_, 2);
        close(saved_);
#endif
    }

    QuietStderr(const QuietStderr&)            = delete;
    QuietStderr& operator=(const QuietStderr&) = delete;

private:
    int saved_ = -1;
};

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
    bool gemma                = false; // the registered Gemma 4 workspace query sizes it
};

// #830, the known escape: Gemma 4's attention decode plan (ops/softmax_attention/gemma4/launch.cu,
// gemma4_attention_decode_plan, which ops::gemma4_attention_workspace_bytes calls, and so
// models::gemma4::target_op_bytes and drafter_pass_bytes) counts its splits from the resident CTAs
// of the decode kernel: it sets the kernel's shared-memory opt-in (allow_dynamic_smem,
// cudaFuncSetAttribute on the current device) and asks the runtime's occupancy
// (kernel_resident_ctas_per_sm, cudaOccupancyMaxActiveBlocksPerMultiprocessor, cached by the real
// ordinal), then multiplies by the substitute's SMs; the Volta SIMT decode
// (gemma4_volta_decode_resident_ctas) does the same. Under a substituted board the splits, and so
// the partials' workspace, follow the host GPU's occupancy; with no GPU it throws, the Gemma 4
// query is refused and the formula stands. The facts' answer (core/kernel_occupancy.h
// resident_ctas_per_sm) needs these kernels in the kernel-facts snapshots, which carry none.
constexpr std::string_view kDecodePlanEscape = "#830 (gemma4_attention_decode_plan)";

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
    out.push_back(
        {"Gemma 4 31B Q4_K", gemma31b, served(32768, 2, SpeculativeBackend::None), 2, true});
    {
        json config                                    = gemma_config(2816, 30, 16, 8, 2, 2112);
        config["text_config"]["num_experts"]           = 128;
        config["text_config"]["top_k_experts"]         = 8;
        config["text_config"]["moe_intermediate_size"] = 704;
        out.push_back({"Gemma 4 26B-A4B Q4_K", with_gguf_formats(facts_of(config, 16 * kGiB)),
                       served(16384, 1, SpeculativeBackend::None), 1, true});
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
        out.push_back(
            {"Gemma 4 31B Q4_K, MTP 3", mtp, served(32768, 1, SpeculativeBackend::Mtp), 1, true});
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
        // The facts the class reads are named_board_facts' (the Qwen planner test's boards).
        const rt::PlanDeviceFacts own = named_board_facts(row);
        check(device->pci_device_id == own.pci_device_id &&
                  device->compute_capability == own.compute_capability && device->sms == own.sms &&
                  device->total_bytes == own.total_bytes,
              row.name + ": --plan-device's facts are named_board_facts'");
        out.push_back({row.name, *device, named_board_class(*device)});
    }
    return out;
}

// ---- The Gemma 4 query's capacity functions, called one by one.

// A weight described by its format and shape alone, in the converter's layout, at an address never
// read (plan_workspace.cpp's DescribedWeight).
struct DescribedWeight {
    WeightParent parent;
    WeightView view;
    Weight weight;

    DescribedWeight(QType qtype, std::uint64_t n, std::uint64_t k) {
        parent.data                  = reinterpret_cast<const std::byte*>(std::uintptr_t{1} << 32);
        const std::uint64_t shape[2] = {n, k};
        parent.geometry              = weight_geometry(qtype, default_weight_layout(qtype), shape);
        view.shape                   = {n, k};
        view.parts                   = {WeightRegion{&parent, 0, n * k}};
        weight                       = native_weight(view, 0.0F);
    }

    DescribedWeight(const DescribedWeight&)            = delete;
    DescribedWeight& operator=(const DescribedWeight&) = delete;
};

// The policy a start's projection carries for a stored format (plan_workspace.cpp's policy_of).
ops::LinearPolicy policy_of(QType qtype) {
    if (qtype == QType::BF16 || qtype == QType::FP32) { return ops::LinearPolicy::A16Only; }
    if (qtype == QType::NVFP4) { return ops::LinearPolicy::AllowA4; }
    return ops::LinearPolicy::AllowA8;
}

struct Projection {
    QType qtype{};
    std::int32_t n = 0, k = 0;
    ops::LinearPolicy policy = ops::LinearPolicy::A16Only;
    bool geglu               = false;
};

// Every capacity function the Gemma 4 query calls for `s` (gemma4_workspace and what it calls),
// at the columns a start sizes: one prefill chunk, the batch, an MTP round. Each answers (its
// bytes, joined); ops::gemma4_attention_workspace_bytes must throw the hidden device's error
// (#830), and no other piece may.
std::string gemma_pieces(const NamedBoard& b, const Subject& s) {
    const rt::PlanModelFacts& m = s.facts;
    const auto chunk =
        static_cast<std::int32_t>(std::min(s.options.prefill_chunk, s.options.max_context));
    const auto batch                  = static_cast<std::int32_t>(s.concurrency);
    const bool mtp                    = s.options.speculative.backend == SpeculativeBackend::Mtp;
    const auto context                = static_cast<std::int32_t>(s.options.max_context);
    std::vector<std::int32_t> columns = {1, batch, chunk};
    // The MTP round as a start lays it out (plan_workspace.cpp gemma4_workspace): the copy window
    // resolved by Gemma 4's rule (an automatic one widens the 3 drafts asked to its own window),
    // the round's rows and columns from the resolved launch windows. The configured depth's
    // columns too.
    std::int32_t width = 1, rows = 0;
    if (mtp) {
        namespace detail        = models::qwen3_5::detail;
        const bool copy_rounds  = detail::family_copy_rounds(models::StateFamily::Gemma4);
        EngineOptions planned   = s.options;
        planned.max_concurrency = s.concurrency;
        planned.speculative     = detail::resolve_copy_window(planned.speculative, copy_rounds);
        const detail::SequencePlanningInputs in =
            detail::sequence_planning_inputs(planned, copy_rounds, 0, false);
        detail::SequencePlanImpl plan;
        plan.max_concurrency         = in.max_concurrency;
        plan.draft_window            = in.draft_window;
        plan.launch_windows          = in.launch_windows;
        plan.launch_window_batches   = in.launch_window_batches;
        plan.speculative_batch_limit = in.speculative_batch_limit;
        plan.speculative_backend     = in.speculative_backend;
        width                        = static_cast<std::int32_t>(plan.draft_window) + 1;
        rows = static_cast<std::int32_t>(detail::speculative_round_batches(plan));
        check(width >= static_cast<std::int32_t>(s.options.speculative.draft_tokens) + 1 &&
                  rows > 0,
              b.name + ", " + s.name + ": the resolved MTP round covers the asked depth");
        columns.push_back(batch *
                          (static_cast<std::int32_t>(s.options.speculative.draft_tokens) + 1));
        columns.push_back(static_cast<std::int32_t>(detail::speculative_round_columns(plan)));
    }
    std::string out =
        "round width " + std::to_string(width) + ", rows " + std::to_string(rows) + "; ";
    const auto piece = [&](const std::string& what, const std::function<std::size_t()>& f) {
        try {
            out += what + " " + std::to_string(f()) + "; ";
        } catch (const std::exception& error) {
            check(false, b.name + ", " + s.name + ", " + what + ": " + error.what());
            out += what + " threw; ";
        }
    };

    std::vector<Projection> trunk;
    for (const rt::PlanLinear& l : m.linears) {
        if (l.expert) { continue; }
        const QType q = artifact::parse_format(l.format);
        trunk.push_back({q, static_cast<std::int32_t>(l.n), static_cast<std::int32_t>(l.k),
                         policy_of(q), l.role == rt::LinearRole::FfnUp && l.gated});
    }
    const QType head_q = artifact::parse_format(m.head_format);
    const Projection head{head_q, static_cast<std::int32_t>(m.head_rows),
                          static_cast<std::int32_t>(m.hidden), policy_of(head_q)};
    const auto vocab = static_cast<std::int32_t>(m.vocab);
    for (const std::int32_t c : columns) {
        const std::string at = " at " + std::to_string(c);
        for (const Projection& p : trunk) {
            const std::string name = std::to_string(p.n) + "x" + std::to_string(p.k);
            piece("linear " + name + at, [&] {
                return ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, 1, c);
            });
            if (!p.geglu) { continue; }
            for (const auto policy : {p.policy, ops::LinearPolicy::A16Only}) {
                piece("geglu " + name + at, [&] {
                    return ops::linear_geglu_workspace_capacity_bytes(p.qtype, p.n, p.k, policy, 1,
                                                                      c);
                });
            }
        }
        for (const auto policy : {head.policy, ops::LinearPolicy::A16Only}) {
            piece("head" + at, [&] {
                return ops::linear_workspace_capacity_bytes(head.qtype, head.n, head.k, policy, 1,
                                                            c);
            });
        }
        if (m.experts != 0) {
            piece("gemma4_moe" + at, [&] { return ops::gemma4_moe_workspace_bytes(1, c); });
            // The generic MoE layer (plan_workspace.cpp generic_moe_bytes).
            piece("generic moe" + at, [&] {
                const auto e        = static_cast<std::uint64_t>(m.experts),
                           h        = static_cast<std::uint64_t>(m.hidden);
                std::uint64_t inter = 0;
                for (const rt::PlanLinear& l : m.linears) {
                    if (l.expert && !l.gated) { inter = static_cast<std::uint64_t>(l.k); }
                }
                const QType up_q   = artifact::parse_format(m.expert_up_formats.front());
                const QType down_q = artifact::parse_format(m.expert_down_formats.front());
                const auto router  = std::make_unique<DescribedWeight>(QType::BF16, e, h);
                const auto up      = std::make_unique<DescribedWeight>(up_q, e * 2 * inter, h);
                const auto down    = std::make_unique<DescribedWeight>(down_q, e * h, inter);
                ops::MoeWeights weights;
                weights.router        = router->weight;
                weights.gate_up       = up->weight;
                weights.down          = down->weight;
                weights.expert_policy = ops::moe_policy(policy_of(up_q), policy_of(down_q));
                ops::MoeConfig config;
                config.top_k          = static_cast<std::int32_t>(m.experts_used);
                config.scoring        = ops::MoeScoring::Softmax;
                config.activation     = ops::MoeActivation::GeGluTanh;
                config.gate_up_layout = ops::MoeGateUpLayout::Concatenated;
                return ops::moe_workspace_bytes(weights, config, 1, c);
            });
        }
    }
    piece("sampling", [&] { return ops::sampling_workspace_capacity_bytes(vocab, 1, batch); });
    if (mtp) {
        piece("greedy acceptance", [&] {
            return ops::speculative_accept_greedy_drafts_workspace_capacity_bytes(
                vocab, 1, width - 1, 1, rows);
        });
    }
    if (mtp && m.drafter_geometry) {
        // The assistant's projections and head, A16 (plan_workspace.cpp gemma_drafter_pass).
        const rt::PlanDrafterGeometry& d = *m.drafter_geometry;
        const auto h                     = static_cast<std::int32_t>(d.hidden),
                   bb                    = static_cast<std::int32_t>(m.hidden),
                   ir                    = static_cast<std::int32_t>(d.intermediate),
                   q_sl = static_cast<std::int32_t>(m.query_heads * m.sliding_key_dim),
                   q_fu = static_cast<std::int32_t>(m.query_heads * m.full_key_dim);
        const DescribedWeight gate_up(QType::BF16, 2 * static_cast<std::uint64_t>(ir), d.hidden);
        const std::vector<Projection> projections = {
            {QType::BF16, h, 2 * bb},
            {QType::BF16, bb, h},
            {QType::BF16, q_sl, h},
            {QType::BF16, h, q_sl},
            {QType::BF16, q_fu, h},
            {QType::BF16, h, q_fu},
            {QType::BF16, 2 * ir, h, ops::LinearPolicy::A16Only,
             ops::linear_geglu_supports(gate_up.weight)},
            {QType::BF16, h, ir},
            {QType::BF16, static_cast<std::int32_t>(d.head_rows), h}};
        for (const std::int32_t pass : {1, batch, rows}) {
            for (const Projection& p : projections) {
                const std::string name = "assistant " + std::to_string(p.n) + "x" +
                                         std::to_string(p.k) + " at " + std::to_string(pass);
                piece(name, [&] {
                    return ops::linear_workspace_capacity_bytes(p.qtype, p.n, p.k, p.policy, 1,
                                                                pass);
                });
                if (p.geglu) {
                    piece(name + " geglu", [&] {
                        return ops::linear_geglu_workspace_capacity_bytes(p.qtype, p.n, p.k,
                                                                          p.policy, 1, pass);
                    });
                }
            }
        }
    }

    // #830: the attention's capacity, of the target's sliding and full layers (and the
    // assistant's, which takes the same geometry), reads the host GPU and throws with none.
    struct Layer {
        std::int32_t head_dim, kv_heads, window;
    };

    const std::vector<Layer> layers = {
        {static_cast<std::int32_t>(m.sliding_key_dim),
         static_cast<std::int32_t>(m.sliding_kv_heads_total / m.sliding_layers),
         static_cast<std::int32_t>(m.window)},
        {static_cast<std::int32_t>(m.full_key_dim),
         static_cast<std::int32_t>(m.full_kv_heads_total / m.full_layers), 0}};
    for (const Layer& l : layers) {
        bool escaped = false;
        try {
            const QuietStderr quiet;
            (void)ops::gemma4_attention_workspace_bytes(
                l.head_dim, static_cast<std::int32_t>(m.query_heads), l.kv_heads,
                ops::Gemma4AttentionEnvelope{static_cast<std::uint32_t>(context), 1}, l.window);
        } catch (const CudaError& error) {
            escaped = error.code() == cudaErrorNoDevice;
        } catch (const std::exception&) {}
        check(escaped, b.name + ", " + s.name + ", attention D=" + std::to_string(l.head_dim) +
                           ": the known escape " + std::string(kDecodePlanEscape) +
                           " reads the hidden device (if it no longer does, the escape is gone: "
                           "remove it and require \"start\" of the Gemma 4 query)");
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

// The workspace the registered query answers under the board's class: the formula for Qwen 3.5
// (its query declines facts without the artifact's directory, #615 W2); for Gemma 4 the query
// refused by the hidden device (#830, whose reach gemma_pieces pins to the attention's capacity).
// Qwen 3.5's query declines facts that carry no converted artifact's directory
// (plan_workspace.cpp).
constexpr std::string_view kQwenWithoutDirectory =
    "formula: the start's forms are a converted artifact's bindings, not in these facts";

void check_workspace(const NamedBoard& b, const Subject& s, const std::string& basis,
                     const std::string& where) {
    const std::string what = b.name + ", " + s.name + ", " + where;
    if (s.gemma) {
        check(starts_with(basis, "formula: an Op's capacity query refused the facts") &&
                  basis.find("cudaErrorNoDevice") != std::string::npos,
              what + ": the Gemma 4 query is refused by the known escape " +
                  std::string(kDecodePlanEscape) + ", not '" + basis + "'");
    } else {
        check(basis == kQwenWithoutDirectory,
              what + ": the formula without the artifact's directory, not '" + basis + "'");
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
            // The registered Gemma 4 query reaches #830 on every call: its CUDA_CHECK lines are
            // silenced.
            std::optional<QuietStderr> quiet;
            if (s.gemma) {
                quiet.emplace();
                guarded(s.name + " capacities", [&] { return gemma_pieces(b, s); });
            }
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
        }
    }
    // The dry run substitutes the named board's class itself (#829): run with none in place, as
    // product::plan_artifact runs it.
    for (const Subject& s : all) {
        std::optional<QuietStderr> quiet;
        if (s.gemma) { quiet.emplace(); }
        {
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
        std::cout << "FAIL: a CUDA device is visible to the test (run it with "
                     "CUDA_VISIBLE_DEVICES=-1)"
                  << std::endl;
        return 1;
    }
    rt::register_plan_workspace_queries();
    rt::register_plan_moe_route_query();
    const std::vector<Subject> all      = subjects();
    const std::vector<NamedBoard> every = boards();
    check(every.size() >= 50, "the registry has its target boards");

    // #829: the dry run of a named board sizes its rungs under the board's class. A probe in place
    // of the workspace query sees the class every call of it runs under: the board's
    // plan_device_class, on every board; and none is left in place after the dry run.
    {
        std::vector<std::string> seen; // the substitute's key of each call ("" none)
        rt::set_plan_workspace_query(
            [&](const rt::PlanModelFacts&, const rt::PlanRung&, const EngineOptions&, std::uint32_t,
                const ninferno::models::qwen3_5::detail::SequencePlanImpl*) {
                const std::string* key = substituted_device_class_key();
                seen.push_back(key != nullptr ? *key : std::string());
                return rt::PlanWorkspaceAnswer{};
            });
        for (const NamedBoard& b : every) {
            seen.clear();
            (void)rt::dry_run_plan(all[0].facts, b.device, all[0].options);
            const std::string want = device_class_key(rt::plan_device_class(b.device));
            check(!seen.empty() && std::all_of(seen.begin(), seen.end(),
                                               [&](const std::string& k) { return k == want; }),
                  b.name + ": the dry run sizes its rungs under the named board's class (#829)");
            check(substituted_device_class_key() == nullptr,
                  b.name + ": no class is left substituted after the dry run");
        }
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

    // The substitute is read: the Gemma 4 capacities (the routes the linear capacity queries take
    // on each board's class) differ between boards, as each dry run does (the boards' memory).
    // Printed: how many distinct answers each key takes over the boards.
    std::map<std::string, std::set<std::string>> values;
    for (const auto& answers : forward) {
        for (const auto& [what, answer] : answers) { values[what].insert(answer); }
    }
    for (const auto& [what, answers] : values) {
        std::cout << "  " << answers.size() << " distinct over the boards: " << what << '\n';
        if (what.find(" capacities") != std::string::npos ||
            what.find(" dry run") != std::string::npos) {
            check(answers.size() > 1, what + ": differs between boards");
        }
    }

    std::cout << every.size() << " boards, " << all.size() << " models, "
              << (forward.empty() ? 0 : forward.front().size()) << " answers a board\n";
    if (!forward.empty()) {
        for (const auto& [what, answer] : forward.front()) {
            std::cout << "  " << every.front().name << ", " << what << ": " << answer.substr(0, 400)
                      << '\n';
        }
    }
    if (failures != 0) {
        std::cout << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "plan device class: pass\n";
    return 0;
}
