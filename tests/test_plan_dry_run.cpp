// The plan without the weights (src/runtime/engine/plan_dry_run.h, src/product/plan/): the seams a
// start shares with it give the same ladder, the dry run is total and monotone over many models,
// cards, contexts, streams and codecs, the quant rule picks the largest that fits, the
// family table agrees with the Engine's, ninferno-ui's launch options are ninferno-serve's, and
// the speculation report follows the start's order. CPU, no network, no GPU.

#include "convert/drafters.h"
#include "product/plan/plan_choice.h"
#include "product/plan/plan_facts.h"
#include "product/plan/plan_quants.h"
#include "product/plan/plan_speculation.h"
#include "runtime/engine/family_registry.h"
#include "runtime/engine/kv_capacity.h"
#include "runtime/engine/plan_dry_run.h"
#include "runtime/engine/quant_rule.h"
#include "runtime/engine/plan_ladder.h"
#include "serve/generation_service.h"
#include "serve/serve_options.h"
#include "spec/registry.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <optional>
#include <string>
#include <vector>

namespace {

namespace rt = ninferno::runtime;
namespace pr = ninferno::product;
using ninferno::EngineOptions;
using ninferno::KvCacheStorage;
using ninferno::KvCapacityPolicy;

constexpr std::uint64_t kMiB = std::uint64_t{1} << 20;
constexpr std::uint64_t kGiB = std::uint64_t{1} << 30;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

// The state the family planner lays out (plan_facts.cpp reads it from a header or a config).
ninferno::models::StateShape state(ninferno::models::StateFamily family, std::uint32_t full,
                                   std::int32_t kv, std::int32_t dim, std::uint32_t sliding,
                                   std::int32_t sliding_kv, std::int32_t sliding_dim,
                                   std::uint32_t window, std::uint32_t recurrent,
                                   std::int32_t value_heads, std::int32_t value_dim,
                                   std::int32_t key_heads, std::int32_t key_dim, std::int32_t conv,
                                   std::int32_t hidden, std::int32_t vocab) {
    return {.family           = family,
            .full_layers      = full,
            .kv_heads         = kv,
            .head_dim         = dim,
            .sliding_layers   = sliding,
            .sliding_kv_heads = sliding_kv,
            .sliding_head_dim = sliding_dim,
            .window           = window,
            .recurrent_layers = recurrent,
            .value_heads      = value_heads,
            .value_dim        = value_dim,
            .key_heads        = key_heads,
            .key_dim          = key_dim,
            .conv_kernel      = conv,
            .hidden           = hidden,
            .output_rows      = vocab,
            .public_tokens    = vocab};
}

using ninferno::models::StateFamily;

// Synthetic shapes (the arithmetic of the lab's models, rounded).
rt::PlanModelFacts qwen27b(std::uint64_t weights) {
    rt::PlanModelFacts m;
    m.model_type               = "qwen3_5_text";
    m.architecture             = "qwen35";
    m.native_context           = 262144;
    m.full_layers              = 16;
    m.full_kv_heads_total      = 16 * 4;
    m.full_key_dim             = 256;
    m.full_value_dim           = 256;
    m.recurrent_layers         = 48;
    m.recurrent_state_elements = 48ULL * 48 * 128 * 128;
    m.recurrent_conv_bytes     = 48ULL * 3 * (6144 + 2 * 16 * 128) * 2;
    m.hidden                   = 5120;
    m.ffn_active               = 17408;
    m.vocab                    = 248320;
    m.params                   = 27e9;
    m.weight_bytes             = weights;
    m.weight_basis             = rt::WeightBasis::FileSize;
    m.state =
        state(StateFamily::Qwen35, 16, 4, 256, 0, 0, 0, 0, 48, 48, 128, 16, 128, 4, 5120, 248320);
    m.offline_bytes = true;
    return m;
}

rt::PlanModelFacts moe35b(std::uint64_t weights) {
    rt::PlanModelFacts m       = qwen27b(weights);
    m.model_type               = "qwen3_5_moe_text";
    m.full_layers              = 10;
    m.full_kv_heads_total      = 10 * 2;
    m.recurrent_layers         = 30;
    m.recurrent_state_elements = 30ULL * 32 * 128 * 128;
    m.recurrent_conv_bytes     = 30ULL * 3 * (4096 + 2 * 16 * 128) * 2;
    m.hidden                   = 2048;
    m.ffn_active               = 8 * 512 + 512;
    m.params                   = 35e9;
    m.state =
        state(StateFamily::Qwen35, 10, 2, 256, 0, 0, 0, 0, 30, 32, 128, 16, 128, 4, 2048, 248320);
    return m;
}

rt::PlanModelFacts gemma31b(std::uint64_t weights) {
    rt::PlanModelFacts m;
    m.model_type             = "gemma4_text";
    m.architecture           = "gemma4";
    m.native_context         = 262144;
    m.full_layers            = 10;
    m.full_kv_heads_total    = 10 * 4;
    m.full_key_dim           = 512;
    m.full_value_dim         = 512;
    m.sliding_layers         = 50;
    m.sliding_kv_heads_total = 50 * 16;
    m.sliding_key_dim        = 256;
    m.sliding_value_dim      = 256;
    m.window                 = 1024;
    m.hidden                 = 5376;
    m.ffn_active             = 21504;
    m.vocab                  = 262144;
    m.params                 = 31e9;
    m.weight_bytes           = weights;
    m.state =
        state(StateFamily::Gemma4, 10, 4, 512, 50, 16, 256, 1024, 0, 0, 0, 0, 0, 0, 5376, 262144);
    m.offline_bytes = true;
    return m;
}

rt::PlanModelFacts gptoss20b(std::uint64_t weights) {
    rt::PlanModelFacts m;
    m.model_type             = "moe_decoder";
    m.architecture           = "gpt-oss";
    m.native_context         = 131072;
    m.full_layers            = 12;
    m.full_kv_heads_total    = 12 * 8;
    m.full_key_dim           = 64;
    m.full_value_dim         = 64;
    m.sliding_layers         = 12;
    m.sliding_kv_heads_total = 12 * 8;
    m.sliding_key_dim        = 64;
    m.sliding_value_dim      = 64;
    m.window                 = 128;
    m.hidden                 = 2880;
    m.ffn_active             = 4 * 2880;
    m.vocab                  = 201088;
    m.params                 = 21e9;
    m.weight_bytes           = weights;
    m.state =
        state(StateFamily::DenseDecoder, 12, 8, 64, 12, 8, 64, 128, 0, 0, 0, 0, 0, 0, 2880, 201088);
    m.offline_bytes = true;
    return m;
}

rt::PlanModelFacts nemotron30b(std::uint64_t weights) {
    rt::PlanModelFacts m;
    m.model_type               = "nemotron_h";
    m.architecture             = "nemotron_h_moe";
    m.native_context           = 262144;
    m.full_layers              = 6;
    m.full_kv_heads_total      = 6 * 2;
    m.full_key_dim             = 128;
    m.full_value_dim           = 128;
    m.recurrent_layers         = 23;
    m.recurrent_state_elements = 23ULL * 64 * 64 * 128;
    m.recurrent_conv_bytes     = 23ULL * 3 * (4096 + 2 * 8 * 128) * 2;
    m.hidden                   = 2688;
    m.ffn_active               = 6 * 1856 + 3712;
    m.vocab                    = 131072;
    m.params                   = 31.6e9;
    m.weight_bytes             = weights;
    m.state =
        state(StateFamily::NemotronH, 6, 2, 128, 0, 0, 0, 0, 23, 64, 64, 8, 128, 0, 2688, 131072);
    m.offline_bytes = true;
    return m;
}

rt::PlanDeviceFacts card(double vram_gib, int cc, bool wddm = false, bool display = false) {
    rt::PlanDeviceFacts d;
    d.name               = "synthetic " + std::to_string(static_cast<int>(vram_gib)) + " GiB";
    d.source             = "test";
    d.total_bytes        = rt::plan_board_total_bytes(vram_gib);
    d.compute_capability = cc;
    d.paravirtual        = wddm;
    d.drives_display     = display;
    return d;
}

// ninferno-serve's options for a context (0: automatic) and streams, as its parser defaults them.
EngineOptions serve_like(std::uint32_t context, std::uint32_t streams) {
    return pr::launch_options({.max_context = context, .max_concurrency = streams}).options;
}

// ---- 1. The seams: the dry run's ladder is the one a start builds from the same functions.

// Every stream holds a context of its own on a non-survival rung (the rule's own fit).
bool own_fit(const rt::DryRunPlan& plan) {
    const rt::DryRunRung* chosen = plan.chosen();
    return chosen != nullptr && !chosen->rung.survival && plan.private_contexts();
}

void test_seams() {
    const rt::PlanModelFacts model   = qwen27b(15 * kGiB);
    const rt::PlanDeviceFacts device = card(24, 89);
    for (const std::uint32_t context : {0U, 8192U, 32768U, 131072U}) {
        for (const std::uint32_t streams : {1U, 2U, 4U, 8U}) {
            EngineOptions options    = serve_like(context, streams);
            const rt::DryRunPlan dry = rt::dry_run_plan(model, device, options);
            check(dry.planned, "planned");
            // What construct_or_plan does with the same options (its seams: plan_axes,
            // plan_ladder_inputs, plan_margins, plan_ladder, prune_plan_ladder, select_plan_rung)
            // over the same per-rung bytes.
            options.kv_cache                       = KvCacheStorage::Int8Group64;
            options.kv_cache_automatic             = false;
            options.recurrent_state                = ninferno::RecurrentStateStorage::FP16;
            options.recurrent_state_automatic      = false;
            options.context_cache.host_state_slots = std::max(
                options.context_cache.host_state_slots, 2U * std::max(2U * streams, streams + 8U));
            const bool host_states = rt::plan_host_states_configured(
                options.context_cache, options.context_cache.host_state_slots, false);
            const rt::PlanAxes axes =
                rt::plan_axes(options, model.native_context, host_states,
                              rt::plan_class_reserve(device.total_bytes, false, false, false));
            check(axes.context_ceiling == dry.axes.context_ceiling &&
                      axes.states_minimum == dry.axes.states_minimum &&
                      axes.states_pinned == dry.axes.states_pinned &&
                      axes.kv_automatic == dry.axes.kv_automatic,
                  "the dry run's axes are a start's");
            const auto candidates =
                rt::plan_ladder(rt::plan_ladder_inputs(axes, streams, true, !axes.chunk_automatic));
            const auto bytes = [&](const rt::PlanRung& rung) {
                return rt::plan_rung_bytes(model, rung, options, streams, !axes.kv_automatic);
            };
            const auto ladder = rt::prune_plan_ladder(candidates, [&](std::size_t i) {
                return std::optional<std::uint64_t>(bytes(candidates[i]).total());
            });
            check(ladder.size() == dry.ladder.size(), "the same ladder length");
            bool same = ladder.size() == dry.ladder.size();
            for (std::size_t i = 0; same && i < ladder.size(); ++i) {
                same = ladder[i] == dry.ladder[i].rung;
            }
            check(same, "the same rungs, context " + std::to_string(context) + " x " +
                            std::to_string(streams));
        }
    }
    // The automatic fit (a start resets the stack limit and measures the warm-up under it).
    {
        EngineOptions pinned           = serve_like(32768, 1);
        pinned.prefill_chunk_automatic = false;
        pinned.kv_capacity             = ninferno::KvCapacityPolicy::explicit_capacity(32768);
        pinned.context_cache.device_state_slots = 2;
        check(rt::plan_automatic_fit(serve_like(0, 1)) && !rt::plan_automatic_fit(pinned),
              "plan_automatic_fit: automatic axes, and none");
    }
    // MTP with an explicit copy window deeper than the draft (a start's validation requires a
    // nonzero one deeper than every depth): the axes' round is the draft window + 1, the
    // definition a start checks its axes against (plan_round_tokens), and the dry run plans.
    for (const auto make : {qwen27b, gemma31b}) {
        EngineOptions o                  = serve_like(32768, 2);
        o.speculative.backend            = ninferno::SpeculativeBackend::Mtp;
        o.speculative.draft_tokens       = 3;
        o.speculative.ngram_draft_tokens = 8;
        const rt::PlanAxes a             = rt::plan_axes(o, 262144, true, 0);
        check(a.round_tokens == 4 && a.round_tokens == rt::plan_round_tokens(o.speculative),
              "an explicit copy window leaves the axes' round at the draft window + 1");
        const rt::DryRunPlan d = rt::dry_run_plan(make(15 * kGiB), device, o);
        check(d.planned && d.axes.round_tokens == 4, "MTP with an explicit copy window plans");
    }
}

// Every family the Engine serves (family_registry.cpp) has its storages in the one table the dry
// run reads, and FP8 rows stand exactly where the chip has E4M3 instructions (sm_89 on).
void test_family_table() {
    for (const auto& family : rt::families()) {
        for (const int cc : {0, 70, 72, 75, 80, 86, 87, 89, 90, 100, 110, 120, 121, 130}) {
            const auto storage = rt::plan_family_storage(family.model_type, cc);
            check(storage.has_value(), std::string(family.model_type) + " has storage");
            if (!storage) { continue; }
            const auto own = rt::family_storage(family.model_type);
            if (!own) { continue; }
            const bool fp8 = own->kv == KvCacheStorage::Fp8E4M3Row256;
            check(storage->kv == (fp8 && cc < 89 ? KvCacheStorage::BFloat16 : own->kv),
                  std::string(family.model_type) + " FP8 rows from sm_89 at cc " +
                      std::to_string(cc));
            check(storage->kv == rt::automatic_kv_cache(family.model_type, cc),
                  std::string(family.model_type) + " KV codec at cc " + std::to_string(cc));
            check(storage->recurrent == rt::automatic_recurrent_state(family.model_type),
                  std::string(family.model_type) + " state");
        }
    }
    check(!rt::plan_family_storage("llama", 89).has_value(), "no family for llama");
}

// ninferno-ui's launch options are ninferno-serve's for the same flags.
ninferno::serve::ServeOptions parse(std::vector<std::string> args) {
    args.insert(args.begin(), {"ninferno-serve", "model.ninferno"});
    std::vector<char*> argv;
    for (std::string& a : args) { argv.push_back(a.data()); }
    return ninferno::serve::parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

void test_launch_options_match_serve() {
    struct Case {
        pr::LaunchAsk ask;
        std::vector<std::string> args;
    };

    const std::vector<Case> cases = {
        {{}, {"--spec", "auto"}},
        {{.max_context = 32768}, {"--max-context", "32768", "--spec", "auto"}},
        {{.max_context = 32768, .max_concurrency = 4},
         {"--max-concurrency", "4", "--max-context", "32768", "--spec", "auto"}},
        {{.max_concurrency = 8}, {"--max-concurrency", "8", "--spec", "auto"}},
        {{.kv_dtype = "fp8"}, {"--kv-dtype", "fp8", "--spec", "auto"}},
        {{.spec = "none"}, {"--spec", "none"}},
        {{.spec = "mtp"}, {"--spec", "mtp"}},
        {{.exact = true}, {"--kv-dtype", "bf16", "--activations", "exact", "--spec", "auto"}},
        {{.activations = "int8"}, {"--activations", "int8", "--spec", "auto"}},
        {{.activations = "exact"}, {"--activations", "exact", "--spec", "auto"}},
    };
    for (const Case& c : cases) {
        const pr::LaunchPlanOptions ours          = pr::launch_options(c.ask);
        const ninferno::serve::ServeOptions serve = parse(c.args);
        const EngineOptions theirs                = ninferno::serve::engine_options_of(serve);
        const EngineOptions& o                    = ours.options;
        std::string label;
        for (const auto& a : c.args) { label += a + " "; }
        check(o.max_context == theirs.max_context &&
                  o.max_context_automatic == theirs.max_context_automatic,
              label + ": context");
        check(o.max_concurrency == theirs.max_concurrency, label + ": concurrency");
        check(o.kv_capacity.mode == theirs.kv_capacity.mode &&
                  o.kv_capacity.explicit_tokens == theirs.kv_capacity.explicit_tokens &&
                  o.kv_capacity.automatic_headroom_bytes ==
                      theirs.kv_capacity.automatic_headroom_bytes,
              label + ": KV capacity");
        check(o.kv_cache_automatic == theirs.kv_cache_automatic &&
                  (o.kv_cache_automatic || o.kv_cache == theirs.kv_cache),
              label + ": KV codec");
        check(o.recurrent_state_automatic == theirs.recurrent_state_automatic,
              label + ": state codec");
        check(o.activations == theirs.activations, label + ": activations");
        check(c.ask.activations_pinned() == serve.activations_given, label + ": activations pinned");
        check(o.prefill_chunk == theirs.prefill_chunk &&
                  o.prefill_chunk_automatic == theirs.prefill_chunk_automatic,
              label + ": chunk");
        check(o.context_cache.enabled == theirs.context_cache.enabled &&
                  o.context_cache.host_tier_automatic == theirs.context_cache.host_tier_automatic &&
                  o.context_cache.host_state_slots == theirs.context_cache.host_state_slots &&
                  o.context_cache.host_kv_capacity_bytes ==
                      theirs.context_cache.host_kv_capacity_bytes &&
                  o.context_cache.device_state_slots == theirs.context_cache.device_state_slots,
              label + ": context cache");
        check(o.use_cuda_graph == theirs.use_cuda_graph, label + ": graphs");
        check(ours.ask.automatic == serve.speculation_from_profile, label + ": spec auto");
        check(ours.ask.given.backend == serve.speculative.backend, label + ": spec backend");
        check(ours.ask.depth_from_profile == serve.draft_tokens_from_model, label + ": depth");
    }
}

// ---- 2. Total and monotone over models x cards x contexts x streams x codecs.

struct Model {
    std::string name;
    rt::PlanModelFacts (*make)(std::uint64_t);
    std::vector<std::uint64_t> quants; // weight bytes, ascending
};

void test_properties() {
    const std::vector<Model> models = {
        {"qwen27b", qwen27b, {9 * kGiB, 12 * kGiB, 15 * kGiB, 18 * kGiB, 22 * kGiB, 29 * kGiB}},
        {"moe35b", moe35b, {11 * kGiB, 16 * kGiB, 20 * kGiB, 24 * kGiB, 36 * kGiB}},
        {"gemma31b", gemma31b, {11 * kGiB, 17 * kGiB, 21 * kGiB, 33 * kGiB}},
        {"gptoss20b", gptoss20b, {12 * kGiB, 13 * kGiB}},
        {"nemotron30b", nemotron30b, {13 * kGiB, 18 * kGiB, 24 * kGiB, 33 * kGiB}},
    };
    const std::vector<std::pair<double, int>> cards = {{8, 89},   {11, 75}, {12, 86}, {16, 89},
                                                       {16, 120}, {24, 89}, {32, 70}, {32, 120},
                                                       {40, 80},  {80, 90}};
    const std::vector<std::uint32_t> contexts       = {0, 8192, 32768, 131072, 262144};
    std::size_t plans                               = 0;
    for (const Model& model : models) {
        for (const auto& [vram, cc] : cards) {
            for (const bool wddm : {false, true}) {
                const rt::PlanDeviceFacts device = card(vram, cc, wddm, wddm);
                const rt::PlanDeviceFacts bigger = card(vram * 2, cc, wddm, wddm);
                for (const std::uint32_t context : contexts) {
                    if (context > model.make(0).native_context) { continue; }
                    for (const std::uint32_t streams : {1U, 2U, 4U, 8U}) {
                        for (std::size_t q = model.quants.size(); q-- > 0;) {
                            const rt::PlanModelFacts facts = model.make(model.quants[q]);
                            const EngineOptions options    = serve_like(context, streams);
                            const rt::DryRunPlan plan = rt::dry_run_plan(facts, device, options);
                            ++plans;
                            const std::string what =
                                model.name + " " + std::to_string(model.quants[q] / kGiB) +
                                " GiB on " + device.name + " cc" + std::to_string(cc) +
                                (wddm ? " wddm" : "") + " ctx " + std::to_string(context) + " x " +
                                std::to_string(streams);
                            // Total: a rung, or a refusal with a reason.
                            check(plan.planned && !plan.ladder.empty(), what + ": planned");
                            check(plan.fits || !plan.reason.empty(), what + ": a reason");
                            if (plan.fits && context != 0) {
                                check(plan.chosen()->rung.max_context == context,
                                      what + ": a pinned context is kept");
                            }
                            // Bytes never grow down the ladder (within each range).
                            for (std::size_t i = 1; i < plan.ladder.size(); ++i) {
                                if (plan.ladder[i].rung.survival !=
                                    plan.ladder[i - 1].rung.survival) {
                                    continue;
                                }
                                check(plan.ladder[i].bytes.total() <
                                          plan.ladder[i - 1].bytes.total(),
                                      what + ": bytes fall down the ladder");
                            }
                            // More VRAM never serves less.
                            const rt::DryRunPlan more = rt::dry_run_plan(facts, bigger, options);
                            if (plan.fits) {
                                check(more.fits, what + ": fits on twice the memory");
                                if (more.fits) {
                                    check(more.chosen()->rung.max_context >=
                                                  plan.chosen()->rung.max_context &&
                                              more.chosen()->rung.kv_pages >=
                                                  plan.chosen()->rung.kv_pages,
                                          what + ": more memory never less context or pool");
                                }
                            }
                        }
                        // Monotone in the quantization: if quant i fits privately, every smaller
                        // one does too.
                        bool larger_fits = false;
                        for (std::size_t q = model.quants.size(); q-- > 0;) {
                            const rt::DryRunPlan plan = rt::dry_run_plan(
                                model.make(model.quants[q]), device, serve_like(context, streams));
                            const bool fits = own_fit(plan);
                            check(!larger_fits || fits,
                                  model.name + ": a smaller quantization fits where a larger does");
                            larger_fits = larger_fits || fits;
                        }
                    }
                }
                // A shorter asked context fits whenever a longer one does.
                for (const std::uint32_t streams : {1U, 4U}) {
                    bool longer_fits = false;
                    for (auto it = contexts.rbegin(); it != contexts.rend(); ++it) {
                        if (*it == 0 || *it > model.make(0).native_context) { continue; }
                        const rt::DryRunPlan plan = rt::dry_run_plan(
                            model.make(model.quants.front()), device, serve_like(*it, streams));
                        const bool fits = own_fit(plan);
                        check(!longer_fits || fits, model.name + ": a shorter context fits");
                        longer_fits = longer_fits || fits;
                    }
                }
                // A narrower KV codec never serves less context.
                const rt::PlanModelFacts facts = model.make(model.quants.front());
                std::uint32_t wider            = 0;
                for (const char* kv : {"bf16", "fp8", "nvfp4"}) {
                    const auto o = pr::launch_options({.max_context = 0, .kv_dtype = kv}).options;
                    const rt::DryRunPlan plan = rt::dry_run_plan(facts, device, o);
                    // A codec the family does not lay out is refused, with its words.
                    if (!plan.planned) {
                        check(!plan.reason.empty(), model.name + ": a refused codec says why");
                        continue;
                    }
                    const std::uint32_t got   = plan.fits ? plan.chosen()->rung.max_context : 0;
                    check(got >= wider, model.name + ": a narrower KV codec never less context");
                    wider = std::max(wider, got);
                }
            }
        }
    }
    check(plans > 1000, "many plans");
    std::cout << "plans checked: " << plans << '\n';
}

// ---- 3. The quant rule.

void test_rule() {
    using rt::QuantCandidate;
    using rt::QuantFit;
    using rt::QuantFitState;
    using rt::QuantPickTier;
    // The targets.
    check(rt::quant_own_targets({.automatic = true, .native = 262144}) ==
              std::vector<std::uint32_t>{32768, 16384, 8192},
          "auto targets 32K, 16K, 8K");
    check(rt::quant_own_targets({.automatic = true, .native = 20000}) ==
              std::vector<std::uint32_t>{20000, 16384, 8192},
          "auto targets under a short native context");
    check(rt::quant_own_targets({.automatic = true, .native = 4096}) ==
              std::vector<std::uint32_t>{4096},
          "auto target: the model's own when shorter than 8K");
    check(rt::quant_own_targets({.automatic = false, .context = 131072}) ==
              std::vector<std::uint32_t>{131072},
          "a pinned context is the one target");

    const std::vector<QuantCandidate> c = {
        {.label = "UD-IQ2_M", .bytes = 10 * kGiB, .label_bits = 2, .bits_per_weight = 2.7},
        {.label = "IQ3_XXS", .bytes = 12 * kGiB, .label_bits = 3, .bits_per_weight = 3.1},
        {.label = "Q3_K_M", .bytes = 15 * kGiB, .label_bits = 3, .bits_per_weight = 3.9},
        {.label = "Q4_K_M", .bytes = 18 * kGiB, .label_bits = 4, .bits_per_weight = 4.8},
        {.label = "Q6_K", .bytes = 24 * kGiB, .label_bits = 6, .bits_per_weight = 6.6},
        {.label           = "BF16",
         .bytes           = 54 * kGiB,
         .label_bits      = 16,
         .bits_per_weight = 16,
         .supported       = false},
    };
    const rt::QuantAsk auto4{.automatic = true, .concurrency = 4, .native = 262144};
    const auto fit = [](QuantFitState state, std::uint32_t own = 0, std::uint32_t context = 0) {
        return QuantFit{.state = state, .context = context, .own_context = own};
    };
    std::vector<QuantFit> fits = {
        fit(QuantFitState::Own, 98304),      fit(QuantFitState::Own, 65536),
        fit(QuantFitState::Own, 65536),      fit(QuantFitState::Own, 49152),
        fit(QuantFitState::Shared, 0, 9000), fit(QuantFitState::Own, 1 << 20)};
    auto pick = rt::largest_fitting_quant(c, fits, auto4);
    check(pick && c[pick->index].label == "Q4_K_M" && pick->tier == QuantPickTier::Own &&
              pick->target_context == 32768,
          "auto x 4: the largest whose streams hold 32K each, not the shared Q6_K");
    fits[3] = fit(QuantFitState::Own, 20000);
    fits[2] = fit(QuantFitState::Own, 20000);
    fits[1] = fit(QuantFitState::Own, 65536);
    pick    = rt::largest_fitting_quant(c, fits, auto4);
    check(pick && c[pick->index].label == "Q4_K_M" && pick->tier == QuantPickTier::OwnStepped &&
              pick->target_context == 16384,
          "no floor quant reaches 32K each: the context steps down before a lower quant");
    fits[3] = fit(QuantFitState::Shared, 0, 12000);
    fits[2] = fit(QuantFitState::Shared, 0, 15000);
    pick    = rt::largest_fitting_quant(c, fits, auto4);
    check(pick && c[pick->index].label == "Q6_K" && pick->tier == QuantPickTier::Shared &&
              !pick->below_floor,
          "nothing at the floor holds 8K each: the shared pool of the largest, not 3 bits own");
    for (std::size_t i = 2; i < 5; ++i) { fits[i] = fit(QuantFitState::No); }
    pick = rt::largest_fitting_quant(c, fits, auto4);
    check(pick && c[pick->index].label == "IQ3_XXS" && pick->tier == QuantPickTier::OwnBelowFloor &&
              pick->below_floor && !pick->low_quality,
          "below the floor only when nothing at it fits");
    fits[1] = fit(QuantFitState::No);
    pick    = rt::largest_fitting_quant(c, fits, auto4);
    check(pick && c[pick->index].label == "UD-IQ2_M" && pick->low_quality,
          "under 3 bits the pick is low quality");
    fits[0] = fit(QuantFitState::No);
    fits[3] = fit(QuantFitState::Desktop, 0, 8192);
    pick    = rt::largest_fitting_quant(c, fits, auto4);
    check(pick && c[pick->index].label == "Q4_K_M" && pick->tier == QuantPickTier::Desktop,
          "last: what fits only without the desktop's reserve");
    fits[3] = fit(QuantFitState::Unknown);
    check(!rt::largest_fitting_quant(c, fits, auto4), "none fits (the unsupported never wins)");
    // A pinned context: own at it, then shared, then below the floor.
    const rt::QuantAsk pinned3{.automatic = false, .context = 131072, .concurrency = 3};
    fits = {fit(QuantFitState::Own, 131072),
            fit(QuantFitState::No),
            fit(QuantFitState::No),
            fit(QuantFitState::Shared, 0, 131072),
            fit(QuantFitState::Shared, 0, 131072),
            fit(QuantFitState::No)};
    pick = rt::largest_fitting_quant(c, fits, pinned3);
    check(pick && c[pick->index].label == "Q6_K" && pick->tier == QuantPickTier::Shared,
          "128K x 3: the shared Q6_K before a 2-bit file with three own contexts");
    // One stream: its context is its own whatever its length.
    const rt::QuantAsk auto1{.automatic = true, .concurrency = 1, .native = 262144};
    fits = {fit(QuantFitState::No), fit(QuantFitState::No),
            fit(QuantFitState::No), fit(QuantFitState::Own, 6144, 6144),
            fit(QuantFitState::No), fit(QuantFitState::No)};
    pick = rt::largest_fitting_quant(c, fits, auto1);
    check(pick && c[pick->index].label == "Q4_K_M" && pick->tier == QuantPickTier::OwnStepped,
          "one stream with a short context: an own pick, never called shared");
    // A strict weak order.
    for (const QuantCandidate& a : c) {
        check(!rt::quant_ranks_above(a, a), "irreflexive: " + a.label);
        for (const QuantCandidate& b : c) {
            check(!(rt::quant_ranks_above(a, b) && rt::quant_ranks_above(b, a)),
                  "asymmetric: " + a.label + " " + b.label);
            for (const QuantCandidate& d : c) {
                if (rt::quant_ranks_above(a, b) && rt::quant_ranks_above(b, d)) {
                    check(rt::quant_ranks_above(a, d), "transitive");
                }
            }
        }
    }

    // The table: the rule's pick fits and no larger quantization fits at the pick's tier.
    const rt::PlanDeviceFacts device = card(24, 89);
    pr::PlanSourceFacts shape;
    shape.model = qwen27b(9 * kGiB);
    std::vector<pr::QuantListing> listing;
    for (const auto& [label, gib, bits] :
         std::vector<std::tuple<std::string, double, int>>{{"IQ2_XXS", 8.5, 2},
                                                           {"IQ3_XXS", 10.5, 3},
                                                           {"Q4_K_M", 15.4, 4},
                                                           {"Q5_K_M", 18.0, 5},
                                                           {"Q6_K", 21.0, 6},
                                                           {"Q8_0", 27.0, 8},
                                                           {"BF16", 50.0, 16}}) {
        listing.push_back({.quant  = label,
                           .format = "gguf",
                           .bytes  = static_cast<std::uint64_t>(gib * kGiB),
                           .bits   = bits});
    }
    listing.push_back({.quant = "mmproj-F16", .format = "gguf", .bytes = kGiB, .drafter = true});
    listing.push_back({.quant = "NVFP4", .format = "gguf", .bytes = 15 * kGiB, .bits = 4});
    for (const std::uint32_t context : {0U, 32768U, 131072U}) {
        for (const std::uint32_t streams : {1U, 4U}) {
            pr::DryRunInputs inputs;
            inputs.device                 = device;
            const pr::LaunchPlanOptions o = pr::launch_options(
                {.max_context = context, .max_concurrency = streams, .spec = "none"});
            inputs.options           = o.options;
            inputs.ask               = o.ask;
            const pr::QuantPlan plan = pr::plan_quants(shape, listing, inputs);
            const std::string what =
                "ctx " + std::to_string(context) + " x " + std::to_string(streams);
            check(plan.rows.size() == listing.size() - 1, "drafters left out of the table");
            check(!plan.rule.empty() && !plan.rule_code.empty(), "a rule line and its code");
            const auto row = [&](const std::string& label) -> const pr::QuantRow& {
                return *std::find_if(
                    plan.rows.begin(), plan.rows.end(),
                    [&](const pr::QuantRow& r) { return r.listing.quant == label; });
            };
            check(row("BF16").fit.state == QuantFitState::Unsupported &&
                      row("BF16").unsupported_code == "above_8_bits",
                  "BF16 GGUF is not served");
            check(row("NVFP4").unsupported_code == "nvfp4_card", "NVFP4 needs an FP4 card");
            if (!plan.pick) { continue; }
            const pr::QuantRow& chosen = plan.rows[plan.pick->index];
            check(chosen.largest && plan.chosen == plan.pick->index, "the pick is marked");
            check(chosen.fit.state == QuantFitState::Own ||
                      chosen.fit.state == QuantFitState::Shared ||
                      chosen.fit.state == QuantFitState::Desktop,
                  what + ": the pick fits");
            const bool own_tier = plan.pick->tier == QuantPickTier::Own ||
                                  plan.pick->tier == QuantPickTier::OwnStepped;
            for (const pr::QuantRow& r : plan.rows) {
                if (!r.candidate.supported ||
                    !rt::quant_ranks_above(r.candidate, chosen.candidate)) {
                    continue;
                }
                if (own_tier) {
                    check(!(r.fit.state == QuantFitState::Own &&
                            r.fit.own_context >= plan.pick->target_context),
                          what + ": no larger quantization holds the target each: " +
                              r.listing.quant + " over " + chosen.listing.quant);
                }
            }
            // Auto with several streams: the launch pins the own context the pick was made for,
            // and a plan pinned there keeps every stream's own.
            if (context == 0 && streams > 1 && chosen.launch_context != 0) {
                check(chosen.launch_context == chosen.fit.own_context, what + ": launch context");
                pr::DryRunInputs pinned = inputs;
                const pr::LaunchPlanOptions p =
                    pr::launch_options({.max_context     = chosen.launch_context,
                                        .max_concurrency = streams,
                                        .spec            = "none"});
                pinned.options                  = p.options;
                const pr::PlannedSpeculation at = pr::plan_model(
                    pr::with_weights(shape, chosen.weight_bytes, chosen.basis), pinned);
                check(own_fit(at.plan), what + ": the pinned launch keeps each stream's own");
            }
            if (context != 0) {
                check(chosen.launch_context == context, what + ": a pinned launch keeps it");
            }
            // The JSON document carries the schema's fields.
            const nlohmann::json doc =
                pr::dry_plan_document(shape, "test/repo", "hub", inputs, chosen.planned, &plan);
            check(doc["schema"] == "ninferno-dry-plan/1", "schema");
            check(doc["chosen"] == chosen.listing.quant, "chosen in the document");
            check(doc["pick"].is_object() && doc["pick"]["quant"] == chosen.listing.quant &&
                      doc["pick"]["card_gb"] == 24,
                  "the pick in the document");
            check(doc["rule_code"] == "largest_fits" &&
                      doc["rule_params"]["quant"] == chosen.listing.quant,
                  "the rule as data");
            check(doc["device"]["card_gb"] == 24, "the card as sold");
            check(doc["plan"].contains("ladder") && doc["plan"].contains("speculation"),
                  "plan with ladder and speculation");
            check(doc["quants"].size() == plan.rows.size(), "every row");
            for (const auto& q : doc["quants"]) {
                for (const char* key : {"state", "own_context", "launch_context", "below_floor",
                                        "low_quality", "drafter_bytes", "unsupported_code",
                                        "weights_basis", "fits", "fits_shared"}) {
                    check(q.contains(key), std::string("quants[].") + key);
                }
            }
            for (const char* key : {"fits", "rung", "rungs", "reserve", "context", "pool_contexts",
                                    "private_contexts", "kv_dtype", "state_dtype", "reason",
                                    "reason_code", "reason_params", "fits_instead"}) {
                check(doc["plan"].contains(key), std::string("plan.") + key);
            }
            for (const char* key : {"ask", "served", "fallbacks", "load", "basis"}) {
                check(doc["plan"]["speculation"].contains(key), std::string("speculation.") + key);
            }
            check(!doc["plan"]["speculation"].contains("classes") &&
                      !doc["plan"]["speculation"].contains("modeled_load"),
                  "no context classes or modeled load: the round-cost model is gone");
        }
    }
    // The quantization by worth (plan_quants: within the rule's tier, each row at its own choice):
    // where the generation decides (Ada: its reference card's factors and Qwen 3.5 dense's curve
    // rest on two cards); the rule's largest elsewhere (Turing: no reference card), and wherever a
    // row's bits are unknown (it has no place on the curve and would look exact).
    {
        // Unknown bits: the model's parameters unknown, so no row's bits per weight (the labels'
        // bits rank the rule's rows, but the choice's accuracy term reads bits per weight).
        pr::PlanSourceFacts no_params = shape;
        no_params.model.params        = 0;
        const auto planned = [&](int cc, std::uint32_t context, bool unknown_bits) {
            pr::DryRunInputs inputs;
            inputs.device                 = card(24, cc);
            inputs.gpu.compute_capability = cc;
            const pr::LaunchPlanOptions o =
                pr::launch_options({.max_context = context, .max_concurrency = 1, .spec = "none"});
            inputs.options = o.options;
            inputs.ask     = o.ask;
            return pr::plan_quants(unknown_bits ? no_params : shape, listing, inputs);
        };
        int weighed = 0;
        for (const std::uint32_t context : {0U, 32768U, 131072U}) {
            const std::string what = "ctx " + std::to_string(context);
            const pr::QuantPlan ada = planned(89, context, false);
            check(ada.pick.has_value(), what + ": Ada picks");
            if (ada.pick && ada.by_worth) {
                ++weighed;
                const pr::QuantRow& pick    = ada.rows[ada.pick->index];
                const pr::QuantRow& largest = ada.rows[ada.largest_index.value_or(ada.pick->index)];
                check(ada.largest_index && *ada.largest_index != ada.pick->index &&
                          ada.rule_code == "best_worth",
                      what + ": the worth's pick is said beside the rule's largest");
                check(rt::below_quality_floor(pick.candidate) == rt::below_quality_floor(largest.candidate),
                      what + ": the worth weighs within the rule's side of the floor");
                check(pick.fit.own_context >= ada.pick->target_context,
                      what + ": the worth's pick holds the target each");
                check(ada.trade && ada.trade->time_ratio > 1 && ada.trade->kld_delta < 0 &&
                          ada.rule.find("lambda 25: 0.001 KLD is worth 2.5 % of a request's time") !=
                              std::string::npos,
                      what + ": the trade said (the largest slower and more accurate): " + ada.rule);
            }
            std::cout << "  Ada 27B " << what << ": "
                      << (ada.pick ? ada.rows[ada.pick->index].listing.quant : std::string("-"))
                      << (ada.by_worth ? " (by worth)" : " (the rule's largest)") << "\n";
            const pr::QuantPlan turing = planned(75, context, false);
            check(turing.pick && !turing.by_worth && turing.rule_code != "best_worth",
                  what + ": Turing keeps the rule's largest");
            const pr::QuantPlan unknown = planned(89, context, true);
            check(!unknown.by_worth, what + ": unknown bits per weight keep the rule's largest");
        }
        std::cout << "  Ada 27B: " << weighed << " of 3 contexts decided by worth\n";
    }
    // A safetensors checkpoint: planned at its converted size, or not planned (never refused).
    {
        const nlohmann::json dense = {{"model_type", "qwen3_5_text"}};
        check(pr::checkpoint_converted_bytes(dense, 54 * kGiB) == 54 * kGiB,
              "a BF16 dense checkpoint runs at its own size");
        const nlohmann::json moe = {{"text_config",
                                     {{"model_type", "qwen3_5_moe_text"},
                                      {"num_experts", 256},
                                      {"num_hidden_layers", 40},
                                      {"hidden_size", 2048},
                                      {"moe_intermediate_size", 512}}}};
        const auto bytes         = pr::checkpoint_converted_bytes(moe, 67 * kGiB);
        check(bytes && *bytes > 17 * kGiB && *bytes < 25 * kGiB,
              "a 35B-A3B BF16 checkpoint converts to about 20 GiB");
        std::vector<pr::QuantListing> one = {
            {.quant = "", .format = "safetensors", .bytes = 54 * kGiB, .bits = 16}};
        pr::DryRunInputs inputs;
        inputs.device                 = device;
        const pr::LaunchPlanOptions o = pr::launch_options({.spec = "none"});
        inputs.options                = o.options;
        inputs.ask                    = o.ask;
        check(pr::plan_quants(shape, one, inputs).rows[0].fit.state == QuantFitState::Unknown,
              "a checkpoint of unknown converted size is not planned, not refused");
        one[0].converted_bytes      = 54 * kGiB;
        const pr::QuantPlan planned = pr::plan_quants(shape, one, inputs);
        check(planned.rows[0].converted && planned.rows[0].weight_bytes == 54 * kGiB &&
                  planned.rows[0].fit.state == QuantFitState::No,
              "planned at its converted size");
    }
    // An unreadable shape: every row unknown, nothing picked (the caller falls back).
    {
        pr::PlanSourceFacts blank;
        pr::DryRunInputs inputs;
        inputs.device                 = device;
        const pr::LaunchPlanOptions o = pr::launch_options({.spec = "none"});
        inputs.options                = o.options;
        inputs.ask                    = o.ask;
        const pr::QuantPlan plan      = pr::plan_quants(blank, listing, inputs);
        check(!plan.pick, "nothing picked without a shape");
        check(plan.rows[2].fit.state == QuantFitState::Unknown &&
                  pr::plan_reason_code(plan.rows[2].planned.plan).code == "shape_unknown",
              "unknown, with its code");
    }
    // One stream under an automatic context on a WDDM card driving the display (a class reserve):
    // a row is own only where the start's own automatic plan keeps every stream's context, never
    // where a pinned probe without the reserve would (the start serves those on a survival rung).
    {
        const rt::PlanDeviceFacts desktop = card(24, 89, true, true);
        pr::DryRunInputs inputs;
        inputs.device                 = desktop;
        const pr::LaunchPlanOptions o = pr::launch_options({.spec = "none"});
        inputs.options                = o.options;
        inputs.ask                    = o.ask;
        const pr::QuantPlan plan      = pr::plan_quants(shape, listing, inputs);
        for (const pr::QuantRow& r : plan.rows) {
            if (r.fit.state != QuantFitState::Own) { continue; }
            const rt::DryRunRung* chosen = r.planned.plan.chosen();
            check(chosen != nullptr && !chosen->rung.survival,
                  r.listing.quant + ": own at one stream only on the start's own plan");
        }
    }
    // Several streams under an automatic context: a row launched at its own context per stream
    // shows the launch's plan (--max-context-cap at that context: still automatic, the step-down
    // rungs kept), not the uncapped pool the streams would share.
    {
        pr::DryRunInputs inputs;
        inputs.device                 = device;
        const pr::LaunchPlanOptions o = pr::launch_options({.max_concurrency = 4, .spec = "none"});
        inputs.options                = o.options;
        inputs.ask                    = o.ask;
        const pr::QuantPlan plan      = pr::plan_quants(shape, listing, inputs);
        for (const pr::QuantRow& r : plan.rows) {
            if (r.launch_context == 0) { continue; }
            const rt::DryRunRung* chosen = r.planned.plan.chosen();
            check(chosen != nullptr && chosen->rung.max_context <= r.launch_context &&
                      r.planned.plan.axes.context_automatic &&
                      r.planned.plan.axes.context_ceiling == r.launch_context,
                  r.listing.quant + ": the plan shown is the launch's, capped at its context");
        }
    }
    // On a 24 GB card at 32K x 1 a 27B fits at Q6_K, not at Q8_0 (the lab's 4090).
    {
        pr::DryRunInputs inputs;
        inputs.device                 = device;
        const pr::LaunchPlanOptions o = pr::launch_options({.max_context = 32768, .spec = "none"});
        inputs.options                = o.options;
        inputs.ask                    = o.ask;
        const pr::QuantPlan plan      = pr::plan_quants(shape, listing, inputs);
        check(plan.chosen && plan.rows[*plan.chosen].listing.quant == "Q6_K",
              "27B at 32K x 1 on 24 GB: Q6_K (" + plan.rule + ")");
    }
}

// ---- 4. Speculation: the start's order, one step served.

void test_speculation_order() {
    pr::PlanSourceFacts facts;
    facts.model                          = qwen27b(15 * kGiB);
    facts.shape.hidden_size              = 5120;
    facts.shape.num_hidden_layers        = 64;
    facts.shape.vocab_size               = 248320;
    facts.shape.builtin_mtp              = true;
    facts.shape.active_params            = 27e9;
    const rt::PlanDeviceFacts device     = card(24, 89);
    const ninferno::spec::GpuKey gpu     = pr::gpu_key_of(device);
    const pr::SpeculationSubject subject = pr::planned_subject(
        pr::PlannedTarget{.repo = "unsloth/Qwen3.8-27B-GGUF", .facts = facts, .gguf = true}, gpu);
    check(subject.components.contains("mtp"), "the MTP layers are a component");
    // A checkpoint fine-tune no registry entry names, its model card naming Qwen3.6-27B as its
    // base: the plan matches the base's entry, as the converter's drafter plan will by the README
    // (not the nearest by shape, Qwen3.8-27B: the same engine, geometry and vocabulary). A GGUF
    // without its drafters has no manifest: the start and the plan both take the entry by shape.
    {
        pr::PlanSourceFacts tuned = facts;
        tuned.shape.base_models   = {"Qwen/Qwen3.6-27B"};
        const pr::SpeculationSubject descendant = pr::planned_subject(
            pr::PlannedTarget{.repo = "someone/Qwen3.6-27B-finetune", .facts = tuned, .gguf = false}, gpu);
        check(descendant.resolved && descendant.resolved->source.find("qwen3.6-27b") != std::string::npos,
              "a checkpoint's model-card lineage finds its base's entry: " +
                  (descendant.resolved ? descendant.resolved->source : std::string("none")));
        // A merge naming a parent no entry knows first: a later parent still matches, as the
        // converter checks every README base model in order.
        tuned.shape.base_models = {"someone/unregistered-27B", "Qwen/Qwen3.6-27B"};
        const pr::SpeculationSubject merge = pr::planned_subject(
            pr::PlannedTarget{.repo = "someone/Qwen3.6-27B-merge", .facts = tuned, .gguf = false}, gpu);
        check(merge.resolved && merge.resolved->source.find("qwen3.6-27b") != std::string::npos,
              "a later base model of a merge finds its entry: " +
                  (merge.resolved ? merge.resolved->source : std::string("none")));
    }
    for (const std::uint32_t context : {0U, 32768U, 262144U}) {
        for (const std::uint32_t streams : {1U, 4U, 8U}) {
            const pr::LaunchPlanOptions o =
                pr::launch_options({.max_context = context, .max_concurrency = streams});
            const pr::PlannedSpeculation planned =
                pr::plan_with_speculation(facts, device, o.options, &subject, o.ask);
            const pr::SpeculationReport& r = planned.speculation;
            check(r.ask == "auto", "auto asked");
            check(!r.fallbacks.empty(), "a fallback step");
            const auto served = std::count_if(r.fallbacks.begin(), r.fallbacks.end(),
                                              [](const auto& s) { return s.served; });
            check(served == 1, "exactly one step served at " + std::to_string(context) + " x " +
                                   std::to_string(streams));
            for (const auto& s : r.fallbacks) {
                if (s.served) { check(s.fits == planned.plan.fits, "the served step's plan"); }
            }
            check(r.label == (r.engine.backend == ninferno::SpeculativeBackend::None
                                  ? std::string("plain")
                                  : r.label),
                  "label");
        }
    }
    // An explicit --spec none is plain, one step.
    const pr::LaunchPlanOptions none = pr::launch_options({.spec = "none"});
    const pr::PlannedSpeculation plain =
        pr::plan_with_speculation(facts, device, none.options, &subject, none.ask);
    check(plain.speculation.label == "plain" && plain.speculation.fallbacks.size() == 1,
          "--spec none is plain");

    // Each quantization's own MTP layers (PlannedSubjects::MtpOf): the facts are one file's, and
    // unsloth's smallest quantizations drop the layers its larger ones carry.
    {
        auto one_file              = facts;
        one_file.shape.builtin_mtp = false;
        pr::PlannedSubjects subjects(
            pr::PlannedTarget{.repo = "unsloth/Qwen3.8-27B-GGUF", .facts = one_file, .gguf = true},
            gpu, {}, {}, [](const std::string& quant) -> std::optional<bool> {
                if (quant == "UD-Q4_K_M") { return true; }
                return std::nullopt;
            });
        check(subjects.of("UD-Q4_K_M", "").components.contains("mtp") &&
                  !subjects.of("UD-IQ1_S", "").components.contains("mtp"),
              "a quantization's own MTP layers, else the facts'");
    }

    // The start's order itself (product::fit_fallbacks), one code for the server, the CLI and this
    // plan: shallower drafts with the same drafter (deepest first, then depth 1), then the model's
    // own MTP head; the candidate returned is the one accepted last.
    const pr::SpeculationSubject drafted = pr::planned_subject(
        pr::PlannedTarget{.repo          = "unsloth/Qwen3.8-27B-GGUF",
                          .facts         = facts,
                          .gguf          = true,
                          .with_drafters = true},
        gpu);
    const auto choice = pr::served_speculation(drafted, true, false);
    check(choice && choice->menu.mode != ninferno::spec::Mode::None &&
              choice->menu.mode != ninferno::spec::Mode::Mtp,
          "the 27B's profile drafts with a separate drafter");
    if (choice) {
        std::vector<std::string> steps;
        const auto only_head = [&](const pr::ServedSpeculation& candidate, pr::FallbackStep at) {
            steps.push_back(std::string(pr::fallback_step_name(at)) + " " +
                            candidate.menu.primary.config.label());
            return at == pr::FallbackStep::Head;
        };
        const auto head = pr::fit_fallbacks(drafted, *choice, false, only_head);
        const std::int32_t k = choice->menu.primary.config.k;
        check(head && head->menu.mode == ninferno::spec::Mode::Mtp && !steps.empty() &&
                  steps.back().rfind("head mtp:", 0) == 0,
              "nothing shallower fits: the MTP head, last");
        // A fit fallback, when the profile has one, comes first (and shallower drafts start
        // from it).
        const std::size_t first = !steps.empty() && steps[0].rfind("fit_fallback ", 0) == 0;
        const auto at_depth = [](const std::string& step, std::int32_t depth) {
            const std::string label = step.substr(step.find(' ') + 1);
            return label.substr(label.find(':') + 1, label.find_first_not_of("0123456789",
                                                                     label.find(':') + 1) -
                                                         label.find(':') - 1) ==
                   std::to_string(depth);
        };
        check(first == 1 || k <= 2 ||
                  (steps.size() >= 3 && steps[0].rfind("shallower ", 0) == 0 &&
                   at_depth(steps[0], k - 1) && at_depth(steps[1], 1)),
              "one draft shallower first, then depth 1");
        steps.clear();
        const auto none = pr::fit_fallbacks(
            drafted, *choice, false,
            [&](const pr::ServedSpeculation&, pr::FallbackStep at) {
                steps.push_back(pr::fallback_step_name(at));
                return false;
            });
        check(!none && !steps.empty() && steps.back() == "head", "nothing fits: plain, after the head");
    }
    // The pool rule (product::pool_rule): plain decode asked only inside the rule's range, the head
    // only when plain decode holds more, kept at four fifths of plain decode's contexts.
    {
        int plain_asked = 0, head_asked = 0;
        const auto rule = [&](std::uint32_t with_drafter, std::uint32_t streams,
                              std::uint32_t plain, std::optional<std::uint32_t> head) {
            return pr::pool_rule(
                drafted, false, with_drafter, streams,
                [&] {
                    ++plain_asked;
                    return plain;
                },
                [&](const pr::ServedSpeculation&) {
                    ++head_asked;
                    return head;
                });
        };
        check(rule(4, 4, 8, 8).keep == pr::PoolKeep::Drafter && plain_asked == 0,
              "every context held: the drafter, nothing else planned");
        check(rule(0, 4, 8, 8).keep == pr::PoolKeep::Drafter && plain_asked == 0,
              "no automatic pool: the drafter");
        check(rule(3, 4, 3, 4).keep == pr::PoolKeep::Drafter && plain_asked == 1 && head_asked == 0,
              "plain decode holds no more: the drafter");
        const auto kept = rule(3, 4, 5, 4);
        check(kept.keep == pr::PoolKeep::Head && kept.head && kept.plain == 5 && head_asked == 1,
              "the head at 4 of plain's 5 (4 streams): kept");
        check(rule(3, 4, 5, 3).keep == pr::PoolKeep::Plain, "the head at 3 of 4: plain decode");
        check(rule(3, 4, 5, std::nullopt).keep == pr::PoolKeep::Plain,
              "the head does not fit: plain decode");
    }
    // The drafters a start attaches beside the target (product::attached_drafter_memory, the
    // server's plan and ninferno-ui's): a GGUF drafter file by its size in the target's listing;
    // a drafter repository read only with a fetch (none here: offline).
    {
        pr::SpeculationSubject attached;
        ninferno::spec::DrafterDecl assistant;
        assistant.name = "assistant";
        assistant.kind = "gemma4_assistant";
        assistant.gguf = "Assistant";
        ninferno::spec::DrafterDecl dflash;
        dflash.name = "dflash";
        dflash.kind = "dflash";
        dflash.repo = "z-lab/Qwen3.5-9B-DFlash";
        attached.attached_drafters = {assistant, dflash};
        const std::vector<pr::QuantListing> listing{
            {.quant = "Q4_K_M", .format = "gguf", .bytes = 5ULL << 30},
            {.quant = "assistant-Q8_0", .format = "gguf", .bytes = 600ULL << 20, .drafter = true},
            {.quant = "mmproj-F16", .format = "gguf", .bytes = 900ULL << 20, .drafter = true}};
        const auto memory = pr::attached_drafter_memory(attached, listing, "qwen3_5_text", 5120, nullptr);
        check(memory.bytes == 600ULL << 20 && !memory.state,
              "the GGUF drafter file's size; the repository's not read offline");
        // Whether a start attaches them (choice/DECISIONS 45): the speculation served drafts with
        // one, by its backend or, each request its own, either thinking flag's.
        pr::SpeculationReport report;
        report.engine.backend = ninferno::SpeculativeBackend::None;
        check(!pr::uses_attached_drafter(attached, report), "plain decode: none attached");
        report.engine.backend = ninferno::SpeculativeBackend::DFlash;
        check(pr::uses_attached_drafter(attached, report), "DFlash with a separate DFlash drafter");
        pr::SpeculationSubject mtp_only;
        mtp_only.attached_drafters = {dflash};
        report.engine.backend      = ninferno::SpeculativeBackend::Mtp;
        check(!pr::uses_attached_drafter(mtp_only, report),
              "the model's own MTP layers: the DFlash drafter not attached");
        report.per_request      = true;
        report.with_thinking    = pr::SpeculationReport::ThinkingVariant{.label = "mtp:5"};
        report.without_thinking = pr::SpeculationReport::ThinkingVariant{.label = "dflash:7"};
        check(pr::uses_attached_drafter(mtp_only, report),
              "each request its own: one flag's DFlash attaches it");
        report.per_request = false;
        check(!pr::uses_attached_drafter(mtp_only, report),
              "one speculation for every request: only the served one counts");
        report.engine.backend = ninferno::SpeculativeBackend::DFlash;
        report.engine.drafter = "local.ninferno";
        check(!pr::uses_attached_drafter(mtp_only, report),
              "a drafter file given (--drafter) drafts in place of the model's own");
        check(pr::backend_of_drafter_kind("gemma4_assistant") == ninferno::SpeculativeBackend::Mtp &&
                  pr::backend_of_drafter_kind("dflash2") == ninferno::SpeculativeBackend::DFlash2 &&
                  pr::backend_of_drafter_kind("eagle") == ninferno::SpeculativeBackend::None,
              "a drafter kind's backend");
        // A Qwen DFlash companion is stored with its projections at Q8 (the recipe's
        // companion_drafter_q8): z-lab/Qwen3.5-9B-DFlash's 2.41 GiB of BF16 take 1.28 GiB; a
        // Gemma drafter stays as published.
        const std::uint64_t published = 2583816465ULL;
        const std::uint64_t stored    = ninferno::convert::companion_drafter_bytes("qwen3_5_text", "dflash", "", published);
        check(stored > 1.27 * (1ULL << 30) && stored < 1.29 * (1ULL << 30),
              "a Qwen DFlash companion at Q8: " + std::to_string(stored));
        check(ninferno::convert::companion_drafter_bytes("gemma4_text", "dflash", "", published) == published,
              "another family's drafter as published");
        // A declared quantization (the 27B's DFlash2 declares q8) replaces every projection,
        // the kernel projections and selector too.
        const std::uint64_t dflash2 = 3848744960ULL; // z-lab/Qwen3.8-27B-DFlash2, 3.584 GiB
        check(ninferno::convert::companion_drafter_bytes("qwen3_5_text", "dflash2", "q8", dflash2) ==
                      static_cast<std::uint64_t>(static_cast<double>(dflash2) * 8.5 / 16.0) &&
                  ninferno::convert::companion_drafter_bytes("qwen3_5_text", "dflash2", "", dflash2) >
                      ninferno::convert::companion_drafter_bytes("qwen3_5_text", "dflash2", "q8", dflash2),
              "a declared q8 quantizes the whole drafter; the recipe keeps DFlash2's kernels BF16");
    }
}

// ---- 5. Facts the device side declares.

void test_device_facts() {
    rt::PlanDeviceFacts d = card(32, 120, true, true);
    check(rt::plan_idle_free_bytes(d) == d.total_bytes - 512 * kMiB - 1536 * kMiB,
          "idle WDDM desktop card");
    d.drives_display = false;
    check(rt::plan_idle_free_bytes(d) == d.total_bytes - 512 * kMiB, "idle headless card");
    check(rt::plan_board_total_bytes(32) <= static_cast<std::uint64_t>(31.36 * kGiB),
          "a named 32 GiB board is no larger than the 5090 reads under Linux");
    rt::PlanModelFacts m = qwen27b(0);
    check(rt::plan_kv_bytes_per_token(m, KvCacheStorage::Int8Group64) == 64ULL * 528,
          "int8 KV per token (D256: 528 per head)");
    check(rt::plan_kv_bytes_per_token(m, KvCacheStorage::BFloat16) == 64ULL * 1024, "bf16");
    check(rt::plan_kv_bytes_per_token(m, KvCacheStorage::Fp8E4M3Row256) == 64ULL * 516, "fp8");
    check(rt::plan_kv_bytes_per_token(m, KvCacheStorage::Nvfp4Group16) == 64ULL * 288, "nvfp4");
    check(rt::plan_kv_bytes_per_token(m, KvCacheStorage::Fp8KeyNvfp4Value) == 64ULL * 402, "k8v4");
    // An unknown shape is not planned, with a reason.
    rt::PlanModelFacts unknown;
    const rt::DryRunPlan plan = rt::dry_run_plan(unknown, card(24, 89), serve_like(0, 1));
    check(!plan.planned && !plan.reason.empty(), "an unknown shape says why");
}

} // namespace

// A row's plan and estimate are of the storages a start of it runs (product::start_axes over the
// row's choice), whatever activation mode is pinned; a mode the family does not run is refused.
void test_rows_at_choice() {
    const rt::PlanDeviceFacts device = card(16, 89);
    struct Family {
        const char* name;
        rt::PlanModelFacts (*make)(std::uint64_t);
        bool int8;
    };
    for (const Family& family : {Family{"qwen27b", qwen27b, true}, Family{"gemma31b", gemma31b, false}}) {
        pr::PlanSourceFacts shape;
        shape.model = family.make(9 * kGiB);
        const std::vector<pr::QuantListing> listing = {
            {.quant = "Q4_K_M", .format = "gguf", .bytes = 9 * kGiB, .bits = 4}};
        for (const char* act : {"", "exact", "fast", "int8"}) {
            for (const std::uint32_t context : {8192U, 65536U}) {
                const std::string what = std::string(family.name) + " act '" + act + "' ctx " +
                                         std::to_string(context);
                pr::DryRunInputs inputs;
                inputs.device = device;
                inputs.gpu    = pr::gpu_key_of(device);
                const pr::LaunchAsk launch{.max_context = context, .spec = "none", .activations = act};
                const pr::LaunchPlanOptions o = pr::launch_options(launch);
                inputs.options                = o.options;
                inputs.ask                    = o.ask;
                inputs.pins.activations       = launch.activations_pinned();
                const pr::QuantPlan plan      = pr::plan_quants(shape, listing, inputs);
                const pr::QuantRow& row       = plan.rows.front();
                const std::string refusal = pr::activation_refusal(
                    pr::with_weights(shape, row.weight_bytes, row.basis), o.options.activations);
                check(refusal.empty() == !(std::string(act) == "int8" && !family.int8),
                      what + ": the shared check refuses only a mode the family does not run");
                if (std::string(act) == "int8" && !family.int8) {
                    check(!row.candidate.supported && row.unsupported_code == "activations_not_run" &&
                              row.why_unsupported == refusal,
                          what + ": refused");
                    // The row's plan keeps the verdict: planned, not fitting, its code and words
                    // (a named quantization's plan shows this row, never a fit at a mode never run).
                    check(row.planned.plan.planned && !row.planned.plan.fits &&
                              row.planned.plan.refused_code == "activations_not_run" &&
                              row.planned.plan.reason == refusal && !row.estimate,
                          what + ": the row's plan is refused");
                    const nlohmann::json doc = pr::dry_plan_document(
                        shape, "test/repo:Q4_K_M", "hub", inputs, row.planned, &plan);
                    check(doc["plan"]["fits"] == false &&
                              doc["plan"]["reason_code"] == "activations_not_run" &&
                              doc["quants"][0]["unsupported_code"] == "activations_not_run",
                          what + ": the document says fits false with the code");
                    // A plan of the same inputs made elsewhere (a CLI's, a fallback) is refused too.
                    pr::PlannedSpeculation fallback = pr::plan_model(
                        pr::with_weights(shape, row.weight_bytes, row.basis), inputs);
                    pr::refuse_unrun_activations(pr::with_weights(shape, row.weight_bytes, row.basis),
                                                 inputs, fallback);
                    check(!fallback.plan.fits && fallback.plan.refused_code == "activations_not_run",
                          what + ": a fallback plan is refused");
                    continue;
                }
                const auto served = pr::choose_at_weights(
                    pr::with_weights(shape, row.weight_bytes, row.basis),
                    pr::inputs_for(inputs, row.listing), inputs.pins, row.listing.quant);
                check(served && served->choice.chosen && row.planned.plan.planned, what + ": chosen");
                if (!served || !served->choice.chosen || !row.planned.plan.planned) { continue; }
                const pr::StartAxes axes = pr::start_axes(*served, inputs.pins);
                check(!axes.kv || *axes.kv == row.planned.plan.kv_cache,
                      what + ": the row's KV codec is the start's");
                check(!axes.state || *axes.state == row.planned.plan.recurrent,
                      what + ": the row's state is the start's");
                check(row.estimate && std::abs(row.estimate->decode_per_stream - served->decode_at_depth) < 1e-6,
                      what + ": the row's decode is the choice's");
                if (!std::string(act).empty()) {
                    const auto& c = served->choice.scored[*served->choice.chosen].candidate;
                    check(c.activations == o.options.activations, what + ": the pin is kept");
                }
            }
        }
    }
}

int main() {
    // The plan's mechanics on the published profiles: the registry without the lab's measured
    // pairs (test_spec_pairs tests those), so a campaign's new pairs move none of these checks
    // (a pair group's preliminary choice would stand where the published profile is asked for).
    {
        std::ifstream in(std::string(NINFERNO_SOURCE_DIR) + "/src/spec/data/drafter_registry.json");
        nlohmann::json doc = nlohmann::json::parse(in);
        doc["pairs"]       = nlohmann::json::array();
        const auto path    = std::filesystem::temp_directory_path() / "ninferno_plan_dry_run_registry.json";
        std::ofstream(path) << doc.dump();
        ninferno::spec::Registry::use_file(path);
    }
    test_seams();
    test_family_table();
    test_launch_options_match_serve();
    test_device_facts();
    test_rule();
    test_speculation_order();
    test_properties();
    test_rows_at_choice();
    if (failures != 0) {
        std::cerr << failures << " failure(s)\n";
        return 1;
    }
    std::cout << "plan dry run OK\n";
    return 0;
}
