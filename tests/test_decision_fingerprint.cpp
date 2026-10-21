// The decision fingerprint (src/runtime/engine/decision_fingerprint.h), on the CPU: the canonical
// text is stable and independent of insertion order; a pinned input keeps a pinned value; every
// key of every section moves the fingerprint, a plan key the manifest too and nothing else the
// manifest; every option, plan and fact the collectors read reaches their section; an environment
// variable set, set empty and unset are three inputs, and one off the decision list is none; the
// SHA-256 matches the FIPS 180-4 vectors; and every NINFERNO_* variable src/ reads is either on
// the decision list or on the list of those known to steer no decision. The file-reading half
// (decision_sources.h, ops::moe_tune_sources): a presets or MoE-tune file moves the fingerprint by
// its content, and a drafter under a non-ASCII name is named without an exception.
#include "core/device_class.h"
#include "core/sha256.h"
#include "ninferno/ops/moe.h"
#include "runtime/engine/decision_fingerprint.h"
#include "core/removed_environment.h"
#include "runtime/engine/decision_sources.h"

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <iterator>
#include <optional>
#include <regex>
#include <map>
#include <set>
#include <string>
#include <vector>

namespace {

namespace rt = ninferno::runtime;

int check(bool ok, const std::string& what) {
    if (!ok) { std::cerr << "FAIL decision_fingerprint: " << what << '\n'; }
    return ok ? 0 : 1;
}

ninferno::DeviceClass sample_device() {
    ninferno::DeviceClass d;
    d.device                  = 3; // the ordinal and the name never enter
    d.major                   = 12;
    d.minor                   = 0;
    d.sms                     = 170;
    d.l2_bytes                = 96ULL << 20;
    d.smem_per_block_optin    = 101376;
    d.smem_per_sm             = 102400;
    d.global_bytes            = 32ULL << 30;
    d.name                    = "NVIDIA GeForce RTX 5090";
    d.pci_device_id           = 0x2B85;
    d.chip                    = "gb202";
    d.max_threads_per_sm      = 1536;
    d.max_blocks_per_sm       = 32;
    d.registers_per_sm        = 65536;
    d.persisting_l2_max_bytes = 64ULL << 20;
    d.memory_bus_bits         = 512;
    d.memory_clock_khz        = 14001000;
    d.sm_clock_khz            = 2407000;
    d.features                = ninferno::device_features_for(12, 0);
    d.facts                   = ninferno::machine_facts_of(d); // the table's row: verified
    return d;
}

const rt::HostFacts kHost{.logical_cores = 32, .ram_bytes = 100ULL << 30};

rt::DecisionInputs sample_inputs() {
    rt::DecisionInputs in;
    in.facts = rt::facts_of(sample_device(), ninferno::device_features_for(12, 0), 13020, kHost);
    in.build = rt::build_of();
    ninferno::EngineOptions options;
    options.maintainer.dense_tune = ninferno::TuneMode::On; // a maintainer option: taint
    in.options = rt::options_of(options, "00112233445566778899aabbccddeeff", "sig", "");
    in.sources = {{"route_table", "gb202.json sha256:ab"}, {"route_table.class", "gb202"}};
    const std::array<std::int32_t, 3> breakpoints{17, 64, 512};
    in.plan =
        rt::plan_of(ninferno::EngineOptions{}, ninferno::MemorySummary{}, breakpoints, "ab", 1);
    in.taint = rt::taint_of(in);
    in.observed = rt::observed_of(ninferno::MemorySummary{});
    in.observed.merge(rt::observed_of(sample_device()));
    return in;
}

// Every key of every section, one at a time, given another value: the fingerprint moves for every
// section but the observed one, the manifest for the plan and the observed section only.
int each_key_moves_the_fingerprint() {
    int failures                  = 0;
    const rt::DecisionInputs base = sample_inputs();
    const auto reference          = rt::decision_fingerprint(base);
    using Member                  = rt::DecisionSection rt::DecisionInputs::*;
    const std::array<std::pair<const char*, Member>, 7> sections{{
        {"facts", &rt::DecisionInputs::facts},
        {"build", &rt::DecisionInputs::build},
        {"options", &rt::DecisionInputs::options},
        {"sources", &rt::DecisionInputs::sources},
        {"plan", &rt::DecisionInputs::plan},
        {"taint", &rt::DecisionInputs::taint},
        {"observed", &rt::DecisionInputs::observed},
    }};
    for (const auto& [name, member] : sections) {
        failures += check(!(base.*member).empty(), std::string(name) + " has keys to vary");
        for (const auto& [key, value] : base.*member) {
            rt::DecisionInputs varied = base;
            (varied.*member)[key]     = value + "x";
            const auto moved          = rt::decision_fingerprint(varied);
            const bool observed       = std::string(name) == "observed";
            const bool plan           = std::string(name) == "plan" || observed;
            failures +=
                check((moved.value != reference.value) != observed,
                      std::string(name) + "." + key +
                          (observed ? " leaves the fingerprint" : " moves the fingerprint"));
            failures += check((moved.manifest != reference.manifest) == plan,
                              std::string(name) + "." + key +
                                  (plan ? " moves the manifest" : " leaves the manifest"));
            // A key added or removed is an input too.
            rt::DecisionInputs removed = base;
            (removed.*member).erase(key);
            const auto gone = rt::decision_fingerprint(removed);
            failures += check(observed ? gone.manifest != reference.manifest
                                       : gone.value != reference.value,
                              std::string(name) + "." + key + " removed moves its hash");
        }
    }
    return failures;
}

// A collector's section moves with each field it reads.
template <class Input, class Collect>
int each_field_reaches(
    const char* what, const Input& base, const Collect& collect,
    const std::vector<std::pair<const char*, std::function<void(Input&)>>>& edits) {
    int failures         = 0;
    const auto reference = collect(base);
    for (const auto& [field, edit] : edits) {
        Input changed = base;
        edit(changed);
        failures += check(collect(changed) != reference,
                          std::string(what) + ": " + field + " reaches the section");
    }
    return failures;
}

int options_reach_their_section() {
    using ninferno::EngineOptions;
    const auto collect = [](const EngineOptions& o) { return rt::options_of(o, "id", "sig", ""); };
    std::vector<std::pair<const char*, std::function<void(EngineOptions&)>>> edits{
        {"activations", [](EngineOptions& o) { o.activations = ninferno::ActivationMode::Exact; }},
        {"purpose", [](EngineOptions& o) { o.purpose = ninferno::EnginePurpose::CausalScoring; }},
        {"max_context", [](EngineOptions& o) { o.max_context = 4096; }},
        {"max_context_automatic", [](EngineOptions& o) { o.max_context_automatic = true; }},
        {"kv_capacity",
         [](EngineOptions& o) { o.kv_capacity = ninferno::KvCapacityPolicy::automatic(); }},
        {"kv_capacity.tokens", [](EngineOptions& o) { o.kv_capacity.explicit_tokens = 4096; }},
        {"max_concurrency", [](EngineOptions& o) { o.max_concurrency = 4; }},
        {"concurrent_prefill", [](EngineOptions& o) { o.concurrent_prefill = true; }},
        {"prefill_chunk", [](EngineOptions& o) { o.prefill_chunk = 512; }},
        {"prefill_chunk_automatic", [](EngineOptions& o) { o.prefill_chunk_automatic = true; }},
        {"kv_lease_window", [](EngineOptions& o) { o.kv_lease_window = 4096; }},
        {"kv_cache", [](EngineOptions& o) { o.kv_cache = ninferno::KvCacheStorage::Int8Group64; }},
        {"kv_cache_automatic", [](EngineOptions& o) { o.kv_cache_automatic = true; }},
        {"sliding_kv_cache",
         [](EngineOptions& o) { o.sliding_kv_cache = ninferno::KvCacheStorage::Fp8E4M3Row256; }},
        {"recurrent_state",
         [](EngineOptions& o) { o.recurrent_state = ninferno::RecurrentStateStorage::FP16; }},
        {"recurrent_state_automatic", [](EngineOptions& o) { o.recurrent_state_automatic = true; }},
        {"rope_yarn_factor", [](EngineOptions& o) { o.rope_yarn_factor = 4.0F; }},
        {"use_cuda_graph", [](EngineOptions& o) { o.use_cuda_graph = false; }},
        {"enable_vision", [](EngineOptions& o) { o.enable_vision = true; }},
        {"context_cache.enabled", [](EngineOptions& o) { o.context_cache.enabled = false; }},
        {"context_cache.device_state_slots",
         [](EngineOptions& o) { o.context_cache.device_state_slots = 0; }},
        {"context_cache.host_state_slots",
         [](EngineOptions& o) { o.context_cache.host_state_slots = 2; }},
        {"context_cache.host_kv_capacity_bytes",
         [](EngineOptions& o) { o.context_cache.host_kv_capacity_bytes = 1; }},
        {"context_cache.host_tier_automatic",
         [](EngineOptions& o) { o.context_cache.host_tier_automatic = true; }},
        {"context_cache.max_private_continuations",
         [](EngineOptions& o) { o.context_cache.max_private_continuations = 2; }},
        {"context_cache.max_shared_prefixes",
         [](EngineOptions& o) { o.context_cache.max_shared_prefixes = 2; }},
        {"context_cache.max_long_anchors",
         [](EngineOptions& o) { o.context_cache.max_long_anchors_per_continuation = 2; }},
        {"speculative.backend",
         [](EngineOptions& o) { o.speculative.backend = ninferno::SpeculativeBackend::Mtp; }},
        {"speculative.draft_tokens", [](EngineOptions& o) { o.speculative.draft_tokens = 3; }},
        {"speculative.request_draft_tokens",
         [](EngineOptions& o) { o.speculative.request_draft_tokens = {2}; }},
        {"speculative.ngram_draft_tokens",
         [](EngineOptions& o) { o.speculative.ngram_draft_tokens = 0; }},
        {"speculative.ngram_min_match",
         [](EngineOptions& o) { o.speculative.ngram_min_match = 8; }},
        {"speculative.proposal_head",
         [](EngineOptions& o) { o.speculative.proposal_head = ninferno::ProposalHead::Optimized; }},
        {"speculative.draft_width",
         [](EngineOptions& o) { o.speculative.draft_width = ninferno::DFlashDraftWidth::Drafts; }},
        {"speculative.acceptance_floor",
         [](EngineOptions& o) { o.speculative.acceptance_floor = 1.5; }},
        {"speculative.step_down", [](EngineOptions& o) { o.speculative.step_down = {{2, 1.3}}; }},
        {"speculative.load",
         [](EngineOptions& o) {
             o.speculative.load = {{2, 1, ninferno::DFlashDraftWidth::Default}};
         }},
        {"speculative.context_limit", [](EngineOptions& o) { o.speculative.context_limit = 8192; }},
        {"speculative.watchdog_window",
         [](EngineOptions& o) { o.speculative.watchdog_window = 8; }},
        {"speculative.watchdog_min_rounds",
         [](EngineOptions& o) { o.speculative.watchdog_min_rounds = 8; }},
        {"speculative.shared_trip_floor",
         [](EngineOptions& o) { o.speculative.shared_trip_floor = 1.1; }},
        {"pacing.steps", [](EngineOptions& o) { o.pacing.steps = 4; }},
        {"pacing.wait", [](EngineOptions& o) { o.pacing.wait = ninferno::DecodeWait::Block; }},
        {"pacing.cuda_sync",
         [](EngineOptions& o) { o.pacing.cuda_sync = ninferno::CudaSyncSchedule::Blocking; }},
        {"maintainer.route_table", [](EngineOptions& o) { o.maintainer.route_table = "t.json"; }},
        {"maintainer.route_dir", [](EngineOptions& o) { o.maintainer.route_dir = "routes"; }},
        {"maintainer.dense_tune",
         [](EngineOptions& o) { o.maintainer.dense_tune = ninferno::TuneMode::Retune; }},
        {"maintainer.moe_tune",
         [](EngineOptions& o) { o.maintainer.moe_tune = ninferno::TuneMode::On; }},
        // maintainer.tune_cache reaches it only while a tuner is on (maintainer_options()).
        {"maintainer.debug_prefill_cuts",
         [](EngineOptions& o) { o.maintainer.debug_prefill_cuts = {700}; }},
        {"maintainer.debug_poison", [](EngineOptions& o) { o.maintainer.debug_poison = 0xFF; }},
        {"maintainer.debug_poison_skip",
         [](EngineOptions& o) { o.maintainer.debug_poison_skip = {"kv"}; }},
    };
    int failures = each_field_reaches("options", EngineOptions{}, collect, edits);
    // The artifact, its prefill signature and a separate drafter enter by id.
    const auto base = rt::options_of(EngineOptions{}, "id", "sig", "");
    failures += check(rt::options_of(EngineOptions{}, "id2", "sig", "") != base, "artifact id");
    failures +=
        check(rt::options_of(EngineOptions{}, "id", "sig2", "") != base, "prefill signature");
    failures += check(rt::options_of(EngineOptions{}, "id", "sig", "d") != base, "drafter id");
    // Paths, the ordinal, queue limits and the speculation sources do not.
    EngineOptions other;
    other.artifact_path                = "/elsewhere/model.ninferno";
    other.device                       = 5;
    other.max_pending_requests         = 3;
    other.speculative.drafter          = "/elsewhere/drafter.ninferno";
    other.speculative.decision_sources = {{"drafter_registry", "x"}};
    failures += check(rt::options_of(other, "id", "sig", "") == base,
                      "paths, the device ordinal, queue limits and sources stay out of options");
    // A maintainer option naming a path enters as given, not by the path: its file is a source.
    EngineOptions here;
    EngineOptions there;
    here.maintainer.route_table  = "/a/t.json";
    there.maintainer.route_table = "/b/other.json";
    here.maintainer.tune_cache   = "/a";
    there.maintainer.tune_cache  = "/b";
    failures +=
        check(rt::options_of(here, "id", "sig", "") == rt::options_of(there, "id", "sig", ""),
              "a maintainer path enters as set, not by its text");
    return failures;
}

int plan_reaches_its_section() {
    using ninferno::MemorySummary;
    const std::array<std::int32_t, 2> breakpoints{17, 64};
    const auto collect = [&](const MemorySummary& m) {
        return rt::plan_of(ninferno::EngineOptions{}, m, breakpoints, "ab", 1);
    };
    std::vector<std::pair<const char*, std::function<void(MemorySummary&)>>> edits{
        {"kv_cache", [](MemorySummary& m) { m.kv_cache = ninferno::KvCacheStorage::Nvfp4Group16; }},
        {"speculative_windows",
         [](MemorySummary& m) { m.speculative_graph_windows = {{.draft_tokens = 3}}; }},
        // The plan ladder's rung and its sizes are the plan: the free memory picks a declared
        // rung, and the fingerprint records which (docs/maintainer/deterministic-decisions.md).
        {"rung", [](MemorySummary& m) { m.plan.rung = 3; }},
        {"rungs", [](MemorySummary& m) { m.plan.rungs = 40; }},
        {"max_context", [](MemorySummary& m) { m.max_context = 4096; }},
        {"prefill_chunk", [](MemorySummary& m) { m.prefill_chunk = 512; }},
        {"kv_tokens", [](MemorySummary& m) { m.kv_capacity = 8192; }},
        {"cached_device_states", [](MemorySummary& m) { m.cached_device_states = 3; }},
    };
    int failures = each_field_reaches("plan", MemorySummary{}, collect, edits);
    // What the free memory measured itself never enters: two identical starts that found a few
    // MiB more or less pick the same rung and keep one fingerprint and one manifest.
    {
        MemorySummary roomier;
        roomier.plan.free_bytes               = 1ULL << 30;
        roomier.available_after_weights_bytes = 1ULL << 30;
        roomier.plan.top_shortfall_bytes      = 5;
        roomier.planned_slack_bytes           = 7;
        failures += check(collect(roomier) == collect(MemorySummary{}) &&
                              rt::observed_of(roomier) == rt::observed_of(MemorySummary{}),
                          "the free memory itself reached the plan or the observed section");
    }
    // The Host tier is sized from the Host memory, and the rung the selection picked before the
    // warm-up and its admission tier are the warm-up's measurement: observed, not planned (a
    // warm-up move keeps nf1 when it ends on the same rung).
    std::vector<std::pair<const char*, std::function<void(MemorySummary&)>>> sized{
        {"rung_selected", [](MemorySummary& m) { m.plan.selected_rung = 2; }},
        {"rung_reserve", [](MemorySummary& m) { m.plan.reserve = "soft"; }},
        {"host_state_slots", [](MemorySummary& m) { m.host_state_capacity_slots = 3; }},
        {"host_kv_bytes", [](MemorySummary& m) { m.host_kv_capacity_bytes = 3; }},
    };
    failures += each_field_reaches(
        "observed", MemorySummary{}, [](const MemorySummary& m) { return rt::observed_of(m); },
        sized);
    for (const auto& [field, edit] : sized) {
        MemorySummary changed;
        edit(changed);
        failures += check(collect(changed) == collect(MemorySummary{}),
                          std::string("plan: ") + field + " stays out of the plan");
    }
    ninferno::EngineOptions resolved;
    resolved.recurrent_state = ninferno::RecurrentStateStorage::BF16;
    failures += check(rt::plan_of(resolved, MemorySummary{}, breakpoints, "ab", 1) !=
                          collect(MemorySummary{}),
                      "plan: recurrent_state reaches the section");
    const std::array<std::int32_t, 2> other{17, 128};
    failures += check(rt::plan_of({}, MemorySummary{}, other, "ab", 1) != collect(MemorySummary{}),
                      "plan: route breakpoints reach the section");
    failures +=
        check(rt::plan_of({}, MemorySummary{}, breakpoints, "cd", 1) != collect(MemorySummary{}),
              "plan: the route table reaches the section");
    // The decode round's step count (16 under a CPU quota, 1 without, or --decode-steps) decides
    // which requests share a step's batch.
    const auto sixteen = rt::plan_of({}, MemorySummary{}, breakpoints, "ab", 16);
    failures +=
        check(sixteen != collect(MemorySummary{}) && sixteen.at("decode.steps_per_round") == "16" &&
                  collect(MemorySummary{}).at("decode.steps_per_round") == "1",
              "plan: decode.steps_per_round reaches the section");
    return failures;
}

int facts_reach_their_section() {
    using ninferno::DeviceClass;
    const auto collect = [](const DeviceClass& d) {
        return rt::facts_of(d, ninferno::device_features_for(12, 0), 13020, kHost);
    };
    std::vector<std::pair<const char*, std::function<void(DeviceClass&)>>> edits{
        {"major", [](DeviceClass& d) { d.major = 8; }},
        {"minor", [](DeviceClass& d) { d.minor = 9; }},
        {"sms", [](DeviceClass& d) { d.sms = 84; }},
        {"l2_bytes", [](DeviceClass& d) { d.l2_bytes = 1; }},
        {"smem_per_block_optin", [](DeviceClass& d) { d.smem_per_block_optin = 1; }},
        {"smem_per_sm", [](DeviceClass& d) { d.smem_per_sm = 1; }},
        {"persisting_l2_max_bytes", [](DeviceClass& d) { d.persisting_l2_max_bytes = 1; }},
        {"global_bytes", [](DeviceClass& d) { d.global_bytes = 1; }},
        {"pci_device_id", [](DeviceClass& d) { d.pci_device_id = 1; }},
        {"chip", [](DeviceClass& d) { d.chip = "gb203"; }},
        {"max_threads_per_sm", [](DeviceClass& d) { d.max_threads_per_sm = 1; }},
        {"max_blocks_per_sm", [](DeviceClass& d) { d.max_blocks_per_sm = 1; }},
        {"registers_per_sm", [](DeviceClass& d) { d.registers_per_sm = 1; }},
        {"features", [](DeviceClass& d) { d.features.tma = !d.features.tma; }},
        {"facts.source",
         [](DeviceClass& d) { d.facts.source = ninferno::FactSource::Extrapolated; }},
        {"facts.reference_clock_khz", [](DeviceClass& d) { d.facts.reference_clock_khz = 1; }},
        {"facts.memory_clock_khz", [](DeviceClass& d) { d.facts.memory_clock_khz = 1; }},
        {"facts.memory_bus_bits", [](DeviceClass& d) { d.facts.memory_bus_bits = 1; }},
        {"facts.share", [](DeviceClass& d) { d.facts.share = 0.5; }},
        {"facts.mma_slow", [](DeviceClass& d) { d.facts.mma_slow = true; }},
    };
    int failures = each_field_reaches("facts", sample_device(), collect, edits);
    // Off the verified table the driver's values feed the facts, and are inputs.
    DeviceClass unknown   = sample_device();
    unknown.pci_device_id = 0;
    unknown.facts         = ninferno::machine_facts_of(unknown);
    std::vector<std::pair<const char*, std::function<void(DeviceClass&)>>> driver{
        {"memory_bus_bits", [](DeviceClass& d) { d.memory_bus_bits = 1; }},
        {"memory_clock_khz", [](DeviceClass& d) { d.memory_clock_khz = 1; }},
        {"sm_clock_khz", [](DeviceClass& d) { d.sm_clock_khz = 1; }},
    };
    failures += check(unknown.facts.source == ninferno::FactSource::Extrapolated,
                      "facts: an unknown id is extrapolated");
    failures += each_field_reaches("facts (unknown id)", unknown, collect, driver);
    // On the verified table they are observed: the manifest's, not the fingerprint's.
    for (const auto& [field, edit] : driver) {
        DeviceClass changed = sample_device();
        edit(changed);
        failures += check(collect(changed) == collect(sample_device()) &&
                              rt::observed_of(changed) != rt::observed_of(sample_device()),
                          std::string("facts: a verified board's driver ") + field +
                              " is observed, not a fact");
    }
    failures += check(rt::observed_of(unknown).empty() &&
                          rt::observed_of(sample_device()).size() == driver.size(),
                      "observed: the driver's values only on the verified table");
    failures += check(rt::facts_of(sample_device(), {}, 13020, kHost) != collect(sample_device()),
                      "facts: build features reach the section");
    failures += check(rt::facts_of(sample_device(), ninferno::device_features_for(12, 0), 12080,
                                   kHost) != collect(sample_device()),
                      "facts: the driver version reaches the section");
    // The host: cores and instruction set as read, the RAM by its power-of-two class of GiB.
    const auto with_host = [](rt::HostFacts host) {
        return rt::facts_of(sample_device(), ninferno::device_features_for(12, 0), 13020, host);
    };
    rt::HostFacts host = kHost;
    host.logical_cores = 16;
    failures += check(with_host(host) != collect(sample_device()), "facts: host.cpu_cores");
    host           = kHost;
    host.ram_bytes = (100ULL << 30) - (300ULL << 20); // a few hundred MiB the kernel keeps
    failures += check(with_host(host) == collect(sample_device()) &&
                          collect(sample_device()).at("host.ram_gib_class") == "64",
                      "facts: RAM of 99.7 and 100 GiB is one class (64)");
    host.ram_bytes = 130ULL << 30;
    failures += check(with_host(host).at("host.ram_gib_class") == "128", "facts: 130 GiB is 128");
    host.ram_bytes = 0;
    failures += check(with_host(host).at("host.ram_gib_class") == "0", "facts: unknown RAM is 0");
    // The CPU bandwidth quota, which the decode round's default step count follows.
    host               = kHost;
    host.cpu_quota_us  = 200'000;
    host.cpu_period_us = 100'000;
    failures += check(with_host(host) != collect(sample_device()) &&
                          with_host(host).at("host.cpu_quota") == "200000/100000" &&
                          collect(sample_device()).at("host.cpu_quota") == "none",
                      "facts: host.cpu_quota");
    host.cpu_quota_us = 150'000;
    failures += check(with_host(host).at("host.cpu_quota") == "150000/100000",
                      "facts: a different quota is a different fact");
    DeviceClass renamed = sample_device();
    renamed.name        = "another board name";
    renamed.device      = 0;
    failures += check(collect(renamed) == collect(sample_device()),
                      "facts: the name and the ordinal stay out");
    return failures;
}

// The maintainer options are taint, by name, when they leave the release's values; the decode
// pacing is an option and no taint. No environment variable enters at all.
int maintainer_options() {
    int failures      = 0;
    const auto taints = [](const ninferno::EngineOptions& o) {
        rt::DecisionInputs in = sample_inputs();
        in.options            = rt::options_of(o, "id", "sig", "");
        return rt::taint_of(in);
    };
    ninferno::EngineOptions release;
    failures += check(!taints(release).contains("maintainer"), "the release's options are clean");
    ninferno::EngineOptions paced;
    paced.pacing.steps = 8;
    paced.pacing.wait  = ninferno::DecodeWait::Block;
    failures += check(!taints(paced).contains("maintainer"), "decode pacing is not taint");
    ninferno::EngineOptions lab;
    lab.maintainer.route_dir    = "routes";
    lab.maintainer.moe_tune     = ninferno::TuneMode::On;
    lab.maintainer.debug_poison = 1;
    failures += check(taints(lab).at("maintainer") == "route_dir,moe_tune,debug_poison",
                      "the maintainer options given are taint by name: " +
                          (taints(lab).contains("maintainer") ? taints(lab).at("maintainer")
                                                              : std::string("none")));
    // --tune-cache is read only while a tuner is on: alone it is not taint.
    ninferno::EngineOptions cache_only;
    cache_only.maintainer.tune_cache = "cache";
    failures += check(!taints(cache_only).contains("maintainer"),
                      "--tune-cache with both tuners off is taint");
    failures += check(rt::options_of(cache_only, "id", "sig", "") ==
                          rt::options_of(release, "id", "sig", ""),
                      "--tune-cache with both tuners off enters the fingerprint");
    cache_only.maintainer.dense_tune = ninferno::TuneMode::On;
    ninferno::EngineOptions tuner_only;
    tuner_only.maintainer.dense_tune = ninferno::TuneMode::On;
    failures += check(rt::options_of(cache_only, "id", "sig", "") !=
                          rt::options_of(tuner_only, "id", "sig", ""),
                      "--tune-cache with a tuner on does not enter the fingerprint");
    failures += check(taints(cache_only).at("maintainer") == "dense_tune,tune_cache",
                      "--tune-cache with a tuner on is not taint");
    // A drafter registry file in place of the compiled-in one is taint, the compiled-in one not.
    rt::DecisionInputs registry          = sample_inputs();
    registry.options                     = rt::options_of(release, "id", "sig", "");
    registry.sources["drafter_registry"] = "builtin sha256:ab";
    failures += check(!rt::taint_of(registry).contains("maintainer"),
                      "the compiled-in drafter registry is taint");
    registry.sources["drafter_registry"] = "file sha256:ab";
    failures += check(rt::taint_of(registry).contains("maintainer") &&
                          rt::taint_of(registry).at("maintainer") == "drafter_registry",
                      "--drafter-registry is not taint");
    return failures;
}

int taint() {
    int failures          = 0;
    rt::DecisionInputs in = sample_inputs();
    in.options["maintainer.dense_tune"] = "off";
    failures += check(rt::taint_of(in).empty(), "a clean input has no taint");
    in.facts["machine.reference_clock_khz"]   = "0";
    in.sources["dense_tune.user"]             = "absent";
    in.sources["route_table.lenders"]         = "gb206 ab";
    in.sources["route_table.class"]           = "gb206";
    in.sources["speculation.profile"]         = "local tune sha256:ab";
    in.sources["moe_tune.user"]               = "moe_tune.json sha256:ab";
    in.sources["spec_fit_fallback"]           = "shallower+captured_load";
    const auto t                              = rt::taint_of(in);
    for (const char* key : {"clocks", "user_tune_cache", "borrowed_routes",
                            "foreign_route_table", "local_spec_tune", "spec_fit_fallback"}) {
        failures += check(t.contains(key), std::string("taint names ") + key);
    }
    failures += check(t.at("user_tune_cache") == "dense,moe", "taint names both tuners' caches");
    failures += check(t.at("spec_fit_fallback") == "shallower+captured_load",
                      "taint names the --spec auto fallback");
    in.sources["route_table.class"] = "sm120";
    failures += check(!rt::taint_of(in).contains("foreign_route_table"),
                      "the architecture's own table is not foreign");
    return failures;
}

// The fingerprint follows the facts the decisions read: an overclocked RTX 4090 and a stock one
// are one fingerprint, a MIG slice of an A100 whose driver reports no clocks is not tainted, and
// an unknown id without clocks (no roofline) is.
int machine_facts_decide() {
    int failures      = 0;
    const auto inputs = [](ninferno::DeviceClass d) {
        d.facts               = ninferno::machine_facts_of(d);
        rt::DecisionInputs in = sample_inputs();
        in.facts = rt::facts_of(d, ninferno::device_features_for(d.major, d.minor), 13020, kHost);
        in.taint = rt::taint_of(in);
        in.observed = rt::observed_of(ninferno::MemorySummary{});
        in.observed.merge(rt::observed_of(d));
        return in;
    };
    ninferno::DeviceClass stock       = sample_device();
    stock.major                       = 8;
    stock.minor                       = 9;
    stock.sms                         = 128;
    stock.pci_device_id               = 0x2684;
    stock.chip                        = "ad102";
    stock.memory_bus_bits             = 384;
    stock.memory_clock_khz            = 10501000;
    stock.sm_clock_khz                = 2520000;
    ninferno::DeviceClass overclocked = stock;
    overclocked.sm_clock_khz          = 2610000;
    overclocked.memory_clock_khz      = 10751000;
    const auto a                      = rt::decision_fingerprint(inputs(stock));
    const auto b                      = rt::decision_fingerprint(inputs(overclocked));
    failures += check(a.value == b.value, "an overclocked RTX 4090 fingerprints as a stock one");
    failures += check(a.manifest != b.manifest, "the overclock is observed in the manifest");
    failures += check(inputs(stock).facts.at("machine.source") == "verified" &&
                          inputs(stock).facts.at("machine.reference_clock_khz") == "2520000" &&
                          !inputs(stock).facts.contains("driver.sm_clock_khz"),
                      "a verified board: the table's clock, not the driver's");

    ninferno::DeviceClass slice  = sample_device();
    slice.major                  = 8;
    slice.minor                  = 0;
    slice.sms                    = 14;
    slice.pci_device_id          = 0x20B0;
    slice.chip                   = "ga100";
    slice.memory_bus_bits        = 0;
    slice.memory_clock_khz       = 0;
    slice.sm_clock_khz           = 0;
    const rt::DecisionInputs mig = inputs(slice);
    failures += check(!mig.taint.contains("clocks"),
                      "a MIG A100 slice with zero driver clocks is not tainted");
    failures += check(!mig.facts.contains("driver.sm_clock_khz") &&
                          mig.observed.contains("driver.sm_clock_khz"),
                      "a MIG slice reads no driver clock: observed, not a fact");
    failures += check(mig.facts.at("machine.source") == "extrapolated" &&
                          mig.facts.at("machine.share") != inputs(stock).facts.at("machine.share"),
                      "a MIG slice: extrapolated, with its share");

    // Two laptops of one name from two OEMs (80 W and 150 W): priced at the bottom of the range,
    // one fingerprint whatever their drivers report.
    ninferno::DeviceClass laptop = stock;
    laptop.sms                   = 76;
    laptop.pci_device_id         = 0x2717;
    laptop.chip                  = "ad103";
    laptop.memory_bus_bits       = 256;
    laptop.memory_clock_khz      = 9001000;
    laptop.sm_clock_khz          = 2040000;
    ninferno::DeviceClass thin   = laptop;
    thin.sm_clock_khz            = 1455000;
    failures += check(rt::decision_fingerprint(inputs(laptop)).value ==
                          rt::decision_fingerprint(inputs(thin)).value,
                      "a 0x2717 laptop's nf1 does not follow sm_clock_khz");
    failures += check(inputs(laptop).facts.at("machine.reference_clock_khz") == "1455000",
                      "the laptop's clock is its range's bottom");

    ninferno::DeviceClass blind = slice;
    blind.pci_device_id         = 0;
    failures += check(inputs(blind).taint.contains("clocks"),
                      "an unknown id without clocks has no roofline: tainted");
    return failures;
}

int stability_and_golden() {
    int failures = 0;
    const auto a = rt::decision_fingerprint(sample_inputs());
    const auto b = rt::decision_fingerprint(sample_inputs());
    failures += check(a.value == b.value && a.manifest == b.manifest && a.canonical == b.canonical,
                      "the same inputs give the same fingerprint");
    failures += check(a.value.size() == 20 && a.value.starts_with("nf1-") &&
                          a.manifest.size() == 20 && a.manifest.starts_with("nm1-"),
                      "nf1-/nm1- and 16 hex digits: " + a.value + " " + a.manifest);
    // Insertion order does not matter; the lines are in byte order.
    rt::DecisionInputs forward;
    rt::DecisionInputs backward;
    const std::vector<std::pair<std::string, std::string>> entries{
        {"b", "2"}, {"a", "1"}, {"a.b", "3"}, {"A", "0"}, {"ab", "4"}};
    for (const auto& [k, v] : entries) { forward.options[k] = v; }
    for (auto it = entries.rbegin(); it != entries.rend(); ++it) {
        backward.options[it->first] = it->second;
    }
    failures += check(rt::canonical_text(forward) == rt::canonical_text(backward),
                      "insertion order does not move the canonical text");
    std::vector<std::string> lines;
    const std::string text = a.canonical;
    for (std::size_t at = 0; at < text.size();) {
        const std::size_t end = text.find('\n', at);
        lines.push_back(text.substr(at, end - at));
        at = end + 1;
    }
    failures += check(!lines.empty() && lines.front() == "schema\tninferno-fingerprint/1" &&
                          std::is_sorted(lines.begin() + 1, lines.end()),
                      "the schema line, then the lines in byte order");

    // A pinned input keeps its value across releases: a change here is a schema change.
    rt::DecisionInputs pinned;
    pinned.facts                    = {{"sms", "170"}, {"chip", "gb202"}};
    pinned.build                    = {{"version", "0.1.0"}};
    pinned.options                  = {{"activations", "fast"}, {"note", "tab\there\nnewline\\"}};
    pinned.sources                  = {{"route_table", "gb202.json sha256:00"}};
    pinned.plan                     = {{"max_context", "4096"}, {"kv_cache", "int8"}};
    pinned.taint                    = {{"maintainer", "dense_tune"}};
    const std::string expected_text = "schema\tninferno-fingerprint/1\n"
                                      "build.version\t0.1.0\n"
                                      "facts.chip\tgb202\n"
                                      "facts.sms\t170\n"
                                      "options.activations\tfast\n"
                                      "options.note\ttab\\there\\nnewline\\\\\n"
                                      "plan.kv_cache\tint8\n"
                                      "plan.max_context\t4096\n"
                                      "sources.route_table\tgb202.json sha256:00\n"
                                      "taint.maintainer\tdense_tune\n";
    const auto golden               = rt::decision_fingerprint(pinned);
    failures +=
        check(golden.canonical == expected_text, "pinned canonical text:\n" + golden.canonical);
    failures +=
        check(golden.value ==
                  "nf1-" + ninferno::sha256_hex(std::string_view(expected_text)).substr(0, 16),
              "the fingerprint is the SHA-256 of the canonical text");
    failures +=
        check(golden.value == "nf1-4572a375e41f6c8c", "pinned fingerprint: " + golden.value);
    failures +=
        check(golden.manifest == "nm1-24db57571fa7a906", "pinned manifest: " + golden.manifest);
    return failures;
}

int sha256_vectors() {
    int failures = 0;
    const std::array<std::pair<std::string, const char*>, 4> vectors{{
        {"", "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"},
        {"abc", "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"},
        {"abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq",
         "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1"},
        {std::string(1000000, 'a'),
         "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0"},
    }};
    for (const auto& [message, digest] : vectors) {
        failures += check(ninferno::sha256_hex(std::string_view(message)) == digest,
                          "SHA-256 of a " + std::to_string(message.size()) + "-byte message");
    }
    // Across the 55/56/64-byte padding boundaries the one-shot digest of a message equals the
    // span form's.
    for (const std::size_t n : {55U, 56U, 63U, 64U, 65U, 119U, 120U}) {
        const std::string m(n, 'q');
        const auto* bytes = reinterpret_cast<const std::uint8_t*>(m.data());
        failures += check(ninferno::sha256(std::span<const std::uint8_t>(bytes, n)) ==
                              ninferno::sha256(std::string_view(m)),
                          "span and text forms agree at " + std::to_string(n) + " bytes");
    }
    return failures;
}

// A removed decision variable that is still set gets one warning naming its replacement; unset
// ones and every other variable get none.
int removed_environment() {
    int failures      = 0;
    const auto is_set = [](std::set<std::string> names) {
        return [names](const char* name) { return names.contains(name); };
    };
    failures += check(ninferno::removed_environment_warnings(is_set({})).empty(),
                      "removed variables: a warning with none set");
    failures += check(ninferno::removed_environment_warnings(is_set({"NINFERNO_API_KEY"})).empty(),
                      "removed variables: a warning for a variable still read");
    const auto warnings = ninferno::removed_environment_warnings(
        is_set({"NINFERNO_DECODE_STEPS", "NINFERNO_MOE_TUNE_CACHE", "NINFERNO_MMQ_VERIFY"}));
    failures +=
        check(warnings.size() == 3 && warnings[0].starts_with("NINFERNO_DECODE_STEPS is set") &&
                  warnings[0].ends_with("use --decode-steps instead") &&
                  warnings[1].ends_with("use --tune-cache instead") &&
                  warnings[2].ends_with("removed with no replacement"),
              "removed variables: the warnings do not name their replacements");
    std::set<std::string> names;
    for (const auto& removed : ninferno::kRemovedEnvironmentVariables) {
        failures +=
            check(names.emplace(removed.name).second && removed.name.starts_with("NINFERNO_"),
                  std::string(removed.name) + " is listed once");
    }
    for (const auto name : rt::non_decision_environment_variables()) {
        failures += check(!names.contains(std::string(name)),
                          std::string(name) + " is both read and removed");
    }
    return failures;
}

// Every NINFERNO_* name quoted in src/ or apps/ is known to steer no decision (a decision is an
// option). A new variable fails here until it is listed, and one that would steer a decision must
// be an option instead.
int environment_inventory() {
    int failures                     = 0;
    const std::filesystem::path root = NINFERNO_SOURCE_DIR;
    std::set<std::string> found;
    std::map<std::string, std::set<std::string>> quoted_in; // name -> files, relative to the tree
    const std::regex quoted("\"(NINFERNO_[A-Z0-9_]+)\"");
    for (const char* top : {"src", "apps"}) {
        for (const auto& entry : std::filesystem::recursive_directory_iterator(root / top)) {
            if (!entry.is_regular_file()) { continue; }
            const auto ext = entry.path().extension().string();
            if (ext != ".cpp" && ext != ".h" && ext != ".cu" && ext != ".cuh") { continue; }
            std::ifstream in(entry.path(), std::ios::binary);
            const std::string text((std::istreambuf_iterator<char>(in)),
                                   std::istreambuf_iterator<char>());
            for (std::sregex_iterator it(text.begin(), text.end(), quoted), end; it != end; ++it) {
                found.insert((*it)[1].str());
                quoted_in[(*it)[1].str()].insert(
                    std::filesystem::relative(entry.path(), root).generic_string());
            }
        }
    }
    std::set<std::string> known;
    for (const auto name : rt::non_decision_environment_variables()) {
        failures += check(known.emplace(name).second, std::string(name) + " is on one list only");
    }
    // The removed decision variables are quoted only to warn that they change nothing
    // (core/removed_environment.h).
    for (const auto& removed : ninferno::kRemovedEnvironmentVariables) {
        failures += check(known.emplace(removed.name).second,
                          std::string(removed.name) + " is on one list only");
        // Quoted nowhere else: a removed name read anywhere would steer something again.
        const auto files = quoted_in.find(std::string(removed.name));
        failures +=
            check(files != quoted_in.end() &&
                      files->second == std::set<std::string>{"src/core/removed_environment.h"},
                  std::string(removed.name) + " is quoted outside src/core/removed_environment.h");
    }
    for (const auto& name : found) {
        failures +=
            check(known.contains(name), name + " is read in src/ or apps/ but is on neither list "
                                               "(runtime/engine/decision_fingerprint.cpp)");
    }
    for (const auto& name : known) {
        // <dump>_ROWS is built at run time from its dump's name (models/common/layer_dump.h).
        const bool rows = name.ends_with("_ROWS") &&
                          found.contains(name.substr(0, name.size() - std::string("_ROWS").size()));
        failures += check(found.contains(name) || rows,
                          name + " is listed but no longer read in src/ or apps/");
    }
    failures += check(found.size() > 15,
                      "the scan found the variables (" + std::to_string(found.size()) + ")");
    return failures;
}

void write_file(const std::filesystem::path& path, std::string_view bytes) {
    std::ofstream out(path, std::ios::binary | std::ios::trunc);
    out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
}

// The fingerprint of the sample inputs with one source set.
std::string with_source(const std::string& key, const std::string& value) {
    rt::DecisionInputs in = sample_inputs();
    in.sources[key]       = value;
    in.taint              = rt::taint_of(in);
    return rt::decision_fingerprint(in).value;
}

// The files decisions read enter by content: a presets file and the MoE tuner's cache move nf1
// when their bytes change, and none of the names throws, a non-ASCII one included.
int file_sources() {
    int failures   = 0;
    const auto dir = std::filesystem::temp_directory_path() /
                     ("ninferno_fingerprint_" + std::to_string(std::rand()));
    std::filesystem::create_directories(dir);

    // --context-cost-presets FILE.
    const auto presets = dir / "presets.json";
    write_file(presets, R"({"schema":"a"})");
    const std::string first = rt::context_cost_presets_source(presets);
    write_file(presets, R"({"schema":"b"})");
    const std::string second = rt::context_cost_presets_source(presets);
    failures += check(first.starts_with("presets.json sha256:") && first != second,
                      "presets: named by file name and content: " + first);
    failures += check(with_source("context_cost.presets", first) !=
                          with_source("context_cost.presets", second),
                      "presets: a changed file moves the fingerprint");
    failures += check(rt::context_cost_presets_source({}) == "none", "presets: none");
    failures += check(rt::context_cost_presets_source(dir / "missing.json") == "unreadable",
                      "presets: a missing file is unreadable");

    // The MoE tuner's cache, read only while --moe-tune times.
    const auto cache = dir / "moe_tune.json";
    ninferno::ops::set_moe_tune(ninferno::TuneMode::Off, dir);
    failures += check(ninferno::ops::moe_tune_sources().empty(), "moe tune: off reads no file");
    ninferno::ops::set_moe_tune(ninferno::TuneMode::On, dir);
    failures += check(ninferno::ops::moe_tune_sources().at("moe_tune.user") == "absent",
                      "moe tune: a cache not written yet is absent");
    write_file(cache, R"({"layers":{"a":1}})");
    const std::string tuned_a = ninferno::ops::moe_tune_sources().at("moe_tune.user");
    write_file(cache, R"({"layers":{"a":2}})");
    const std::string tuned_b = ninferno::ops::moe_tune_sources().at("moe_tune.user");
    failures += check(tuned_a.starts_with("moe_tune.json sha256:") && tuned_a != tuned_b,
                      "moe tune: named by file name and content: " + tuned_a);
    failures +=
        check(with_source("moe_tune.user", tuned_a) != with_source("moe_tune.user", tuned_b),
              "moe tune: a changed cache moves the fingerprint");
    rt::DecisionInputs tuned       = sample_inputs();
    tuned.sources["moe_tune.user"] = tuned_a;
    failures += check(rt::taint_of(tuned).at("user_tune_cache") == "moe",
                      "moe tune: the user's cache is taint");
    ninferno::ops::set_moe_tune(ninferno::TuneMode::Off, {});

    // A drafter under a non-ASCII name: named by its artifact id, "unreadable" when it is none,
    // never an exception (path::string() throws on Windows outside the ANSI code page).
    const std::filesystem::path drafter = dir / std::filesystem::path(u8"черновик-日本.ninferno");
    write_file(drafter, "not an artifact");
    std::string id = "threw";
    try {
        id = rt::drafter_identity(drafter);
    } catch (...) {}
    failures += check(id == "unreadable", "drafter: a non-artifact under a non-ASCII name: " + id);
    failures += check(rt::drafter_identity({}).empty(), "drafter: none is empty");
    failures += check(rt::drafter_identity(dir / "missing.ninferno") == "unreadable",
                      "drafter: a missing file is unreadable");
    const auto named = ninferno::named_file_sha256(drafter);
    failures += check(named && named->starts_with("\xd1\x87\xd0\xb5\xd1\x80"),
                      "a non-ASCII file name is named in UTF-8");

    std::error_code ignored;
    std::filesystem::remove_all(dir, ignored);
    return failures;
}

} // namespace

int main() {
    int failures = 0;
    failures += stability_and_golden();
    failures += each_key_moves_the_fingerprint();
    failures += options_reach_their_section();
    failures += plan_reaches_its_section();
    failures += facts_reach_their_section();
    failures += machine_facts_decide();
    failures += maintainer_options();
    failures += taint();
    failures += sha256_vectors();
    failures += environment_inventory();
    failures += removed_environment();
    failures += file_sources();
    std::cout << (failures == 0 ? "OK" : "FAIL") << " decision_fingerprint (" << failures
              << " failures)\n";
    return failures == 0 ? 0 : 1;
}
