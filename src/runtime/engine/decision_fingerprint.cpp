#include "runtime/engine/decision_fingerprint.h"

#include "core/sha256.h"

#include <array>
#include <filesystem>
#include <utility>
#include <bit>
#include <cstdio>
#include <string>
#include <vector>

#if __has_include(<cuda_runtime_api.h>)
#    include <cuda_runtime_api.h> // CUDART_VERSION only; nothing here calls CUDA
#endif

// Compile definitions of this source alone (src/runtime/CMakeLists.txt).
#ifndef NINFERNO_VERSION_STRING
#    define NINFERNO_VERSION_STRING "unknown"
#endif
#ifndef NINFERNO_CUDA_EMBED_PTX
#    define NINFERNO_CUDA_EMBED_PTX 0
#endif
#ifndef NINFERNO_CUDA_ARCHITECTURES
#    define NINFERNO_CUDA_ARCHITECTURES "unknown"
#endif

namespace ninferno::runtime {

// The commit this tree was built at, read at every build (cmake/GitCommit.cmake writes its
// definition); "unknown" without git or a checkout.
const char* build_git_commit() noexcept;

namespace {

// Read by the products, steering no decision: outputs (dumps, records, debug files), I/O modes,
// credentials, update checks and where a tool finds its inputs (the desktop UI's access key,
// catalog and updater, apps/ui). Every decision is an option. The <dump>_ROWS names are built at
// run time from their dump's name (models/common/layer_dump.h).
constexpr std::array<std::string_view, 22> kNonDecisionVariables{
    "NINFERNO_DENSE_DUMP_ROWS",
    "NINFERNO_GEMMA4_DUMP_ROWS",
    "NINFERNO_QWEN_DUMP_ROWS",
    "NINFERNO_API_KEY",
    "NINFERNO_DEBUG_CUDA_ERROR_FILE",
    "NINFERNO_DEBUG_TOKEN_DOMAIN_FILE",
    "NINFERNO_DEBUG_FREEZE_FILE",
    "NINFERNO_DEBUG_NAN_FILE",
    "NINFERNO_DEBUG_TRAP_FILE",
    "NINFERNO_DENSE_DUMP",
    "NINFERNO_GEMMA4_DUMP",
    "NINFERNO_PDFIUM",
    "NINFERNO_QWEN_DUMP",
    "NINFERNO_QWEN_DUMP_KINDS",
    "NINFERNO_ROUTE_RECORD",
    "NINFERNO_SCORE_ARGMAX",
    "NINFERNO_SCORE_LOGITS",
    "NINFERNO_UPDATE_CHECK",
    "NINFERNO_UPDATE_URL",
    "NINFERNO_UI_KEY",
    "NINFERNO_CATALOG_URL",
    "NINFERNO_UPDATER",
};

// The maintainer options' keys in the options section with their release values: one given
// otherwise is taint.
constexpr std::array<std::pair<std::string_view, std::string_view>, 8> kMaintainerDefaults{{
    {"route_table", "unset"},
    {"route_dir", "unset"},
    {"dense_tune", "off"},
    {"moe_tune", "off"},
    {"tune_cache", "unset"},
    {"debug_prefill_cuts", ""},
    {"debug_poison", "unset"},
    {"debug_poison_skip", ""},
}};

std::string number(std::uint64_t value) { return std::to_string(value); }

std::string number(int value) { return std::to_string(value); }

std::string number(std::uint32_t value) { return std::to_string(value); }

std::string flag(bool value) { return value ? "1" : "0"; }

std::string given(const std::filesystem::path& path) { return path.empty() ? "unset" : "set"; }

std::string bits(float value) {
    char text[16];
    std::snprintf(text, sizeof(text), "0x%08x", std::bit_cast<std::uint32_t>(value));
    return text;
}

std::string bits(double value) {
    char text[24];
    std::snprintf(text, sizeof(text), "0x%016llx",
                  static_cast<unsigned long long>(std::bit_cast<std::uint64_t>(value)));
    return text;
}

template <class T>
std::string optional_number(const std::optional<T>& value) {
    return value ? number(*value) : std::string("unset");
}

std::string joined(const std::vector<std::string>& items) {
    std::string out;
    for (const auto& item : items) { out += (out.empty() ? "" : ",") + item; }
    return out;
}

std::string_view activation_name(ActivationMode mode) {
    switch (mode) {
    case ActivationMode::Fast:
        return "fast";
    case ActivationMode::Exact:
        return "exact";
    case ActivationMode::Int8:
        return "int8";
    }
    return "invalid";
}

std::string_view purpose_name(EnginePurpose purpose) {
    switch (purpose) {
    case EnginePurpose::Generation:
        return "generation";
    case EnginePurpose::CausalScoring:
        return "causal_scoring";
    }
    return "invalid";
}

std::string_view backend_name(SpeculativeBackend backend) {
    switch (backend) {
    case SpeculativeBackend::None:
        return "none";
    case SpeculativeBackend::Mtp:
        return "mtp";
    case SpeculativeBackend::DFlash:
        return "dflash";
    case SpeculativeBackend::DFlash2:
        return "dflash2";
    }
    return "invalid";
}

std::string_view head_name(ProposalHead head) {
    return head == ProposalHead::Optimized ? "optimized" : "full";
}

std::string_view width_name(DFlashDraftWidth width) {
    switch (width) {
    case DFlashDraftWidth::Default:
        return "default";
    case DFlashDraftWidth::Block:
        return "block";
    case DFlashDraftWidth::Drafts:
        return "drafts";
    }
    return "invalid";
}

void features_into(DecisionSection& out, const std::string& prefix, const DeviceFeatures& f) {
    out[prefix + "clusters"]        = flag(f.clusters);
    out[prefix + "mma_bf16"]        = flag(f.mma_bf16);
    out[prefix + "mma_bf16_native"] = flag(f.mma_bf16_native);
    out[prefix + "mma_f8f6f4"]      = flag(f.mma_f8f6f4);
    out[prefix + "mma_fp8"]         = flag(f.mma_fp8);
    out[prefix + "mma_mxf4nvf4"]    = flag(f.mma_mxf4nvf4);
    out[prefix + "mma_tensor"]      = flag(f.mma_tensor);
    out[prefix + "pdl"]             = flag(f.pdl);
    out[prefix + "tma"]             = flag(f.tma);
}

void driver_into(DecisionSection& out, const DeviceClass& d) {
    out["driver.memory_bus_bits"]  = number(d.memory_bus_bits);
    out["driver.memory_clock_khz"] = number(d.memory_clock_khz);
    out["driver.sm_clock_khz"]     = number(d.sm_clock_khz);
}

void escaped_into(std::string& out, std::string_view text) {
    for (const char c : text) {
        switch (c) {
        case '\\':
            out += "\\\\";
            break;
        case '\t':
            out += "\\t";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        default:
            out += c;
        }
    }
}

void section_into(std::string& out, std::string_view name, const DecisionSection& section) {
    for (const auto& [key, value] : section) {
        escaped_into(out, name);
        out += '.';
        escaped_into(out, key);
        out += '\t';
        escaped_into(out, value);
        out += '\n';
    }
}

std::string schema_line() { return "schema\t" + std::string(kDecisionFingerprintSchema) + "\n"; }

std::string short_hash(std::string_view text) { return sha256_hex(text).substr(0, 16); }

} // namespace

std::string canonical_text(const DecisionInputs& inputs) {
    std::string out = schema_line();
    // The section names in byte order, none a prefix of another: each section's lines follow the
    // one before in the order of the whole "<section>.<key>" strings.
    section_into(out, "build", inputs.build);
    section_into(out, "facts", inputs.facts);
    section_into(out, "options", inputs.options);
    section_into(out, "plan", inputs.plan);
    section_into(out, "sources", inputs.sources);
    section_into(out, "taint", inputs.taint);
    return out;
}

DecisionFingerprint decision_fingerprint(const DecisionInputs& inputs) {
    DecisionFingerprint out;
    out.canonical       = canonical_text(inputs);
    out.value           = "nf1-" + short_hash(out.canonical);
    std::string planned = schema_line();
    section_into(planned, "plan", inputs.plan);
    section_into(out.observed, "observed", inputs.observed);
    out.manifest = "nm1-" + short_hash(planned + out.observed);
    return out;
}

DecisionSection facts_of(const DeviceClass& d, const DeviceFeatures& build, int driver_version,
                         const HostFacts& host) {
    DecisionSection out;
    out["cc_major"]                = number(d.major);
    out["cc_minor"]                = number(d.minor);
    out["sms"]                     = number(d.sms);
    out["l2_bytes"]                = number(static_cast<std::uint64_t>(d.l2_bytes));
    out["smem_per_block_optin"]    = number(static_cast<std::uint64_t>(d.smem_per_block_optin));
    out["smem_per_sm"]             = number(static_cast<std::uint64_t>(d.smem_per_sm));
    out["persisting_l2_max_bytes"] = number(static_cast<std::uint64_t>(d.persisting_l2_max_bytes));
    out["global_bytes"]            = number(static_cast<std::uint64_t>(d.global_bytes));
    out["pci_device_id"]           = number(d.pci_device_id);
    out["chip"]                    = d.chip;
    out["id"]                      = d.id();
    out["max_threads_per_sm"]      = number(d.max_threads_per_sm);
    out["max_blocks_per_sm"]       = number(d.max_blocks_per_sm);
    out["registers_per_sm"]        = number(d.registers_per_sm);
    // The facts the decisions read (DeviceClass::facts), not the driver's clocks: an overclocked
    // board and a stock one of a listed id fingerprint alike.
    const MachineFacts& m              = d.facts;
    out["machine.source"]              = std::string(to_string(m.source));
    out["machine.reference_clock_khz"] = number(m.reference_clock_khz);
    out["machine.memory_clock_khz"]    = number(m.memory_clock_khz);
    out["machine.memory_bus_bits"]     = number(m.memory_bus_bits);
    out["machine.share"]               = bits(m.share);
    out["machine.mma_slow"]            = flag(m.mma_slow);
    // The driver's values are inputs only where they fed a fact (MachineFacts::driver_read: an
    // unknown id snaps them, a laptop row without its range's bottom bounds its clock by them, a
    // board on another memory than its row's prices its memory by them); a verified board, a MIG
    // slice or a laptop priced at its range's bottom never reads them.
    if (m.driver_read) { driver_into(out, d); }
    out["driver_version"] = number(driver_version);
    features_into(out, "feature.", d.features);
    features_into(out, "build_feature.", build);
    out["host.cpu_cores"]     = number(host.logical_cores);
    out["host.cpu_quota"]     = host.cpu_quota_us != 0 && host.cpu_period_us != 0
                                    ? number(host.cpu_quota_us) + "/" + number(host.cpu_period_us)
                                    : std::string("none");
    const std::uint64_t gib   = host.ram_bytes >> 30U;
    out["host.ram_gib_class"] = number(gib == 0 ? std::uint64_t{0} : std::bit_floor(gib));
    return out;
}

DecisionSection build_of() {
    DecisionSection out;
    out["version"]    = NINFERNO_VERSION_STRING;
    out["git_commit"] = build_git_commit();
#ifdef NINFERNO_BUILD_ARCH
    out["arch"] = number(NINFERNO_BUILD_ARCH);
#else
    out["arch"] = "unknown";
#endif
#ifdef NINFERNO_BUILD_ARCH_SPECIFIC
    out["arch_specific"] = "1";
#else
    out["arch_specific"] = "0";
#endif
    out["cuda_architectures"] = NINFERNO_CUDA_ARCHITECTURES; // every entry, comma-joined
    out["embed_ptx"]          = flag(NINFERNO_CUDA_EMBED_PTX != 0);
#ifdef NINFERNO_WITHOUT_NVFP4
    out["nvfp4"] = "0";
#else
    out["nvfp4"] = "1";
#endif
#ifdef CUDART_VERSION
    out["cudart"] = number(CUDART_VERSION);
#else
    out["cudart"] = "unknown";
#endif
    return out;
}

DecisionSection options_of(const EngineOptions& o, std::string_view source_identity,
                           std::string_view prefill_signature,
                           std::string_view drafter_source_identity) {
    DecisionSection out;
    out["source_identity"]       = std::string(source_identity);
    out["prefill_signature"]     = std::string(prefill_signature);
    out["purpose"]               = std::string(purpose_name(o.purpose));
    out["activations"]           = std::string(activation_name(o.activations));
    out["max_context"]           = number(o.max_context);
    out["max_context_automatic"] = flag(o.max_context_automatic);
    out["kv_capacity.mode"]      = std::string(kv_capacity_mode_name(o.kv_capacity.mode));
    out["kv_capacity.tokens"]    = number(o.kv_capacity.explicit_tokens);
    out["kv_capacity.headroom_bytes"] =
        number(static_cast<std::uint64_t>(o.kv_capacity.automatic_headroom_bytes));
    out["max_concurrency"]           = number(o.max_concurrency);
    out["concurrent_prefill"]        = flag(o.concurrent_prefill);
    out["prefill_chunk"]             = number(o.prefill_chunk);
    out["prefill_chunk_automatic"]   = flag(o.prefill_chunk_automatic);
    out["kv_lease_window"]           = number(o.kv_lease_window);
    out["kv_cache"]                  = std::string(kv_cache_storage_name(o.kv_cache));
    out["kv_cache_automatic"]        = flag(o.kv_cache_automatic);
    out["sliding_kv_cache"]          = std::string(kv_cache_storage_name(o.sliding_kv_cache));
    out["recurrent_state"]           = std::string(recurrent_state_storage_name(o.recurrent_state));
    out["recurrent_state_automatic"] = flag(o.recurrent_state_automatic);
    out["rope_yarn_factor"]          = bits(o.rope_yarn_factor);
    out["use_cuda_graph"]            = flag(o.use_cuda_graph);
    out["enable_vision"]             = flag(o.enable_vision);
    const ContextCacheOptions& c     = o.context_cache;
    out["context_cache.enabled"]     = flag(c.enabled);
    out["context_cache.device_state_slots"] = optional_number(c.device_state_slots);
    out["context_cache.host_state_slots"]   = number(c.host_state_slots);
    out["context_cache.host_kv_capacity_bytes"] =
        number(static_cast<std::uint64_t>(c.host_kv_capacity_bytes));
    out["context_cache.host_tier_automatic"]       = flag(c.host_tier_automatic);
    out["context_cache.max_private_continuations"] = optional_number(c.max_private_continuations);
    out["context_cache.max_shared_prefixes"]       = optional_number(c.max_shared_prefixes);
    out["context_cache.max_long_anchors_per_continuation"] =
        optional_number(c.max_long_anchors_per_continuation);
    const SpeculativeOptions& s     = o.speculative;
    out["speculative.backend"]      = std::string(backend_name(s.backend));
    out["speculative.draft_tokens"] = number(s.draft_tokens);
    std::vector<std::string> depths;
    for (const std::uint32_t k : s.request_draft_tokens) { depths.push_back(number(k)); }
    out["speculative.request_draft_tokens"] = joined(depths);
    out["speculative.ngram_draft_tokens"]   = number(s.ngram_draft_tokens);
    out["speculative.ngram_min_match"]      = number(s.ngram_min_match);
    out["speculative.proposal_head"]        = std::string(head_name(s.proposal_head));
    out["speculative.draft_width"]          = std::string(width_name(s.draft_width));
    out["speculative.drafter_source_identity"] = std::string(drafter_source_identity);
    out["speculative.acceptance_floor"]     = bits(s.acceptance_floor);
    std::vector<std::string> steps;
    for (const SpeculationStep& step : s.step_down) {
        steps.push_back(number(step.draft_tokens) + ":" + bits(step.acceptance_floor));
    }
    out["speculative.step_down"] = joined(steps);
    std::vector<std::string> load;
    for (const SpeculationLoad& bound : s.load) {
        load.push_back(number(bound.lanes) + ":" + number(bound.draft_tokens) + ":" +
                       std::string(width_name(bound.draft_width)));
    }
    out["speculative.load"]                = joined(load);
    out["speculative.context_limit"]       = number(s.context_limit);
    out["speculative.watchdog_window"]     = number(s.watchdog_window);
    out["speculative.watchdog_min_rounds"] = number(s.watchdog_min_rounds);
    out["speculative.shared_trip_floor"]   = bits(s.shared_trip_floor);
    const DecodePacingOptions& p           = o.pacing;
    out["pacing.decode_steps"]             = p.steps == 0 ? std::string("auto") : number(p.steps);
    out["pacing.decode_wait"]              = decode_wait_name(p.wait);
    out["pacing.cuda_sync"]                = cuda_sync_schedule_name(p.cuda_sync);
    const MaintainerOptions& m             = o.maintainer;
    out["maintainer.route_table"]          = given(m.route_table);
    out["maintainer.route_dir"]            = given(m.route_dir);
    out["maintainer.dense_tune"]           = tune_mode_name(m.dense_tune);
    out["maintainer.moe_tune"]             = tune_mode_name(m.moe_tune);
    // --tune-cache is read only while a tuner is on: alone it decides nothing and is not hashed.
    const bool tuner_on          = m.dense_tune != TuneMode::Off || m.moe_tune != TuneMode::Off;
    out["maintainer.tune_cache"] = given(tuner_on ? m.tune_cache : std::filesystem::path());
    std::vector<std::string> cuts;
    for (const std::uint64_t cut : m.debug_prefill_cuts) { cuts.push_back(number(cut)); }
    out["maintainer.debug_prefill_cuts"] = joined(cuts);
    out["maintainer.debug_poison"] =
        m.debug_poison ? number(static_cast<std::uint32_t>(*m.debug_poison)) : "unset";
    out["maintainer.debug_poison_skip"] = joined(m.debug_poison_skip);
    return out;
}

DecisionSection plan_of(const EngineOptions& resolved, const MemorySummary& memory,
                        std::span<const std::int32_t> route_breakpoints,
                        std::string_view route_table, std::uint32_t decode_steps_per_round) {
    DecisionSection out;
    out["decode.steps_per_round"] = number(decode_steps_per_round);
    out["kv_cache"]         = std::string(kv_cache_storage_name(memory.kv_cache));
    out["recurrent_state"]  = std::string(recurrent_state_storage_name(resolved.recurrent_state));
    std::vector<std::string> windows;
    for (const auto& window : memory.speculative_graph_windows) {
        windows.push_back(number(window.draft_tokens));
    }
    out["speculative_windows"] = joined(windows);
    std::vector<std::string> breakpoints;
    for (const std::int32_t t : route_breakpoints) { breakpoints.push_back(number(t)); }
    out["route_breakpoints"] = joined(breakpoints);
    out["route_table"]       = std::string(route_table);
    // The plan ladder's rung the plan runs at, and its sizes: the one startup observation that
    // changes decisions, recorded as the declared rung it selected. Which rung the selection
    // picked before the warm-up and the tier it admitted it at are observed (observed_of): two
    // starts that serve the same rung decide alike.
    const PlanSummary& plan     = memory.plan;
    out["rung"]                 = number(plan.rung);
    out["rungs"]                = number(plan.rungs);
    out["max_context"]          = number(memory.max_context);
    out["prefill_chunk"]        = number(memory.prefill_chunk);
    out["kv_tokens"]            = number(memory.kv_capacity);
    out["cached_device_states"] = number(memory.cached_device_states);
    return out;
}

DecisionSection observed_of(const MemorySummary& memory) {
    DecisionSection out;
    out["rung_selected"]        = number(memory.plan.selected_rung);
    out["rung_reserve"]         = memory.plan.reserve;
    out["host_state_slots"]     = number(memory.host_state_capacity_slots);
    out["host_kv_bytes"]        = number(static_cast<std::uint64_t>(memory.host_kv_capacity_bytes));
    return out;
}

DecisionSection observed_of(const DeviceClass& device) {
    DecisionSection out;
    if (!device.facts.driver_read) { driver_into(out, device); }
    return out;
}

DecisionSection context_cost_sources_of(const ContextCostSummary& context_cost) {
    DecisionSection out;
    // A carried machine without its board distance (" at <distance>", for the logs): its key alone
    // picks it, and the distance follows from facts the fingerprint already holds.
    std::string_view machine = context_cost.machine;
    if (machine.starts_with("carried from ")) {
        machine = machine.substr(0, machine.rfind(" at "));
    }
    out["context_cost.machine"]  = machine.empty() ? std::string("unset") : std::string(machine);
    out["context_cost.transfer"] = context_cost_preset_source_name(context_cost.transfer_source);
    out["context_cost.prefill"]  = context_cost_preset_source_name(context_cost.prefill_source);
    return out;
}

DecisionSection artifact_ids_of(std::string_view artifact_id,
                                std::string_view drafter_artifact_id) {
    DecisionSection out;
    out["artifact_id"] = std::string(artifact_id);
    if (!drafter_artifact_id.empty()) {
        out["drafter_artifact_id"] = std::string(drafter_artifact_id);
    }
    return out;
}

std::span<const std::string_view> non_decision_environment_variables() {
    return kNonDecisionVariables;
}

DecisionSection taint_of(const DecisionInputs& in) {
    DecisionSection out;
    const auto value = [](const DecisionSection& section, const char* key) {
        const auto it = section.find(key);
        return it == section.end() ? std::string() : it->second;
    };
    std::vector<std::string> maintainer;
    for (const auto& [key, release] : kMaintainerDefaults) {
        const auto it = in.options.find("maintainer." + std::string(key));
        if (it != in.options.end() && it->second != release) { maintainer.emplace_back(key); }
    }
    // --drafter-registry: a registry file in place of the compiled-in one (sources.drafter_registry
    // "file sha256:..."; product/speculation/speculation.cpp).
    if (value(in.sources, "drafter_registry").starts_with("file")) {
        maintainer.emplace_back("drafter_registry");
    }
    if (!maintainer.empty()) { out["maintainer"] = joined(maintainer); }
    std::vector<std::string> tuners;
    for (const char* tuner : {"dense", "moe"}) {
        if (in.sources.contains(std::string(tuner) + "_tune.user")) { tuners.emplace_back(tuner); }
    }
    if (!tuners.empty()) { out["user_tune_cache"] = joined(tuners); }
    // No roofline (MachineFacts::has_roofline): the cost model stands aside. A known id whose
    // driver reports no clocks (MIG, some WSL drivers) has the table's and is not tainted.
    if (value(in.facts, "machine.reference_clock_khz") == "0" ||
        value(in.facts, "machine.memory_clock_khz") == "0" ||
        value(in.facts, "machine.memory_bus_bits") == "0") {
        out["clocks"] = "0";
    }
    if (const auto lenders = value(in.sources, "route_table.lenders"); !lenders.empty()) {
        out["borrowed_routes"] = "1";
    }
    // The device's own class chain (DeviceClass::class_chain): id, chip, architecture, "*".
    if (const auto table_class = value(in.sources, "route_table.class"); !table_class.empty()) {
        const std::string arch = "sm" + value(in.facts, "cc_major") + value(in.facts, "cc_minor");
        if (table_class != value(in.facts, "id") && table_class != value(in.facts, "chip") &&
            table_class != arch && table_class != "*") {
            out["foreign_route_table"] = table_class;
        }
    }
    // A context-cost machine measured on another board that still prices a component
    // (context_cost_sources_of): the nearest measured board's numbers, not a rule over facts. One
    // whose every component an external --context-cost-presets entry replaced prices nothing.
    constexpr std::string_view carried = "carried from ";
    const char* const carried_source =
        context_cost_preset_source_name(ContextCostPresetSource::CarriedMachine);
    if (const auto machine = value(in.sources, "context_cost.machine");
        machine.starts_with(carried) &&
        (value(in.sources, "context_cost.transfer") == carried_source ||
         value(in.sources, "context_cost.prefill") == carried_source)) {
        out["carried_context_cost"] = machine.substr(carried.size());
    }
    if (value(in.sources, "speculation.profile").starts_with("local tune")) {
        out["local_spec_tune"] = "1";
    }
    if (const auto fallback = value(in.sources, "spec_fit_fallback"); !fallback.empty()) {
        out["spec_fit_fallback"] = fallback;
    }
    return out;
}

} // namespace ninferno::runtime
