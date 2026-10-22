#include "product/speculative_options.h"
#include "serve/generation_service.h"
#include "serve/serve_options.h"
#include "serve/translate.h"

#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <string>
#include <initializer_list>
#include <utility>
#include <vector>

#include <unistd.h>

namespace {

using namespace ninferno::serve;

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

ServeOptions parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return parse_serve_options(static_cast<int>(argv.size()), argv.data());
}

} // namespace

int main() {
    int failures = 0;

    const ServeOptions version = parse({"ninferno-serve", "--version"});
    failures += check(version.version_requested && !version.help_requested &&
                          version.model_source.empty(),
                      "--version alone is not a version request");
    failures += check(parse({"ninferno-serve", "model.ninferno", "--help"}).help_requested &&
                          parse({"ninferno-serve", "--device", "0", "-h"}).help_requested,
                      "--help or -h after other arguments is not a help request");

    // --plan prints the plan ladder and exits; --json needs it, --prepare-only excludes it.
    {
        const ServeOptions plan = parse({"ninferno-serve", "model.ninferno", "--plan", "--json"});
        const ServeOptions text = parse({"ninferno-serve", "model.ninferno", "--plan"});
        failures += check(plan.plan_only && plan.plan_json && text.plan_only && !text.plan_json &&
                              !parse({"ninferno-serve", "model.ninferno"}).plan_only,
                          "--plan [--json] is not parsed");
        const auto refused = [&](std::vector<std::string> arguments) {
            try {
                (void)parse(std::move(arguments));
            } catch (const std::invalid_argument&) { return true; }
            return false;
        };
        failures +=
            check(refused({"ninferno-serve", "model.ninferno", "--json"}) &&
                      refused({"ninferno-serve", "model.ninferno", "--plan", "--prepare-only"}),
                  "--json without --plan or --plan with --prepare-only is accepted");
    }

    const ServeOptions defaults = parse({"ninferno-serve", "model.ninferno"});
    failures += check(defaults.allow_prefix_reuse, "prefix reuse is not enabled by default");
    failures +=
        check(!defaults.preserve_thinking, "thinking history is unexpectedly preserved by default");
    failures += check(!defaults.enable_vision, "Vision is not disabled by default");
    failures += check(defaults.request_log_jsonl.empty(),
                      "request JSONL logging is not disabled by default");
    failures += check(defaults.context_cost_presets.empty(),
                      "external context-cost presets are unexpectedly configured by default");
    failures += check(defaults.log_stats_interval_ms == 5000,
                      "periodic throughput interval default mismatch");
    failures += check(defaults.drain_timeout_ms == 60000 && defaults.health_stall_ms == 60000,
                      "drain timeout or health stall threshold default mismatch");
    failures += check(defaults.media_cache_bytes == ninferno::kDefaultMediaCacheBytes &&
                          defaults.media_live_bytes == ninferno::kDefaultMediaLiveBytes &&
                          defaults.media_preprocess_threads == 0,
                      "media preparation resource defaults mismatch");
    failures += check(defaults.max_context == 0 &&
                          defaults.kv_capacity.mode == ninferno::KvCapacityMode::Automatic &&
                          defaults.kv_capacity.explicit_tokens == 0,
                      "default context and KV capacity are not automatic");
    failures += check(defaults.context_cache.host_state_slots == ninferno::kDefaultHostStateSlots &&
                          defaults.context_cache.host_kv_capacity_bytes ==
                              ninferno::kDefaultHostKvCapacityBytes,
                      "Host context-cache defaults mismatch");
    failures += check(defaults.speculative.backend == ninferno::SpeculativeBackend::None,
                      "speculative decoding is not disabled by default");
    // No timing, no alternative route table and no debug aid unless an option asks.
    failures += check(defaults.pacing == ninferno::DecodePacingOptions{} &&
                          defaults.maintainer == ninferno::MaintainerOptions{} &&
                          defaults.drafter_registry.empty(),
                      "pacing or maintainer options are not the defaults");
    {
        const ServeOptions lab = parse({"ninferno-serve",       "model.ninferno",
                                        "--decode-steps",       "8",
                                        "--decode-wait",        "block",
                                        "--cuda-sync",          "yield",
                                        "--route-table",        "t.json",
                                        "--route-dir",          "r",
                                        "--dense-tune",         "retune",
                                        "--moe-tune",           "on",
                                        "--tune-cache",         "c",
                                        "--debug-prefill-cuts", "700,64",
                                        "--debug-poison",       "nan",
                                        "--debug-poison-skip",  "kv,state",
                                        "--drafter-registry",   "reg.json"});
        failures += check(lab.pacing.steps == 8 && lab.pacing.wait == ninferno::DecodeWait::Block &&
                              lab.pacing.cuda_sync == ninferno::CudaSyncSchedule::Yield,
                          "the pacing flags are not read");
        const ninferno::MaintainerOptions& m = lab.maintainer;
        failures += check(m.route_table == "t.json" && m.route_dir == "r" &&
                              m.dense_tune == ninferno::TuneMode::Retune &&
                              m.moe_tune == ninferno::TuneMode::On && m.tune_cache == "c" &&
                              m.debug_prefill_cuts == std::vector<std::uint64_t>{700, 64} &&
                              m.debug_poison == std::uint8_t{0xFF} &&
                              m.debug_poison_skip == std::vector<std::string>{"kv", "state"} &&
                              lab.drafter_registry == "reg.json",
                          "the maintainer flags are not read");
        for (const char* bad :
             {"--decode-steps", "--decode-wait", "--dense-tune", "--debug-poison"}) {
            bool refused = false;
            try {
                (void)parse({"ninferno-serve", "model.ninferno", bad, "x"});
            } catch (const std::invalid_argument&) { refused = true; }
            failures += check(refused, "an invalid pacing or maintainer value is accepted");
        }
    }
    failures += check(defaults.response_store_max_records == kDefaultResponseStoreRecords &&
                          defaults.response_store_max_bytes == kDefaultResponseStoreBytes,
                      "Responses store defaults mismatch");
    failures += check(!defaults.model_id_override.has_value(),
                      "model id override is unexpectedly configured by default");
    failures += check(!defaults.default_thinking_budget,
                      "thinking budget is unexpectedly limited by default");
    failures += check(
        !defaults.sampling_overrides.temperature && !defaults.sampling_overrides.top_p &&
            !defaults.sampling_overrides.top_k && !defaults.sampling_overrides.presence_penalty &&
            !defaults.sampling_overrides.frequency_penalty,
        "server defaults unexpectedly override registered model sampling");
    failures += check(resolve_public_model_id(defaults, "artifact-model") == "artifact-model",
                      "artifact model id was not selected by default");

    failures += check(defaults.activations == ninferno::kDefaultActivationMode,
                      "serve activations default is not the product default");
    for (const auto& [spelling, mode] :
         std::initializer_list<std::pair<const char*, ninferno::ActivationMode>>{
             {"auto", ninferno::ActivationMode::Auto},
             {"16", ninferno::ActivationMode::Bits16},
             {"8", ninferno::ActivationMode::Bits8},
             {"4", ninferno::ActivationMode::Bits4},
             {"exact", ninferno::ActivationMode::Bits16},
             {"fast", ninferno::ActivationMode::Auto},
             {"int8", ninferno::ActivationMode::Bits8}}) {
        failures += check(
            parse({"ninferno-serve", "model.ninferno", "--activations", spelling}).activations ==
                mode,
            "--activations auto|16|8|4|exact|fast|int8");
    }
    bool unknown_activations_rejected = false;
    try {
        (void)parse({"ninferno-serve", "model.ninferno", "--activations", "a4"});
    } catch (const std::invalid_argument&) { unknown_activations_rejected = true; }
    failures += check(unknown_activations_rejected, "--activations accepted an unknown mode");
    failures += check(serve_usage_text("ninferno-serve").find("--activations auto|16|8|4") !=
                          std::string::npos,
                      "serve help omits --activations");

    const ServeOptions fp8 = parse({"ninferno-serve", "model.ninferno", "--kv-dtype", "fp8"});
    failures += check(fp8.kv_cache == ninferno::KvCacheStorage::Fp8E4M3Row256,
                      "--kv-dtype fp8 did not select row-scaled E4M3 KV");
    const ServeOptions nvfp4 = parse({"ninferno-serve", "model.ninferno", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninferno::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ServeOptions k8v4 = parse({"ninferno-serve", "model.ninferno", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninferno::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    failures += check(k8v4.sliding_kv_cache == ninferno::KvCacheStorage::BFloat16,
                      "the sliding rings are not BF16 by default");
    const ServeOptions rings =
        parse({"ninferno-serve", "model.ninferno", "--sliding-kv-dtype", "fp8"});
    failures += check(rings.sliding_kv_cache == ninferno::KvCacheStorage::Fp8E4M3Row256 &&
                          !rings.kv_cache.has_value(),
                      "--sliding-kv-dtype fp8 did not select FP8 rings alone");
    // Without --kv-dtype, or with auto, the model family chooses the KV storage.
    failures += check(
        !parse({"ninferno-serve", "model.ninferno"}).kv_cache.has_value() &&
            !parse({"ninferno-serve", "model.ninferno", "--kv-dtype", "auto"}).kv_cache.has_value(),
        "the KV storage is not the family's by default and with --kv-dtype auto");
    const ServeOptions bf16 = parse({"ninferno-serve", "model.ninferno", "--kv-dtype", "bf16"});
    failures += check(bf16.kv_cache == ninferno::KvCacheStorage::BFloat16,
                      "--kv-dtype bf16 did not select BF16 KV");
    bool rings_nvfp4_rejected = false;
    try {
        (void)parse({"ninferno-serve", "model.ninferno", "--sliding-kv-dtype", "nvfp4"});
    } catch (const std::invalid_argument&) { rings_nvfp4_rejected = true; }
    failures += check(rings_nvfp4_rejected, "--sliding-kv-dtype accepted nvfp4");
    const std::string kv_help = serve_usage_text("ninferno-serve");
    failures += check(kv_help.find("nvfp4") != std::string::npos &&
                          kv_help.find("k8v4") != std::string::npos,
                      "serve help omits a production KV storage mode");

    const ServeOptions model_alias =
        parse({"ninferno-serve", "model.ninferno", "--model-id", "deployment-alias"});
    failures +=
        check(model_alias.model_id_override == "deployment-alias" &&
                  resolve_public_model_id(model_alias, "artifact-model") == "deployment-alias",
              "explicit model id did not override the artifact identity");

    const ServeOptions context_cost =
        parse({"ninferno-serve", "model.ninferno", "--context-cost-presets", "local-costs.json"});
    failures += check(context_cost.context_cost_presets == "local-costs.json",
                      "--context-cost-presets did not preserve its path");

    const ServeOptions thinking_budget =
        parse({"ninferno-serve", "model.ninferno", "--default-thinking-budget", "37"});
    failures += check(thinking_budget.default_thinking_budget == 37,
                      "--default-thinking-budget did not preserve its positive value");
    bool zero_thinking_budget_rejected = false;
    try {
        (void)parse({"ninferno-serve", "model.ninferno", "--default-thinking-budget", "0"});
    } catch (const std::invalid_argument&) { zero_thinking_budget_rejected = true; }
    failures += check(zero_thinking_budget_rejected, "zero --default-thinking-budget was accepted");

    bool empty_model_id_rejected = false;
    try {
        (void)parse({"ninferno-serve", "model.ninferno", "--model-id", ""});
    } catch (const std::invalid_argument&) { empty_model_id_rejected = true; }
    failures += check(empty_model_id_rejected, "empty --model-id was accepted");

    bool empty_api_key_rejected = false;
    try {
        (void)parse({"ninferno-serve", "model.ninferno", "--api-key", ""});
    } catch (const std::invalid_argument&) { empty_api_key_rejected = true; }
    failures += check(empty_api_key_rejected, "empty --api-key was accepted (no authentication)");

    // The key from a file (its trailing line break dropped) or from NINFERNO_API_KEY, never both a
    // flag and a file; the environment yields to a flag.
    const std::string key_file = "/tmp/ninferno_test_api_key_" + std::to_string(::getpid());
    std::ofstream(key_file) << "s3cret-key\n";
    failures +=
        check(parse({"ninferno-serve", "model.ninferno", "--api-key-file", key_file}).api_key ==
                  "s3cret-key",
              "--api-key-file did not read the key without its line break");
    bool twice_rejected = false;
    try {
        (void)parse(
            {"ninferno-serve", "model.ninferno", "--api-key", "a", "--api-key-file", key_file});
    } catch (const std::invalid_argument&) { twice_rejected = true; }
    failures += check(twice_rejected, "--api-key with --api-key-file was accepted");
    std::ofstream(key_file) << "\n";
    bool empty_file_rejected = false;
    try {
        (void)parse({"ninferno-serve", "model.ninferno", "--api-key-file", key_file});
    } catch (const std::invalid_argument&) { empty_file_rejected = true; }
    failures += check(empty_file_rejected, "an empty --api-key-file was accepted");
    std::remove(key_file.c_str());
    ::setenv("NINFERNO_API_KEY", "env-key", 1);
    failures += check(parse({"ninferno-serve", "model.ninferno"}).api_key == "env-key",
                      "NINFERNO_API_KEY was not read");
    failures +=
        check(parse({"ninferno-serve", "model.ninferno", "--api-key", "flag"}).api_key == "flag",
              "NINFERNO_API_KEY overrode --api-key");
    ::setenv("NINFERNO_API_KEY", "", 1);
    bool empty_env_rejected = false;
    try {
        (void)parse({"ninferno-serve", "model.ninferno"});
    } catch (const std::invalid_argument&) { empty_env_rejected = true; }
    failures += check(empty_env_rejected, "an empty NINFERNO_API_KEY was accepted");
    ::unsetenv("NINFERNO_API_KEY");

    const ServeOptions dflash = parse({"ninferno-serve", "model.ninferno", "--spec", "dflash",
                                       "--draft-tokens", "15", "--lm-head-draft"});
    failures += check(dflash.speculative.backend == ninferno::SpeculativeBackend::DFlash,
                      "--spec dflash did not select DFlash");
    failures += check(dflash.speculative.draft_tokens == 15,
                      "--draft-tokens did not preserve the DFlash window");
    failures += check(dflash.speculative.proposal_head == ninferno::ProposalHead::Optimized,
                      "--lm-head-draft did not select the optimized proposal head");

    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto options = parse({"ninferno-serve", "model.ninferno", "--spec", "dflash2",
                                    "--draft-tokens", std::to_string(k), "--lm-head-draft"});
        failures +=
            check(options.speculative.backend == ninferno::SpeculativeBackend::DFlash2 &&
                      options.speculative.draft_tokens == k &&
                      options.speculative.proposal_head == ninferno::ProposalHead::Optimized,
                  "serve options did not preserve DFlash2 configuration");
    }

    // An explicit --spec without --draft-tokens takes its depth from the model at startup; a depth
    // out of range, or one without --spec, is still refused.
    for (const char* backend : {"mtp", "dflash", "dflash2"}) {
        const auto options = parse({"ninferno-serve", "model.ninferno", "--spec", backend});
        failures += check(options.draft_tokens_from_model && !options.speculation_from_profile &&
                              options.speculative.draft_tokens == 0,
                          "an explicit --spec without --draft-tokens was not left to the model");
    }
    failures +=
        check(!parse({"ninferno-serve", "model.ninferno", "--spec", "mtp", "--draft-tokens", "3"})
                   .draft_tokens_from_model,
              "an explicit --draft-tokens was left to the model");
    for (const auto& flags : std::vector<std::vector<std::string>>{
             {"--spec", "dflash2", "--draft-tokens", "16"}, {"--draft-tokens", "3"}}) {
        std::vector<std::string> args{"ninferno-serve", "model.ninferno"};
        args.insert(args.end(), flags.begin(), flags.end());
        bool refused = false;
        try {
            (void)parse(args);
        } catch (const std::invalid_argument&) { refused = true; }
        failures += check(refused, "an invalid --draft-tokens was accepted");
    }
    failures += check(
        ninferno::product::default_draft_tokens(ninferno::SpeculativeBackend::Mtp) == 3 &&
            ninferno::product::default_draft_tokens(ninferno::SpeculativeBackend::DFlash2) == 7,
        "the default draft depths changed");

    for (const char* plain : {"none", "off"}) {
        const ServeOptions options = parse({"ninferno-serve", "model.ninferno", "--spec", plain});
        failures += check(!options.speculation_from_profile &&
                              options.speculative.backend == ninferno::SpeculativeBackend::None,
                          "--spec none and --spec off run plain, as in the CLI");
    }

    // No --spec: the profile, as --spec auto; anything configuring an explicit speculation, or
    // --spec none, keeps it off.
    const ServeOptions by_default = parse({"ninferno-serve", "model.ninferno"});
    failures += check(by_default.speculation_from_profile && by_default.speculation_by_default &&
                          !by_default.per_request_speculation &&
                          by_default.speculative.backend == ninferno::SpeculativeBackend::None,
                      "serve takes the model's speculation profile by default");
    for (const auto& flags :
         std::vector<std::vector<std::string>>{{"--spec", "none"},
                                               {"--spec", "mtp", "--draft-tokens", "3"},
                                               {"--spec", "dflash", "--drafter", "d.ninferno"}}) {
        std::vector<std::string> args{"ninferno-serve", "model.ninferno"};
        args.insert(args.end(), flags.begin(), flags.end());
        const ServeOptions options = parse(args);
        failures += check(!options.speculation_from_profile && !options.speculation_by_default,
                          "an explicit speculation flag keeps the profile off");
    }
    // A drafter with no backend would run nothing: refused, as --draft-tokens alone is.
    bool lone_drafter_refused = false;
    try {
        (void)parse({"ninferno-serve", "model.ninferno", "--drafter", "d.ninferno"});
    } catch (const std::invalid_argument&) { lone_drafter_refused = true; }
    failures += check(lone_drafter_refused, "--drafter without --spec was accepted");
    const ServeOptions profiled = parse({"ninferno-serve", "model.ninferno", "--spec", "auto"});
    failures += check(profiled.speculation_from_profile && !profiled.per_request_speculation &&
                          profiled.speculative.backend == ninferno::SpeculativeBackend::None,
                      "--spec auto did not select the profile at its default");
    const ServeOptions fixed =
        parse({"ninferno-serve", "model.ninferno", "--spec", "auto", "--spec-select", "thinking"});
    failures += check(fixed.speculation_from_profile && fixed.per_request_speculation,
                      "--spec-select thinking was not kept");
    failures += check(!profiled.speculation_load, "--spec auto keeps the profile's load table");
    const ServeOptions loaded =
        parse({"ninferno-serve", "model.ninferno", "--spec", "auto", "--spec-load", "2:5,4:0"});
    failures += check(loaded.speculation_load && loaded.speculation_load->size() == 2 &&
                          (*loaded.speculation_load)[1].lanes == 4 &&
                          (*loaded.speculation_load)[1].depth == 0,
                      "--spec-load was not parsed");
    const ServeOptions unbounded =
        parse({"ninferno-serve", "model.ninferno", "--spec", "auto", "--spec-load", "none"});
    failures += check(unbounded.speculation_load && unbounded.speculation_load->empty(),
                      "--spec-load none did not replace the profile's table with no bound");
    const ServeOptions widths = parse(
        {"ninferno-serve", "model.ninferno", "--spec", "auto", "--spec-load", "2:5:drafts,4:0"});
    failures += check(widths.speculation_load && widths.speculation_load->size() == 2 &&
                          (*widths.speculation_load)[0].width == ninferno::spec::DraftWidth::Drafts,
                      "--spec-load did not read a bound's draft width");
    failures += check(profiled.speculative.shared_trip_floor == 1.4 &&
                          parse({"ninferno-serve", "model.ninferno", "--spec", "auto",
                                 "--spec-shared-floor", "0"})
                                  .speculative.shared_trip_floor == 0.0,
                      "--spec-shared-floor was not parsed (default 1.4)");
    failures += check(
        profiled.speculative.draft_width == ninferno::DFlashDraftWidth::Default &&
            parse({"ninferno-serve", "model.ninferno", "--spec", "dflash", "--draft-tokens", "3",
                   "--draft-width", "drafts"})
                    .speculative.draft_width == ninferno::DFlashDraftWidth::Drafts &&
            parse({"ninferno-serve", "model.ninferno", "--spec", "auto", "--draft-width", "block"})
                    .speculative.draft_width == ninferno::DFlashDraftWidth::Block,
        "--draft-width was not parsed (default: the engine's choice)");
    for (const std::vector<std::string>& bad :
         {std::vector<std::string>{"ninferno-serve", "model.ninferno", "--spec", "auto",
                                   "--draft-tokens", "3"},
          std::vector<std::string>{"ninferno-serve", "model.ninferno", "--spec", "dflash",
                                   "--draft-tokens", "3", "--draft-width", "half"},
          std::vector<std::string>{"ninferno-serve", "model.ninferno", "--spec", "dflash",
                                   "--draft-tokens", "3", "--draft-width", "Block"},
          std::vector<std::string>{"ninferno-serve", "model.ninferno", "--spec", "mtp",
                                   "--draft-tokens", "3", "--draft-width", "drafts"},
          std::vector<std::string>{"ninferno-serve", "model.ninferno", "--spec", "auto",
                                   "--spec-load", "4:3,2:5"},
          std::vector<std::string>{"ninferno-serve", "model.ninferno", "--spec", "auto",
                                   "--spec-load", "1:3"},
          std::vector<std::string>{"ninferno-serve", "model.ninferno", "--spec", "auto",
                                   "--spec-load", "2:0,4:3"},
          std::vector<std::string>{"ninferno-serve", "model.ninferno", "--spec-select",
                                   "sometimes"},
          std::vector<std::string>{"ninferno-serve", "model.ninferno", "--spec", "auto",
                                   "--drafter", "d.ninferno"},
          std::vector<std::string>{"ninferno-serve", "model.ninferno", "--ngram-min-match", "0"}}) {
        bool rejected = false;
        try {
            (void)parse(bad);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "an invalid --spec auto combination was accepted");
    }

    // Without --spec the profile is on, so its bounds and widths apply as under --spec auto.
    const ServeOptions default_bounds =
        parse({"ninferno-serve", "model.ninferno", "--spec-load", "2:3", "--draft-width", "block"});
    failures += check(default_bounds.speculation_from_profile &&
                          default_bounds.speculation_load.has_value(),
                      "--spec-load and --draft-width bound the default profile");

    const ServeOptions dflash_vision = parse({"ninferno-serve", "model.ninferno", "--spec",
                                              "dflash", "--draft-tokens", "15", "--vision"});
    failures +=
        check(dflash_vision.enable_vision &&
                  dflash_vision.speculative.backend == ninferno::SpeculativeBackend::DFlash &&
                  dflash_vision.speculative.draft_tokens == 15,
              "serve options did not preserve combined DFlash and Vision features");

    bool implicit_backend_rejected = false;
    try {
        (void)parse({"ninferno-serve", "model.ninferno", "--draft-tokens", "3"});
    } catch (const std::invalid_argument&) { implicit_backend_rejected = true; }
    failures += check(implicit_backend_rejected, "--draft-tokens selected a backend implicitly");

    const ServeOptions configured = parse({"ninferno-serve",
                                           "model.ninferno",
                                           "--no-prefix-reuse",
                                           "--vision",
                                           "--usage-chunk-choice",
                                           "--max-concurrency",
                                           "4",
                                           "--max-pending-requests",
                                           "12",
                                           "--pending-timeout-ms",
                                           "2500",
                                           "--max-context",
                                           "4096",
                                           "--kv-capacity",
                                           "8192",
                                           "--log-stats-interval-ms",
                                           "0",
                                           "--drain-timeout-ms",
                                           "1500",
                                           "--health-stall-ms",
                                           "0",
                                           "--preserve-thinking",
                                           "--media-cache-mib",
                                           "256",
                                           "--media-live-mib",
                                           "512",
                                           "--media-preprocess-threads",
                                           "6"});
    failures += check(!configured.allow_prefix_reuse,
                      "--no-prefix-reuse did not disable server prefix reuse");
    failures += check(configured.context_cache.host_state_slots == 0 &&
                          configured.context_cache.host_kv_capacity_bytes == 0,
                      "root-only server mode retained default Host capacities");
    failures += check(configured.enable_vision, "--vision did not enable Vision");
    failures +=
        check(configured.usage_chunk_choice, "--usage-chunk-choice did not reach serving options");
    failures += check(configured.preserve_thinking == true,
                      "--preserve-thinking did not reach serving options");
    failures +=
        check(configured.max_concurrency == 4, "--max-concurrency did not reach serving options");
    failures += check(configured.max_context == 4096 &&
                          configured.kv_capacity.mode == ninferno::KvCapacityMode::Explicit &&
                          configured.kv_capacity.explicit_tokens == 8192,
                      "context and KV capacity options were not kept distinct");
    failures += check(configured.max_pending_requests == 12,
                      "--max-pending-requests did not reach serving options");
    failures += check(configured.pending_timeout_ms == 2500,
                      "--pending-timeout-ms did not reach serving options");
    failures += check(configured.log_stats_interval_ms == 0,
                      "--log-stats-interval-ms did not disable periodic reporting");
    failures += check(configured.drain_timeout_ms == 1500 && configured.health_stall_ms == 0,
                      "--drain-timeout-ms or --health-stall-ms did not reach serving options");
    failures += check(configured.media_cache_bytes == (256ULL << 20) &&
                          configured.media_live_bytes == (512ULL << 20) &&
                          configured.media_preprocess_threads == 6,
                      "media preparation limits did not reach serving options");

    const ServeOptions logging =
        parse({"ninferno-serve", "model.ninferno", "--log-level", "debug"});
    failures += check(logging.log_level == ninferno::product::LogLevel::Debug,
                      "log level did not reach serving options");

    const ServeOptions context_cache =
        parse({"ninferno-serve", "model.ninferno", "--device-state-slots", "3",
               "--host-state-slots", "5", "--host-kv-mib", "64", "--max-private-continuations", "9",
               "--max-shared-prefixes", "4", "--max-long-anchors-per-continuation", "2"});
    failures += check(context_cache.context_cache.enabled &&
                          context_cache.context_cache.device_state_slots == 3 &&
                          context_cache.context_cache.host_state_slots == 5 &&
                          context_cache.context_cache.host_kv_capacity_bytes == (64ULL << 20) &&
                          context_cache.context_cache.max_private_continuations == 9 &&
                          context_cache.context_cache.max_shared_prefixes == 4 &&
                          context_cache.context_cache.max_long_anchors_per_continuation == 2,
                      "context-cache capacities did not reach serving options");
    bool disabled_cache_capacity_rejected = false;
    try {
        (void)parse(
            {"ninferno-serve", "model.ninferno", "--no-prefix-reuse", "--host-kv-mib", "64"});
    } catch (const std::invalid_argument&) { disabled_cache_capacity_rejected = true; }
    failures += check(disabled_cache_capacity_rejected,
                      "root-only server mode accepted context-cache capacity options");

    const ServeOptions response_store =
        parse({"ninferno-serve", "model.ninferno", "--response-store-max-records", "42",
               "--response-store-max-mib", "8"});
    failures += check(response_store.response_store_max_records == 42 &&
                          response_store.response_store_max_bytes == (8ULL << 20),
                      "Responses store limits did not reach serving options");

    const ServeOptions sampling =
        parse({"ninferno-serve", "model.ninferno", "--temperature", "0", "--top-p", "0.9",
               "--top-k", "20", "--min-p", "0.1", "--presence-penalty", "1.25",
               "--frequency-penalty", "-0.5", "--seed", "0"});
    failures += check(sampling.sampling_overrides.temperature == 0.0F &&
                          sampling.sampling_overrides.top_p == 0.9F &&
                          sampling.sampling_overrides.top_k == 20 &&
                          sampling.sampling_overrides.min_p == 0.1F &&
                          sampling.sampling_overrides.presence_penalty == 1.25F &&
                          sampling.sampling_overrides.frequency_penalty == -0.5F &&
                          sampling.sampling_overrides.seed == 0,
                      "server sampling flags did not preserve explicit values and zeros");
    bool oversized_top_k_rejected = false;
    try {
        (void)parse({"ninferno-serve", "model.ninferno", "--top-k", "65"});
    } catch (const std::invalid_argument&) { oversized_top_k_rejected = true; }
    failures += check(oversized_top_k_rejected,
                      "server accepted top_k beyond the executable candidate domain");

    GenerationRequest request;
    request.max_tokens   = 1;
    const auto semantics = resolve_prompt_semantics(request, defaults);
    failures += check(!semantics.reasoning_effort && !semantics.enable_thinking &&
                          !semantics.reasoning_effort,
                      "omitted reasoning effort did not resolve to the template default");
    failures +=
        check(to_request_options(request, defaults, semantics, true).execution.allow_prefix_reuse,
              "resolved read-write cache policy did not reach Engine options");
    failures +=
        check(!to_request_options(request, defaults, semantics, false).execution.allow_prefix_reuse,
              "resolved disabled cache policy inherited external enablement");
    const ninferno::RequestOptions inherited_sampling =
        to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse);
    failures += check(inherited_sampling.execution.sampling.temperature == 0.0F &&
                          inherited_sampling.execution.sampling.top_p == 0.9F &&
                          inherited_sampling.execution.sampling.seed == 0,
                      "server sampling overrides did not reach Engine options");
    request.sampling.temperature = 1.1;
    failures += check(to_request_options(request, sampling, semantics, sampling.allow_prefix_reuse)
                              .execution.sampling.temperature == 1.1F,
                      "request sampling override did not win over the server override");
    failures += check(
        to_request_options(request, thinking_budget, semantics, thinking_budget.allow_prefix_reuse)
                .execution.thinking.budget == 37,
        "thinking-enabled request did not inherit the server budget");
    request.enable_thinking = false;
    const auto non_thinking = resolve_prompt_semantics(request, thinking_budget);
    failures += check(!non_thinking.reasoning_effort,
                      "disabled thinking retained an effective reasoning effort");
    failures += check(!to_request_options(request, thinking_budget, non_thinking,
                                          thinking_budget.allow_prefix_reuse)
                           .execution.thinking.budget,
                      "non-thinking request inherited the server thinking budget");
    request.enable_thinking.reset();
    request.reasoning_effort   = ninferno::ReasoningEffort::Low;
    const auto explicit_effort = resolve_prompt_semantics(request, defaults);
    failures += check(explicit_effort.reasoning_effort == ninferno::ReasoningEffort::Low &&
                          explicit_effort.enable_thinking == true,
                      "explicit reasoning effort did not remain the effective effort");
    request.reasoning_effort.reset();
    failures += check(resolve_prompt_semantics(request, configured).preserve_thinking == true,
                      "server preserve-thinking default was not resolved");
    request.preserve_thinking = false;
    failures += check(resolve_prompt_semantics(request, configured).preserve_thinking == false,
                      "request preserve-thinking override did not win");

    {
        // Every option the help describes (a line opening with it) is in the synopsis too.
        const std::string help     = serve_usage_text("ninferno-serve");
        const std::string synopsis = help.substr(0, help.find("\n       serves"));
        const std::string lead     = "\n       --";
        for (std::size_t at = help.find(lead); at != std::string::npos;
             at             = help.find(lead, at + 1)) {
            const std::size_t begin = at + lead.size() - 2;
            const std::string option =
                help.substr(begin, help.find_first_of(" ,:;", begin) - begin);
            failures += check(synopsis.find(option) != std::string::npos,
                              ("serve synopsis omits " + option).c_str());
        }
    }
    failures +=
        check(serve_usage_text("ninferno-serve").find("--no-prefix-reuse") != std::string::npos,
              "serve help omits --no-prefix-reuse");
    failures += check(serve_usage_text("ninferno-serve").find("--host-kv-mib") != std::string::npos,
                      "serve help omits context-cache capacities");
    failures += check(serve_usage_text("ninferno-serve").find("device-state=2x max-concurrency") !=
                          std::string::npos,
                      "serve help omits context-cache defaults");
    failures +=
        check(serve_usage_text("ninferno-serve").find("--preserve-thinking") != std::string::npos,
              "serve help omits --preserve-thinking");
    failures += check(serve_usage_text("ninferno-serve").find("--default-thinking-budget") !=
                          std::string::npos,
                      "serve help omits --default-thinking-budget");
    failures += check(serve_usage_text("ninferno-serve").find("--vision") != std::string::npos,
                      "serve help omits --vision");
    failures +=
        check(serve_usage_text("ninferno-serve").find("--usage-chunk-choice") != std::string::npos,
              "serve help omits --usage-chunk-choice");
    failures += check(serve_usage_text("ninferno-serve").find("--log-stats-interval-ms") !=
                          std::string::npos,
                      "serve help omits --log-stats-interval-ms");
    failures += check(serve_usage_text("ninferno-serve").find("--log-level") != std::string::npos,
                      "serve help omits the log-level control");
    failures += check(serve_usage_text("ninferno-serve").find("--media-preprocess-threads") !=
                          std::string::npos,
                      "serve help omits media preparation controls");
    failures += check(serve_usage_text("ninferno-serve").find("--kv-capacity") != std::string::npos,
                      "serve help omits --kv-capacity");
    // The automatic pool's driver reserve: any MiB value with an automatic
    // capacity, refused with an explicit one.
    failures += check(parse({"ninferno-serve", "model.ninferno", "--kv-auto-headroom-mib", "256"})
                                  .kv_capacity.automatic_headroom_bytes == (256ULL << 20) &&
                          parse({"ninferno-serve", "model.ninferno", "--kv-capacity", "auto",
                                 "--kv-auto-headroom-mib", "0"})
                                  .kv_capacity.automatic_headroom_bytes == 0,
                      "--kv-auto-headroom-mib does not set the automatic headroom");
    for (const std::vector<std::string>& refused :
         {std::vector<std::string>{"ninferno-serve", "model.ninferno", "--kv-capacity", "8192",
                                   "--kv-auto-headroom-mib", "256"},
          std::vector<std::string>{"ninferno-serve", "model.ninferno", "--max-context", "4096",
                                   "--kv-auto-headroom-mib", "256"},
          std::vector<std::string>{"ninferno-serve", "model.ninferno", "--kv-auto-headroom-mib",
                                   "-1"},
          std::vector<std::string>{"ninferno-serve", "model.ninferno", "--kv-auto-headroom-mib",
                                   "1.5"}}) {
        bool rejected = false;
        try {
            (void)parse(refused);
        } catch (const std::invalid_argument&) { rejected = true; }
        failures += check(rejected, "--kv-auto-headroom-mib accepted an explicit capacity or a "
                                    "malformed value");
    }
    failures += check(serve_usage_text("ninferno-serve").find("--response-store-max-mib") !=
                          std::string::npos,
                      "serve help omits Responses store limits");
    failures += check(serve_usage_text("ninferno-serve").find("--context-cost-presets") !=
                          std::string::npos,
                      "serve help omits external context-cost presets");
    failures += check(serve_usage_text("ninferno-serve").find("metadata.name") != std::string::npos,
                      "serve help omits the artifact-derived model id default");

    const ServeOptions inherited =
        parse({"ninferno-serve", "model.ninferno", "--max-context", "16384"});
    failures += check(inherited.kv_capacity.mode == ninferno::KvCapacityMode::Explicit &&
                          inherited.kv_capacity.explicit_tokens == 16384,
                      "omitted --kv-capacity did not follow --max-context");
    // More slots: every slot's full context when it fits (an automatic capacity, bounded by one
    // context below and --max-concurrency x --max-context above).
    const ServeOptions per_slot = parse(
        {"ninferno-serve", "model.ninferno", "--max-context", "4096", "--max-concurrency", "8"});
    failures += check(per_slot.kv_capacity.mode == ninferno::KvCapacityMode::Automatic &&
                          per_slot.kv_capacity.explicit_tokens == 0 &&
                          per_slot.kv_capacity.automatic_headroom_bytes ==
                              ninferno::kDefaultKvCapacityHeadroomBytes,
                      "omitted --kv-capacity with several slots did not size for every slot");
    const ServeOptions per_slot_kv = parse({"ninferno-serve", "model.ninferno", "--max-concurrency",
                                            "8", "--max-context", "4096", "--kv-capacity", "8192"});
    failures += check(per_slot_kv.kv_capacity.mode == ninferno::KvCapacityMode::Explicit &&
                          per_slot_kv.kv_capacity.explicit_tokens == 8192,
                      "explicit --kv-capacity with several slots did not stay explicit");

    const ServeOptions kv_bounded =
        parse({"ninferno-serve", "model.ninferno", "--kv-capacity", "32768"});
    failures += check(kv_bounded.max_context == 0 &&
                          kv_bounded.kv_capacity.mode == ninferno::KvCapacityMode::Explicit &&
                          kv_bounded.kv_capacity.explicit_tokens == 32768,
                      "--kv-capacity alone did not leave the context automatic");
    bool zero_context_refused = false;
    try {
        (void)parse({"ninferno-serve", "model.ninferno", "--max-context", "0"});
    } catch (const std::invalid_argument&) { zero_context_refused = true; }
    failures += check(zero_context_refused, "--max-context 0 was accepted");

    // MoE expert offload was removed: each of its options is refused by name, with or without a
    // value, and the help no longer offers it.
    const std::vector<std::pair<std::vector<std::string>, std::string>> removed = {
        {{"ninferno-serve", "model.ninferno", "--moe-offload", "auto"}, "--moe-offload"},
        {{"ninferno-serve", "model.ninferno", "--moe-offload", "off"}, "--moe-offload"},
        {{"ninferno-serve", "model.ninferno", "--moe-offload=4"}, "--moe-offload"},
        {{"ninferno-serve", "model.ninferno", "--moe-offload-compute", "cpu"},
         "--moe-offload-compute"},
        {{"ninferno-serve", "model.ninferno", "--moe-offload-threads", "4"},
         "--moe-offload-threads"},
    };
    for (const auto& [args, name] : removed) {
        std::string message;
        try {
            (void)parse(args);
        } catch (const std::invalid_argument& error) { message = error.what(); }
        failures += check(message == name + " was removed (MoE expert offload is not supported "
                                            "in this version)",
                          (name + " was not refused as removed: " + message).c_str());
    }
    failures += check(serve_usage_text("ninferno-serve").find("moe-offload") == std::string::npos,
                      "serve help still offers MoE expert offload");

    const ServeOptions automatic =
        parse({"ninferno-serve", "model.ninferno", "--kv-capacity", "auto"});
    failures += check(automatic.kv_capacity.mode == ninferno::KvCapacityMode::Automatic &&
                          automatic.kv_capacity.explicit_tokens == 0 &&
                          automatic.kv_capacity.automatic_headroom_bytes ==
                              ninferno::kDefaultKvCapacityHeadroomBytes,
                      "--kv-capacity auto did not select automatic sizing");

    const ServeOptions chunk_default = parse({"ninferno-serve", "model.ninferno"});
    const ServeOptions chunk_given =
        parse({"ninferno-serve", "model.ninferno", "--prefill-chunk", "1024"});
    failures += check(chunk_default.prefill_chunk == 1024 && !chunk_default.prefill_chunk_explicit &&
                          chunk_given.prefill_chunk == 1024 && chunk_given.prefill_chunk_explicit,
                      "--prefill-chunk did not mark an explicit chunk, or its absence did");
    const ServeOptions unleased =
        parse({"ninferno-serve", "model.ninferno", "--kv-lease-window", "0"});
    failures += check(chunk_default.kv_lease_window == 4096 && unleased.kv_lease_window == 0,
                      "--kv-lease-window did not reach serving options, or its default is not 4096");

    const ServeOptions logged = parse({"ninferno-serve", "model.ninferno", "--request-log-jsonl",
                                       "requests.jsonl", "--api-key", "do-not-log"});
    failures += check(logged.request_log_jsonl == "requests.jsonl",
                      "--request-log-jsonl did not preserve its path");
    failures +=
        check(serve_usage_text("ninferno-serve").find("--request-log-jsonl") != std::string::npos,
              "serve help omits --request-log-jsonl");
    bool secret_present    = false;
    bool redaction_present = false;
    for (const std::string& argument : logged.startup_argv) {
        secret_present    = secret_present || argument == "do-not-log";
        redaction_present = redaction_present || argument == "<redacted>";
    }
    failures += check(!secret_present, "startup argv retained the API key");
    failures += check(redaction_present, "startup argv omitted the API-key redaction marker");

    // What --spec auto says when its choice does not fit and it decodes plainly, and when plain
    // decode does not fit either.
    failures += check(ninferno::serve::auto_plain_reason("dflash:7", "out of memory", false) ==
                          "--spec auto (dflash:7) does not fit on this device, so the server "
                          "decodes plainly (out of memory)",
                      "the plain-after-auto reason");
    failures += check(ninferno::serve::auto_plain_reason("dflash:7", "out of memory", true) ==
                          "--spec auto (dflash:7) does not fit on this device, so the server "
                          "decodes plainly (out of memory); with --prefill-chunk left automatic "
                          "the server also narrows the chunk before it decodes plainly",
                      "the plain-after-auto reason with an explicit --prefill-chunk");
    failures += check(ninferno::serve::auto_then_plain_refusal("mtp:3", "short by 2 GiB") ==
                          "--spec auto (mtp:3) does not fit on this device, and plain decode "
                          "does not fit either: short by 2 GiB",
                      "the refusal when plain decode does not fit after --spec auto");

    if (failures == 0) { std::cout << "ok\n"; }
    return failures == 0 ? 0 : 1;
}
