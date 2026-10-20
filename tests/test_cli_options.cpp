#include "options.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

ninferno::cli::Options parse(std::vector<std::string> arguments) {
    std::vector<char*> argv;
    argv.reserve(arguments.size());
    for (std::string& argument : arguments) { argv.push_back(argument.data()); }
    return ninferno::cli::parse_options(static_cast<int>(argv.size()), argv.data());
}

bool rejects(const std::function<void()>& operation) {
    try {
        operation();
    } catch (const std::invalid_argument&) { return true; }
    return false;
}

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

} // namespace

int main() {
    int failures = 0;
    const ninferno::cli::Options configured =
        parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--thinking-budget", "37"});
    failures += check(configured.thinking_budget == 37,
                      "--thinking-budget did not preserve its positive value");
    failures +=
        check(parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "-h"}).help_requested,
              "-h after other arguments is not a help request");
    failures += check(
        parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--ngram-draft-tokens", "0"})
                .speculative.ngram_draft_tokens == 0,
        "--ngram-draft-tokens 0 (no copy rounds) was not accepted");
    failures += check(ninferno::cli::usage_text("ninferno-cli").find("--thinking-budget") !=
                          std::string::npos,
                      "CLI help omits --thinking-budget");
    failures += check(rejects([] {
                          (void)parse({"ninferno-cli", "model.ninferno", "--prompt", "hello",
                                       "--thinking-budget", "0"});
                      }),
                      "zero --thinking-budget was accepted");
    failures += check(rejects([] {
                          (void)parse({"ninferno-cli", "model.ninferno", "--prompt", "hello",
                                       "--thinking-budget", "8", "--no-thinking"});
                      }),
                      "--thinking-budget was accepted with --no-thinking");
    const ninferno::cli::Options with_effort =
        parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--thinking-budget", "8",
               "--reasoning-effort", "medium"});
    failures += check(with_effort.thinking_budget == 8 && with_effort.reasoning_effort,
                      "thinking budget did not coexist with reasoning effort");
    const ninferno::cli::Options dflash_vision =
        parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--vision", "--spec",
               "dflash", "--draft-tokens", "7"});
    failures +=
        check(dflash_vision.enable_vision &&
                  dflash_vision.speculative.backend == ninferno::SpeculativeBackend::DFlash &&
                  dflash_vision.speculative.draft_tokens == 7,
              "CLI did not preserve the combined DFlash and Vision startup features");
    for (const auto k : {1U, 2U, 7U, 15U}) {
        const auto dflash2 = parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--spec",
                                    "dflash2", "--draft-tokens", std::to_string(k)});
        failures += check(dflash2.speculative.backend == ninferno::SpeculativeBackend::DFlash2 &&
                              dflash2.speculative.draft_tokens == k,
                          "CLI did not preserve the DFlash2 draft count");
    }
    for (const auto k : {0U, 16U}) {
        failures +=
            check(rejects([&] {
                      (void)parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--spec",
                                   "dflash2", "--draft-tokens", std::to_string(k)});
                  }),
                  "CLI accepted an unsupported DFlash2 draft count");
    }
    const auto plain_default = parse({"ninferno-cli", "model.ninferno", "--prompt", "hello"});
    failures += check(!plain_default.spec_explicit && !plain_default.acceptance_floor,
                      "without --spec the model's profile decides");
    const auto none =
        parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--spec", "none"});
    failures += check(none.spec_explicit &&
                          none.speculative.backend == ninferno::SpeculativeBackend::None,
                      "--spec none turns speculation off explicitly");
    const auto automatic = parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--spec",
                                  "auto", "--acceptance-floor", "1.5"});
    failures += check(!automatic.spec_explicit && automatic.acceptance_floor &&
                          *automatic.acceptance_floor == 1.5F,
                      "--spec auto and --acceptance-floor");
    failures += check(
        parse({"ninferno-cli", "model.ninferno", "--prompt", "hello"}).activations ==
                ninferno::kDefaultActivationMode &&
            parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--activations", "exact"})
                    .activations == ninferno::ActivationMode::Exact &&
            parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--activations", "fast"})
                    .activations == ninferno::ActivationMode::Fast,
        "--activations exact|fast");
    failures += check(rejects([] {
                          (void)parse({"ninferno-cli", "model.ninferno", "--prompt", "hello",
                                       "--activations", "a16"});
                      }),
                      "--activations accepted an unknown mode");
    const ninferno::cli::Options nvfp4 =
        parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--kv-dtype", "nvfp4"});
    failures += check(nvfp4.kv_cache == ninferno::KvCacheStorage::Nvfp4Group16,
                      "--kv-dtype nvfp4 did not select group-16 NVFP4 KV");
    const ninferno::cli::Options k8v4 =
        parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--kv-dtype", "k8v4"});
    failures += check(k8v4.kv_cache == ninferno::KvCacheStorage::Fp8KeyNvfp4Value,
                      "--kv-dtype k8v4 did not select asymmetric K8V4 KV");
    // Without --kv-dtype, or with auto, the model family chooses the KV storage.
    failures += check(
        !parse({"ninferno-cli", "model.ninferno", "--prompt", "hello"}).kv_cache.has_value() &&
            !parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--kv-dtype", "auto"})
                 .kv_cache.has_value(),
        "the KV storage is not the family's by default and with --kv-dtype auto");
    const std::string help = ninferno::cli::usage_text("ninferno-cli");
    failures +=
        check(help.find("nvfp4") != std::string::npos && help.find("k8v4") != std::string::npos,
              "CLI help omits a production KV storage mode");
    const ninferno::cli::Options logging =
        parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--log-level", "debug"});
    failures += check(logging.log_level == ninferno::product::LogLevel::Debug,
                      "CLI log level was not parsed");
    failures += check(help.find("--log-level") != std::string::npos,
                      "CLI help omits the log-level control");
    failures += check(rejects([] {
                          (void)parse({"ninferno-cli", "model.ninferno", "--prompt", "hello",
                                       "--log-level", "verbose"});
                      }),
                      "CLI accepted an unknown log level");
    failures += check(
        rejects([] {
            (void)parse({"ninferno-cli", "model.ninferno", "--prompt", "hello", "--top-k", "65"});
        }),
        "CLI accepted top_k beyond the executable candidate domain");
    return failures == 0 ? 0 : 1;
}
