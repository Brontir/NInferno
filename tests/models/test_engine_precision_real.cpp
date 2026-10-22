// --activations auto|16|8|4 through the Engine (ninferno-serve's path) on real artifacts of any
// family: every mode starts and answers, or (8 and 4 only) is refused before the load with the
// resolution's own text ("no matrix ... can compute with"). Nothing else may stop a start: the
// frozen sequence plan once refused 8 and 4 on the dense, Gemma 4 and Nemotron-H families
// ("Program model does not match the frozen sequence plan"), whose programs reported 16 or Auto
// for the mode they were loaded under. The startup resolution is printed per mode.
//
// NINFERNO_TEST_PRECISION_ARTIFACTS names the artifacts, separated by ';' (one dense, one Gemma 4,
// one Nemotron-H at least, quantized so 8 and 4 change something); without it the test skips.
// GPU: each start takes a 2048-token context and one request at a time.

#include "ninferno/engine.h"
#include "product/activation_options.h"

#include <cstdlib>
#include <exception>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

std::vector<std::string> split(const std::string& text) {
    std::vector<std::string> out;
    std::stringstream stream(text);
    for (std::string item; std::getline(stream, item, ';');) {
        if (!item.empty()) { out.push_back(item); }
    }
    return out;
}

void run(const std::string& artifact, ninferno::ActivationMode mode) {
    using namespace ninferno;
    const std::string label = artifact + " --activations " + product::activation_mode_name(mode);
    EngineOptions options;
    options.artifact_path                        = artifact;
    options.max_context                          = 2048;
    options.kv_capacity                          = KvCapacityPolicy::explicit_capacity(2048);
    options.max_concurrency                      = 1;
    options.activations                          = mode;
    options.context_cache.host_state_slots       = 0;
    options.context_cache.host_kv_capacity_bytes = 0;
    try {
        Engine engine(options);
        const auto precision = engine.model_metadata().activations;
        std::cout << label << ": " << product::describe_activation_precision(precision) << '\n';
        for (const auto& line : precision.kept_wider) { std::cout << "  kept: " << line << '\n'; }
        for (const auto& line : precision.notes) { std::cout << "  note: " << line << '\n'; }
        check(precision.requested == mode, label + ": the resolution names another mode");
        const auto prompt = engine.tokenize_text("The capital of France is");
        RequestOptions request;
        request.execution.requested_output_tokens = 8;
        request.execution.sampling.temperature    = 0.0F;
        const GenerationResult result =
            engine.generate(engine.prepare_tokens(prompt, false), request);
        check(!result.generated_token_ids.empty(), label + ": no token generated");
        std::cout << "  " << result.generated_token_ids.size() << " tokens\n";
    } catch (const std::invalid_argument& error) {
        const std::string what = error.what();
        const bool refused     = (mode == ActivationMode::Bits8 || mode == ActivationMode::Bits4) &&
                             what.find("can compute with") != std::string::npos;
        const bool gpu = (mode == ActivationMode::Bits8 || mode == ActivationMode::Bits4) &&
                         what.find("this GPU runs no matrix") != std::string::npos;
        std::cout << label << ": refused: " << what << '\n';
        check(refused || gpu, label + ": refused by something other than the resolution: " + what);
    } catch (const std::exception& error) { check(false, label + ": " + error.what()); }
}

} // namespace

int main() {
    const char* list = std::getenv("NINFERNO_TEST_PRECISION_ARTIFACTS");
    if (list == nullptr || *list == 0) {
        std::cout << "SKIP: set NINFERNO_TEST_PRECISION_ARTIFACTS (artifacts separated by ';')\n";
        return 77;
    }
    for (const auto& artifact : split(list)) {
        for (const auto mode : {ninferno::ActivationMode::Auto, ninferno::ActivationMode::Bits16,
                                ninferno::ActivationMode::Bits8, ninferno::ActivationMode::Bits4}) {
            run(artifact, mode);
        }
    }
    if (failures == 0) { std::cout << "engine precision real: all checks passed\n"; }
    return failures == 0 ? 0 : 1;
}
