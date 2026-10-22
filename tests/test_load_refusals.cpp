// What a load refuses before it touches the device: the options only the Qwen 3.5 family
// implements (not --activations, which every family takes), named for the family that would ignore
// them, and option values it cannot read. No GPU.

#include "ninferno/option_enums.h"
#include "runtime/engine/hf_text_frontend.h"

#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

int check(bool condition, const char* message) {
    if (condition) { return 0; }
    std::cerr << message << '\n';
    return 1;
}

// The invalid_argument message `operation` throws, or "" when it does not throw one.
std::string refusal(const auto& operation) {
    try {
        operation();
    } catch (const std::invalid_argument& error) { return error.what(); }
    return {};
}

bool names(const std::string& message, std::string_view a, std::string_view b) {
    return message.find(a) != std::string::npos && message.find(b) != std::string::npos;
}

int qwen_only_options() {
    using ninferno::ActivationMode;
    using ninferno::EngineOptions;
    using ninferno::RecurrentStateStorage;
    using ninferno::runtime::refuse_qwen_only_options;
    int failures = 0;

    const auto refused = [](const EngineOptions& options) {
        return refusal([&] { refuse_qwen_only_options(options, "Gemma 4"); });
    };
    failures += check(refused(EngineOptions{}).empty(), "default options refused");

    EngineOptions yarn;
    yarn.rope_yarn_factor = 4.0F;
    failures += check(names(refused(yarn), "--rope-yarn-factor", "Gemma 4"),
                      "--rope-yarn-factor not refused with the family named");

    // --activations is every family's: 8 and 4 resolve per matrix (activation_precision.h).
    for (const ActivationMode mode : {ActivationMode::Bits8, ActivationMode::Bits4}) {
        EngineOptions narrow;
        narrow.activations = mode;
        failures += check(refused(narrow).empty(), "--activations 8|4 refused as Qwen-only");
    }

    for (const RecurrentStateStorage storage :
         {RecurrentStateStorage::BF16, RecurrentStateStorage::FP16}) {
        EngineOptions state;
        state.recurrent_state = storage;
        failures += check(names(refused(state), "--state-dtype", "Gemma 4"),
                          "--state-dtype bf16|f16 not refused with the family named");
    }
    return failures;
}

// The option spellings of the tuner modes and the pacing choices: each name parses back, and any
// other value is refused with the option named rather than read as a default.
int option_spellings() {
    using ninferno::TuneMode;
    int failures = 0;
    for (const TuneMode mode : {TuneMode::Off, TuneMode::On, TuneMode::Retune}) {
        failures +=
            check(ninferno::parse_tune_mode(ninferno::tune_mode_name(mode), "--dense-tune") == mode,
                  "a tuner mode does not parse back from its name");
    }
    for (const char* value : {"", "1", "ON", "true", "retune "}) {
        failures +=
            check(names(refusal([&] { (void)ninferno::parse_tune_mode(value, "--moe-tune"); }),
                        "--moe-tune", "'" + std::string(value) + "'"),
                  "an unknown tuner value was not refused with the option named");
    }
    failures += check(ninferno::parse_decode_wait("block") == ninferno::DecodeWait::Block,
                      "--decode-wait block does not parse");
    failures += check(names(refusal([] { (void)ninferno::parse_decode_wait("busy"); }),
                            "--decode-wait", "'busy'"),
                      "an unknown --decode-wait was not refused with the option named");
    failures +=
        check(ninferno::parse_cuda_sync_schedule("yield") == ninferno::CudaSyncSchedule::Yield,
              "--cuda-sync yield does not parse");
    return failures;
}

} // namespace

int main() {
    const int failures = qwen_only_options() + option_spellings();
    if (failures == 0) { std::cout << "load refusals OK\n"; }
    return failures == 0 ? 0 : 1;
}
