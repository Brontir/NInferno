// What a load refuses before it touches the device: the options only the Qwen 3.5 family
// implements, named for the family that would ignore them, and a tuner variable it cannot read.
// No GPU.

#include "core/tune_timer.h"
#include "runtime/engine/hf_text_frontend.h"
#include "support/posix.h"

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

    EngineOptions int8;
    int8.activations = ActivationMode::Int8;
    failures += check(names(refused(int8), "--activations int8", "Gemma 4"),
                      "--activations int8 not refused with the family named");

    for (const RecurrentStateStorage storage :
         {RecurrentStateStorage::BF16, RecurrentStateStorage::FP16}) {
        EngineOptions state;
        state.recurrent_state = storage;
        failures += check(names(refused(state), "--state-dtype", "Gemma 4"),
                          "--state-dtype bf16|f16 not refused with the family named");
    }
    return failures;
}

int tuner_variable() {
    using ninferno::TuneMode;
    using ninferno::tune_mode;
    constexpr const char* variable = "NINFERNO_TEST_TUNE";
    int failures                   = 0;

    const auto mode_of = [&](const char* value) {
        if (value == nullptr) {
            ::unsetenv(variable);
        } else {
            ::setenv(variable, value, 1);
        }
        return tune_mode(variable);
    };
    failures += check(mode_of(nullptr) == TuneMode::Off, "unset tuner variable is not off");
    failures += check(mode_of("") == TuneMode::Off, "empty tuner variable is not off");
    failures += check(mode_of("off") == TuneMode::Off, "tuner variable off is not off");
    failures += check(mode_of("on") == TuneMode::On, "tuner variable on is not on");
    failures += check(mode_of("retune") == TuneMode::Retune, "tuner variable retune is not retune");
    for (const char* value : {"1", "ON", "true", "retune "}) {
        failures += check(names(refusal([&] { (void)mode_of(value); }), variable,
                                "'" + std::string(value) + "'"),
                          "an unknown tuner value was not refused with the variable named");
    }
    ::unsetenv(variable);
    return failures;
}

} // namespace

int main() {
    const int failures = qwen_only_options() + tuner_variable();
    if (failures == 0) { std::cout << "load refusals OK\n"; }
    return failures == 0 ? 0 : 1;
}
