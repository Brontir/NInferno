// --activations auto|16|8|4: the policy each mode gives a matrix, the narrowest width a format
// admits on a device's features, the startup resolution over an artifact's directory (counts, the
// warning lines of a partial request, the refusal of one no matrix can take), and the families'
// load profiles. CPU: the device features are passed in, the directory is built here.

#include "artifact/formats.h"
#include "artifact/schema.h"
#include "core/device_class.h"
#include "models/common/activation_census.h"
#include "models/common/weight_use.h"
#include "models/dense/model.h"
#include "models/gemma4/model.h"
#include "models/nemotron_h/model.h"
#include "ops/linear/activation_width.h"
#include "product/activation_options.h"
#include "runtime/engine/activation_precision.h"

#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

namespace {

using ninferno::ActivationMode;
using ninferno::QType;
using ninferno::ops::LinearPolicy;

int check(bool condition, const std::string& message) {
    if (condition) { return 0; }
    std::cerr << "FAIL: " << message << '\n';
    return 1;
}

const ninferno::DeviceFeatures kBlackwell = ninferno::device_features_for(12, 0); // FP4 MMA
const ninferno::DeviceFeatures kAda       = ninferno::device_features_for(8, 9);  // FP8, no FP4
const ninferno::DeviceFeatures kAmpere    = ninferno::device_features_for(8, 6);  // no FP8
// The resolution's devices: the features above and the generation row's FP32-accumulate rate
// (GeForce parts run it at half rate, where int8 prefill pays; the A100 at full rate).
const ninferno::ops::ActivationDevice kRtx50{kBlackwell, false};
const ninferno::ops::ActivationDevice kRtx40{kAda, false};
const ninferno::ops::ActivationDevice kA100{ninferno::device_features_for(8, 0), true};

int widths() {
    using ninferno::ops::narrowest_activation_width;
    int failures  = 0;
    const auto at = [](QType q, LinearPolicy p, bool divisor, const ninferno::DeviceFeatures& f,
                       std::int64_t n = 4096, std::int64_t k = 4096) {
        return narrowest_activation_width(q, p, n, k, divisor, f);
    };
    failures += check(at(QType::NVFP4, LinearPolicy::AllowA4, true, kBlackwell).bits == 4,
                      "calibrated NVFP4 takes four bits on sm_120");
    const auto ada = at(QType::NVFP4, LinearPolicy::AllowA4, true, kAda);
    failures += check(ada.bits == 8 && ada.reason.find("sm_120a") != std::string_view::npos,
                      "NVFP4 below sm_120 holds at eight bits, saying why");
    failures += check(at(QType::NVFP4, LinearPolicy::AllowA4, false, kBlackwell).bits == 16,
                      "NVFP4 allowed four bits without a divisor stays BF16 on sm_120a");
    failures += check(at(QType::NVFP4, LinearPolicy::AllowA8, false, kBlackwell).bits == 8,
                      "NVFP4 held at eight bits takes W4A8");
    failures += check(at(QType::GGML_Q4_K, LinearPolicy::AllowA8, false, kAmpere).bits == 8,
                      "Q4_K takes eight bits");
    const auto q4k_a4 = at(QType::GGML_Q4_K, LinearPolicy::AllowA4, false, kBlackwell);
    failures +=
        check(q4k_a4.bits == 8 && q4k_a4.reason.find("NVFP4 only") != std::string_view::npos,
              "Q4_K under four bits stays at eight, saying why");
    failures += check(at(QType::BF16, LinearPolicy::AllowA4, false, kBlackwell).bits == 16,
                      "BF16 has no narrower route");
    failures += check(at(QType::GGML_TQ1_0, LinearPolicy::AllowA8, false, kBlackwell).bits == 16,
                      "ternary weights have no eight-bit route");
    failures += check(at(QType::GGML_Q4_K, LinearPolicy::A16Only, false, kBlackwell).bits == 16,
                      "A16Only is sixteen bits");
    failures += check(
        at(QType::FP8_E4M3FN_ROW_BF16, LinearPolicy::AllowA8, false, kAda, 5120, 17408).bits == 8,
        "a registered FP8 shape takes E4M3 activations on sm_89");
    failures += check(
        at(QType::FP8_E4M3FN_ROW_BF16, LinearPolicy::AllowA8, false, kAmpere, 5120, 17408).bits ==
            16,
        "FP8 below sm_89 stays BF16");
    failures += check(
        at(QType::FP8_E4M3FN_ROW_BF16, LinearPolicy::AllowA8, false, kAda, 1000, 1024).bits == 16,
        "an unregistered FP8 shape stays BF16");
    return failures;
}

// A directory of `matrices`: one object, binding and Use each ("text/" or not).
struct Matrix {
    std::string name;
    QType format;
    ninferno::artifact::ActivationPolicy policy;
    bool divisor = false;
};

ninferno::artifact::Directory directory_of(const std::vector<Matrix>& matrices) {
    namespace a = ninferno::artifact;
    a::Directory d;
    for (const auto& m : matrices) {
        a::TensorObject tensor;
        tensor.id     = m.name;
        tensor.shape  = {4096, 4096};
        tensor.format = std::string(a::format_name(m.format));
        d.objects.emplace_back(tensor);
        a::Binding binding;
        binding.whole_object = true;
        binding.parts.push_back({a::ObjectHandle{d.objects.size() - 1}, 0, 4096ULL * 4096ULL});
        d.bindings[m.name] = binding;
        a::Use use;
        use.parameter         = m.name;
        use.input             = "x";
        use.activation_policy = m.policy;
        if (m.divisor) { use.auxiliaries["activation_input_divisor"] = a::Binding{}; }
        d.uses[{m.name, "x"}] = use;
    }
    return d;
}

bool kept(const ninferno::ActivationPrecision& p, std::string_view what) {
    for (const auto& line : p.kept_wider) {
        if (line.find(what) != std::string::npos) { return true; }
    }
    return false;
}

std::string refusal(const ninferno::artifact::Directory& d, ActivationMode mode,
                    const ninferno::ops::ActivationDevice& device) {
    try {
        (void)ninferno::runtime::resolve_activation_precision(d, mode, device);
    } catch (const std::invalid_argument& error) { return error.what(); }
    return {};
}

int resolution() {
    using ninferno::artifact::ActivationPolicy;
    using ninferno::runtime::resolve_activation_precision;
    int failures = 0;
    // An NVFP4 checkpoint calibrated for four bits beside a GGUF block, a BF16 head and an
    // in-artifact drafter matrix (counted on its own line, following the mode).
    const auto mixed = directory_of({
        {"text/layers/0/attention/qkv", QType::NVFP4, ActivationPolicy::AllowA4, true},
        {"text/layers/0/mlp/down", QType::NVFP4, ActivationPolicy::AllowA4, true},
        {"text/layers/0/mlp/gate", QType::GGML_Q4_K, ActivationPolicy::AllowA8},
        {"text/layers/0/mlp/up", QType::NVFP4, ActivationPolicy::A16Only, false},
        {"text/output_head", QType::GGML_Q6_K, ActivationPolicy::A16Only},
        {"mtp/layers/0/mlp/down", QType::NVFP4, ActivationPolicy::AllowA4, true},
    });

    const auto aut = resolve_activation_precision(mixed, ActivationMode::Auto, kRtx50);
    failures += check(aut.bits4 == 2 && aut.bits8 == 1 && aut.bits16 == 2 && aut.kept_wider.empty(),
                      "auto: the recorded policies, no warning");
    failures += check(aut.drafter_bits4 == 1 && aut.drafter_bits8 == 0 && aut.drafter_bits16 == 0,
                      "auto: the in-artifact drafter on its own line");
    failures += check(ninferno::product::describe_activation_precision(aut) ==
                          "auto -> 4-bit on 2, 8-bit on 1, 16-bit on 2 matrices",
                      "auto: the startup summary");

    const auto exact = resolve_activation_precision(mixed, ActivationMode::Bits16, kRtx50);
    failures +=
        check(exact.bits16 == 5 && exact.bits4 == 0 && exact.bits8 == 0 && exact.kept_wider.empty(),
              "16: every matrix at sixteen bits");
    failures += check(exact.drafter_bits16 == 1 && exact.drafter_bits4 == 0,
                      "16: the in-artifact drafter follows the mode");

    const auto eight = resolve_activation_precision(mixed, ActivationMode::Bits8, kRtx50);
    failures += check(eight.bits4 == 0 && eight.bits8 == 4 && eight.bits16 == 1,
                      "8: every matrix but the head at eight bits");
    failures += check(eight.kept_wider.size() == 1 && kept(eight, "text/output_head") &&
                          kept(eight, "recorded 16-bit"),
                      "8: the head is named as left at sixteen bits");

    const auto four = resolve_activation_precision(mixed, ActivationMode::Bits4, kRtx50);
    failures += check(four.bits4 == 2 && four.bits8 == 2 && four.bits16 == 1,
                      "4: calibrated NVFP4 at four, the rest at eight, the head at sixteen");
    failures += check(kept(four, "1 q4_k") || kept(four, "NVFP4 only"),
                      "4: the GGUF block is named as left at eight bits");
    failures += check(kept(four, "no calibrated activation divisor"),
                      "4: NVFP4 without a divisor is named as left at eight bits");

    // 4 on a card without the FP4 MMA: no matrix takes it, so it is refused, pointing at 8.
    const auto no_fp4 = refusal(mixed, ActivationMode::Bits4, kRtx40);
    failures += check(no_fp4.find("--activations 4") != std::string::npos &&
                          no_fp4.find("--activations 8") != std::string::npos,
                      "4 below sm_120 is refused and names 8: " + no_fp4);
    failures += check(refusal(mixed, ActivationMode::Bits8, kRtx40).empty(), "8 on sm_89 starts");

    // A GGUF checkpoint: 8 starts, 4 is refused (no NVFP4 weights).
    const auto gguf =
        directory_of({{"text/layers/0/mlp/down", QType::GGML_Q4_K, ActivationPolicy::A16Only},
                      {"text/layers/0/mlp/up", QType::GGML_IQ3_S, ActivationPolicy::AllowA8}});
    failures += check(resolve_activation_precision(gguf, ActivationMode::Bits8, kRtx50).bits8 == 2,
                      "8: a GGUF checkpoint at eight bits");
    failures += check(!refusal(gguf, ActivationMode::Bits4, kRtx50).empty(),
                      "4: a GGUF checkpoint is refused");

    // BF16 weights: 8 and 4 change nothing and are refused; auto and 16 start.
    const auto bf16 =
        directory_of({{"text/layers/0/mlp/down", QType::BF16, ActivationPolicy::A16Only}});
    failures += check(!refusal(bf16, ActivationMode::Bits8, kRtx50).empty() &&
                          !refusal(bf16, ActivationMode::Bits4, kRtx50).empty(),
                      "8 and 4 on BF16 weights are refused");
    failures += check(refusal(bf16, ActivationMode::Auto, kRtx50).empty() &&
                          refusal(bf16, ActivationMode::Bits16, kRtx50).empty(),
                      "auto and 16 on BF16 weights start");
    return failures;
}

// The Ops' own widths: the int8 prefill of A16Only G64 weights where it pays, the expert census,
// the notes (auto slower than 8, a mode equal to auto) and the four-mode preflight.
int ops_widths() {
    using ninferno::artifact::ActivationPolicy;
    using ninferno::runtime::resolve_activation_precision;
    int failures = 0;
    // The official Qwen3.8-27B artifact records its G64 matrices A16Only: auto prefills them on
    // int8 activations on a GeForce part, not on an A100 and never under 16.
    const auto g64 = directory_of({
        {"text/layers/0/mlp/down", QType::Q4_G64_FP16, ActivationPolicy::A16Only},
        {"text/output_head", QType::Q6_G64_FP16, ActivationPolicy::A16Only},
    });
    const auto rtx = resolve_activation_precision(g64, ActivationMode::Auto, kRtx50);
    failures += check(rtx.bits8 == 1 && rtx.bits16 == 1,
                      "auto: an A16Only G64 matrix prefills at eight bits on an RTX, the head not");
    failures += check(resolve_activation_precision(g64, ActivationMode::Auto, kA100).bits16 == 2,
                      "auto: no int8 prefill on an A100 (full-rate FP32-accumulate HMMA)");
    failures += check(resolve_activation_precision(g64, ActivationMode::Bits16, kRtx50).bits16 == 2,
                      "16: never the int8 prefill");
    failures +=
        check(!resolve_activation_precision(g64, ActivationMode::Bits8, kRtx50).same_as_auto,
              "8 differs from auto on A16Only G64 (an eight-bit decode route)");
    // A GGUF recorded for eight bits but its head and BF16 router: 8 is auto's policies where a
    // route reads them (the router has none to move), so the same kernels.
    const auto gguf  = directory_of({
        {"text/layers/0/mlp/down", QType::GGML_Q4_K, ActivationPolicy::AllowA8},
        {"text/layers/0/mlp/router", QType::BF16, ActivationPolicy::A16Only},
        {"text/output_head", QType::GGML_Q6_K, ActivationPolicy::A16Only},
    });
    const auto eight = resolve_activation_precision(gguf, ActivationMode::Bits8, kRtx50);
    failures += check(eight.same_as_auto && eight.notes.size() == 1,
                      "8 runs auto's kernels on a GGUF recorded for eight bits, and says so");

    // The census: what the expert Ops run, by parameter name, over the linear path's admission.
    const auto moe = directory_of({
        {"text/layers/0/mlp/experts/0/gate", QType::NVFP4, ActivationPolicy::AllowA4, true},
        {"text/layers/0/mlp/experts/0/down", QType::NVFP4, ActivationPolicy::AllowA4, true},
        {"text/layers/0/mlp/shared_down", QType::NVFP4, ActivationPolicy::AllowA4, true},
        {"text/layers/0/attention/qkv", QType::NVFP4, ActivationPolicy::AllowA4, true},
    });
    failures += check(ninferno::runtime::has_routed_experts(moe) &&
                          !ninferno::runtime::has_routed_experts(g64),
                      "routed experts are found by name");
    const auto before = resolve_activation_precision(moe, ActivationMode::Auto, kRtx50);
    failures +=
        check(before.bits4 == 4, "before the load every matrix counts as the linear path's");
    ninferno::models::ActivationCensus census;
    ninferno::models::census_width("text/layers/0/mlp/experts/0/gate", 16, "the Op keeps BF16");
    ninferno::models::census_width("text/layers/0/mlp/experts/0/down", 16, "the Op keeps BF16");
    ninferno::models::census_policy("text/layers/0/mlp/shared_down", LinearPolicy::A16Only);
    const auto after =
        resolve_activation_precision(moe, ActivationMode::Auto, kRtx50, &census, false);
    failures += check(after.bits4 == 1 && after.bits16 == 3,
                      "after the load the census's widths and policies count");
    const auto four =
        resolve_activation_precision(moe, ActivationMode::Bits4, kRtx50, &census, false);
    failures += check(kept(four, "the Op keeps BF16"), "a census reason reaches the warning");

    // NVFP4 experts beside an integer matrix: 4 is never "= auto" on them (the load picks the
    // fused four-bit bank or the generic MoE at eight bits), and when the load's experts take no
    // four bits, 4 still runs and says so.
    const auto experts = directory_of({
        {"text/layers/0/mlp/experts/0/up", QType::NVFP4, ActivationPolicy::AllowA4, true},
        {"text/layers/0/attention/qkv", QType::GGML_Q4_K, ActivationPolicy::AllowA8},
    });
    failures +=
        check(!resolve_activation_precision(experts, ActivationMode::Bits4, kRtx50).same_as_auto,
              "4 is not auto on NVFP4 experts");
    {
        ninferno::models::ActivationCensus generic;
        ninferno::models::census_width("text/layers/0/mlp/experts/0/up", 8, "no four-bit route");
        const auto loaded =
            resolve_activation_precision(experts, ActivationMode::Bits4, kRtx50, &generic, false);
        bool said = false;
        for (const auto& note : loaded.notes) {
            said = said || note.find("after the load") != std::string::npos;
        }
        failures += check(loaded.bits4 == 0 && loaded.bits8 == 2 && said,
                          "4 with no four-bit expert route after the load says so");
    }

    // Uncalibrated NVFP4 recorded for four bits: auto keeps BF16 on sm_120a, 8 runs W4A8.
    const auto raw =
        directory_of({{"text/layers/0/mlp/down", QType::NVFP4, ActivationPolicy::AllowA4, false}});
    const auto slow = resolve_activation_precision(raw, ActivationMode::Auto, kRtx50);
    failures += check(slow.bits16 == 1 && slow.notes.size() == 1 &&
                          slow.notes.front().find("auto is slower than 8") != std::string::npos,
                      "auto says when 8 is faster here");

    // The preflight: every mode in order, a refusal as text, never for auto or 16.
    const auto rows = ninferno::runtime::preflight_activation_modes(g64, kRtx50);
    failures +=
        check(rows.size() == 4 && rows[0].mode == ActivationMode::Auto && !rows[0].refused &&
                  !rows[1].refused && !rows[2].refused && rows[3].refused &&
                  rows[3].refusal.find("--activations 4") != std::string::npos,
              "preflight: auto, 16 and 8 start; 4 is refused on G64 weights");
    return failures;
}

int names() {
    using ninferno::product::activation_mode_alias;
    using ninferno::product::activation_mode_name;
    using ninferno::product::parse_activation_mode;
    int failures = 0;
    for (const auto mode : {ActivationMode::Auto, ActivationMode::Bits16, ActivationMode::Bits8,
                            ActivationMode::Bits4}) {
        failures += check(parse_activation_mode(activation_mode_name(mode)) == mode,
                          "a mode's name parses back");
    }
    failures += check(parse_activation_mode("exact") == ActivationMode::Bits16 &&
                          parse_activation_mode("fast") == ActivationMode::Auto &&
                          parse_activation_mode("int8") == ActivationMode::Bits8,
                      "the deprecated names");
    failures += check(activation_mode_alias("exact") == "16" && !activation_mode_alias("16"),
                      "an alias names its new spelling");
    bool refused = false;
    try {
        (void)parse_activation_mode("bf16");
    } catch (const std::invalid_argument&) { refused = true; }
    failures += check(refused, "an unknown spelling is refused");
    return failures;
}

int profiles() {
    int failures       = 0;
    namespace dense    = ninferno::models::dense;
    namespace gemma4   = ninferno::models::gemma4;
    namespace nemotron = ninferno::models::nemotron_h;
    failures += check(
        dense::activation_mode_of(dense::ActivationProfile::Artifact) == ActivationMode::Auto &&
            dense::activation_mode_of(dense::ActivationProfile::A16) == ActivationMode::Bits16 &&
            dense::activation_mode_of(dense::ActivationProfile::A8) == ActivationMode::Bits8 &&
            dense::activation_mode_of(dense::ActivationProfile::A4) == ActivationMode::Bits4,
        "dense profiles");
    failures += check(
        gemma4::activation_mode_of(gemma4::ActivationProfile::A16Full) == ActivationMode::Bits16 &&
            gemma4::activation_mode_of(gemma4::ActivationProfile::A8) == ActivationMode::Bits8 &&
            gemma4::activation_mode_of(gemma4::ActivationProfile::A4) == ActivationMode::Bits4,
        "Gemma 4 profiles");
    failures += check(
        nemotron::activation_mode_of(nemotron::ActivationProfile::A8) == ActivationMode::Bits8 &&
            nemotron::activation_mode_of(nemotron::ActivationProfile::A4) == ActivationMode::Bits4,
        "Nemotron-H profiles");
    return failures;
}

} // namespace

int main() {
    const int failures = widths() + resolution() + ops_widths() + names() + profiles();
    if (failures == 0) { std::cout << "activation precision: ok\n"; }
    return failures == 0 ? 0 : 1;
}
