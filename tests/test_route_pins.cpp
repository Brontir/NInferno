// The maintainer's route pins (--prefill-route, --decode-route, --moe-path; ops/route_pins.h),
// without a device: their spellings and refusals, the shared engine-flag parser, the startup's
// refusals that need no weight (--activations exact / int8), the ledger and its coverage check, the
// MoE path choice, and a registered shape's activation choice under a pin. CPU-only.

#include "ninferno/option_enums.h"
#include "ninferno/types.h"
#include "ops/route_pins.h"
#include "product/engine_flags.h"
#include "runtime/engine/model_instance.h"

#include <functional>
#include <iostream>
#include <stdexcept>
#include <string>
#include <string_view>

namespace {

using namespace ninferno;

int failures = 0;

void check(bool condition, const std::string& message) {
    if (condition) { return; }
    std::cerr << "FAIL: " << message << '\n';
    ++failures;
}

// Whether `f` throws std::invalid_argument whose text holds every one of `expected`.
bool refuses(const std::function<void()>& f, std::initializer_list<std::string_view> expected) {
    try {
        f();
    } catch (const std::invalid_argument& error) {
        const std::string what = error.what();
        for (const std::string_view e : expected) {
            if (what.find(e) == std::string::npos) {
                std::cerr << "  refused with: " << what << '\n';
                return false;
            }
        }
        return true;
    }
    return false;
}

void spellings() {
    for (const PrefillRoute r : {PrefillRoute::Auto, PrefillRoute::A16, PrefillRoute::W4a8,
                                 PrefillRoute::I8, PrefillRoute::Mmq, PrefillRoute::W4a4}) {
        check(parse_prefill_route(prefill_route_name(r)) == r,
              std::string("--prefill-route round trip ") + prefill_route_name(r));
    }
    for (const DecodeRoute r : {DecodeRoute::Auto, DecodeRoute::A16, DecodeRoute::R8,
                                DecodeRoute::Vec, DecodeRoute::Mmq}) {
        check(parse_decode_route(decode_route_name(r)) == r,
              std::string("--decode-route round trip ") + decode_route_name(r));
    }
    for (const MoePath p : {MoePath::Auto, MoePath::Fused, MoePath::Generic}) {
        check(parse_moe_path(moe_path_name(p)) == p,
              std::string("--moe-path round trip ") + moe_path_name(p));
    }
    check(refuses([] { (void)parse_prefill_route("fp8"); }, {"--prefill-route", "'fp8'"}),
          "--prefill-route fp8 was accepted");
    check(refuses([] { (void)parse_decode_route("rmma"); }, {"--decode-route", "a16"}),
          "--decode-route rmma was accepted");
    check(refuses([] { (void)parse_moe_path("sparse"); }, {"--moe-path", "generic"}),
          "--moe-path sparse was accepted");
    check(refuses([] { (void)parse_prefill_route("MMQ"); }, {"--prefill-route"}),
          "spellings are case-sensitive, as every option's");
}

void engine_flags() {
    DecodePacingOptions pacing;
    MaintainerOptions m;
    const auto parse = [&](std::string_view flag, std::string value) {
        return product::parse_engine_flag(flag, [&] { return value; }, pacing, m);
    };
    check(parse("--prefill-route", "mmq") && m.prefill_route == PrefillRoute::Mmq,
          "--prefill-route mmq was not parsed");
    check(parse("--decode-route", "vec") && m.decode_route == DecodeRoute::Vec,
          "--decode-route vec was not parsed");
    check(parse("--moe-path", "generic") && m.moe_path == MoePath::Generic,
          "--moe-path generic was not parsed");
    check(MaintainerOptions{}.prefill_route == PrefillRoute::Auto &&
              MaintainerOptions{}.decode_route == DecodeRoute::Auto &&
              MaintainerOptions{}.moe_path == MoePath::Auto,
          "a pin is not auto by default");
    check(!(m == MaintainerOptions{}), "a pin does not enter MaintainerOptions' equality (the "
                                       "process-options lease compares it)");
    for (const char* flag : {"--prefill-route", "--decode-route", "--moe-path"}) {
        check(std::string_view(product::kEngineFlagsUsage).find(flag) != std::string_view::npos,
              std::string("the engine usage omits ") + flag);
    }
    check(std::string_view(product::kScoringEngineFlagsUsage).find("--decode-route") ==
                  std::string_view::npos &&
              product::is_decode_only_flag("--decode-route") &&
              !product::is_decode_only_flag("--prefill-route"),
          "a scoring program must refuse --decode-route and take --prefill-route");
}

void engine_refusals() {
    EngineOptions base;
    base.max_concurrency = 1;
    {
        EngineOptions o            = base;
        o.activations              = ActivationMode::Exact;
        o.maintainer.prefill_route = PrefillRoute::A16;
        check(refuses([&] { (void)runtime::admit_engine_options(o); },
                      {"--prefill-route a16", "--activations exact"}),
              "a prefill pin under --activations exact was accepted");
        o.maintainer.prefill_route = PrefillRoute::Auto;
        o.maintainer.decode_route  = DecodeRoute::R8;
        check(refuses([&] { (void)runtime::admit_engine_options(o); },
                      {"--decode-route r8", "--activations exact"}),
              "a decode pin under --activations exact was accepted");
        o.maintainer.decode_route = DecodeRoute::Auto;
        o.maintainer.moe_path     = MoePath::Generic;
        check(!refuses([&] { (void)runtime::admit_engine_options(o); }, {"--moe-path"}),
              "--moe-path under --activations exact was refused");
    }
    {
        EngineOptions o            = base;
        o.activations              = ActivationMode::Int8;
        o.maintainer.prefill_route = PrefillRoute::W4a4;
        check(refuses([&] { (void)runtime::admit_engine_options(o); },
                      {"--prefill-route w4a4", "--activations int8", "w4a8"}),
              "--prefill-route w4a4 under --activations int8 was accepted");
        o.maintainer.prefill_route = PrefillRoute::W4a8;
        try {
            const EngineOptions n = runtime::admit_engine_options(o);
            check(n.maintainer.prefill_route == PrefillRoute::W4a8,
                  "a pin did not pass through the normalization");
        } catch (const std::exception& error) {
            check(false,
                  std::string("--prefill-route w4a8 under int8 was refused: ") + error.what());
        }
    }
}

void ledger() {
    using ops::family_bit;
    ops::set_route_pins({PrefillRoute::Mmq, DecodeRoute::R8, MoePath::Auto});
    check(ops::route_pins() == ops::RoutePins{PrefillRoute::Mmq, DecodeRoute::R8, MoePath::Auto},
          "set_route_pins did not set the pins");
    // No call of either band: no refusal, each pin says it was not checked.
    {
        const auto report = ops::check_route_pin_coverage(ops::route_pins());
        check(report.size() == 2 && report[0].find("none checked") != std::string::npos,
              "a band without calls is not reported as unchecked");
    }
    // Calls the pin could not take: refused, naming what could.
    ops::note_prefill_pin(false, family_bit(PrefillRoute::A16) | family_bit(PrefillRoute::W4a8));
    check(refuses([] { (void)ops::check_route_pin_coverage(ops::route_pins()); },
                  {"--prefill-route mmq", "a16, w4a8"}),
          "a prefill pin that took no call was not refused with the families that could");
    // One call it took: reported, not refused.
    ops::note_prefill_pin(true, family_bit(PrefillRoute::Mmq));
    ops::note_decode_pin(true, family_bit(DecodeRoute::R8) | family_bit(DecodeRoute::A16));
    try {
        const auto report = ops::check_route_pin_coverage(ops::route_pins());
        check(report.size() == 2 && report[0].find("a16, w4a8, mmq") != std::string::npos &&
                  report[1].find("a16, r8") != std::string::npos,
              "the coverage report does not name the families met");
    } catch (const std::exception& error) {
        check(false, std::string("a pin that took calls was refused: ") + error.what());
    }
    // Setting the pins clears the ledger.
    ops::set_route_pins({PrefillRoute::Mmq, DecodeRoute::Auto, MoePath::Auto});
    check(!ops::route_pin_ledger().prefill.seen && ops::route_pin_ledger().prefill.could == 0,
          "set_route_pins left the ledger");
    ops::set_route_pins({});
    check(!ops::route_pins().any() && ops::check_route_pin_coverage(ops::route_pins()).empty(),
          "no pin reports nothing");
}

void moe_path() {
    ops::set_route_pins({});
    check(ops::moe_takes_fused(true, "") && !ops::moe_takes_fused(false, "why"),
          "auto does not take the fused kernels exactly where they run");
    ops::set_route_pins({PrefillRoute::Auto, DecodeRoute::Auto, MoePath::Generic});
    check(!ops::moe_takes_fused(true, ""), "--moe-path generic took the fused kernels");
    check(ops::route_pin_ledger().moe_choices == 1 && ops::route_pin_ledger().moe_fused_ready == 1,
          "the MoE choice is not in the ledger");
    ops::set_route_pins({PrefillRoute::Auto, DecodeRoute::Auto, MoePath::Fused});
    check(ops::moe_takes_fused(true, ""), "--moe-path fused did not take the fused kernels");
    check(refuses([] { (void)ops::moe_takes_fused(false, "top-k 4, SparseMoe takes 8"); },
                  {"--moe-path fused", "top-k 4", "generic"}),
          "--moe-path fused on a layer the fused kernels cannot run was accepted");
    // A model without a choice: refused at the coverage check.
    ops::set_route_pins({PrefillRoute::Auto, DecodeRoute::Auto, MoePath::Generic});
    check(refuses([] { (void)ops::check_route_pin_coverage(ops::route_pins()); },
                  {"--moe-path generic", "nothing to pin"}),
          "--moe-path on a model without a MoE choice was accepted");
    ops::set_route_pins({});
}

void shape_activations() {
    ops::set_route_pins({});
    check(!ops::pinned_shape_quantized(512, PrefillRoute::W4a4) &&
              !ops::pinned_shape_quantized(1, PrefillRoute::W4a4),
          "no pin decided a registered shape's activations");
    ops::set_route_pins({PrefillRoute::A16, DecodeRoute::Auto, MoePath::Auto});
    check(ops::pinned_shape_quantized(17, PrefillRoute::W4a4) == false &&
              !ops::pinned_shape_quantized(16, PrefillRoute::W4a4),
          "--prefill-route a16 must keep BF16 from 17 tokens and leave decode widths");
    ops::set_route_pins({PrefillRoute::W4a4, DecodeRoute::A16, MoePath::Auto});
    check(ops::pinned_shape_quantized(512, PrefillRoute::W4a4) == true &&
              !ops::pinned_shape_quantized(512, std::nullopt) &&
              ops::pinned_shape_quantized(4, PrefillRoute::W4a4) == false,
          "--prefill-route w4a4 / --decode-route a16 on a registered NVFP4 shape");
    ops::note_shape_activations(512, PrefillRoute::W4a4, true);
    ops::note_shape_activations(4, PrefillRoute::W4a4, false);
    const auto l = ops::route_pin_ledger();
    check(l.prefill.took && l.decode.took &&
              (l.prefill.could & ops::family_bit(PrefillRoute::W4a4)) != 0,
          "a registered shape's call is not in the ledger");
    ops::set_route_pins({});
}

} // namespace

int main() {
    spellings();
    engine_flags();
    engine_refusals();
    ledger();
    moe_path();
    shape_activations();
    std::cout << (failures ? "FAIL" : "OK") << " route_pins\n";
    return failures ? 1 : 0;
}
