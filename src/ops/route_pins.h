#pragma once

// ninferno::ops - route pins: a maintainer's test pin of a kernel family (--prefill-route,
// --decode-route, --moe-path; option_enums.h), so that an A/B run can check that the automatic
// selection takes the better family. Auto everywhere is the release's selection, untouched.
//
// A pin is process-wide, as the route tables are: an Engine sets it at construction while no other
// Engine of the process is alive (runtime/engine/engine.cpp), and the ops read it when they resolve
// a route, eagerly and inside graph captures alike. A pinned family takes every call of its band
// that it can compute; a call it cannot compute (another format, a K or alignment its tiles do not
// take, a width past its kernel, a policy without its activations) keeps the automatic route, and
// the ledger below records both, so that the startup refuses a pin that took no call and names the
// families that could (check_route_pin_coverage). Nothing here times anything or reads the
// environment.

#include "ninferno/option_enums.h"

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

namespace ninferno::ops {

struct RoutePins {
    PrefillRoute prefill = PrefillRoute::Auto;
    DecodeRoute decode   = DecodeRoute::Auto;
    MoePath moe          = MoePath::Auto;

    [[nodiscard]] bool any() const noexcept {
        return prefill != PrefillRoute::Auto || decode != DecodeRoute::Auto || moe != MoePath::Auto;
    }

    bool operator==(const RoutePins&) const = default;
};

// The bands: a prefill pin applies to calls of kPrefillPinMinTokens tokens or more (the prefill
// route profile's band, prefill_routes.h), a decode pin to calls of 1..kDecodePinMaxTokens (decode
// and speculative verify).
inline constexpr std::int32_t kPrefillPinMinTokens = 17;
inline constexpr std::int32_t kDecodePinMaxTokens  = 16;

// Sets the process's pins and clears the ledger.
void set_route_pins(const RoutePins& pins) noexcept;
[[nodiscard]] RoutePins route_pins() noexcept;

// The families a pin's band met since set_route_pins, as bit masks over the enum's values
// (bit 1 << static_cast<int>(family)): `seen` whether a call of the band was resolved, `took`
// whether the pinned family took one, `could` the families that could compute some call.
struct PinBand {
    bool seen           = false;
    bool took           = false;
    std::uint32_t could = 0;
};

struct RoutePinLedger {
    PinBand prefill;
    PinBand decode;
    // MoE layers whose family chose between its fused kernels and the generic MoE Op, and how
    // many of them could take the fused kernels.
    std::uint32_t moe_choices     = 0;
    std::uint32_t moe_fused_ready = 0;
};

[[nodiscard]] constexpr std::uint32_t family_bit(PrefillRoute f) noexcept {
    return 1U << static_cast<unsigned>(f);
}

[[nodiscard]] constexpr std::uint32_t family_bit(DecodeRoute f) noexcept {
    return 1U << static_cast<unsigned>(f);
}

// One call of a pinned band: whether the pinned family took it, and the families that could.
void note_prefill_pin(bool took, std::uint32_t could) noexcept;
void note_decode_pin(bool took, std::uint32_t could) noexcept;
// One MoE layer's choice under any pin (`fused_ready`: its fused kernels could run it).
void note_moe_choice(bool fused_ready) noexcept;
[[nodiscard]] RoutePinLedger route_pin_ledger() noexcept;

// The MoE path of one layer whose fused kernels can (`fused_ready`) or cannot run it: true for the
// fused kernels. Auto takes them where they can run (the release's choice); Generic never does;
// Fused refuses (std::invalid_argument naming `why_not`, what keeps the fused kernels out, and that
// the generic MoE runs it) a layer they cannot run. Notes the choice in the ledger.
[[nodiscard]] bool moe_takes_fused(bool fused_ready, const std::string& why_not);

// For a registered shape's own choice between BF16 activations and its quantized kernel at T
// tokens (NVFP4: A4; FP8: A8, which no pin names: `quantized` empty): the pin's answer, nullopt
// where no pin decides (Auto, a family the shape has no kernel of, a width outside the band).
[[nodiscard]] std::optional<bool>
pinned_shape_quantized(std::int32_t tokens, std::optional<PrefillRoute> quantized) noexcept;
// The ledger entry of such a call: `quantized` whether it ran its quantized kernel.
void note_shape_activations(std::int32_t tokens, std::optional<PrefillRoute> family,
                            bool quantized) noexcept;

// "a16, w4a8" for the families of a mask.
[[nodiscard]] std::string prefill_families(std::uint32_t mask);
[[nodiscard]] std::string decode_families(std::uint32_t mask);

// After the startup resolved its routes (the kernel warm-up's prefill widths and decode graphs,
// the served Program's graphs): throws std::invalid_argument when a pin's band had calls and the
// pin took none of them, naming the families that could compute them, or when --moe-path was given
// and no MoE layer chose a path. Returns one report line per pin given (what took it; a band no
// startup call reached says so).
[[nodiscard]] std::vector<std::string> check_route_pin_coverage(const RoutePins& pins);

} // namespace ninferno::ops
